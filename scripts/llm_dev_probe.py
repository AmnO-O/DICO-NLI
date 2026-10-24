"""Ask a large LLM to label the dev set, to find out whether it knows something
the fine-tuned cross-encoder does not.

Modes, so the API call and the scoring stay separable:

    export   write one record per dev row (instance_id, texts, prompt)
    models   list the model ids this key can reach
    call     send the prompts to the Gemini API, store the raw answers
    score    score the parsed predictions against the official reference

Examples
--------
    python scripts/llm_dev_probe.py export
    python scripts/llm_dev_probe.py models
    python scripts/llm_dev_probe.py call --model gemini-3.5-flash-lite --workers 8
    python scripts/llm_dev_probe.py call --limit 20 --dry-run      # inspect a payload
    python scripts/llm_dev_probe.py call --resume                  # retry only the failures
    python scripts/llm_dev_probe.py score
    python scripts/llm_dev_probe.py score --compare results/<run>/predictions_dev_ensemble.csv

The API key is read from GEMINI_API_KEY (or GOOGLE_API_KEY). Nothing extra is
installed: requests are plain urllib, so this runs in a bare environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_functions import score_files  # noqa: E402

REFERENCE = ROOT / "final_data" / "dev" / "dico_nli_dev_track1_reference.csv"
OUT_DIR = ROOT / "results" / "llm_probe"
PROMPTS = OUT_DIR / "dev_prompts.jsonl"
RAW = OUT_DIR / "dev_raw.jsonl"
PREDICTIONS = OUT_DIR / "dev_predictions.csv"

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ENDPOINT_LIST = "https://generativelanguage.googleapis.com/v1beta/models"
BASELINE_ENSEMBLE = 0.8424
BASELINE_SINGLE = 0.8382

PROMPT = """You are labelling a pair of short phrases taken from news headlines and image captions. Each phrase is an isolated fragment: do not assume a surrounding sentence that is not there.

Choose exactly one label.

EQUIVALENCE - the two phrases mean the same thing, so each entails the other.
FORWARD_ENTAILMENT - phrase 1 entails phrase 2, but phrase 2 does not entail phrase 1.
BACKWARD_ENTAILMENT - phrase 2 entails phrase 1, but phrase 1 does not entail phrase 2.
NEGATIVE_OTHER - neither phrase entails the other. This includes pairs that are related, similar or share most of their words without being equivalent, and pairs that contradict.

Phrase 1: {text1}
Phrase 2: {text2}

Reply with JSON only, no prose: {{"label": "<one of the four labels>"}}"""


# --------------------------------------------------------------------------- helpers
def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment")
    return key


# --------------------------------------------------------------------------- export
def export() -> None:
    dev = pd.read_csv(REFERENCE, dtype=str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("w", encoding="utf-8") as handle:
        for _, row in dev.iterrows():
            record = {
                "instance_id": row["instance_id"],
                "text1": row["text1"],
                "text2": row["text2"],
                "prompt": PROMPT.format(text1=row["text1"], text2=row["text2"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(dev)} prompts -> {PROMPTS}")


# --------------------------------------------------------------------------- models
def models(pattern: str | None) -> None:
    key = api_key()
    url = ENDPOINT_LIST + "?pageSize=200"
    request = urllib.request.Request(url, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    names = [entry["name"].split("/", 1)[1] for entry in payload.get("models", [])]
    names = [name for name in names if pattern is None or pattern.lower() in name.lower()]
    for name in sorted(names):
        print(name)
    print(f"\n{len(names)} model(s)")


# ----------------------------------------------------------------------------- call
def parse_label(text: str) -> str | None:
    """Pull exactly one label out of the reply, tolerating fences and prose."""
    text = re.sub(r"```(?:json)?", "", text)
    try:
        payload = json.loads(text.strip())
        if isinstance(payload, dict) and str(payload.get("label", "")).upper() in LABELS:
            return str(payload["label"]).upper()
    except (json.JSONDecodeError, AttributeError):
        pass

    hits = [label for label in LABELS if label in text.upper()]
    # A verbose answer can name two labels ("not EQUIVALENCE, but NEGATIVE_OTHER"),
    # so accept only an unambiguous single hit.
    return hits[0] if len(hits) == 1 else None


def request_once(model: str, prompt: str, key: str, timeout: float = 60.0) -> str:
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 64},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"no candidates in reply: {str(payload)[:300]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def call_one(model: str, key: str, record: dict, retries: int = 4) -> dict:
    delay = 2.0
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            text = request_once(model, record["prompt"], key)
            return {"instance_id": record["instance_id"], "raw": text,
                    "label": parse_label(text), "error": None}
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:  # noqa: BLE001 - network failures vary
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return {"instance_id": record["instance_id"], "raw": "", "label": None, "error": last_error}


def call(model: str, workers: int, limit: int | None, dry_run: bool, resume: bool) -> None:
    if not PROMPTS.exists():
        raise SystemExit(f"{PROMPTS} is missing - run `export` first.")
    records = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    if limit:
        records = records[:limit]

    if dry_run:
        print(json.dumps({"model": model, "prompt": records[0]["prompt"]}, indent=2)[:1500])
        return

    key = api_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    already: dict[str, dict] = {}
    if resume and RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("label"):
                already[row["instance_id"]] = row
    todo = [record for record in records if record["instance_id"] not in already]
    print(f"{len(already)} already answered, {len(todo)} to request")

    fresh = {}
    started = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(call_one, model, key, record): record for record in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                fresh[row["instance_id"]] = row
                if done % 25 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}  {time.time() - started:.0f}s")

    merged = {**already, **fresh}
    ordered = [merged[record["instance_id"]] for record in records if record["instance_id"] in merged]

    with RAW.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unparsed = [row for row in ordered if row["label"] is None]
    keep = [row for row in ordered if row["label"] is not None]
    pd.DataFrame(keep)[["instance_id", "label"]].to_csv(PREDICTIONS, index=False, lineterminator="\n")
    (OUT_DIR / "dev_predictions.meta.json").write_text(
        json.dumps({"model": model, "rows": len(keep), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                   indent=2),
        encoding="utf-8",
    )

    print(f"\nparsed {len(keep)}/{len(ordered)} -> {PREDICTIONS}")
    for row in unparsed[:5]:
        print(f"  unparsed {row['instance_id']}: {row['error'] or row['raw'][:120]!r}")
    if unparsed and all((row["error"] or "").startswith("HTTP 404") for row in unparsed):
        print(f"  every request returned 404 - `{model}` is probably not a valid model id; "
              f"run `python scripts/llm_dev_probe.py models`")
    elif unparsed:
        print("  re-run with --resume after fixing the cause")


# ---------------------------------------------------------------------------- score
def score(compare: Path | None, predictions: Path, fill_missing: str | None) -> None:
    if not predictions.exists():
        raise SystemExit(f"{predictions} is missing - run `call` first.")

    meta_path = predictions.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"predictions : {predictions.name}  (model {meta.get('model')}, {meta.get('generated_at')})")
    else:
        print(f"WARNING: {predictions.name} has no .meta.json - it was not written by this script's "
              f"`call` mode, so check that it is not a copy of the gold labels")

    gold = pd.read_csv(REFERENCE, dtype=str)[["instance_id", "label"]].rename(columns={"label": "gold"})
    pred = pd.read_csv(predictions, dtype=str)
    if "instance_id" not in pred or "label" not in pred:
        raise SystemExit(f"{predictions} must have columns instance_id,label")

    missing = sorted(set(gold.instance_id) - set(pred.instance_id))
    extra = sorted(set(pred.instance_id) - set(gold.instance_id))
    if extra:
        print(f"WARNING: {len(extra)} prediction ids are not in the reference, ignored "
              f"(e.g. {extra[0]})")
    if missing:
        (OUT_DIR / "missing_ids.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"WARNING: {len(missing)} reference rows have no prediction "
              f"(ids -> {OUT_DIR / 'missing_ids.txt'})")
        if fill_missing is None:
            raise SystemExit(
                f"refusing to score an incomplete file; re-run `call --resume` to fill the gap, "
                f"or pass --fill-missing {LABELS[-1]} to score anyway"
            )
        pred = pd.concat(
            [pred, pd.DataFrame({"instance_id": missing, "label": fill_missing})], ignore_index=True
        )
        predictions = OUT_DIR / "dev_predictions_filled.csv"
        pred[["instance_id", "label"]].to_csv(predictions, index=False, lineterminator="\n")
        print(f"filled file -> {predictions}")

    report = score_files(REFERENCE, predictions, output_dir=OUT_DIR / "official_scores")

    print(f"weighted_f1 : {report.weighted_f1:.4f}")
    print(f"soft_cons   : {report.soft_cons:.4f}")
    print(f"hard_cons   : {report.hard_cons:.4f}")
    print(f"baseline    : ensemble {BASELINE_ENSEMBLE:.4f} / single {BASELINE_SINGLE:.4f}")
    print(f"delta       : {report.weighted_f1 - BASELINE_ENSEMBLE:+.4f} vs ensemble")

    merged = gold.merge(pred[["instance_id", "label"]].rename(columns={"label": "pred"}),
                        on="instance_id", how="left")

    print("\nper label        support   predicted   precision  recall     f1")
    for label in LABELS:
        tp = int(((merged.gold == label) & (merged.pred == label)).sum())
        support = int((merged.gold == label).sum())
        predicted = int((merged.pred == label).sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"  {label:22} {support:5}   {predicted:8}   {precision:8.3f}  {recall:6.3f}  {f1:6.3f}")

    cell = int(((merged.gold == "NEGATIVE_OTHER") & (merged.pred == "EQUIVALENCE")).sum())
    reverse = int(((merged.gold == "EQUIVALENCE") & (merged.pred == "NEGATIVE_OTHER")).sum())
    print(f"\nNEG -> EQ : {cell}    EQ -> NEG : {reverse}")

    print("\nconfusion (rows = gold, cols = predicted)")
    table = pd.crosstab(merged.gold, merged.pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
    table.index = [label[:6] for label in LABELS]
    table.columns = [label[:6] for label in LABELS]
    print(table.to_string())

    if compare is None:
        return
    if not compare.exists():
        print(f"\n(compare file not found: {compare})")
        return

    model = pd.read_csv(compare, dtype=str)[["instance_id", "label"]].rename(columns={"label": "model"})
    both = merged.merge(model, on="instance_id", how="left").dropna(subset=["model"])
    llm_right = both["pred"] == both["gold"]
    model_right = both["model"] == both["gold"]
    print(f"\nagreement with {compare.name}  ({len(both)} shared rows)")
    print(f"  both right             : {int((llm_right & model_right).sum())}")
    print(f"  LLM right, model wrong : {int((llm_right & ~model_right).sum())}")
    print(f"  model right, LLM wrong : {int((~llm_right & model_right).sum())}")
    print(f"  both wrong             : {int((~llm_right & ~model_right).sum())}")

    gained = both[llm_right & ~model_right]
    if len(gained):
        print("\n  rows the LLM fixes (gold / what the model said):")
        for (gold_label, model_label), group in gained.groupby(["gold", "model"]):
            print(f"    gold {gold_label:20} model {model_label:20} x{len(group)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["export", "models", "call", "score"])
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pattern", default=None, help="filter for the models mode")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--fill-missing", default=None, choices=LABELS)
    args = parser.parse_args()

    if args.mode == "export":
        export()
    elif args.mode == "models":
        models(args.pattern)
    elif args.mode == "call":
        call(args.model, args.workers, args.limit, args.dry_run, args.resume)
    else:
        score(args.compare, args.predictions, args.fill_missing)


if __name__ == "__main__":
    main()
