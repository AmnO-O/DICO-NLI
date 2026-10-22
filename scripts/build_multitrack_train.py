"""Build combined multilingual training sets from the four DiCo-NLI tracks.

Every track ships the *same* 1,760 underlying decisions rendered in a different
language pair: track 1 en-en, track 2 es-es, track 3 eu-eu, and track 4 the six
cross-lingual directions. Concatenating them multiplies the number of training
rows without touching the label space.

The train/dev split happens at the decision level, so this cannot leak dev: the
script asserts that track 1 dev shares zero ``pair_id`` with any train track.

Outputs (regenerable, so they are gitignored):

    <out-dir>/dico_nli_train_multilingual.csv   tracks 1-3   9,126 rows
    <out-dir>/dico_nli_train_multitrack.csv     tracks 1-4  27,378 rows

Usage:
    python scripts/build_multitrack_train.py
    python scripts/build_multitrack_train.py --data-root /kaggle/input/... --out-dir /kaggle/working/derived
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABELS = {"FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "EQUIVALENCE", "NEGATIVE_OTHER"}

# How many directed renderings each decision has inside a track. Tracks 1-3 are
# monolingual (one rendering per direction), track 4 carries all six ordered
# language pairs.
DIRECTIONS_PER_TRACK = {1: 1, 2: 1, 3: 1, 4: 6}

COLUMNS = [
    "instance_id",
    "pair_id",
    "text1",
    "text2",
    "text1_lang",
    "text2_lang",
    "label",
    "track",
]


def load_track(data_root: Path, track: int) -> pd.DataFrame:
    path = data_root / "train" / f"dico_nli_train_track{track}_participant_labeled.csv"
    df = pd.read_csv(path, dtype=str)
    df["track"] = track
    for column in ("text1_lang", "text2_lang"):
        if column not in df.columns:
            raise ValueError(f"{path}: missing {column}")
    return df[COLUMNS]


def check_track(df: pd.DataFrame, track: int) -> None:
    """Assert the structural invariant the dataset is built on."""
    if not set(df["label"]) <= LABELS:
        raise ValueError(f"track {track}: unexpected labels {set(df['label']) - LABELS}")

    repeats = df["pair_id"].map(df["pair_id"].value_counts())
    expected_singleton = DIRECTIONS_PER_TRACK[track]
    expected_paired = 2 * expected_singleton

    neg = df["label"] == "NEGATIVE_OTHER"
    if not (repeats[neg] == expected_singleton).all():
        raise ValueError(f"track {track}: a NEG row is not a singleton rendering")
    if not (repeats[~neg] == expected_paired).all():
        raise ValueError(f"track {track}: a non-NEG row is not a doubled rendering")


def check_disjoint_from_dev(data_root: Path, train: pd.DataFrame) -> None:
    dev = pd.read_csv(data_root / "dev" / "dico_nli_dev_track1_reference.csv", dtype=str)
    shared = set(dev["pair_id"].dropna()) & set(train["pair_id"].dropna())
    if shared:
        raise ValueError(f"{len(shared)} pair_ids leak from train into track 1 dev")


def build(data_root: Path, tracks: tuple[int, ...]) -> pd.DataFrame:
    frames = [load_track(data_root, track) for track in tracks]
    for frame, track in zip(frames, tracks):
        check_track(frame, track)

    combined = pd.concat(frames, ignore_index=True)
    if combined["instance_id"].duplicated().any():
        raise ValueError("duplicate instance_id across tracks")
    check_disjoint_from_dev(data_root, combined)
    return combined.sort_values(["track", "instance_id"], kind="stable").reset_index(drop=True)


def describe(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}: {len(df):,} rows, {df['pair_id'].nunique():,} decisions")
    per_track = df.groupby(["track", "text1_lang", "text2_lang"]).size()
    for (track, lang1, lang2), count in per_track.items():
        print(f"   track{track}  {lang1}-{lang2}  {count:6,} rows")
    print("   labels: " + "  ".join(f"{k}={v:,}" for k, v in df["label"].value_counts().items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("final_data"),
                        help="Directory holding the released train/ and dev/ folders.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the combined CSVs. Defaults to <data-root>/train. "
                             "On Kaggle pass a writable path such as /kaggle/working/derived.")
    args = parser.parse_args()

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir or (data_root / "train")
    out_dir.mkdir(parents=True, exist_ok=True)

    multilingual = build(data_root, (1, 2, 3))
    describe("multilingual (tracks 1-3)", multilingual)
    path = out_dir / "dico_nli_train_multilingual.csv"
    multilingual.to_csv(path, index=False, lineterminator="\n")
    print(f"   -> {path}")

    multitrack = build(data_root, (1, 2, 3, 4))
    describe("multitrack (tracks 1-4)", multitrack)
    path = out_dir / "dico_nli_train_multitrack.csv"
    multitrack.to_csv(path, index=False, lineterminator="\n")
    print(f"   -> {path}")

    print("\nchecks passed: label space, singleton/pair invariant per track, no dev leakage")


if __name__ == "__main__":
    main()
