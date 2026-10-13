# DiCo-NLI — Working Repository

Working copy for **SemEval 2027 Task 2: Directional-Consistent Fine-Grained Natural Language Inference (DiCo-NLI)**.

This repository contains the task-relevant resources needed to build, run, and locally evaluate a system. Organizer-only material (website, advertising, LaTeX, references) is intentionally excluded.

## Contents

| Path | Description |
|------|-------------|
| [`starter_kit/`](starter_kit/) | Minimal Hugging Face fine-tuning pipeline, prediction writer, optional official-scorer integration, dependency pins, and SLURM templates. |
| [`final_data/`](final_data/) | Organizer-provided training and development data per track. |
| [`trial_data/`](trial_data/) | Sample/trial data with scored random and majority baselines. |
| [`evaluation_functions/`](evaluation_functions/) | Official scorer and submission validator. |

## Tracks

| Track | Name | Description |
|-------|------|-------------|
| 1 | English | Monolingual English phrase-pair NLI. |
| 2 | Spanish | Monolingual Spanish phrase-pair NLI. |
| 3 | Basque | Monolingual Basque phrase-pair NLI. |
| 4 | Mixed multilingual | Premise and hypothesis may appear in any EN/ES/EU combination. |

## Labels

```text
EQUIVALENCE
FORWARD_ENTAILMENT
BACKWARD_ENTAILMENT
NEGATIVE_OTHER
```

## Quick start

```bash
pip install -r starter_kit/requirements.txt
```

Local scoring with the official scorer (Python 3.10+):

```bash
python3 -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions results/my_dev_track1_predictions.csv \
  --output-dir results/dev_track1_scores
```

See [`starter_kit/README.md`](starter_kit/README.md) and [`evaluation_functions/README.md`](evaluation_functions/README.md) for details.

## Evaluation

The official scorer reports three main scores per track:

- **Weighted F1** — item-level label-prediction quality.
- **SoftCons** — directional self-consistency under the reversal operator.
- **HardCons** — paired directional correctness under reversal.

## Resources

- Competition website: https://www.codabench.org/competitions/18038/
- LREC 2026 pilot paper: https://lrec.elra.info/lrec2026-main-423
- PhrasIS: https://doi.org/10.1093/jigpal/jzae037
