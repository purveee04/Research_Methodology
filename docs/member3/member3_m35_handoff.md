# GC-SWGAN Member 3 — Experiment Tracking and A/B Evaluation Handoff

**Status (15 November 2026):** Full matched A/B training and one evaluation-only test inference
per selected classifier are complete. Final evidence is in `../../results/final_ab_2026-11-15/`.

## Verified data and model selection
- Dataset SHA-256: `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.
- Corrected split NPZ SHA-256: `1ea636a1cd3df30fdd6b155b06896692198e141d536d28a60899f17955c1ede8`.
- Held-out test set: 1,731 images; index SHA-256 recorded in `PRETEST_VERIFICATION.json`.
- A selected checkpoint: step 8,500, validation macro-F1 0.1918.
- B selected checkpoint: step 10,000, validation macro-F1 0.2797.
- Selection used validation macro-F1. Test inference was frozen, evaluation-only, and did not tune parameters.
- Training code commit: `25882a893fe9980acc430008d4869b9a411bb3e0`.

## Test metrics
| Metric | A | B |
|---|---:|---:|
| Accuracy | 0.2577 | 0.3200 |
| Macro-F1 | 0.1862 | 0.2689 |
| Balanced accuracy | 0.2321 | 0.3030 |
| Weighted-F1 | 0.2043 | 0.2898 |
| ECE | 0.0415 | 0.0569 |
| Brier score | 0.8135 | 0.8014 |

B scored higher on the four classification metrics in this matched, single-seed experiment.
This should be described as an observed result, not a statistically established general effect.
ECE was higher for B. Per-class details and confusion matrices are included in the results directory.

## Limitations
- Pseudo-label acceptance was low: A 89/200,000 (0.0445%); B 96/200,000 (0.0480%). Do not attribute B's difference mainly to pseudo-labelling.
- No correct class-4 predictions were made by either model on the test set; per-class errors are substantial.
- Generator structural and gradient checks passed, but visual and reconstruction diagnostics do not establish effective denoising or morphology preservation.
- 253 possible sky-footprint overlap pairs were conservatively grouped, not individually confirmed as duplicate galaxies.
- One seed was used. Further seeds would be needed to estimate variability or establish a general effect.

## Reproduction assets
- Final comparison, per-class CSV, full JSON results, pre-test verification, test-start record, and plots: `../../results/final_ab_2026-11-15/`.
- Evaluation-only Kaggle notebook and script: `final_test_kit/`.
- Dataset and large trained checkpoints are not stored in Git. Retain the dataset in its approved storage and checkpoints in Kaggle/team artifact storage.
