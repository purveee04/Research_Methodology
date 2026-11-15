# Member 3 — Experiment Tracking and Final Evaluation Status

## Completed
- M3.5 log reader, validation plots, summary outputs, loss-identity checks, and evidence audit.
- Final A/B runs completed for 10,000 steps at 96×96 using the team-reviewed corrected split.
- Evaluation-only inference completed once for each frozen best-validation-macro-F1 classifier on the same 1,731-image held-out test partition.
- Final metrics, per-class metrics, confusion matrices, comparison plot, and pre-test provenance are stored in `../../results/final_ab_2026-11-15/`.
- The final test evaluation notebook, script, and instructions are stored in `final_test_kit/`.

## Interpretation and limitations
- B exceeded A on test accuracy, macro-F1, balanced accuracy, and weighted-F1 in this matched single-seed result. This does not establish a general effect across seeds.
- Pseudo-label acceptance was very low: A accepted 89/200,000 logged examples (0.0445%); B accepted 96/200,000 (0.0480%). Do not attribute the result mainly to pseudo-labelling.
- B's test ECE was higher (0.0569) than A's (0.0415); calibration did not improve on every measure.
- Per-class errors remain substantial. Neither model predicted class 4 correctly on the held-out set.
- Generator structural/gradient checks passed, while visual and L1 diagnostics did not establish successful denoising or reliable morphology preservation.
- The 253 geometric-overlap pairs were grouped conservatively and were not all manually verified as duplicate galaxies.
- Test inference was evaluation-only; test results were not used for model selection or tuning.

## Reproduction inputs
The training dataset is not included. Its SHA-256, the corrected split NPZ SHA-256, held-out test-index hash, selected steps, and training-code commit are recorded in `PRETEST_VERIFICATION.json` and `FINAL_TEST_RESULTS.json`.
The selected classifier checkpoint hashes and full metric values are in `FINAL_TEST_RESULTS.json`.
Large model checkpoints are intentionally not stored in this repository; retain them in Kaggle or the team's artifact storage.
