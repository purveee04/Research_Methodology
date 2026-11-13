# Member 3 — Evaluation and loss audit (Prompt 2)

**Project:** GC-SWGAN image-to-image galaxy classifier, Member 3 deliverable.  
**Prepared:** 2026-11-12. **Status:** New evaluation utilities tested on synthetic data; integrated TensorFlow loss audit pending execution in the user's Kaggle clone.

## Scope, sources, and limitations
- Used Member 2's 12-Nov-2026 handoff and the attached Member 3 data-audit notebook to understand the intended interfaces and split policy.
- I could **not download the current GitHub source** in my execution environment (network resolution/retrieval unavailable). Accordingly, **no line-by-line audit of the actual current `training_module.py` and `train_gc_swgan_i2i.py` is claimed here**. `member3_source_audit.py` extracts evidence from the actual cloned files when run on Kaggle; reading the indicated lines and running the existing TensorFlow tests remain required.
- No HDF5 files or completed Experiment A/B checkpoints were available locally. No actual galaxy-model accuracy, sample reconstructions, optimizer persistence, or Gradient Penalty behavior was measured here.
- No Generator, Critic, training-loop, or split code was modified.

## Confirmed information from the project handoff (not newly verified)
- Reported training code: `training_module.py`, runner: `train_gc_swgan_i2i.py`, tests: `test_training_module.py`.
- Reported Member 2 tests: 79 passed in Kaggle, but earlier session; a five-step **debug** Experiment A completed at 96×96. These results do not establish final accuracy or full training viability.
- Intended updates: labelled cross-entropy plus high-confidence pseudo-labels; WGAN-GP on clean real versus generated images; Generator A adversarial only, B adversarial + λrec×L1.
- Data-audit report supplied in the conversation: 17,675 images, three near-duplicate-looking train/eval galaxy pairs, and 63 candidate 192px sky-footprint overlaps. **Leakage not cleared; the existing random split must not be called approved.**

## Member 3 evaluation implementation
`data_eval.py` provides:
- `evaluate_probabilities(y_true, probabilities)` with strict shape/probability validation; accuracy, macro-F1 averaged across **all 10 classes**, macro-F1 over **classes present in ground truth** (both explicitly labeled), balanced accuracy, weighted F1, per-class precision/recall/F1/support, 10×10 confusion matrix. Beware: the existing runner may define macro-F1 differently; compare conventions before using numbers side-by-side.
- `plot_confusion_matrix`, `plot_log_curve`, `plot_pseudo_acceptance` and `compare_validation_runs`.
- `fair_config_diff` flags **unexpected config differences**; always review the allowed set. It cannot prove that weights are initialized identically or the data path resolves to the same bytes.
- All plots are derived from provided arrays/CSV files; no training or test selection is performed.

## Tests executed
`python -m unittest test_data_eval -v` — **9 tests, 9 passed, 0 failures** locally, on synthetic data. Tests cover perfect/mistaken predictions, missing classes, bad inputs, CSV plot generation, pseudo-label rates, validation comparisons, and config-diff detection.

## Runtime loss-verification checklist (M3.4)
Fill **PASS / FAIL / NEEDS REVIEW**, with source line numbers and/or test logs. Static name matches alone are **not** PASS.

| Requirement | Evidence to capture on Kaggle | Current verdict |
|---|---|---|
| Critic compares clean real vs generated | Inspect actual train step inputs and gradient path; unit test with controlled images | NEEDS REVIEW |
| WGAN-GP penalty and coefficient | Inspect interpolated-image gradient, norm penalty, sign and multiplication; test finite gradients | NEEDS REVIEW |
| Generator A adversarial only | Test A reconstruction contribution = exactly zero and total G loss equals G adv | NEEDS REVIEW |
| Generator B includes weighted L1 | Test B G total = G adv + λrec×mean(abs(generated-clean)), λrec > 0 | NEEDS REVIEW |
| True labels from labelled train only | Trace labelled index selection and supervised CE input; inspect batch assertion | NEEDS REVIEW |
| Pseudo-labels from unlabelled train only | Trace index memberships; threshold and mask; reject val/test IDs | NEEDS REVIEW |
| Models and optimizer objects persist | Check objects constructed outside per-batch step; optimizer iterations/slots after consecutive steps | NEEDS REVIEW |
| A/B same initial conditions | Compare saved source-commit, image hashes, exact split indices, seeds, optimizer configs, LR, resolution, number of steps, weights initialization | NEEDS REVIEW |
| Final test untouched during tuning | Inspect runner's test use and saved decisions; run tests only after checkpoint/parameters fixed | NEEDS REVIEW |

### Potential integration risks to review, not confirmed defects
1. **Macro-F1 convention:** 10-class macro-F1 (including classes absent from a tiny synthetic set) may differ from sklearn macro-F1 calculated on observed classes. This can alter comparisons for small diagnostic batches, not necessarily full validation sets.
2. **CSV schema:** The plotting module recognizes common columns (`step`, `val_acc`, `val_macro_f1`, etc.). If the real runner logs other names, inspect the CSV header and add aliases. The module fails loudly rather than fabricating plots.
3. **Fair comparison:** Loss mode change may require matching initial checkpoints and random streams. Merely matching a seed/config doesn't independently prove identical starting weights.
4. **Leakage:** The three visually suspicious pairs and geometric overlaps require an approved grouped split or documented exclusion policy before full runs. Do not use the original split for final scores without resolving this issue.
5. **External cutouts:** Keep omitted until dimensions and overlap provenance are verified.

## Kaggle execution
1. Open `member3_evaluation_kaggle.ipynb` in Kaggle.
2. Attach `member3_prompt2_package.zip` as a private Kaggle Dataset (Input → Upload → New Dataset), or upload the four `.py` files individually.
3. Run setup cell, then the synthetic tests, source audit, and demonstration plots.
4. Run existing repo TensorFlow tests. Document version, pass/fail, and stack traces. If GPU is inaccessible, basic unit tests may run on CPU.
5. Once actual training CSVs exist, set `VAL_A_CSV`, `VAL_B_CSV`, `LOSS_A_CSV` to their **actual paths** and generate validation-only plots. Do not evaluate the test set for model selection.
6. Save all artifacts under `/kaggle/working/member3_eval_output/` and download the produced ZIP.

## Deliverables
`data_eval.py`, `test_data_eval.py`, `member3_source_audit.py`, `member3_evaluation_kaggle.ipynb`, this report. A ZIP bundles these files.

## Still required before Prompt 3 is complete
- Review/resolve M3.1/M3.2 sky-overlap findings, freeze the same approved split for both runs.
- Kaggle-source line audit + real TensorFlow unit tests.
- Team approval for sigma, τ, pseudo-label policy, 96 vs 192px, steps, and λrec.
- Identical A/B initial conditions with configurations and metrics captured from real runs.
- Real final test evaluation only after selecting checkpoints/configuration based on validation.
- GitHub branch/PR and reviewed integration; no `main` edits were made here.
