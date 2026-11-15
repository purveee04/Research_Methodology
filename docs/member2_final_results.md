# Member 2 — Final Training and Evaluation Handoff

**Status:** Both full A/B runs and the final held-out test evaluation are complete.

## Training protocol

- Dataset: Galaxy10 NoDuplicated; SHA-256 `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.
- Corrected split NPZ SHA-256: `1ea636a1cd3df30fdd6b155b06896692198e141d536d28a60899f17955c1ede8`.
- Split counts: 14,217 training pool (1,777 labelled; 12,440 unlabelled), 1,727 validation, 1,731 test.
- Resolution 96×96; 10,000 steps; seed 0; 20 unlabelled + 1 labelled per step; sigma 0.1; pseudo-label threshold 0.95; refresh every step; pseudo-loss weight 1.0; GP weight 10; learning rate 1e-4.
- A: adversarial Generator objective only. B: same settings plus L1 reconstruction loss at lambda_rec=1.0.
- External cutouts excluded. Validation macro-F1 selected the classifier checkpoint: A step 8,500; B step 10,000. Test evaluation used the same frozen test partition and did not tune or select checkpoints.
- Training code commit recorded by the evaluation preflight: `25882a893fe9980acc430008d4869b9a411bb3e0`.

## Results

| Metric | A | B |
|---|---:|---:|
| Validation macro-F1 at selected step | 0.1918 | 0.2797 |
| Test accuracy | 0.2577 | 0.3200 |
| Test macro-F1 | 0.1862 | 0.2689 |
| Test balanced accuracy | 0.2321 | 0.3030 |
| Test weighted-F1 | 0.2043 | 0.2898 |
| Test ECE | 0.0415 | 0.0569 |
| Test Brier score | 0.8135 | 0.8014 |

B scored higher on the four primary classification metrics in this one matched-seed comparison. The result should be described as observed in this setup, not as a statistically established general improvement. ECE was worse for B.

## Pseudo-labeling and Generator findings

- A accepted 89/200,000 logged candidate examples (0.0445%); B accepted 96/200,000 (0.0480%). Pseudo-labeling contributed little classifier supervision under this threshold, so avoid claiming the observed B gain was caused by pseudo-labeling.
- Generator shape, noise, finiteness, and gradient checks passed at 96×96. However, the reviewed full-run grids show artifacts and inconsistent preservation of morphology; generated-versus-clean L1 diagnostics do not demonstrate successful denoising. Structural checks and image-quality performance are separate findings.
- The corrected split groups 253 potential geometric-overlap pairs conservatively; not all were individually confirmed as duplicate galaxies. Image 11475 was retained in the unlabelled pool per team decision.

## Evidence files

- `../results/final_ab_2026-11-15/FINAL_TEST_RESULTS.json`: complete metrics, provenance, confusion matrices, and checkpoint hashes.
- `../results/final_ab_2026-11-15/FINAL_TEST_COMPARISON.csv` and `FINAL_TEST_PER_CLASS.csv`: tabular results.
- `../results/final_ab_2026-11-15/test_confusion_A.png`, `test_confusion_B.png`, and `final_test_macro_f1_comparison.png`: evaluation figures.
- `../results/generator_review/ab_sample_review_contact_sheet.jpg`: Member 1's selected-sample review contact sheet.
- `../docs/member3/final_test_kit/`: evaluation-only code and instructions.
- `../results/final_ab_2026-11-15/PRETEST_VERIFICATION.json`: dataset, split, test-index, run, and selection evidence.

The dataset and large checkpoints are excluded from Git. The actual training CSVs and run `args.json` files were not part of the downloaded evaluation handoff; retain the Kaggle run folders or add those logs separately if exporting them later.
