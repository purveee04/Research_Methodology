# Member 3: final A/B evaluation — read before running

## What was already verified from Member 2's handoff

- Both 10,000-step runs supplied logs, configurations, split evidence, and `best_clf.index` / `best_clf.data-00000-of-00001` files.
- A: selected checkpoint step **8,500**, validation macro-F1 **0.1918**.
- B: selected checkpoint step **10,000**, validation macro-F1 **0.2797**.
- Galaxy10 SHA-256: `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.
- Corrected NPZ SHA-256: `1ea636a1cd3df30fdd6b155b06896692198e141d536d28a60899f17955c1ede8`.
- Image 11475 remains in unlabelled training as the team decided; external cutouts excluded.
- 253 geometric overlap pairs grouped conservatively; not all visually established as duplicates.
- Pseudo-labels accepted: A 89/200,000 (0.0445%); B 96/200,000 (0.048%).

**The full held-out test metrics have NOT been computed here.** The 2.7 GB HDF5 is necessary and must be attached in Kaggle. Do not interpret validation figures as test scores.

## How to run in Kaggle

1. Upload this kit ZIP as a private Kaggle dataset, attach it to a new Kaggle notebook, or import the included notebook directly.
2. Upload/attach `member2_full_AB_review.zip` as a private Kaggle dataset.
3. Attach the same exact Galaxy10 dataset used by Member 2 (file might be called `Galaxy10_DECaLS.h5` rather than `Galaxy10_DECals_NoDuplicated.h5`). The checksum, not filename, identifies it.
4. Enable **Internet** for the repository clone and use the existing TensorFlow/legacy Keras environment. If `tf_keras` is missing, install matching `tf_keras`.
5. In `member3_final_test_kaggle.ipynb`, run setup → **preflight (NO TEST PREDICTIONS)** and review PASS.
6. Once satisfied, run the explicitly labelled **ONE-TIME FINAL TEST** cell. Do NOT run this to tune thresholds, losses, training steps, architecture, or checkpoint choice.
7. Save/download the output ZIP. It contains test metrics for both classifiers, 10 per-class rows per model, confusion plots, and the evaluation provenance report.

## Evaluator implementation

- Uses **exact training code commit** `25882a893fe9980acc430008d4869b9a411bb3e0`; the notebook checks out that commit in a separate directory.
- Uses `training_module.load_corrected_split` for HDF5 SHA, split label/index/group checks, and compares `tm.array_fingerprint` of each train/val/test array against the checkpoint manifests.
- Uses `tm.build_trunk(96)` and `tm.build_heads(..., 10)` to reconstruct the classifier graph, then `clf.load_weights(best_clf_prefix)` for each selected checkpoint.
- Uses `train_gc_swgan_i2i.predict_probs()` and `full_metrics()` directly; **no training loop, Generator updates, or optimizer steps**.
- Evaluates the **same exact 1,731-image corrected held-out test partition** for A and B, with exactly the same crop `[32:224]`, area resizing to 96×96 and uint8 rounding used by the training runner.
- Selects checkpoints using the SAVED validation-best state. Refuses a changed selection rule, wrong experiment, incomplete logs, wrong training SHA, or mismatched split.
- Writes a `TEST_STARTED.json` marker and refuses another test inference with the same output directory. If interrupted, investigate the cause rather than running repeated test evaluations to optimize metrics.

## Outputs

- `PRETEST_VERIFICATION.json`: training code version, dataset/split identity and checkpoint selection preflight.
- `TEST_STARTED.json`: records authorized test access.
- `FINAL_TEST_RESULTS.json`: full real metrics, per-class stats, confusion matrices, checkpoint hashes and limitations.
- `FINAL_TEST_COMPARISON.csv`: accuracy, macro/weighted-F1, balanced accuracy, calibration metrics for A and B.
- `FINAL_TEST_PER_CLASS.csv`: precision, recall, F1 and support for 10 classes for A and B.
- `test_confusion_A.png`, `test_confusion_B.png`, `final_test_macro_f1_comparison.png`.

## Limitations

- One paired seed is not a statistical significance study; don't claim B is definitively better on other seeds or datasets.
- The result depends on a conservative geometric guard, not human confirmation of all overlap pairs.
- Very few pseudo-labels crossed the 0.95 confidence threshold, so little pseudo-label supervision occurred.
- Grey/blurry generated-image samples are a separate image-quality issue; test classification scores do not demonstrate denoising.
- The selected A and B classifiers come from training runs with their own weights. `best_clf` is deliberately classifier-only; a full GAN checkpoint is not required for classification inference.
