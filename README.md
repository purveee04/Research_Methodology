## Generator Module – Testing Results

### 1. Configuration

| Parameter | Value |
|---|---|
| Generator type | Image-to-image |
| Selected resolution | 96 × 96 |
| Input channels | 3 (RGB) |
| Batch size (timing test) | 64 |
| GPU devices detected | 2 |

### 2. Unit Testing

The Generator module was tested using randomly generated dummy images.

The following checks were performed:

- Gaussian noise generation.
- Input and output shape consistency.
- Output values within [-1, 1].
- Reconstruction L1 loss calculation.
- Gradient flow through Generator variables.
- Generator input dependence.
- Bottleneck compression.

**Initial unit tests:** Passed.

### 3. 96 × 96 Generator Test

| Parameter | Result |
|---|---|
| Input shape | (2, 96, 96, 3) |
| Noisy image shape | (2, 96, 96, 3) |
| Generated image shape | (2, 96, 96, 3) |
| Output minimum | -0.000763 |
| Output maximum | 0.000735 |

**Result:** Passed.

### 4. GPU Performance Test

The forward and backward computations were measured over 5 iterations after a warm-up pass.

| Parameter | Result |
|---|---|
| Resolution | 96 × 96 |
| Batch size | 64 |
| Average execution time | 0.08879 seconds |

Note: This measures Generator forward and backward computations only. It does not include Critic training, optimizer updates or data loading.

### 5. Generator testing and image-quality review

The Generator's unit and integration checks were performed at 96×96, including input/output
shape, Gaussian noise, finite values, and gradient flow through all 28 trainable variables.
These checks passed, but they establish structural behavior rather than image quality.
After full A/B training, visual review found image artifacts and no consistent evidence that
the Generator preserves galaxy morphology or denoises effectively. The selected sample grids
are available at `results/generator_review/ab_sample_review_contact_sheet.jpg`.

### 6. Full A/B and held-out test results (15 November 2026)

Two matched 10,000-step runs completed at 96×96 using the corrected split, seed 0,
20 unlabelled + 1 labelled example per step, sigma 0.1, pseudo-label threshold 0.95,
and gradient-penalty weight 10. Experiment A used adversarial loss only; Experiment B
added L1 reconstruction loss with lambda_rec=1.0. External cutouts were excluded.
Checkpoint selection used validation macro-F1. Member 3 then evaluated each frozen
selected classifier once on the same 1,731-image held-out test partition.

| Metric | A | B |
|---|---:|---:|
| Selected step (validation macro-F1) | 8,500 (0.1918) | 10,000 (0.2797) |
| Test accuracy | 0.2577 | 0.3200 |
| Test macro-F1 | 0.1862 | 0.2689 |
| Test balanced accuracy | 0.2321 | 0.3030 |
| Test weighted-F1 | 0.2043 | 0.2898 |

B scored higher on these four test metrics in this matched single-seed experiment.
This is an observed result, not evidence of a statistically established general effect.
Pseudo-label acceptance was very low (A: 89/200,000; B: 96/200,000), so the gain
should not be attributed mainly to pseudo-labelling. B's test ECE was higher than A's
(0.0569 vs 0.0415); calibration therefore did not improve on every metric.

The Generator's 96×96 structural and gradient checks passed, but Member 1's review of
the full-run sample grids and diagnostics found artefacts and no consistent evidence
of denoising or morphology preservation. The evaluation and review artifacts are in
`results/final_ab_2026-11-15/` and `results/generator_review/`. The final test kit is in
`docs/member3/final_test_kit/`. The 253 possible geometric-overlap pairs were grouped
conservatively; they were not all individually confirmed as duplicate galaxies.

No further training or test inference is currently planned. The remaining work is to
complete the written research report and describe the single-seed, low pseudo-label
acceptance, modest classification performance, and Generator image-quality limitations.

## Member 2 – Corrected-Split Training Integration

The Member 2 trainer now requires the corrected split files from
`artifacts/member3/corrected_split/`. It verifies the dataset SHA-256 against the split manifest,
checks the split arrays, sizes, class counts and geometric groups against the validation report,
and records the split and dataset fingerprints in each run's provenance. Validation and test arrays
are held out for evaluation and are not passed to the training batch source. External cutouts are
excluded; the unlabelled stream uses only the corrected Galaxy10 training pool.

Image 11475 is retained in the unlabelled training pool, following the team's 13 November decision.
The trainer records its class and raw pixel range and number of distinct pixel values in
`split_audit.json` when run against the Galaxy10 HDF5 file.

The best classifier checkpoint is selected using validation macro-F1. No A/B pilot or full training
run has been performed by this integration update. The corrected split is still subject to final
team review; the trainer requires `--confirm-split-review` after that review before it will start.
The dataset SHA-256 and image 11475 pixel details must still be confirmed in the Kaggle environment
using the actual HDF5 dataset.

Local verification after the corrected-split changes: **82 unit tests ran, 61 passed and 21 TensorFlow
tests were skipped** because TensorFlow and the Generator dependency are unavailable in this
environment. The shipped Member 3 NPZ, manifest and validation report passed a structural compatibility
check with the trainer. This did not verify the Galaxy10 HDF5 checksum or run the TensorFlow trainer.
