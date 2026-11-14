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

### 5. Testing Limitations

- Testing was performed using dummy images, not the actual galaxy dataset.
- The Generator has not yet been trained to perform denoising.
- Gaussian noise sigma and reconstruction loss weight lambda_rec remain subject to team validation.
- Actual image quality and denoising performance will be evaluated after integration with the training pipeline.

### 6. Next Steps

- Run short A/B pilots after the integrated trainer and corrected split pass team review.
- Evaluate generated galaxy images.
- Compare Experiments A and B.
- Evaluate denoising performance using validation data.

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
