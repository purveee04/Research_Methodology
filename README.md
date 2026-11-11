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

- Integrate the Generator with the updated training pipeline.
- Evaluate generated galaxy images.
- Compare Experiments A and B.
- Evaluate denoising performance using validation data.
