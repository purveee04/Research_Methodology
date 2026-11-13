# Member 3 — M3.5 Experiment Tracking and A/B Reporting

## What is complete
- Standalone A/B log reader, validation plots, summary CSV, Generator A/B loss identities and configuration/evidence audit.
- Twelve local automated tests passed using *synthetic fixtures* to test software correctness, not research performance.
- Kaggle notebook to create unapproved configuration templates now, inspect future real run outputs, then generate reports from real log files.

## What is **not** complete
- No full A/B runs or final validation/test metrics are supplied. This package does not generate them.
- Corrected split remains a conservative candidate pending team review, image 11475 decision and Member 2 integration verification.
- Reconstruction lambda_rec, 96/192 resolution, number of steps, sigma, pseudo-label policies, etc., require team approval.
- Report does not independently evaluate saved checkpoint on test. Later use the trained model's outputs under an approved test-evaluation protocol.

## Running in Kaggle
1. Upload `member3_m35_package.zip` as a private Kaggle dataset, attach it, and open `member3_m35_kaggle.ipynb` in the same notebook environment.
2. Execute setup and tests. Verify 12 tests pass.
3. Generate config review templates and fill actual values with the team. The templates intentionally contain `null`/`false` until evidence is available.
4. For completed runs, attach the A/B outputs and enter *actual* `val.csv` and `loss.csv` paths plus completed metadata JSON paths.
5. Run real reporting after approving the split, checking both data hashes, and verifying fresh equivalent A/B runs. The comparison gate blocks unverified reportable results. With an explicit diagnostic flag, charts are clearly marked DIAGNOSTIC ONLY.
6. Export `member3_m35_results.zip` to Member 2 and the team.

## Results interpretation
- `validation_results.csv` includes **best validation values**, not test accuracy; the best step for each metric may differ. Choose the checkpoint selection metric *before* final evaluation.
- `loss.csv` is checked for `g_rec=0, g_loss=g_adv` in A and `g_loss=g_adv+lambda_rec*g_rec` in B.
- It is possible for this arithmetic to pass but still have model/integration defects. Unit tests and the Member 3 source audit remain separate evidence.
- A/B run settings must match except loss mode and reconstruction weight. Even identical configuration JSONs don't prove actual data input was correctly connected, so explicit integration evidence is mandatory.
- Values obtained using the old random split, smoke runs, or synthetic fixtures must not be mixed with reportable final A/B results.

## Required inputs to collect from Member 2
- Run A/B `val.csv`, `loss.csv` and preferably `diagnostics.csv` and `results.json`.
- Run config and checkpoint metadata, Git commit IDs, dataset SHA256 and exact `corrected_split_indices.npz` SHA256.
- Confirmation that corrected NPZ was used (not the random split), checksum check passed, no extra cutouts used, both runs started fresh, same resolution/seed/optimizer/steps.
- Agreement on B reconstruction weight and checkpoint selection criterion.

## Dataset recorded in Member 2's handoff
`Galaxy10_DECals_NoDuplicated.h5`: SHA-256 `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.
This is a known *reported* hash; independently verify actual data identity before setting `dataset_hash_verified=true`.
