# GC-SWGAN Member 3 — Experiment A/B Tracking Handoff

**Status (12 November 2026):** Reporting code implemented/tested. The complete A/B experiment results remain **PENDING**.

## Provenance and corrected data
- Input intended: Galaxy10 NoDuplicated, 17,675 images.
- Reported dataset SHA256: `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.
- Corrected split: pool 14,217 (labelled 1,777, unlabelled 12,440); validation 1,727; test 1,731.
- Conservative split groups potential 256-pixel overlap; 253 candidate pairs and 81 images reassigned. Corrected split and blank image 11475 await team review; extra cutouts excluded.
- Original smoke test and 96px baseline were not final A/B runs.

## Experiment matrix — must be frozen before training
| Setting | A | B | Evidence/approval |
|---|---|---|---|
| Galaxy10 file SHA256 | same | same | Verify at runtime |
| Corrected split NPZ SHA256 | same | same | Verify at runtime |
| Split + run seeds | same | same | Confirm |
| Resolution, steps, batch ratio | same | same | Confirm |
| Sigma, pseudo threshold/refresh/weight | same | same | Confirm |
| Optimizers, schedule, gradient penalty | same | same | Confirm |
| Generator objective | Adversarial only | Adversarial + lambda_rec × L1 | B coefficient must be chosen |
| External cutouts | Excluded | Excluded | Verify |
| New model initialization | New/fresh | New/fresh (equivalent seed) | Verify |

## Output requested from Member 2
For **each** completed A/B run: `val.csv`, `loss.csv`, `diagnostics.csv` if generated, final `results.json`, exact config, dataset and split hash verification, selected checkpoint details, code commit, and run outputs folder. Keep A/B separate.

## Reportable vs diagnostic
- The M3.5 code can produce *diagnostic* curves on provided CSVs even before approval, but will label them DIAGNOSTIC ONLY.
- To mark a matched validation comparison reportable, all configuration fields must match, approvals must be recorded, and loss equations must pass.
- Validation results are not final test results; test only after model selection is frozen. Do not invent final scores or use the test set to tune parameters.

## Outstanding team decisions
1. Sign off conservative sky-footprint grouping and image 11475 handling.
2. Demonstrate Member 2 trainer consumes corrected NPZ, checks dataset SHA256, rejects mismatches, excludes extra cutouts.
3. Choose resolution (96px vs 192px), full steps, sigma, pseudo-label policies, lambda_rec, checkpoint selection metric and seed policy.
4. Complete matched A/B experiments, save logs/checkpoints.
5. Generate real validation comparison and final held-out test evaluation under approved protocol.
