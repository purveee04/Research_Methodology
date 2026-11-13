# GC-SWGAN — Proposed A/B protocol and split review (12 Nov 2026)

**Status: PROPOSED — not approved and not a final training authorization.** This protocol uses the information recorded in Member 3's M3.1 audit and M3.2 execution screenshots, plus Member 2's handoff. Confirm the choices with all three members before running full A/B jobs.

## 1. Data and split

- Primary source: `Galaxy10_DECals_NoDuplicated.h5`, 17,675 RGB images of shape (256,256,3).
- **Required SHA-256:** `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.
- Only use Member 3's **corrected** `corrected_split_indices.npz` and corresponding manifest/validation report, never the original M3.1 random `split_indices.npz`.
- Current candidate partition sizes: pool 14,217 (labelled 1,777; unlabelled 12,440), validation 1,727, test 1,731. Exact index arrays take precedence over these descriptive counts.
- M3.2's implemented geometric grouping rule (256-pixel footprints) passed `PASS_CONSERVATIVE_GEOMETRIC_GUARD`; 253 candidate overlap edges were considered and 81 rows changed partition. This checks an explicit geometric rule **not all forms of leakage**. Three suspicious near-duplicate pairs share the training pool.
- External `decals_images.h5` cutouts: **exclude** in both A and B. Data provenance/overlap is unresolved.
- **Image 11475:** the M3.1 JSON reports a constant-value image. M3.1's split table assigns it class 6, unlabelled training. Inspect the true pixel array in Kaggle; record min/max/number of unique values and plot. If it is unusable, exclude it **from any image sampler, including Critic and pseudo-labeling**, using a versioned exclusion policy integrated by Member 2 and revalidate counts/manifests. Do not edit the source HDF5 or silently alter the saved NPZ; a policy that changes its sampled membership requires validation and new manifest identification. If retained, document why.

## 2. Feasible primary experiment design (for team approval)

| Choice | Proposed primary value | Why / caveat |
|---|---|---|
| Image resolution | **96×96** for primary matched A/B | Existing 10k baseline and smoke test used 96; reduces Kaggle compute. Explicitly label as deviation from 192×192 target. Optional follow-up: paired 192×192 A/B, only if GPU budget permits; do not compare 96 A vs 192 B. |
| Steps | **10,000 updates** per run as initial target | Mirrors prior completed baseline; subject to pilot runtime and session limits. If infeasible, agree on a smaller *identical* budget for A and B before either final run. |
| Batch ratio | **20 unlabelled + 1 labelled** each step | Instructor confirmed. |
| Sigma | **0.1** with [-1,1] image normalization | Already smoke-tested; research choice, not instructor-approved. |
| Pseudo-label threshold | **0.95** initially | Reuses provisional code default. Five-step smoke accepted none: monitor acceptance as training progresses. If persistently zero, predeclare a threshold sensitivity experiment (for both A and B); don't alter only one run midstream. |
| Pseudo-label refresh | Every training step | Existing implemented default. |
| Pseudo-label CE weight | 1.0 | Current provisional default; log clearly. |
| Stream exhaustion | Reshuffle policy | Existing default; check that labels remain paired and split membership is respected. |
| GP weight | 10 | Baseline-style current default. |
| Learning rate | 0.0001, Adam beta1=0.5 | Existing implementation; track LR scheduler. |
| Run seed | 0 for first matched A/B pair | Same seed for matched initializations; if feasible replicate with matched 1 and 2. Never report single-seed variation as statistical certainty. |
| Split seed | 16 (already built) | Use actual corrected NPZ, do not recompute. |
| Validation frequency | Suggested every **500 steps**, plus final evaluation | Confirm CLI support; checkpoints need consistent selection. |
| Best checkpoint selection | **Validation macro-F1** preferred, or validation accuracy if keeping current runner without modifying it | Pick one **before** experiment; current runner reportedly chooses accuracy. Macro-F1 better handles imbalance. Require both A/B use same criterion. |
| Reconstruction strength B | **Not yet fixed** | Do not launch reportable B until selected. Start with `lambda_rec=1.0` as a *pilot candidate*, inspect magnitude of `g_adv`, `g_rec`, total and image samples. Freeze a justified value before final A/B; do not use test data to choose it. A has no reconstruction term. |
| Seeds | At least one matched pair; three if resources | One is a pilot/minimum, 3 enables simple variation reporting. |
| Test set | Final locked-checkpoint assessment **only** | Do not tune on test. |

**Critical check:** Existing code performs one critic update per generator update per training step. The original GC-SWGAN update schedule has **not been independently verified from the baseline script**; the team should inspect it, then either preserve it or explicitly document a changed schedule. Make the schedule identical in A and B.

## 3. Computing plan for limited Kaggle GPUs

1. Member 2 integrates the corrected NPZ and SHA-256 safeguards and exports run manifests and stable checkpoints. Check wrong-hash and missing-split runs abort safely.
2. Member 3 inspects image 11475 and records exclusion/retention; team signs off the conservative overlap grouping policy and class balance.
3. Team freezes all shared knobs and checkpoint-selection metric; choose B's `lambda_rec` after a short **pilot only**. If selecting after looking at pilots, run final A and B **from scratch** with final agreed settings.
4. On available T4 GPU, run 10–20 step integration test of each variant (`--skip-test`); check model updates, finite losses, `g_rec==0` for A, `g_loss≈g_adv+lambda*g_rec` for B, pseudo-label logs, checkpoints and verified split hash.
5. Benchmark ~100 training steps in a pilot and extrapolate. Choose feasible identical update budget for A/B; save checkpoints periodically. If runner supports exact resume, validate restore with optimizer state. Never resume old random-split checkpoints.
6. Perform A and B on the **same exact file, split, RNG seed, resolution, steps and policy**. Log source commit and environment; save separate folders.
7. Select checkpoints by frozen validation criterion. Then evaluate test **once for each final selected model**. Publish accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrices, pseudo-label acceptance and reconstruction diagnostics.
8. Run Member 3 M3.5 on both real `val.csv`, `loss.csv`, and full configs; mark results reportable only when checks and final sign-off are documented.

## 4. Approval gates

- [ ] Pixel inspection and explicit policy for image 11475 recorded.
- [ ] Team accepts the conservative sky-overlap group policy (or documents a different corrected split).
- [ ] SHA-256 verified against source; corrected indices and all labels checked; test and validation protected from *every* training input.
- [ ] Member 2's corrected-split integration tests pass; no old split regenerated; external cutouts excluded.
- [ ] Original critic schedule confirmed or documented as changed.
- [ ] Resolution, step budget, sigma, threshold, refresh/weight, `lambda_rec`, seed list, evaluation cadence and model-selection rule signed off.
- [ ] Paired pilots and checkpoints pass; training can resume safely if needed.

## 5. Message after sign-off

> We reviewed image 11475 and documented its handling, accepted the conservative corrected split for the specified Galaxy10 SHA-256, and verified the trainer loads exactly the corrected NPZ for both variants. We have frozen the A/B configuration, including lambda_rec and the model-selection criterion. Please proceed with matched A/B pilots, then the full runs if all checks pass. Do not use the external cutouts or old random-split checkpoints.

**This message is only for use when every gate is actually checked.**
