# Member 3 — Data Audit (Task 1)

Prepared 12 Nov 2026 against `main` @ `1b61b178e9aac4a58dd63b6387d27b680084bcad` (the Member 2 merge commit, confirmed
by `git rev-parse HEAD` on a fresh clone). Scope: data integrity only. The Generator and the training algorithm are unchanged.

## 0. Read this first: what is and is not established

I **did not have the HDF5 files**. Every statement below is one of three kinds:

| Tag | Meaning |
|---|---|
| **VERIFIED** | I read the code and/or ran it myself (synthetic data) and saw the result |
| **DOCUMENT** | Taken from a team document, the baseline notebook's saved output, or the official dataset page; I did not see the data |
| **TO RUN** | Needs the real files; the notebook cell that answers it is named |

## 1. Bottom line

| Question | Answer now | Status |
|---|---|---|
| Are train / val / test separate in the code? | Yes. Disjoint by construction, re-checked in code at every launch, and now re-proved by `split_membership_report` | VERIFIED (logic) / TO RUN (real file) |
| Does `split_indices` match the baseline's split logic? | Yes at the level of counts: it reproduces the baseline's printed 14,188 / 1,773 / 1,774 / 1,774 and its labelled class counts exactly | VERIFIED |
| Why 17,675 vs 17,736? | They are **two different files**. 17,736 is the official release and what the baseline run used. 17,675 is the file in Member 2's handoff. Which 61 images differ, and why, is unknown | DOCUMENT; cause TO RUN (Step 6) |
| Is `decals_images.h5` one file? | **No.** The baseline saw 20,000 × 256 × 256; Member 2's handoff saw 1,178 × 69 × 69 with no coordinates | DOCUMENT |
| May the external cutouts be used? | **NOT APPROVED.** No evidence of non-overlap exists for either file, and the 69 px file cannot be checked at all | see §3 F5 |
| Can the split be reproduced exactly? | Now yes: `split_indices.npz` + table + manifest, with a regenerate-and-compare check | VERIFIED (logic) / TO RUN (real file) |

## 2. How the data flows (trace)

`train_gc_swgan_i2i.py` → `build_data()` (line 172):

1. `labels = f["ans"]`, `ra/dec/pxscale` read from `Galaxy10_DECaLS.h5`; **all** images loaded with `load_h5_images` (line 126: crop `[32:224]` → 192 px, area-resize to 96 if requested).
2. `tm.split_indices(labels, test_frac, val_frac, label_frac, split_seed)` (training_module.py:475): stratified test, then stratified val, then a stratified labelled subset of the remaining train pool. Defaults 0.1 / 0.1 / 0.125, seed 16.
3. `tm.unlabeled_pool_indices(pool, lab, include_labelled)` (:500): by default the unlabelled stream = train pool **minus** the labelled subset (the baseline included the labelled images too).
4. Extra cutouts from `--unlabeled-h5` are appended to the unlabelled stream if that file exists.
5. `TrainingData(x_lab, y_lab, x_unl)` (:304) is the only thing `TrainBatchSource` accepts; val/test arrays sit in a separate dict used only for evaluation. **VERIFIED by reading**: there is no code path from `eval_data` into a batch stream.
6. Guards run at every launch: `assert_no_leakage` (byte-identical images, :337) and `run_overlap_check` → `check_cutout_overlap` (:408).
7. `dataset_identity` (:194) stores SHA-256 of both files plus hashes of label and index arrays in the run manifest.

**Expected split sizes** (they depend only on N; compare with the Kaggle output):

| File | N | train pool | labelled | val | test | unlabelled (excl. labelled) |
|---|---|---|---|---|---|---|
| Baseline | 17,736 | 14,188 | 1,773 | 1,774 | 1,774 | 12,415 |
| Member 2's file | 17,675 | 14,139 | 1,767 | 1,768 | 1,768 | 12,372 |

## 3. Findings

### F1 — Split logic is sound — VERIFIED
`split_indices` fed with synthetic labels built from the baseline's printed class counts gives pool 14,188, labelled 1,773, val 1,774, test 1,774 and labelled class counts `[108, 185, 264, 203, 34, 204, 183, 263, 142, 187]`, identical to `baseline_10000steps.ipynb` (cell 19). This is locked in as a test (`TestBaselineReproduction`). Member 2's own 36 numpy-only data tests (split, overlap, fingerprint, manifest) also pass here without TensorFlow.
*Limit:* this proves the **logic and counts**, not that the real 17,675 file's *indices* equal the baseline's. They cannot, see F2.

### F2 — 17,736 vs 17,675 is two different files — DOCUMENT; cause TO RUN
- Baseline notebook output (cell 2): `Image shape: (17736, 256, 256, 3)`, class counts `[1081 1853 2645 2027 334 2043 1829 2628 1423 1873]`.
- The [official astroNN page](https://astronn.readthedocs.io/en/latest/galaxy10.html) and the paper (§2) both say 17,736, with `ra`, `dec`, `redshift`, `pxscale`. That page gives no per-version counts and no explanation for a smaller file.
- Member 2's handoff reports 17,675 and calls the input "repaired". I found no repair script in the repo.
- **Why it matters:** the split picks *row numbers*. If 61 rows were removed from the middle of the file, every later row shifts, so the same seed selects different images. Baseline val/test and the new val/test are then not the same images, on top of the 96 px vs 192 px difference already noted in the team guide.
- **Unknown until Step 6:** which 61 rows, whether they were duplicates, corrupted images or pixel-changed images, and whether any **labels** changed. The notebook classifies each difference (removed / modified-pixels / duplicate copy) and counts label changes.
- **Recommendation:** pick ONE Galaxy10 file for A and B, record its SHA-256, and say in the report that the baseline used the other file (or re-run the baseline on the chosen file).

### F3 — `decals_images.h5` exists in two different versions — DOCUMENT
| Source | Shape | Metadata |
|---|---|---|
| Baseline notebook, cell 2 | (20000, 256, 256, 3) | not inspected; unlabelled stream was 34,188 = 14,188 + 20,000 |
| Member 2 handoff | (1178, 69, 69, 3) | only an `images` key; no ra / dec / pixel scale |

Same name, different content. The script that made the 20,000-image file is referenced by comments in `training_module.py` (`kaggle_prep_data.py`) but is **not in the repository**, so its provenance and any exclusion it applied cannot be verified.

### F4 — A 69 px file is silently cropped to 37 px — VERIFIED
`load_h5_images` slices `[32:224]` without checking the source size. On a 69 × 69 image numpy returns 37 × 37 (I reproduced this); the trainer then area-resizes it to 96 and trains on an upscaled 37 px patch with no error. Proposed fix: `member3_proposed_patch_load_h5_images.patch` (§5). Not applied.

### F5 — External cutouts: NOT APPROVED
- **1,178 × 69 × 69 file:** no coordinates and no pixel scale, so the footprint check cannot run (Member 2's guard would also refuse it). The exact-duplicate check cannot catch a different-size cutout of the same galaxy. The crop in F4 applies. Not usable.
- **20,000 × 256 × 256 file:** its metadata was never inspected in the documents I have, the baseline output prints no overlap check, and its prep script is missing. Not approved until Step 7 returns `CLEARED_PENDING_SIGNOFF` **and** the team signs off.
- **Why leakage is plausible a priori** (a risk estimate, not a finding): Galaxy10 is a subset of Galaxy Zoo DECaLS (paper §5.3); the paper draws its unlabelled images randomly from that catalogue (314,000 galaxies per its introduction); I found no sentence about excluding Galaxy10 galaxies. If 20,000 cutouts were an unfiltered uniform sample, about 20,000 × 17,736 / 314,000 ≈ **1,130** would be Galaxy10 galaxies and about **226** of those would be validation/test galaxies. Those images would reach the Critic and the pseudo-labeller and inflate metrics. The assumptions (uniform, unfiltered) may not hold; the check decides.
- The new gate is **stricter than Member 2's**: a pixel scale typed with `--unlabeled-pixscale` is not accepted (it is unverifiable), the image size must be square and 224–256 px, and an exact-duplicate check against val/test must have run.

### F6 — Leakage *inside* Galaxy10 is not checked by existing code — TO RUN
Existing guards compare byte-identical images only. They do not use sky positions *within* Galaxy10, so a train galaxy that is a repeat of, or sits on top of, a val/test galaxy would pass. `galaxy10_sky_overlap` reports, for train-pool galaxies against val/test: (a) centre within 3″ (the same galaxy twice; treat as a problem), (b) 256 px footprint overlap and (c) 192 px footprint overlap (what the model actually sees). (b) and (c) use Member 2's conservative circumscribed-circle bound, so they are **upper bounds**: informational, for the team to decide on. Nothing is dropped automatically, because dropping images would change the split.

### F7 — Split indices were not preserved — FIXED (new files)
The run manifest stores only SHA-256 hashes of the index arrays, so a finished run cannot tell you *which* images were validation/test, and scikit-learn can change a split between versions. `write_split_artifacts` saves the indices, labels, per-image content hashes, seeds and library versions; `verify_split_artifacts` regenerates the split and compares (also detects edited files, changed labels or settings).

### F8 — Smaller guard observations — VERIFIED by reading
| # | Observation | Effect |
|---|---|---|
| a | `--skip-overlap-check` makes `build_data` exit even when there is no extra file (mode `skipped` is tested before `not_applicable`) | Harmless but confusing. Do not pass that flag |
| b | `--unlabeled-pixscale` is trusted | A wrong number silently gives a wrong footprint check |
| c | `check_cutout_overlap` assumes 256 px for the extra cutouts | Conservative for 224–255 px, too small for > 256 px (my gate rejects > 256) |
| d | `assert_no_leakage` hashes the cropped/resized arrays | Fine for identical sources; blind to re-cut or re-scaled copies |

### Guard behaviour in Member 2's runner (read from `build_data` / `run_overlap_check`)
| Situation | Outcome |
|---|---|
| No `decals_images.h5` | Trains on train-pool images only (the smoke run) |
| Extra file without ra/dec | Exits: "full cutout-overlap check did not run" |
| ra/dec present, no pxscale, no CLI value | Exits (centre-only check is refused) |
| ra/dec present + `--unlabeled-pixscale` typed | **Runs** using the typed value (F8b) |
| Extra file of the wrong size with full metadata | **Runs**; crop is silently wrong (F4) |

## 4. Decisions only the team can make

| Decision | Recommendation |
|---|---|
| Which Galaxy10 file for A and B | One file; record its SHA-256; explain the baseline file difference |
| External cutouts | Do not use any until Step 7 says `CLEARED_PENDING_SIGNOFF` and the prep script/provenance is shared. Run A/B with Galaxy10 train-pool images only (the smoke-run configuration) if not cleared |
| Galaxy10 train-pool galaxies flagged by F6 | Decide a rule *before* looking at results (e.g. exclude pool galaxies whose 192 px footprint touches val/test). Exclusion changes the split, so re-run the audit and re-save the artifacts |
| 96 vs 192 px | Already in the team checklist; keep any baseline comparison clearly caveated |

## 5. Proposed change to Member 2's code (not applied)

`member3_proposed_patch_load_h5_images.patch` (applies cleanly with `git apply --check`; the patched file compiles): `load_h5_images` raises a clear error if the source images are smaller than 224 px instead of cropping them to 37 px. 5 lines, no change for 256 px inputs. I did not run it under TensorFlow (not installed in my environment); the guard condition was tested separately: 69 px and 223 px refused, 224 px and 256 px accepted.
Optional, only if `verify_split_artifacts` ever reports a difference on Kaggle: add a `--split-file` option that loads `split_indices.npz` instead of recomputing the split.

## 6. Files, tests, what to send

| File | Purpose |
|---|---|
| `data_audit.py` | Audit functions; reuses `training_module` (no TensorFlow needed) |
| `test_data_audit.py` | 29 tests, synthetic data only |
| `member3_kaggle_audit.ipynb` | The exact Kaggle cells with plain-language explanations (edit Step 4 only) |
| `member3_data_audit.md` | This report |
| `member3_proposed_patch_load_h5_images.patch` | Optional fix for F4 |

**Tests run here:** `python -m unittest test_data_audit` → 29 OK (3 of 4 deliberate code breakages were caught by the tests; the 4th is covered by a second, independent check). Member 2's data tests (`TestSplitAndPositions`, `TestCutoutOverlap`, `TestFingerprints`, `TestResumeManifest`) → 36 OK. All notebook code cells were executed end to end on synthetic files. **Not run:** anything on the real data, and Member 2's TensorFlow tests.

**To send:** everyone: the Step 9 summary + `member3_audit_results.zip` + §4 decisions. Member 2: F2, F4, F8, the patch, and "which file did you train on, and how was it repaired?". Member 1: nothing changes (the Generator interface is unaffected; only 256 px sources are acceptable).

## 7. Fill in after the Kaggle run

| Item | Result |
|---|---|
| Galaxy10 file used by Member 2: N, SHA-256 | |
| Split sizes (expect §2) / membership OK? | |
| Blank images / duplicate groups / across splits / conflicting labels | |
| Train-pool galaxies: same centre / footprint 256 / footprint 192 | |
| Split artifacts reproduced? (`verify_split_artifacts`) | |
| Baseline file N, SHA-256; unmatched rows A / B; label changes | |
| 61-image difference: removed / modified / duplicate copies | |
| `extra_20000` verdict and reasons | |
| `extra_1178` verdict and reasons | |

## 8. Not checked

Rotated/flipped copies of the same image (hashes are orientation-specific); a different cutout of the same galaxy when no coordinates exist (no method here can detect it); whether pseudo-labels or Critic batches ever touch val/test at run time (that is the M3.4 loss and gradient-flow audit in the team guide, not part of this data audit); anything about model quality.
