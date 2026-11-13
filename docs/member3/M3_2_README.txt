M3.2 — Kaggle instructions (from your uploaded M3.1 audit)

1. Download member3_m32_package.zip from ChatGPT.
2. In SAME Kaggle notebook where the real 2.73GB Galaxy10 NoDuplicated HDF5 is available, click Input > Upload > New Dataset; upload the package ZIP, make it private, then Add Input to attach it.
3. EITHER open member3_m32_kaggle.ipynb in the same session (copy cells into existing notebook), OR paste its Step 1-6 code cells into your current notebook. Opening a fresh Kaggle notebook creates a new runtime and may not have the HDF5 under /kaggle/working.
4. Step 1 locates and unpacks the code + your original member3_audit_results.zip. Step 2 runs unit tests. Step 3 finds original HDF5. Step 4 computes the corrected split after checking the exact SHA256. Step 5 checks flagged pairs and balance. Step 6 outputs member3_corrected_split_handoff.zip.
5. Share the generated handoff ZIP with Member 2. It contains the candidate corrected_split_indices.npz, corrected_split_manifest.json, overlap_review.csv, corrected_split_validation.json and TEAM_HANDOFF.txt.
6. IMPORTANT: do not start full A/B runs. The new split must be reviewed and integrated into trainer's data loading; the trainer currently regenerates random splits unless explicitly changed. Both experiments must start afresh with the SAME corrected NPZ.

POLICY: group *potential* overlaps conservatively using circumscribed circles for each 256px image and provided per-image pixel scale. This may group distinct galaxies. Original train/val/test membership is retained where possible, group-majority/tie-train fixes overlaps. The labelled/unlabelled training pools are then rebuilt to ~12.5% labels per class. No changes to the original HDF5.

M3.1 finding: one constant-value image at index 11475, located in original training unlabelled stream. It is flagged for review but NOT silently deleted.
