# Member 3 — GC-SWGAN (repository-ready contribution)

This folder contains the **recovered Member 3 source files**, tests, notebooks, documentation, and the **real Kaggle-generated candidate corrected split**. It does not contain the 2.73 GB Galaxy10 HDF5 file or any final A/B trained results.

## Layout
- Root `data_audit.py`, `member3_m32.py`, `data_eval.py`, `member3_source_audit.py`, `member3_m35.py`: source modules. Their `test_*.py` files are colocated to allow `python -m unittest ...` from repository root.
- `notebooks/member3/`: Kaggle notebooks; paths inside them assume Kaggle inputs/runtime and may need adjustment.
- `docs/member3/`: project documentation, M3.1 audit summary, and proposed research protocol; `evidence/galaxy10_current_audit.json` is original audit evidence, NOT final corrected indices.
- `artifacts/member3/corrected_split/`: **candidate** corrected index NPZ, original-dataset checksum manifest, overlap-review CSV, validation report, and trainer handoff. These were generated from real Galaxy10 NoDuplicated data in Kaggle. Group policy is conservative and still requires team approval. SHA-256 dataset identity is `a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4`.

## IMPORTANT
- Do **not** use the old M3.1 uncorrected `split_indices.npz` for final A/B training. It has not been included in this push-ready folder.
- The corrected split here is a proposed artifact, not training-ready until reviewed and the runner has integrated exact NPZ loading and hash verification.
- Image index 11475 was flagged constant-value and remains in the training pool pending review.
- External cutouts stay excluded.
- The `README_M35.md` and `member3_m35_handoff.md` are preparation documents; actual A/B experiment logs do not exist in this package.
- The old `member3_loss_audit.md` is a preliminary audit; review it against the latest main branch before declaring runtime verification complete.

## Suggested checks at repository root
```
python -m unittest test_data_audit test_member3_m32 test_data_eval test_member3_m35 -v
python -m unittest test_training_module -v
```
Use your Kaggle environment for TensorFlow if your computer lacks compatible dependencies.

## Suggested Git workflow
From a clean clone of `https://github.com/purveee04/Research_Methodology`, create `data-evaluation` branch and copy **contents** of this folder into the repository root. Run tests and inspect `git status` and `git diff --cached --stat` before committing. Push branch, create a PR into `main`, and have Member 2 review any potential duplication of corrected split artifacts before merging.
