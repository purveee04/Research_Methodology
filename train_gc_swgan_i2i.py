#!/usr/bin/env python3
"""
train_gc_swgan_i2i.py -- Member 2 trainer: image-to-image generator (generator_module.py),
20:1 unlabelled:labelled steps, confidence-filtered pseudo-labels, Experiment A / B.

NEW file. gc_swgan_fixed.py is not imported, run or modified. Defaults write to ./outputs_i2i and
./checkpoints_i2i so old outputs/checkpoints are never overwritten.

Pieces COPIED FROM gc_swgan_fixed.py (unchanged logic): load_h5_images, ece_score, brier_score,
full_metrics, predict_probs, the checkpoint/resume/logging pattern.

    python train_gc_swgan_i2i.py --experiment A \\
        --split-npz artifacts/member3/corrected_split/corrected_split_indices.npz \\
        --split-manifest artifacts/member3/corrected_split/corrected_split_manifest.json \\
        --split-validation artifacts/member3/corrected_split/corrected_split_validation.json ...
    python train_gc_swgan_i2i.py --experiment B --lambda-rec <TEAM VALUE> [same split arguments] ...

Every value marked PROVISIONAL in training_module.PROVISIONAL is printed at launch and saved in the
checkpoint manifest. None of them is team- or instructor-approved.

Resume safety: <ckpt-dir>/<name>/manifest.json records the full config, the data-pipeline settings and
the dataset identity (SHA-256 of the labelled HDF5 file, optional external HDF5, labels and split
indices). A resume is refused if ANY of them differs. The corrected split checksum is compared before
training.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import argparse
import csv
import json
import shutil
import time

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score, accuracy_score)

import generator_module as gm
import training_module as tm

NCLASSES, CH = tm.NCLASSES, tm.CH


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", default="gc_swgan_i2i_run")
    p.add_argument("--labeled-h5", default="./Galaxy10_DECaLS.h5")
    p.add_argument("--split-npz", required=True,
                   help="Member 3 corrected_split_indices.npz; random splits are not used")
    p.add_argument("--split-manifest", required=True,
                   help="matching corrected_split_manifest.json with expected Galaxy10 SHA-256")
    p.add_argument("--split-validation", required=True,
                   help="matching corrected_split_validation.json")
    p.add_argument("--confirm-split-review", action="store_true",
                   help="required only after the team completes its final review of the candidate split")
    p.add_argument("--out-dir", default="./outputs_i2i")
    p.add_argument("--ckpt-dir", default="./checkpoints_i2i")
    p.add_argument("--experiment", required=True, choices=list(tm.EXPERIMENTS),
                   help="A: adversarial generator loss only | B: + lambda_rec * L1(clean, G(noisy))")
    p.add_argument("--lambda-rec", type=float, default=None,
                   help="reconstruction weight; B defaults to the team-agreed 1.0, must be omitted for A")
    p.add_argument("--img-size", type=int, default=96, choices=list(tm.SUPPORTED_IMG_SIZES))  # PROVISIONAL
    p.add_argument("--steps", type=int, default=10000)                                        # PROVISIONAL
    p.add_argument("--n-unlabeled", type=int, default=20)                                     # PROVISIONAL
    p.add_argument("--n-labeled", type=int, default=1)                                        # PROVISIONAL
    p.add_argument("--sigma", type=float, default=0.1)                                        # PROVISIONAL
    p.add_argument("--pseudo-threshold", type=float, default=0.95)                            # PROVISIONAL
    p.add_argument("--pseudo-refresh", default="every_step", choices=list(tm.REFRESH_MODES))  # PROVISIONAL
    p.add_argument("--pseudo-refresh-every", type=int, default=100)                           # PROVISIONAL
    p.add_argument("--pseudo-loss-weight", type=float, default=1.0)                           # PROVISIONAL
    p.add_argument("--exhaustion-policy", default="reshuffle",
                   choices=list(tm.EXHAUSTION_POLICIES))                                      # PROVISIONAL
    p.add_argument("--aug-labeled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aug-unlabeled", action="store_true")
    p.add_argument("--gp-weight", "--lam", dest="gp_weight", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--decay", type=float, default=1.000004)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--benchmark", type=int, default=0, help="time N steps (after warm-up) and exit")
    return p.parse_args(argv)


def make_config(args):
    try:
        if args.experiment == "B" and args.lambda_rec is None:
            args.lambda_rec = 1.0
        return tm.TrainConfig(
            experiment=args.experiment, img_size=args.img_size, n_unlabeled=args.n_unlabeled,
            n_labeled=args.n_labeled, sigma=args.sigma, lambda_rec=args.lambda_rec,
            pseudo_threshold=args.pseudo_threshold, pseudo_refresh=args.pseudo_refresh,
            pseudo_refresh_every=args.pseudo_refresh_every, pseudo_loss_weight=args.pseudo_loss_weight,
            exhaustion_policy=args.exhaustion_policy, aug_labeled=args.aug_labeled,
            aug_unlabeled=args.aug_unlabeled, gp_weight=args.gp_weight, lr=args.lr,
            min_lr=args.min_lr, decay=args.decay, seed=args.seed)
    except ValueError as e:
        sys.exit("Invalid configuration: %s" % e)


def make_data_settings(args):
    return tm.make_data_settings(split_source="member3_corrected_npz",
                                 include_external_cutouts=False,
                                 include_labelled_in_unlabeled=False)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_h5_images(path, key, size, chunk=1000):       # COPIED FROM gc_swgan_fixed.py
    """Centre-crop 256->192 (as in the paper), optionally resize, return uint8."""
    with h5py.File(path, "r") as f:
        ds = f[key]
        n = ds.shape[0]
        out = np.empty((n, size, size, CH), np.uint8)
        for i in range(0, n, chunk):
            x = ds[i:i + chunk, 32:224, 32:224, :]
            if size != 192:
                x = tf.image.resize(x.astype(np.float32), (size, size), method="area").numpy()
                x = np.clip(np.rint(x), 0, 255)
            out[i:i + chunk] = x.astype(np.uint8)
    return out


def _read_optional(f, key):
    return np.array(f[key]) if key in f else None


def run_overlap_check(args, g_ra, g_dec, g_px, ix, extra_meta):
    """External cutouts are excluded, so Member 3's validated fixed split is authoritative."""
    return {"mode": "not_applicable", "ok": True,
            "message": "External cutouts are excluded; the corrected Galaxy10 split handles sky-overlap groups."}


def build_data(args, want_identity=True):
    """Load the corrected fixed split and return its provenance with train/evaluation data.

    eval_arrays (val/test) live in a SEPARATE dict that is never passed to TrainBatchSource; they are
    used only for evaluation.
    """
    with h5py.File(args.labeled_h5, "r") as f:
        labels = np.array(f["ans"]).astype(np.int64)
        g_ra, g_dec, g_px = (_read_optional(f, k) for k in ("ra", "dec", "pxscale"))
    try:
        ix, split_provenance = tm.load_corrected_split(
            args.split_npz, args.split_manifest, args.split_validation, labels, args.labeled_h5)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        sys.exit("Corrected split validation failed; refusing to train: %s" % e)
    if 11475 not in set(ix["unl_pool"].tolist()):
        sys.exit("The agreed corrected split must retain image 11475 in the unlabelled training pool.")
    with h5py.File(args.labeled_h5, "r") as f:
        image_11475 = np.asarray(f["images"][11475])
    image_11475_doc = {
        "index": 11475,
        "class": int(labels[11475]),
        "partition": "unl_pool",
        "policy": "Retain in the unlabelled training pool, per the team's 13 Nov 2026 decision; document its pixel values.",
        "raw_shape": list(image_11475.shape),
        "raw_min": int(image_11475.min()),
        "raw_max": int(image_11475.max()),
        "raw_unique_pixel_values": int(np.unique(image_11475).size),
    }
    split_provenance["image_11475"] = image_11475_doc
    print("image 11475 retained: class %d, min %d, max %d, unique values %d" % (
        image_11475_doc["class"], image_11475_doc["raw_min"], image_11475_doc["raw_max"],
        image_11475_doc["raw_unique_pixel_values"]))
    x_all = load_h5_images(args.labeled_h5, "images", args.img_size)
    unl_idx = ix["unl_pool"]

    parts, extra_meta, have_extra = [], None, False
    if len(unl_idx):
        parts.append(x_all[unl_idx])
    print("NOTE: external cutouts are excluded -> unlabelled data uses corrected Galaxy10 train pool only")
    if not parts:
        sys.exit("No unlabelled data available.")

    train = tm.TrainingData(x_all[ix["lab"]], labels[ix["lab"]], np.concatenate(parts, axis=0))
    eval_data = {"x_val": x_all[ix["val"]], "y_val": labels[ix["val"]],
                 "x_test": x_all[ix["test"]], "y_test": labels[ix["test"]]}
    del x_all

    # ---- leakage guards (run every launch) ----
    tm.assert_no_leakage(train, x_val=eval_data["x_val"], x_test=eval_data["x_test"])
    print("exact-duplicate check: no unlabelled/labelled train image is byte-identical to a val/test image")
    overlap = run_overlap_check(args, g_ra, g_dec, g_px, ix, extra_meta)
    print("sky-overlap check [%s]: %s" % (overlap["mode"], overlap["message"]))
    if overlap["mode"] in ("skipped", "unavailable", "center_only_limited"):
        sys.exit("Refusing to train: the full cutout-overlap check did not run.")
    if not overlap["ok"]:
        sys.exit("Refusing to train: possible validation/test leakage. Rebuild decals_images.h5 excluding "
                 "those galaxies' footprints.")

    identity = None
    if want_identity:
        print("hashing dataset files for the run manifest (SHA-256) ...", flush=True)
        identity = tm.dataset_identity(args.labeled_h5, None,
                                       labels, ix, unl_idx,
                                       labeled_fingerprint=split_provenance["dataset_fingerprint"])
    print("labelled train: %d | unlabelled train: %d | val: %d | test: %d" % (
        len(train.x_lab), len(train.x_unl), len(eval_data["x_val"]), len(eval_data["x_test"])))
    print("labelled class counts:", np.bincount(train.y_lab, minlength=NCLASSES).tolist())
    print("corrected split: %s | dataset SHA-256: %s" % (
        split_provenance["validation_status"], split_provenance["dataset_sha256_expected"]))
    print("corrected split indices: pool %d | labelled %d | unlabelled %d | val %d | test %d" % (
        len(ix["pool"]), len(ix["lab"]), len(ix["unl_pool"]), len(ix["val"]), len(ix["test"])))
    return train, eval_data, identity, overlap, split_provenance


# --------------------------------------------------------------------------- #
# Evaluation (COPIED FROM gc_swgan_fixed.py, with clf passed in instead of a global)
# --------------------------------------------------------------------------- #
def predict_probs(clf, x_u8, bs=128):
    outs = []
    for i in range(0, len(x_u8), bs):
        x = tm.to_float(tf.constant(x_u8[i:i + bs]))
        outs.append(tf.nn.softmax(clf(x, training=False)).numpy())   # dropout OFF
    return np.concatenate(outs)


def ece_score(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier_score(probs, y):
    oh = np.eye(NCLASSES)[y]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def full_metrics(probs, y):
    pred = probs.argmax(1)
    rep = classification_report(y, pred, labels=list(range(NCLASSES)),
                                output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_recall": [float(rep[str(c)]["recall"]) for c in range(NCLASSES)],
        "per_class_precision": [float(rep[str(c)]["precision"]) for c in range(NCLASSES)],
        "ece": ece_score(probs, y), "brier": brier_score(probs, y),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(NCLASSES))).tolist(),
    }


def save_sample_grid_and_diagnostics(step, models, cfg, x_fixed_u8, out_dir, diag_writer, diag_file):
    """Clean / noisy / generated grid + identity_score + input_dependence on FIXED TRAINING images
    (handoff section 9.3). Inference mode."""
    clean = tm.to_float(tf.constant(x_fixed_u8))
    noisy = gm.add_gaussian_noise(clean, cfg.sigma, seed=1)
    fake = models.gen(noisy, training=False)
    g_vs_c, n_vs_c = gm.identity_score(clean, noisy, fake)
    noisy_b = gm.add_gaussian_noise(tf.roll(clean, 1, axis=0), cfg.sigma, seed=2)
    dep = gm.input_dependence(models.gen, noisy, noisy_b)
    diag_writer.writerow([step, "%.5f" % g_vs_c, "%.5f" % n_vs_c, "%.5f" % dep])
    diag_file.flush()
    n = len(x_fixed_u8)
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for j in range(n):
        for r, im in enumerate((clean[j], noisy[j], fake[j])):
            ax[r, j].imshow(np.clip(im.numpy() * 0.5 + 0.5, 0, 1))
            ax[r, j].axis("off")
    ax[0, 0].set_title("clean / noisy / generated", loc="left", fontsize=8)
    fig.savefig(os.path.join(out_dir, "samples_step_%06d.png" % step), dpi=80, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if not args.confirm_split_review:
        sys.exit("The corrected split is still marked as a candidate pending final team review. "
                 "Review the split artifacts first; after team signoff, rerun with --confirm-split-review.")
    cfg = make_config(args)
    data_settings = make_data_settings(args)
    OUT = os.path.join(args.out_dir, args.name)
    CKPT = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    print("\nExperiment %s | settings in use (status shown for each setting):" % cfg.experiment)
    print("\n".join(tm.provisional_report(cfg)))
    print("  %-22s = %-12s  PROVISIONAL: full-run step budget pending final team confirmation\n" %
          ("steps", args.steps))

    train, ev, identity, overlap, split_provenance = build_data(args, want_identity=not args.benchmark)
    for source, target in ((args.split_npz, "corrected_split_indices.npz"),
                           (args.split_manifest, "corrected_split_manifest.json"),
                           (args.split_validation, "corrected_split_validation.json")):
        shutil.copy2(source, os.path.join(OUT, target))
    json.dump(split_provenance, open(os.path.join(OUT, "split_audit.json"), "w"), indent=2)
    models = tm.build_models(cfg)
    trainer = tm.Trainer(models, cfg)
    print("G params: %d | shared trunk params: %d" % (models.gen.count_params(), models.trunk.count_params()))

    # ---- benchmark ----
    if args.benchmark:
        src = tm.TrainBatchSource(train, cfg, seed=args.seed)
        for i in range(5):
            out = trainer.step(src.next(), i + 1)
        float(out["c_loss"])
        t0 = time.time()
        for i in range(args.benchmark):
            out = trainer.step(src.next(), i + 6)
        float(out["c_loss"])
        dt = (time.time() - t0) / args.benchmark
        print("\n=== %.3f s/step at %dx%d, %d unlabelled + %d labelled  ->  10k steps = %.1f h ===" % (
            dt, cfg.img_size, cfg.img_size, cfg.n_unlabeled, cfg.n_labeled, dt * 1e4 / 3600))
        return

    # ---- checkpoint / resume (pattern COPIED FROM gc_swgan_fixed.py, plus the manifest guard) ----
    step_var = tf.Variable(0, dtype=tf.int64)
    ckpt = tm.make_checkpoint(models, step_var)         # includes the pseudo-label snapshot if used
    mgr = tf.train.CheckpointManager(ckpt, CKPT, max_to_keep=3)
    state_path, manifest_path = os.path.join(CKPT, "state.json"), os.path.join(CKPT, "manifest.json")
    state = {"best_val_macro_f1": -1.0, "best_step": 0}
    manifest = tm.build_manifest(cfg, data_settings, identity,
                                 info={"overlap_check": {k: v for k, v in overlap.items()},
                                       "corrected_split": split_provenance})
    resuming = mgr.latest_checkpoint is not None
    try:
        tm.prepare_run_manifest(resuming, manifest_path, manifest)
        if resuming:
            tm.restore_checkpoint(ckpt, mgr.latest_checkpoint, models, cfg, step_var)
    except tm.ResumeError as e:
        sys.exit(str(e))
    if resuming:
        if os.path.exists(state_path):
            state = json.load(open(state_path))
        print("Resumed from %s at step %d (settings, data and pseudo-label snapshot verified)" % (
            mgr.latest_checkpoint, int(step_var.numpy())))
    start = int(step_var.numpy())
    # different data order after each resume, but reproducible
    source = tm.TrainBatchSource(train, cfg, seed=args.seed * 100003 + start)

    # ---- logs (append mode) ----
    paths = {k: os.path.join(OUT, k + ".csv") for k in ("loss", "val", "diagnostics")}
    new = {k: not os.path.exists(v) for k, v in paths.items()}
    files = {k: open(v, "a", newline="") for k, v in paths.items()}
    lw, vw, dw = (csv.writer(files[k]) for k in ("loss", "val", "diagnostics"))
    fieldnames = tm.log_fieldnames(cfg.n_classes)
    if new["loss"]:
        lw.writerow(fieldnames)
    if new["val"]:
        vw.writerow(["step", "val_acc", "val_macro_f1", "val_balanced_acc"])
    if new["diagnostics"]:
        dw.writerow(["step", "l1_generated_vs_clean", "l1_noisy_vs_clean", "input_dependence"])
    json.dump({"args": vars(args), "config": cfg.to_dict()}, open(os.path.join(OUT, "args.json"), "w"), indent=2)

    x_fixed = train.x_unl[np.random.default_rng(0).choice(len(train.x_unl), 8, replace=False)]  # TRAIN images
    t = {"log": time.time(), "start": time.time()}

    def save_ckpt():
        tm.save_checkpoint(mgr, models)
        json.dump(state, open(state_path, "w"))

    def on_log(step, row):
        now = time.time()
        sps = (now - t["log"]) / args.log_every
        t["log"] = now
        row["sec_per_step"] = "%.3f" % sps
        flat = tm.flatten_log_row(row, cfg.n_classes)
        lw.writerow([flat.get(k, "") for k in fieldnames])
        files["loss"].flush()
        print("step %d | C %.4f (sup %.4f, pl %.4f) | D %.4f | G %.4f (adv %.4f, rec %.4f) | lr %.2e | "
              "%.2f s/step" % (step, row["c_loss"], row["c_sup"], row["c_pseudo"], row["d_loss"],
                               row["g_loss"], row["g_adv"], row["g_rec"], row["lr"], sps), flush=True)
        print("   pseudo-labels: accepted %d/%d (%.1f%%) | predicted counts %s | accepted counts %s" % (
            row["pl_accepted"], row["pl_seen"], 100 * row["pl_accept_rate"],
            row["pl_pred_counts"], row["pl_accepted_counts"]), flush=True)

    def on_step(step):
        nonlocal state
        step_var.assign(step)
        if step % args.eval_every == 0:
            m = full_metrics(predict_probs(models.clf, ev["x_val"]), ev["y_val"])   # evaluation only
            vw.writerow([step, "%.4f" % m["accuracy"], "%.4f" % m["macro_f1"], "%.4f" % m["balanced_accuracy"]])
            files["val"].flush()
            print("   VAL acc %.4f | macro-F1 %.4f | bal-acc %.4f" % (
                m["accuracy"], m["macro_f1"], m["balanced_accuracy"]), flush=True)
            if m["macro_f1"] > state["best_val_macro_f1"]:
                state = {"best_val_macro_f1": m["macro_f1"], "best_step": step}
                models.clf.save_weights(os.path.join(CKPT, "best_clf"))
            save_sample_grid_and_diagnostics(step, models, cfg, x_fixed, OUT, dw, files["diagnostics"])
        if step % args.save_every == 0:
            save_ckpt()

    def should_stop(step):
        return bool(args.max_hours) and (time.time() - t["start"]) > args.max_hours * 3600

    status, last = tm.run_training(trainer, source, start, args.steps, args.log_every,
                                   on_log=on_log, on_step=on_step, should_stop=should_stop)
    save_ckpt()
    for f in files.values():
        f.close()
    print("Training ended: %s at step %d%s" % (
        status, last, " (re-run the same command to resume)" if status == "stopped" else ""))
    if status == "stopped" or args.skip_test:
        return

    # ---- final evaluation: test set touched ONCE ----
    results = {"args": vars(args), "config": cfg.to_dict(), "data_settings": data_settings,
               "dataset_identity": identity, "overlap_check": overlap,
               "corrected_split": split_provenance,
               "end_status": status, "last_step": last,
               "best_val_step": state["best_step"],
               "best_val_macro_f1": state["best_val_macro_f1"]}
    results["final_model"] = full_metrics(predict_probs(models.clf, ev["x_test"]), ev["y_test"])
    if os.path.exists(os.path.join(CKPT, "best_clf.index")):
        models.clf.load_weights(os.path.join(CKPT, "best_clf"))
        results["best_val_model"] = full_metrics(predict_probs(models.clf, ev["x_test"]), ev["y_test"])
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    for k in ("final_model", "best_val_model"):
        if k in results:
            r = results[k]
            print("%-16s acc %.4f | macro-F1 %.4f | wF1 %.4f | bal-acc %.4f | ECE %.3f" % (
                k, r["accuracy"], r["macro_f1"], r["weighted_f1"], r["balanced_accuracy"], r["ece"]))


if __name__ == "__main__":
    main()
