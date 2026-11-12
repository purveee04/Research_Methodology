"""
training_module.py  --  Member 2 (batching, pseudo-labelling, training orchestration)

NEW file. It does not modify or import gc_swgan_fixed.py (that script parses argv and builds
models at import time, so it cannot be imported). Everything reused from it is a VERBATIM COPY
marked "COPIED FROM gc_swgan_fixed.py" with the original function name.

What is here
    TrainConfig                 all knobs; validation; PROVISIONAL defaults are listed below
    TrainingData                the ONLY container the streams accept: x_lab, y_lab, x_unl
    split_indices / assert_no_leakage / check_cutout_overlap          leakage guards
    BatchStream / TrainBatchSource                  20 unlabelled + 1 labelled per step
    pseudo_label_from_logits / classifier_loss / compose_generator_loss
    build_models / Trainer / run_training           classifier -> critic -> generator step
    build_manifest / prepare_run_manifest           what a checkpoint was made with; resume guard
    make_checkpoint / save_checkpoint / restore_checkpoint     incl. the pseudo-label snapshot
    PseudoLabelWindow           acceptance rate + class counts accumulated on the GPU

Per-step order (same as gc_swgan_fixed.py, plus a pseudo-label prediction before step 1):
    0. pseudo-label the 20 CLEAN unlabelled images (softmax of logits, training=False)
    1. classifier step   : CE(labelled, true label) + w * mean CE(accepted unlabelled, pseudo-label)
    2. critic step       : Wasserstein loss + gradient penalty, real = CLEAN unlabelled images,
                           fake = G(add_gaussian_noise(clean))           (one critic step, as before)
    3. generator step    : g_adv = -mean(critic(G(noisy)))   [A]
                           g_adv + lambda_rec * L1(clean, G(noisy))      [B]

Numpy-only parts (config, streams, splits, leakage checks, manifests) import without TensorFlow, so
they can be unit-tested anywhere. TensorFlow parts need `pip install tf_keras` on TF >= 2.16.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")      # must precede the tensorflow import

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError:                                    # numpy-only parts stay usable
    tf = None
else:
    _tf_ver = tuple(int(v) for v in tf.__version__.split(".")[:2])
    if _tf_ver >= (2, 16):
        try:
            import tf_keras  # noqa: F401
        except ImportError:
            raise ImportError("TensorFlow >= 2.16 needs the legacy Keras package:  pip install tf_keras")
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.initializers import RandomNormal

try:
    import generator_module as gm                      # the merged Member 1 module (unchanged)
except ImportError:
    gm = None

NCLASSES = 10
CH = 3
SUPPORTED_IMG_SIZES = (192, 96, 48)                    # same set as generator_module._CFG
EXPERIMENTS = ("A", "B")
EXHAUSTION_POLICIES = ("replacement", "reshuffle", "stop")
REFRESH_MODES = ("every_step", "every_n_steps")

# --------------------------------------------------------------------------- #
# Config.  Values copied from gc_swgan_fixed.py are marked [baseline].
# Values marked PROVISIONAL are placeholders chosen by Member 2 so the code runs; they are NOT
# team-approved, NOT instructor-approved, and NOT tuned.  Change them in the launch command.
# --------------------------------------------------------------------------- #
PROVISIONAL = {
    "img_size":            "only completed baseline is 96x96; 192 untested with the new generator",
    "n_unlabeled":         "team guide: 20 'for now', may be tuned",
    "n_labeled":           "team guide: 1 'for now', may be tuned",
    "sigma":               "no sigma approved; choose on validation pilots, then freeze",
    "lambda_rec":          "NO DEFAULT: required for Experiment B, must be unset for A",
    "pseudo_threshold":    "no value was ever specified",
    "pseudo_refresh":      "live classifier at the start of every step",
    "pseudo_refresh_every": "only used when pseudo_refresh == 'every_n_steps'",
    "pseudo_loss_weight":  "how the pseudo-label CE is weighted against the labelled CE is undecided",
    "exhaustion_policy":   "baseline sampled with replacement forever; 20:1 epoch behaviour undecided",
}


@dataclass
class TrainConfig:
    experiment: str = "A"                    # "A": adversarial G loss only | "B": + lambda_rec * L1
    img_size: int = 96                       # PROVISIONAL
    n_unlabeled: int = 20                    # PROVISIONAL (team guide, may be tuned)
    n_labeled: int = 1                       # PROVISIONAL (team guide, may be tuned)
    sigma: float = 0.1                       # PROVISIONAL, pixel units of [-1, 1]
    lambda_rec: Optional[float] = None       # REQUIRED > 0 for B, must be None for A
    pseudo_threshold: float = 0.95           # PROVISIONAL, accepted iff max prob >= threshold
    pseudo_refresh: str = "every_step"       # PROVISIONAL: every_step | every_n_steps
    pseudo_refresh_every: int = 100          # PROVISIONAL (only for every_n_steps)
    pseudo_loss_weight: float = 1.0          # PROVISIONAL
    exhaustion_policy: str = "reshuffle"     # PROVISIONAL: replacement | reshuffle | stop
    aug_labeled: bool = True                 # [baseline] labelled stream was augmented
    aug_unlabeled: bool = False              # [baseline] --aug-unlabeled default off
    gp_weight: float = 10.0                  # [baseline] --lam
    lr: float = 1e-4                         # [baseline]
    min_lr: float = 1e-6                     # [baseline]
    decay: float = 1.000004                  # [baseline]
    n_classes: int = NCLASSES
    seed: int = 0

    def __post_init__(self):
        self.experiment = str(self.experiment).upper()
        _choice("experiment", self.experiment, EXPERIMENTS)
        _choice("img_size", self.img_size, SUPPORTED_IMG_SIZES)
        _choice("pseudo_refresh", self.pseudo_refresh, REFRESH_MODES)
        _choice("exhaustion_policy", self.exhaustion_policy, EXHAUSTION_POLICIES)
        if self.n_unlabeled < 1 or self.n_labeled < 1:
            raise ValueError("n_unlabeled and n_labeled must be >= 1")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        if not 0.0 <= self.pseudo_threshold <= 1.0:
            raise ValueError("pseudo_threshold must be in [0, 1]")
        if self.pseudo_refresh_every < 1:
            raise ValueError("pseudo_refresh_every must be >= 1")
        if self.pseudo_loss_weight < 0:
            raise ValueError("pseudo_loss_weight must be >= 0")
        if self.lr <= 0 or self.min_lr <= 0 or self.decay <= 0 or self.gp_weight < 0:
            raise ValueError("lr, min_lr, decay must be > 0 and gp_weight >= 0")
        if self.experiment == "B":
            if self.lambda_rec is None or not self.lambda_rec > 0:
                raise ValueError("Experiment B needs lambda_rec > 0 (no default: the team has not chosen one)")
        else:
            if self.lambda_rec not in (None, 0, 0.0):
                raise ValueError("Experiment A is adversarial-only: lambda_rec must be unset (got %r)"
                                 % (self.lambda_rec,))
            self.lambda_rec = None

    def lr_at(self, step):                   # [baseline] lr_at(step)
        return max(self.lr / (self.decay ** step), self.min_lr)

    def to_dict(self):
        return asdict(self)


def _choice(name, value, choices):
    if value not in choices:
        raise ValueError("%s must be one of %s, got %r" % (name, list(choices), value))


def provisional_report(cfg):
    """Lines for the launch banner: every PROVISIONAL value actually in use."""
    return ["  %-22s = %-12s  PROVISIONAL (not approved): %s" % (k, getattr(cfg, k), why)
            for k, why in PROVISIONAL.items()]


# --------------------------------------------------------------------------- #
# Run manifest: what a checkpoint was made with, and the resume guard
# --------------------------------------------------------------------------- #
class ResumeError(RuntimeError):
    """Raised when a checkpoint cannot be resumed exactly with the current settings/data."""


MANIFEST_FORMAT = 2
DATA_SETTING_KEYS = ("split_seed", "test_frac", "val_frac", "label_frac",
                     "include_labelled_in_unlabeled", "no_pool_unlabeled")
IDENTITY_KEYS = ("labeled_file", "unlabeled_file", "labels", "idx_pool", "idx_lab",
                 "idx_val", "idx_test", "idx_unl_pool")


def make_data_settings(**kw):
    """The data-pipeline settings that decide WHICH images are train/val/test/labelled/unlabelled.
    All keys are mandatory, so a forgotten one is an error instead of a silent gap in the guard."""
    missing, extra = set(DATA_SETTING_KEYS) - set(kw), set(kw) - set(DATA_SETTING_KEYS)
    if missing or extra:
        raise ValueError("data settings: missing %s, unexpected %s" % (sorted(missing), sorted(extra)))
    return {k: kw[k] for k in DATA_SETTING_KEYS}


def file_fingerprint(path, chunk=1 << 24):
    """Content identity of a file: size + SHA-256 (streamed; ~10-20 s per GB-scale file)."""
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            size += len(block)
    return {"size": size, "sha256": h.hexdigest()}


def array_fingerprint(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(repr((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def dataset_identity(labeled_path, unlabeled_path, labels, ix, unl_pool_idx):
    """Dataset identity for the manifest: SHA-256 of both HDF5 files (None if there is no unlabelled
    file) plus hashes of the labels and of the (sorted) index sets that define the partition."""
    have_unl = bool(unlabeled_path) and os.path.exists(unlabeled_path)
    return {
        "labeled_file": file_fingerprint(labeled_path),
        "unlabeled_file": file_fingerprint(unlabeled_path) if have_unl else None,
        "labels": array_fingerprint(np.asarray(labels, np.int64)),
        "idx_pool": array_fingerprint(np.sort(ix["pool"])),
        "idx_lab": array_fingerprint(np.sort(ix["lab"])),
        "idx_val": array_fingerprint(np.sort(ix["val"])),
        "idx_test": array_fingerprint(np.sort(ix["test"])),
        "idx_unl_pool": array_fingerprint(np.sort(np.asarray(unl_pool_idx))),
    }


def _normalize(obj):
    """Round-trip through JSON so tuples/lists, numpy scalars etc. compare the way the file will."""
    return json.loads(json.dumps(obj, sort_keys=True, default=_json_default))


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % (o,))


def build_manifest(cfg, data_settings, identity, info=None):
    """Everything that determines what a checkpoint's training run was.
    Compared on resume: format, config (EVERY TrainConfig field), data, identity.
    NOT compared: info (free-form notes such as the overlap-check result) and the total step budget."""
    if set(data_settings) != set(DATA_SETTING_KEYS):
        raise ValueError("data_settings must be made with make_data_settings()")
    if set(identity) != set(IDENTITY_KEYS):
        raise ValueError("identity must be made with dataset_identity()")
    return _normalize({"format": MANIFEST_FORMAT, "config": cfg.to_dict(), "data": data_settings,
                       "identity": identity, "info": info or {}})


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


_MISSING = object()


def diff_manifests(saved, new):
    """Human-readable list of every difference between two manifests (ignores 'info')."""
    a = _flatten({k: v for k, v in saved.items() if k != "info"})
    b = _flatten({k: v for k, v in new.items() if k != "info"})
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, _MISSING), b.get(k, _MISSING)
        if va != vb:
            out.append("%s: checkpoint=%s now=%s" % (
                k, "<absent>" if va is _MISSING else repr(va), "<absent>" if vb is _MISSING else repr(vb)))
    return out


def check_resume_compatible(saved, new):
    if not isinstance(saved, dict) or saved.get("format") != MANIFEST_FORMAT:
        raise ResumeError("Checkpoint manifest is missing or has an unknown format; refusing to resume.")
    diffs = diff_manifests(saved, _normalize(new))
    if diffs:
        raise ResumeError("Refusing to resume: the checkpoint was made with different settings or data:\n  "
                          + "\n  ".join(diffs))


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def prepare_run_manifest(has_checkpoint, path, manifest):
    """Fresh run: write the manifest, return 'fresh'. Existing checkpoint: the saved manifest must exist
    and match exactly, else ResumeError; the saved manifest is never overwritten. Returns 'resume'."""
    new = _normalize(manifest)
    if not has_checkpoint:
        _atomic_write_json(path, new)
        return "fresh"
    if not os.path.exists(path):
        raise ResumeError("A checkpoint exists but %s is missing: cannot verify its settings, refusing to "
                          "resume." % path)
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError) as e:
        raise ResumeError("Cannot read %s (%s); refusing to resume." % (path, e))
    check_resume_compatible(saved, new)
    return "resume"


# --------------------------------------------------------------------------- #
# Data container + leakage guards
# --------------------------------------------------------------------------- #
class LeakageError(RuntimeError):
    pass


@dataclass
class TrainingData:
    """The ONLY thing the batch streams accept. It has no validation or test fields by design:
    x_lab/y_lab  labelled TRAIN images + true labels
    x_unl        unlabelled TRAIN images (labels discarded) [+ extra cutouts]
    All images uint8 (N, S, S, 3)."""
    x_lab: np.ndarray
    y_lab: np.ndarray
    x_unl: np.ndarray

    def __post_init__(self):
        for name in ("x_lab", "x_unl"):
            a = np.asarray(getattr(self, name))
            if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != CH:
                raise ValueError("%s must be uint8 (N,S,S,%d), got %s %s" % (name, CH, a.dtype, a.shape))
            setattr(self, name, a)
        if self.x_lab.shape[1:] != self.x_unl.shape[1:]:
            raise ValueError("labelled and unlabelled images differ in shape")
        self.y_lab = np.asarray(self.y_lab).astype(np.int64)
        if self.y_lab.shape != (len(self.x_lab),):
            raise ValueError("y_lab must have shape (len(x_lab),)")
        if len(self.y_lab) and (self.y_lab.min() < 0 or self.y_lab.max() >= NCLASSES):
            raise ValueError("labels must be in [0, %d)" % NCLASSES)
        if len(self.x_lab) == 0 or len(self.x_unl) == 0:
            raise ValueError("need at least one labelled and one unlabelled training image")


def image_hashes(x):
    """16-byte content hash of every image (exact byte equality)."""
    return {hashlib.blake2b(np.ascontiguousarray(x[i]).tobytes(), digest_size=16).digest()
            for i in range(len(x))}


def assert_no_leakage(train, **protected):
    """EXACT-DUPLICATE check (unchanged): raise LeakageError if any training/unlabelled image is
    byte-identical to (or shares memory with) a protected array, e.g.
    assert_no_leakage(train, x_val=x_val, x_test=x_test).
    It cannot see a different cutout of the same galaxy; see check_cutout_overlap for that."""
    if not isinstance(train, TrainingData):
        raise TypeError("train must be a TrainingData")
    mine = {"x_lab": train.x_lab, "x_unl": train.x_unl}
    hashes = {k: image_hashes(v) for k, v in mine.items()}
    for pname, parr in protected.items():
        parr = np.asarray(parr)
        ph = image_hashes(parr)
        for k, arr in mine.items():
            if np.may_share_memory(arr, parr):
                raise LeakageError("%s shares memory with protected set %r" % (k, pname))
            common = hashes[k] & ph
            if common:
                raise LeakageError("%d image(s) in %s are byte-identical to protected set %r"
                                   % (len(common), k, pname))


# ---- sky-position / cutout-footprint overlap ---------------------------------- #
def _unit(ra, dec):                                    # [kaggle_prep_data.py] unit()
    ra, dec = np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)


def positional_match_mask(ra, dec, ref_ra, ref_dec, arcsec=3.0):
    """True for each (ra, dec) whose CENTRE is within `arcsec` of some reference CENTRE.
    Same geometry as kaggle_prep_data.py (cKDTree on unit vectors, chord distance)."""
    from scipy.spatial import cKDTree
    ra = np.asarray(ra, float)
    if len(ra) == 0 or len(ref_ra) == 0:
        return np.zeros(len(ra), bool)
    tree = cKDTree(_unit(ref_ra, ref_dec))
    chord = 2 * np.sin(np.deg2rad(arcsec / 3600.0) / 2)
    dist, _ = tree.query(_unit(ra, dec))
    return dist <= chord


def count_positional_matches(ra, dec, ref_ra, ref_dec, arcsec=3.0):
    return int(positional_match_mask(ra, dec, ref_ra, ref_dec, arcsec).sum())


def footprint_overlap_pairs(ra, dec, half_arcsec, ref_ra, ref_dec, ref_half_arcsec):
    """(i, j) pairs whose square cutouts MAY overlap on the sky.
    Each cutout is a square of half-side `half` arcsec centred on (ra, dec). Orientation is not
    assumed: each square is replaced by its circumscribed circle (radius half*sqrt(2)), so a pair is
    reported iff separation < (half_i + half_j)*sqrt(2). This is conservative: it never misses a real
    overlap whatever the rotation, but it can flag squares that only nearly touch diagonally."""
    from scipy.spatial import cKDTree
    ra = np.asarray(ra, float)
    ref_ra = np.asarray(ref_ra, float)
    if len(ra) == 0 or len(ref_ra) == 0:
        return []
    ha = np.broadcast_to(np.asarray(half_arcsec, float), (len(ra),))
    hr = np.broadcast_to(np.asarray(ref_half_arcsec, float), (len(ref_ra),))
    ua, ur = _unit(ra, dec), _unit(ref_ra, ref_dec)
    reach_max = (ha.max() + hr.max()) * np.sqrt(2.0)
    chord_max = 2 * np.sin(np.deg2rad(reach_max / 3600.0) / 2)
    cand = cKDTree(ua).query_ball_tree(cKDTree(ur), chord_max)
    pairs = []
    for i, js in enumerate(cand):
        for j in js:
            chord = float(np.linalg.norm(ua[i] - ur[j]))
            sep_arcsec = np.rad2deg(2 * np.arcsin(min(chord / 2, 1.0))) * 3600.0
            if sep_arcsec < (ha[i] + hr[j]) * np.sqrt(2.0):
                pairs.append((i, int(j)))
    return pairs


def check_cutout_overlap(extra_ra, extra_dec, g_ra, g_dec, protected_idx, other_idx=None,
                         extra_pxscale=None, extra_pxscale_source=None, g_pxscale=None,
                         size_px=256, center_arcsec=3.0):
    """Do the extra unlabelled cutouts overlap validation/test galaxies on the sky?

    protected_idx: indices (into the Galaxy10 arrays) of the validation + test galaxies -> hard failure.
    other_idx:     remaining Galaxy10 galaxies (train pool) -> reported only.

    mode "footprint": needs the pixel scale of the extra cutouts (extra_pxscale, scalar or per-cutout)
        AND Galaxy10's per-galaxy pxscale. Full size_px x size_px footprints, orientation-independent
        circumscribed-circle bound. Relies on that metadata being correct; it does not look at pixels.
    mode "center_only_limited": otherwise. Only the CENTRES are compared. This does NOT show that the
        256x256 cutouts exclude validation/test galaxies, and the message says so.
    Returns a dict with mode, ok (False iff a validation/test overlap was found), counts, message."""
    extra_ra, extra_dec = np.asarray(extra_ra, float), np.asarray(extra_dec, float)
    n = len(extra_ra)
    g_ra, g_dec = np.asarray(g_ra, float), np.asarray(g_dec, float)
    prot = np.asarray(protected_idx, dtype=int)
    oth = np.asarray([] if other_idx is None else other_idx, dtype=int)
    res = {"n_extra": int(n), "n_protected_galaxies": int(len(prot))}

    if extra_pxscale is not None and g_pxscale is not None:
        ex_px = np.asarray(extra_pxscale, float)
        g_px = np.asarray(g_pxscale, float)
        if ex_px.ndim and ex_px.shape != (n,):
            raise ValueError("extra_pxscale must be a scalar or have one value per extra cutout")
        if (ex_px <= 0).any() or (g_px <= 0).any():
            raise ValueError("pixel scales must be > 0")
        half_e = 0.5 * size_px * ex_px
        half_g = 0.5 * size_px * g_px

        def n_overlapping(idx):
            if len(idx) == 0:
                return 0
            pairs = footprint_overlap_pairs(extra_ra, extra_dec, half_e, g_ra[idx], g_dec[idx], half_g[idx])
            return len({i for i, _ in pairs})
        n_p, n_o = n_overlapping(prot), n_overlapping(oth)
        res.update(mode="footprint", n_overlap_protected=int(n_p), n_overlap_other=int(n_o), ok=(n_p == 0))
        src = extra_pxscale_source or "unspecified source"
        basis = ("%dx%d-pixel footprints, orientation-independent circumscribed-circle bound; Galaxy10 "
                 "pxscale read from file; extra-cutout pixscale = %.4f arcsec/px (%s). Metadata-based: it "
                 "does not inspect pixels and is only as good as that metadata."
                 % (size_px, size_px, float(np.median(ex_px)), src))
        if n_p:
            res["message"] = ("LEAKAGE RISK (footprint check): %d of %d extra cutouts overlap a validation/test "
                              "galaxy footprint. Basis: %s" % (n_p, n, basis))
        else:
            res["message"] = ("Footprint check: no extra-cutout footprint intersects a validation/test galaxy "
                              "footprint (%d other-Galaxy10 overlaps, informational). Basis: %s" % (n_o, basis))
        return res

    n_p = int(positional_match_mask(extra_ra, extra_dec, g_ra[prot], g_dec[prot], center_arcsec).sum())
    n_o = int(positional_match_mask(extra_ra, extra_dec, g_ra[oth], g_dec[oth], center_arcsec).sum()) if len(oth) else 0
    res.update(mode="center_only_limited", n_overlap_protected=n_p, n_overlap_other=n_o, ok=(n_p == 0))
    if n_p:
        res["message"] = ("LEAKAGE RISK (limited center-match check): %d of %d extra cutout centres lie within "
                          "%.1f arcsec of a validation/test galaxy centre." % (n_p, n, center_arcsec))
    else:
        res["message"] = ("LIMITED center-match check only: no extra-cutout CENTRE lies within %.1f arcsec of a "
                          "validation/test galaxy CENTRE. This does NOT show that the %dx%d-pixel cutouts "
                          "exclude those galaxies, because the unlabelled file does not store its cutout pixel "
                          "scale. Pass --unlabeled-pixscale (the value used when the cutouts were made) to "
                          "enable the footprint check. Not evidence that leakage is absent."
                          % (center_arcsec, size_px, size_px))
    return res


def split_indices(labels, test_frac, val_frac, label_frac, split_seed):
    """COPIED FROM gc_swgan_fixed.py build_data(): identical sklearn calls, so the same
    --split-seed gives the same train/val/test/labelled partition as the baseline.
    Returns dict of index arrays: pool (train pool), lab (labelled subset of pool), val, test."""
    from sklearn.model_selection import train_test_split
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=split_seed)
    pool_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_frac / (1.0 - test_frac),
        stratify=labels[trainval_idx], random_state=split_seed)
    if label_frac < 1.0:
        lab_idx, _ = train_test_split(
            pool_idx, train_size=label_frac, stratify=labels[pool_idx],
            random_state=split_seed)
    else:
        lab_idx = pool_idx
    out = {"pool": pool_idx, "lab": lab_idx, "val": val_idx, "test": test_idx}
    p, v, t = set(pool_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    if (p & v) or (p & t) or (v & t) or not set(lab_idx.tolist()) <= p:
        raise LeakageError("train/val/test index sets are not disjoint")      # cannot happen; guard
    return out


def unlabeled_pool_indices(pool_idx, lab_idx, include_labelled):
    """Train-pool images whose labels are DISCARDED and which may be pseudo-labelled.
    The baseline included the labelled subset too (x_all[pool_idx]); by default we exclude it so
    'unlabelled' means unlabelled. Set include_labelled=True to reproduce the baseline pool."""
    pool_idx = np.asarray(pool_idx)
    if include_labelled:
        return pool_idx
    return pool_idx[~np.isin(pool_idx, np.asarray(lab_idx))]


# --------------------------------------------------------------------------- #
# Batch streams
# --------------------------------------------------------------------------- #
class StreamExhausted(Exception):
    """Raised by policy 'stop' when a full pass over the data is complete."""


def augment_dihedral_u8(batch, rng):
    """Per-sample random flip / flip / transpose, same three ops as augment_u8 in gc_swgan_fixed.py
    (numpy version; labels are unaffected)."""
    out = np.empty_like(batch)
    for i in range(len(batch)):
        im = batch[i]
        if rng.random() < 0.5:
            im = im[:, ::-1]
        if rng.random() < 0.5:
            im = im[::-1]
        if rng.random() < 0.5:
            im = np.transpose(im, (1, 0, 2))
        out[i] = im
    return out


class BatchStream:
    """Yields fixed-size batches from ONE array.
    policy 'replacement': infinite, rng.integers draws (exactly the baseline's sampling)
    policy 'reshuffle'  : one pass without replacement per epoch, incomplete tail dropped, then new order
    policy 'stop'       : one such pass, then StreamExhausted (forever)"""

    def __init__(self, x, y, batch_size, seed, policy="reshuffle", augment=False):
        _choice("policy", policy, EXHAUSTION_POLICIES)
        if y is not None and len(y) != len(x):
            raise ValueError("x and y differ in length")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if policy != "replacement" and len(x) < batch_size:
            raise ValueError("policy %r needs at least batch_size=%d images, have %d"
                             % (policy, batch_size, len(x)))
        self.x, self.y, self.bs, self.policy, self.augment = x, y, batch_size, policy, augment
        self.n = len(x)
        self._rng = np.random.default_rng(seed)
        self.epochs_completed = 0
        self.batches_yielded = 0
        self.last_indices = None
        self._exhausted = False
        self._order, self._pos = None, 0
        if policy != "replacement":
            self._new_order()

    def _new_order(self):
        self._order, self._pos = self._rng.permutation(self.n), 0

    def next_batch(self):
        if self._exhausted:
            raise StreamExhausted("stream already exhausted")
        if self.policy == "replacement":
            idx = self._rng.integers(0, self.n, self.bs)
        else:
            if self._pos + self.bs > self.n:
                self.epochs_completed += 1
                if self.policy == "stop":
                    self._exhausted = True
                    raise StreamExhausted("one full pass completed (%d batches)" % self.batches_yielded)
                self._new_order()
            idx = self._order[self._pos:self._pos + self.bs]
            self._pos += self.bs
        self.last_indices = idx
        xb = self.x[idx]
        if self.augment:
            xb = augment_dihedral_u8(xb, self._rng)
        self.batches_yielded += 1
        return xb, (self.y[idx] if self.y is not None else None)


class TrainBatchSource:
    """One training step's data: (x_lab_u8 [n_lab], y_lab [n_lab], x_unl_u8 [n_unl]).
    Accepts ONLY a TrainingData, so validation/test arrays have no way in."""

    def __init__(self, data, cfg, seed=None):
        if not isinstance(data, TrainingData):
            raise TypeError("TrainBatchSource only accepts TrainingData (train images only)")
        seed = cfg.seed if seed is None else seed
        self.lab = BatchStream(data.x_lab, data.y_lab, cfg.n_labeled, seed,
                               cfg.exhaustion_policy, cfg.aug_labeled)
        self.unl = BatchStream(data.x_unl, None, cfg.n_unlabeled, seed + 1,
                               cfg.exhaustion_policy, cfg.aug_unlabeled)

    def next(self):
        x_lab, y_lab = self.lab.next_batch()
        x_unl, _ = self.unl.next_batch()
        return x_lab, y_lab, x_unl


# --------------------------------------------------------------------------- #
# TensorFlow parts
# --------------------------------------------------------------------------- #
def _require_tf():
    if tf is None:
        raise RuntimeError("TensorFlow is required here (pip install tensorflow tf_keras)")


def _require_gm():
    _require_tf()
    if gm is None:
        raise RuntimeError("generator_module.py must be importable (put it next to this file)")


def to_float(x_u8):                                    # COPIED FROM gc_swgan_fixed.py
    return tf.cast(x_u8, tf.float32) / 127.5 - 1.0


# ---- pseudo-labelling ------------------------------------------------------ #
def pseudo_label_from_logits(logits, threshold):
    """logits (B,K) -> (probs, pred, conf, accepted). probs = softmax(logits);
    pred = argmax class; accepted = (max prob >= threshold)."""
    probs = tf.nn.softmax(logits, axis=-1)
    conf = tf.reduce_max(probs, axis=-1)
    pred = tf.argmax(probs, axis=-1)                   # int64
    accepted = conf >= tf.cast(threshold, conf.dtype)
    return probs, pred, conf, accepted


def pseudo_label_counts(pred, accepted, num_classes):
    """(pred_counts[K], accepted_counts[K]) as float32; static shapes so it is XLA friendly."""
    oh = tf.one_hot(pred, num_classes, dtype=tf.float32)
    return (tf.reduce_sum(oh, axis=0),
            tf.reduce_sum(oh * tf.cast(accepted, tf.float32)[:, None], axis=0))


class PseudoLabelWindow:
    """Running totals kept in tf.Variables and updated INSIDE the tf.function, so there is no
    per-step GPU sync (the baseline avoided .numpy() per step too). Read and reset at log time."""

    def __init__(self, num_classes):
        _require_tf()
        self.k = num_classes
        self.seen = tf.Variable(0.0, trainable=False)
        self.accepted = tf.Variable(0.0, trainable=False)
        self.pred_counts = tf.Variable(tf.zeros([num_classes]), trainable=False)
        self.accepted_counts = tf.Variable(tf.zeros([num_classes]), trainable=False)

    def update(self, n_seen, pred_counts, accepted_counts):
        self.seen.assign_add(tf.cast(n_seen, tf.float32))
        self.accepted.assign_add(tf.reduce_sum(accepted_counts))
        self.pred_counts.assign_add(pred_counts)
        self.accepted_counts.assign_add(accepted_counts)

    def read_and_reset(self):
        seen, acc = float(self.seen.numpy()), float(self.accepted.numpy())
        out = {"pl_seen": int(round(seen)), "pl_accepted": int(round(acc)),
               "pl_accept_rate": (acc / seen) if seen > 0 else float("nan"),
               "pl_pred_counts": [int(round(v)) for v in self.pred_counts.numpy()],
               "pl_accepted_counts": [int(round(v)) for v in self.accepted_counts.numpy()]}
        for v in (self.seen, self.accepted):
            v.assign(0.0)
        for v in (self.pred_counts, self.accepted_counts):
            v.assign(tf.zeros([self.k]))
        return out


# ---- losses ----------------------------------------------------------------- #
def classifier_loss(logits_lab, y_lab, logits_unl, pseudo_labels, accepted, pseudo_weight):
    """total = CE_labelled_mean + pseudo_weight * (sum over ACCEPTED of CE / max(1, n_accepted)).
    The labelled term is exactly the baseline's classifier loss. A float mask (not boolean_mask)
    keeps shapes static. With nothing accepted the pseudo term is exactly 0."""
    ce_lab = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        y_lab, logits_lab, from_logits=True))
    ce_unl = tf.keras.losses.sparse_categorical_crossentropy(
        pseudo_labels, logits_unl, from_logits=True)
    m = tf.cast(accepted, tf.float32)
    ce_pseudo = tf.reduce_sum(m * ce_unl) / tf.maximum(tf.reduce_sum(m), 1.0)
    return ce_lab + pseudo_weight * ce_pseudo, ce_lab, ce_pseudo


def compose_generator_loss(g_adv, clean, generated, experiment, lambda_rec):
    """Experiment A: g_adv only (reconstruction helper is NOT called; g_rec reported as 0).
    Experiment B: g_adv + lambda_rec * reconstruction_l1(clean, generated).
    Returns (g_total, g_rec)."""
    if experiment == "A":
        return g_adv, tf.zeros([], tf.float32)
    _require_gm()
    g_rec = gm.reconstruction_l1(clean, generated)
    return g_adv + lambda_rec * g_rec, g_rec


# ---- models (architecture COPIED FROM gc_swgan_fixed.py) ------------------- #
if tf is not None:
    INIT = RandomNormal(mean=0.0, stddev=0.02)   # shared instance, as in the baseline (see notes)

_D_CFG = {5: [(64, 1), (64, 2), (128, 2), (128, 2), (256, 2), (256, 2), (256, 1)],
          4: [(64, 1), (64, 2), (128, 2), (256, 2), (256, 2), (256, 1)],
          3: [(64, 1), (64, 2), (128, 2), (256, 2), (256, 1)]}
_N_UP = {192: 5, 96: 4, 48: 3}


def build_trunk(img_size):
    """COPIED FROM gc_swgan_fixed.py build_trunk(): shared feature extractor of critic + classifier
    (LayerNorm: required for the gradient penalty)."""
    _require_tf()
    m = Sequential(name="trunk")
    m.add(layers.Input((img_size, img_size, CH)))
    cfg = _D_CFG[_N_UP[img_size]]
    for i, (f, s) in enumerate(cfg):
        m.add(layers.Conv2D(f, 4, strides=s, padding="same", use_bias=False,
                            kernel_initializer=INIT))
        m.add(layers.LayerNormalization())
        m.add(layers.LeakyReLU(0.2))
        if i < len(cfg) - 1:
            m.add(layers.Dropout(0.15))
    m.add(layers.Flatten())
    m.add(layers.Dropout(0.15))
    return m


def build_heads(trunk, img_size, n_classes):
    """critic = trunk + Dense(1) (no sigmoid); classifier = trunk + Dense(K) (LOGITS).
    Both are called once on a dummy input so their weights exist before optimizers read
    .trainable_variables -- the baseline crashed here on TF 2.20 (notebook In[8]); the notebook's
    patch did the same dummy call."""
    critic = Sequential([trunk, layers.Dense(1)], name="critic")
    clf = Sequential([trunk, layers.Dense(n_classes)], name="classifier")
    dummy = tf.zeros((1, img_size, img_size, CH))
    clf(dummy)
    critic(dummy)
    return critic, clf


def expected_snapshot_step(step, every):
    """Step at which the pseudo-label snapshot was last refreshed, given the refresh rule
    'copy live classifier -> snapshot after every `every`-th completed step' (and once at step 0)."""
    return (int(step) // int(every)) * int(every)


@dataclass
class Models:
    gen: object
    trunk: object
    critic: object
    clf: object
    opt_c: object
    opt_d: object
    opt_g: object
    clf_snapshot: object = None          # only for pseudo_refresh == "every_n_steps"
    snapshot_step: object = None         # tf.Variable: step of the last snapshot refresh (saved)
    snapshot_valid: object = None        # tf.Variable: set to 1 only when a checkpoint is SAVED

    @property
    def pseudo_label_model(self):
        return self.clf if self.clf_snapshot is None else self.clf_snapshot

    def sync_snapshot(self, step=0):
        """Copy live classifier weights into the snapshot and record the refresh step."""
        if self.clf_snapshot is None:
            return
        src, dst = self.clf.variables, self.clf_snapshot.variables
        assert len(src) == len(dst)
        for d, s in zip(dst, src):
            d.assign(s)
        self.snapshot_step.assign(int(step))


def build_models(cfg):
    """Current generator_module generator + baseline trunk/critic/classifier + three Adam(beta_1=0.5),
    exactly as the baseline built them."""
    _require_gm()
    from tensorflow.keras.optimizers import Adam
    gen = gm.build_image_generator(cfg.img_size)
    assert gen.output_shape[1:] == (cfg.img_size, cfg.img_size, CH), gen.output_shape
    trunk = build_trunk(cfg.img_size)
    critic, clf = build_heads(trunk, cfg.img_size, cfg.n_classes)
    opt_c = Adam(cfg.lr, beta_1=0.5)
    opt_d = Adam(cfg.lr, beta_1=0.5)
    opt_g = Adam(cfg.lr, beta_1=0.5)
    for opt, vs in ((opt_c, clf.trainable_variables),
                    (opt_d, critic.trainable_variables),
                    (opt_g, gen.trainable_variables)):
        try:
            opt.build(vs)           # create slot variables so checkpoint restore is exact
        except Exception:
            pass
    snap = snap_step = snap_valid = None
    if cfg.pseudo_refresh == "every_n_steps":
        snap_trunk = build_trunk(cfg.img_size)
        _, snap = build_heads(snap_trunk, cfg.img_size, cfg.n_classes)
        snap_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        snap_valid = tf.Variable(0, dtype=tf.int64, trainable=False)
    m = Models(gen, trunk, critic, clf, opt_c, opt_d, opt_g, snap, snap_step, snap_valid)
    m.sync_snapshot(0)
    return m


# ---- checkpointing (incl. the pseudo-label snapshot) -------------------------- #
def make_checkpoint(models, step_var):
    """tf.train.Checkpoint of everything that defines training state. In 'every_n_steps' mode this
    also tracks the classifier snapshot and its refresh position."""
    _require_tf()
    kw = dict(step=step_var, generator=models.gen, trunk=models.trunk, critic=models.critic,
              classifier=models.clf, opt_c=models.opt_c, opt_d=models.opt_d, opt_g=models.opt_g)
    if models.clf_snapshot is not None:
        kw.update(clf_snapshot=models.clf_snapshot, snapshot_step=models.snapshot_step,
                  snapshot_valid=models.snapshot_valid)
    return tf.train.Checkpoint(**kw)


def save_checkpoint(mgr, models):
    """Marks the snapshot as part of this checkpoint (so restore can tell a checkpoint that lacks it)
    and saves."""
    if models.snapshot_valid is not None:
        models.snapshot_valid.assign(1)
    return mgr.save()


def restore_checkpoint(ckpt, path, models, cfg, step_var):
    """Restore, then VERIFY the pseudo-label snapshot instead of silently re-syncing it to the live
    weights. Raises ResumeError if the checkpoint has no snapshot or its refresh position does not
    match the step. Returns the restored step."""
    ckpt.restore(path).expect_partial()
    step = int(step_var.numpy())
    if models.clf_snapshot is not None:
        if int(models.snapshot_valid.numpy()) != 1:
            raise ResumeError("pseudo_refresh='every_n_steps' but the checkpoint contains no saved "
                              "pseudo-label snapshot (made in another mode or by older code): refusing "
                              "to resume, because re-syncing it would change which labels are accepted.")
        want = expected_snapshot_step(step, cfg.pseudo_refresh_every)
        got = int(models.snapshot_step.numpy())
        if got != want:
            raise ResumeError("snapshot refresh position is inconsistent: checkpoint step %d needs a "
                              "snapshot taken at step %d but it records step %d; refusing to resume."
                              % (step, want, got))
    return step


# ---- the training step ------------------------------------------------------ #
def make_train_step(models, cfg, window):
    """Returns a tf.function  step(x_lab_u8, y_lab, x_unl_u8, lr) -> dict of scalar tensors.
    Models and optimizers are captured, never rebuilt: state persists across calls."""
    _require_gm()
    gen, critic, clf = models.gen, models.critic, models.clf
    opt_c, opt_d, opt_g = models.opt_c, models.opt_d, models.opt_g
    pl_model = models.pseudo_label_model
    K = cfg.n_classes

    @tf.function
    def train_step(x_lab_u8, y_lab, x_unl_u8, lr):
        for o in (opt_c, opt_d, opt_g):
            o.learning_rate.assign(lr)
        x_lab = to_float(x_lab_u8)
        x_unl = to_float(x_unl_u8)               # CLEAN unlabelled training images
        B = tf.shape(x_unl)[0]
        n_lab = tf.shape(x_lab)[0]

        # (0) pseudo-labels for the UNLABELLED images only; inference mode (dropout off)
        _, pred, conf, accepted = pseudo_label_from_logits(
            tf.stop_gradient(pl_model(x_unl, training=False)), cfg.pseudo_threshold)
        pred = tf.stop_gradient(pred)
        pred_counts, acc_counts = pseudo_label_counts(pred, accepted, K)
        window.update(B, pred_counts, acc_counts)

        # (1) classifier: labelled (true label) + accepted pseudo-labelled (trunk + classifier head).
        #     One forward pass over [labelled; unlabelled]: LayerNorm => no coupling between samples.
        with tf.GradientTape() as t:
            logits = clf(tf.concat([x_lab, x_unl], axis=0), training=True)
            c_loss, c_sup, c_pl = classifier_loss(
                logits[:n_lab], y_lab, logits[n_lab:], pred, accepted, cfg.pseudo_loss_weight)
        g = t.gradient(c_loss, clf.trainable_variables)
        opt_c.apply_gradients(zip(g, clf.trainable_variables))

        # (2) critic: Wasserstein + gradient penalty. real = CLEAN images, fake = G(noisy).
        noisy = gm.add_gaussian_noise(x_unl, cfg.sigma)          # fresh noise
        fake = gen(noisy, training=True)                         # outside any tape -> no G grads
        eps = tf.random.uniform([B, 1, 1, 1])
        x_hat = eps * x_unl + (1.0 - eps) * fake
        with tf.GradientTape() as t:
            with tf.GradientTape() as t_gp:
                t_gp.watch(x_hat)
                s_hat = critic(x_hat, training=True)
            gr = t_gp.gradient(s_hat, x_hat)
            norms = tf.sqrt(tf.reduce_sum(tf.square(gr), axis=[1, 2, 3]) + 1e-12)
            gp = tf.reduce_mean(tf.square(norms - 1.0))
            s = critic(tf.concat([fake, x_unl], axis=0), training=True)
            s_fake, s_real = s[:B], s[B:]
            d_loss = tf.reduce_mean(s_fake) - tf.reduce_mean(s_real) + cfg.gp_weight * gp
        g = t.gradient(d_loss, critic.trainable_variables)
        opt_d.apply_gradients(zip(g, critic.trainable_variables))

        # (3) generator: fresh noise on the same clean images; pairing for B is (clean, G(noisy))
        noisy2 = gm.add_gaussian_noise(x_unl, cfg.sigma)
        with tf.GradientTape() as t:
            fake2 = gen(noisy2, training=True)
            g_adv = -tf.reduce_mean(critic(fake2, training=True))
            g_loss, g_rec = compose_generator_loss(g_adv, x_unl, fake2, cfg.experiment, cfg.lambda_rec)
        g = t.gradient(g_loss, gen.trainable_variables)
        opt_g.apply_gradients(zip(g, gen.trainable_variables))

        return {"c_loss": c_loss, "c_sup": c_sup, "c_pseudo": c_pl, "d_loss": d_loss,
                "g_loss": g_loss, "g_adv": g_adv, "g_rec": g_rec,
                "n_accepted": tf.reduce_sum(acc_counts), "mean_conf": tf.reduce_mean(conf)}

    return train_step


LOSS_KEYS = ("c_loss", "c_sup", "c_pseudo", "d_loss", "g_loss", "g_adv", "g_rec")


class Trainer:
    """Holds the persistent models + window and runs one sequential step per call.
    step indices are 1-based, as in the baseline (lr_at(step))."""

    def __init__(self, models, cfg):
        _require_gm()
        self.models, self.cfg = models, cfg
        self.window = PseudoLabelWindow(cfg.n_classes)
        self.train_step = make_train_step(models, cfg, self.window)

    def step(self, batch, step_idx):
        x_lab, y_lab, x_unl = batch
        out = self.train_step(x_lab, y_lab, x_unl, tf.constant(self.cfg.lr_at(step_idx), tf.float32))
        if self.cfg.pseudo_refresh == "every_n_steps" and step_idx % self.cfg.pseudo_refresh_every == 0:
            self.models.sync_snapshot(step_idx)  # snapshot used from step_idx+1 on; position recorded
        return out

    def collect_log_row(self, step_idx, out):
        """Syncs with the device. Pseudo-label stats cover all images since the previous call."""
        row = {"step": step_idx}
        row.update({k: float(out[k]) for k in LOSS_KEYS})
        row["lr"] = self.cfg.lr_at(step_idx)
        row.update(self.window.read_and_reset())
        return row


def log_fieldnames(num_classes):
    return (["step", "c_loss", "c_sup", "c_pseudo", "d_loss", "g_loss", "g_adv", "g_rec", "lr",
             "sec_per_step", "pl_seen", "pl_accepted", "pl_accept_rate"]
            + ["pl_pred_%d" % i for i in range(num_classes)]
            + ["pl_acc_%d" % i for i in range(num_classes)])


def flatten_log_row(row, num_classes):
    flat = {k: v for k, v in row.items() if not isinstance(v, list)}
    for i in range(num_classes):
        flat["pl_pred_%d" % i] = row["pl_pred_counts"][i]
        flat["pl_acc_%d" % i] = row["pl_accepted_counts"][i]
    return flat


def run_training(trainer, source, start_step, total_steps, log_every,
                 on_log=None, on_step=None, should_stop=None):
    """Sequential loop: same models/optimizers every step. Returns (status, last_step) with status in
    'completed' | 'stream_exhausted' | 'stopped'. on_log(step, row) runs every log_every steps;
    on_step(step) runs after every step (eval/checkpoint hooks); should_stop(step) -> bool."""
    last = start_step
    for step in range(start_step + 1, total_steps + 1):
        try:
            batch = source.next()
        except StreamExhausted:
            return "stream_exhausted", last
        out = trainer.step(batch, step)
        last = step
        if on_log is not None and step % log_every == 0:
            on_log(step, trainer.collect_log_row(step, out))
        if on_step is not None:
            on_step(step)
        if should_stop is not None and should_stop(step):
            return "stopped", last
    return "completed", last
