"""
Unit tests for training_module.py  (Member 2).  Synthetic data only: no dataset download, no GPU.

    python -m unittest test_training_module -v

Part 1 (config, streams, leakage, splits) needs only numpy + scikit-learn + scipy.
Part 2 (pseudo-labels, losses, train step, persistent state) needs TensorFlow + tf_keras and uses
48x48 images so it is fast on CPU. If TensorFlow is missing, Part 2 is SKIPPED (not passed).
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")      # before importing tensorflow
import copy
import dataclasses
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

import training_module as tm

HAVE_TF = tm.tf is not None and tm.gm is not None
S = 8                                                   # tiny images for the numpy-only tests


def rand_u8(n, size=S, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def make_train(n_lab=30, n_unl=100, size=S, seed=0):
    return tm.TrainingData(rand_u8(n_lab, size, seed), np.arange(n_lab) % 10, rand_u8(n_unl, size, seed + 1))


def cfg_for(**kw):
    base = dict(experiment="A", img_size=48)
    base.update(kw)
    return tm.TrainConfig(**base)


# =========================================================================== #
# Part 1: no TensorFlow needed
# =========================================================================== #
class TestConfig(unittest.TestCase):
    def test_defaults_valid_and_A_has_no_lambda(self):
        c = tm.TrainConfig()
        self.assertEqual(c.experiment, "A")
        self.assertIsNone(c.lambda_rec)
        self.assertEqual((c.n_unlabeled, c.n_labeled), (20, 1))

    def test_experiment_modes(self):
        with self.assertRaises(ValueError):
            tm.TrainConfig(experiment="C")
        with self.assertRaises(ValueError):
            tm.TrainConfig(experiment="B")                       # B needs an explicit lambda_rec
        with self.assertRaises(ValueError):
            tm.TrainConfig(experiment="B", lambda_rec=0.0)
        with self.assertRaises(ValueError):
            tm.TrainConfig(experiment="A", lambda_rec=5.0)       # A must not carry a reconstruction weight
        self.assertEqual(tm.TrainConfig(experiment="B", lambda_rec=2.0).lambda_rec, 2.0)

    def test_value_ranges(self):
        for bad in (dict(pseudo_threshold=1.5), dict(pseudo_threshold=-0.1), dict(sigma=-1.0),
                    dict(img_size=100), dict(exhaustion_policy="x"), dict(pseudo_refresh="x"),
                    dict(n_unlabeled=0), dict(n_labeled=0), dict(pseudo_loss_weight=-1)):
            with self.assertRaises(ValueError, msg=str(bad)):
                tm.TrainConfig(**bad)

    def test_lr_schedule_matches_baseline_formula(self):
        c = tm.TrainConfig()
        self.assertAlmostEqual(c.lr_at(0), 1e-4)
        self.assertAlmostEqual(c.lr_at(10000), 1e-4 / (1.000004 ** 10000), places=12)
        self.assertEqual(c.lr_at(10 ** 7), 1e-6)          # clamped at min_lr (same formula as baseline)

    def test_provisional_report_lists_every_provisional_value(self):
        text = "\n".join(tm.provisional_report(tm.TrainConfig()))
        for k in tm.PROVISIONAL:
            self.assertIn(k, text)
        self.assertIn("PROVISIONAL", text)


class TestBatchCounts(unittest.TestCase):
    def test_20_to_1_per_step(self):
        src = tm.TrainBatchSource(make_train(), tm.TrainConfig(), seed=3)
        for _ in range(60):
            x_lab, y_lab, x_unl = src.next()
            self.assertEqual(x_lab.shape, (1, S, S, 3))
            self.assertEqual(y_lab.shape, (1,))
            self.assertEqual(x_unl.shape, (20, S, S, 3))
            self.assertEqual((x_lab.dtype, x_unl.dtype, y_lab.dtype), (np.uint8, np.uint8, np.int64))

    def test_ratio_is_configurable(self):
        src = tm.TrainBatchSource(make_train(), tm.TrainConfig(n_unlabeled=7, n_labeled=3), seed=0)
        x_lab, y_lab, x_unl = src.next()
        self.assertEqual((len(x_lab), len(y_lab), len(x_unl)), (3, 3, 7))


class TestLabelPairing(unittest.TestCase):
    """The labelled image and its TRUE label must stay paired."""

    def test_pairing_without_augmentation(self):
        n = 40
        x = np.zeros((n, S, S, 3), np.uint8)
        for i in range(n):
            x[i] = i                                       # image i is the constant i
        y = (7 * np.arange(n) + 3) % 10
        data = tm.TrainingData(x, y, rand_u8(60))
        src = tm.TrainBatchSource(data, tm.TrainConfig(aug_labeled=False, n_labeled=4), seed=1)
        for _ in range(100):
            xb, yb, _ = src.next()
            idx = xb[:, 0, 0, 0].astype(int)               # recover the source index from the pixels
            np.testing.assert_array_equal(yb, y[idx])
            np.testing.assert_array_equal(idx, src.lab.last_indices)

    def test_pairing_with_augmentation(self):
        n = 25
        rng = np.random.default_rng(5)
        x = rng.integers(0, 256, (n, S, S, 3), dtype=np.uint8)       # unique asymmetric images
        y = (3 * np.arange(n) + 1) % 10

        def variants(im):                                  # all 8 dihedral variants
            out = []
            for t in (False, True):
                b = np.transpose(im, (1, 0, 2)) if t else im
                for fw in (False, True):
                    for fh in (False, True):
                        v = b[:, ::-1] if fw else b
                        out.append((v[::-1] if fh else v).tobytes())
            return out
        lookup = {v: i for i in range(n) for v in variants(x[i])}
        data = tm.TrainingData(x, y, rand_u8(60))
        src = tm.TrainBatchSource(data, tm.TrainConfig(aug_labeled=True, n_labeled=1), seed=2)
        for _ in range(100):
            xb, yb, _ = src.next()
            i = lookup[np.ascontiguousarray(xb[0]).tobytes()]      # KeyError => not a dihedral variant
            self.assertEqual(int(yb[0]), int(y[i]))
            self.assertEqual(i, int(src.lab.last_indices[0]))


class TestExhaustionPolicies(unittest.TestCase):
    def test_replacement_never_exhausts(self):
        s = tm.BatchStream(rand_u8(10), None, 20, seed=0, policy="replacement")   # bs > n is allowed
        for _ in range(500):
            self.assertEqual(len(s.next_batch()[0]), 20)

    def test_reshuffle_covers_epoch_without_repeats(self):
        s = tm.BatchStream(rand_u8(45), None, 20, seed=0, policy="reshuffle")
        for epoch in range(3):
            seen = []
            for _ in range(2):                              # floor(45/20) = 2 batches, 5 dropped
                s.next_batch()
                seen += s.last_indices.tolist()
            self.assertEqual(len(set(seen)), 40)
            self.assertEqual(s.epochs_completed, epoch)
        s.next_batch()
        self.assertEqual(s.epochs_completed, 3)

    def test_stop_raises_after_one_pass_and_keeps_raising(self):
        s = tm.BatchStream(rand_u8(45), None, 20, seed=0, policy="stop")
        s.next_batch(), s.next_batch()
        for _ in range(3):
            with self.assertRaises(tm.StreamExhausted):
                s.next_batch()

    def test_non_replacement_policies_need_enough_images(self):
        for pol in ("reshuffle", "stop"):
            with self.assertRaises(ValueError):
                tm.BatchStream(rand_u8(5), None, 20, seed=0, policy=pol)

    def test_seeded_and_reproducible(self):
        a = tm.BatchStream(rand_u8(100), None, 20, seed=9, policy="reshuffle")
        b = tm.BatchStream(rand_u8(100), None, 20, seed=9, policy="reshuffle")
        c = tm.BatchStream(rand_u8(100), None, 20, seed=10, policy="reshuffle")
        a.next_batch(), b.next_batch(), c.next_batch()
        np.testing.assert_array_equal(a.last_indices, b.last_indices)
        self.assertFalse(np.array_equal(a.last_indices, c.last_indices))


class TestValidationAndTestCannotEnter(unittest.TestCase):
    def setUp(self):
        self.train = make_train(30, 100, seed=10)
        self.x_val, self.x_test = rand_u8(20, seed=200), rand_u8(20, seed=300)

    def test_container_has_no_val_or_test_fields(self):
        self.assertEqual({f.name for f in dataclasses.fields(tm.TrainingData)}, {"x_lab", "y_lab", "x_unl"})

    def test_source_rejects_anything_but_TrainingData(self):
        loose = {"x_lab": self.train.x_lab, "y_lab": self.train.y_lab, "x_unl": self.train.x_unl,
                 "x_val": self.x_val}
        for bad in (loose, self.x_val, (self.train.x_lab, self.train.y_lab, self.x_val)):
            with self.assertRaises(TypeError):
                tm.TrainBatchSource(bad, tm.TrainConfig())

    def test_clean_split_passes_the_guard(self):
        tm.assert_no_leakage(self.train, x_val=self.x_val, x_test=self.x_test)

    def test_guard_catches_val_or_test_image_in_unlabelled_pool(self):
        x_unl = self.train.x_unl.copy()
        x_unl[17] = self.x_val[3]
        bad = tm.TrainingData(self.train.x_lab, self.train.y_lab, x_unl)
        with self.assertRaises(tm.LeakageError):
            tm.assert_no_leakage(bad, x_val=self.x_val, x_test=self.x_test)
        x_unl = self.train.x_unl.copy()
        x_unl[0] = self.x_test[5]
        bad = tm.TrainingData(self.train.x_lab, self.train.y_lab, x_unl)
        with self.assertRaises(tm.LeakageError):
            tm.assert_no_leakage(bad, x_val=self.x_val, x_test=self.x_test)

    def test_guard_catches_val_image_in_labelled_set(self):
        x_lab = self.train.x_lab.copy()
        x_lab[2] = self.x_val[0]
        bad = tm.TrainingData(x_lab, self.train.y_lab, self.train.x_unl)
        with self.assertRaises(tm.LeakageError):
            tm.assert_no_leakage(bad, x_val=self.x_val)

    def test_guard_catches_shared_memory(self):
        big = rand_u8(150, seed=77)
        bad = tm.TrainingData(big[:30], np.zeros(30, int), big[30:130])
        with self.assertRaises(tm.LeakageError):
            tm.assert_no_leakage(bad, x_val=big[100:150])      # overlaps x_unl in the same buffer

    def test_every_emitted_image_is_a_train_image_never_val_or_test(self):
        train_h = tm.image_hashes(np.concatenate([self.train.x_lab, self.train.x_unl]))
        prot_h = tm.image_hashes(self.x_val) | tm.image_hashes(self.x_test)
        for policy in ("replacement", "reshuffle"):
            cfg = tm.TrainConfig(exhaustion_policy=policy, aug_labeled=False)
            src = tm.TrainBatchSource(self.train, cfg, seed=4)
            for _ in range(150):
                x_lab, _, x_unl = src.next()
                emitted = tm.image_hashes(np.concatenate([x_lab, x_unl]))
                self.assertTrue(emitted <= train_h)
                self.assertFalse(emitted & prot_h)


class TestSplitAndPositions(unittest.TestCase):
    def setUp(self):
        self.labels = np.random.default_rng(0).integers(0, 10, 2000)

    def test_partition_is_disjoint_and_complete(self):
        ix = tm.split_indices(self.labels, 0.1, 0.1, 0.125, 16)
        pool, val, test, lab = (set(ix[k].tolist()) for k in ("pool", "val", "test", "lab"))
        self.assertFalse(pool & val or pool & test or val & test)
        self.assertEqual(len(pool | val | test), len(self.labels))
        self.assertTrue(lab <= pool)
        self.assertAlmostEqual(len(val) / len(self.labels), 0.1, places=2)
        self.assertAlmostEqual(len(lab) / len(pool), 0.125, places=2)

    def test_same_seed_same_split(self):
        a = tm.split_indices(self.labels, 0.1, 0.1, 0.125, 16)
        b = tm.split_indices(self.labels, 0.1, 0.1, 0.125, 16)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_unlabelled_pool_excludes_labelled_unless_asked(self):
        ix = tm.split_indices(self.labels, 0.1, 0.1, 0.125, 16)
        only = tm.unlabeled_pool_indices(ix["pool"], ix["lab"], include_labelled=False)
        self.assertFalse(set(only.tolist()) & set(ix["lab"].tolist()))
        self.assertEqual(len(only), len(ix["pool"]) - len(ix["lab"]))
        for forbidden in ("val", "test"):
            self.assertFalse(set(only.tolist()) & set(ix[forbidden].tolist()))
        self.assertEqual(len(tm.unlabeled_pool_indices(ix["pool"], ix["lab"], True)), len(ix["pool"]))

    def test_positional_match_geometry(self):
        ref_ra, ref_dec = np.array([10.0, 200.0]), np.array([20.0, -30.0])
        ra = np.array([10.0, 10.0, 200.0, 45.0])
        dec = np.array([20.0 + 1 / 3600, 20.0 + 10 / 3600, -30.0, 5.0])   # 1", 10", exact, far
        self.assertEqual(tm.count_positional_matches(ra, dec, ref_ra, ref_dec, 3.0), 2)


# --------------------------- finding 2: footprint vs centre overlap --------------------------- #
ARC = 1 / 3600.0


class TestCutoutOverlap(unittest.TestCase):
    def test_footprints_overlap_although_centres_are_far_apart(self):
        # 50" apart: the centre test (3") sees nothing, but 256 px * 0.262"/px cutouts overlap heavily
        e_ra, e_dec = np.array([10.0]), np.array([20.0])
        g_ra, g_dec = np.array([10.0]), np.array([20.0 + 50 * ARC])
        self.assertEqual(tm.count_positional_matches(e_ra, e_dec, g_ra, g_dec, 3.0), 0)
        half = 0.5 * 256 * 0.262
        self.assertEqual(tm.footprint_overlap_pairs(e_ra, e_dec, half, g_ra, g_dec, half), [(0, 0)])

    def test_far_apart_footprints_do_not_overlap(self):
        half = 0.5 * 256 * 0.262                           # ~33.5"
        g_dec = np.array([20.0 + 200 * ARC])               # > (33.5+33.5)*sqrt(2) = 94.8"
        self.assertEqual(tm.footprint_overlap_pairs([10.0], [20.0], half, [10.0], g_dec, half), [])

    def test_per_galaxy_pixel_scales_are_used(self):
        # same 100" separation: small pixscale -> no overlap; large pixscale on the reference -> overlap
        small = 0.5 * 256 * 0.1                            # 12.8"
        big = 0.5 * 256 * 0.6                              # 76.8"
        dec2 = np.array([20.0 + 100 * ARC])
        self.assertEqual(tm.footprint_overlap_pairs([10.0], [20.0], small, [10.0], dec2, small), [])
        self.assertEqual(len(tm.footprint_overlap_pairs([10.0], [20.0], small, [10.0], dec2, big)), 1)

    def test_bound_is_orientation_independent_and_conservative(self):
        half = 100.0                                       # circumscribed-circle reach = 2*100*sqrt(2) = 283"
        d = 150.0                                          # inside the reach: flagged whatever the rotation
        self.assertEqual(len(tm.footprint_overlap_pairs([0.0], [0.0], half, [0.0], [d * ARC], half)), 1)
        d = 290.0                                          # outside the reach: disjoint in any orientation
        self.assertEqual(tm.footprint_overlap_pairs([0.0], [0.0], half, [0.0], [d * ARC], half), [])

    def _scene(self):
        g_ra = np.array([10.0, 10.0, 10.0, 90.0])
        g_dec = np.array([20.0, 20.0 + 60 * ARC, 20.0 + 500 * ARC, 0.0])
        return g_ra, g_dec, np.full(4, 0.262)

    def test_footprint_mode_fails_on_protected_overlap_that_centre_mode_misses(self):
        g_ra, g_dec, g_px = self._scene()
        e_ra, e_dec = np.array([10.0]), np.array([20.0 + 30 * ARC])      # 30" from galaxies 0 and 1
        lim = tm.check_cutout_overlap(e_ra, e_dec, g_ra, g_dec, protected_idx=[0, 1])
        self.assertEqual(lim["mode"], "center_only_limited")
        self.assertTrue(lim["ok"])                                       # centres > 3" apart: limited check passes
        fp = tm.check_cutout_overlap(e_ra, e_dec, g_ra, g_dec, protected_idx=[0, 1],
                                     extra_pxscale=0.262, extra_pxscale_source="test", g_pxscale=g_px)
        self.assertEqual(fp["mode"], "footprint")
        self.assertFalse(fp["ok"])
        self.assertEqual(fp["n_overlap_protected"], 1)
        self.assertIn("LEAKAGE RISK", fp["message"])

    def test_overlap_with_non_protected_galaxy_is_reported_not_fatal(self):
        g_ra, g_dec, g_px = self._scene()
        e_ra, e_dec = np.array([10.0]), np.array([20.0 + 30 * ARC])
        r = tm.check_cutout_overlap(e_ra, e_dec, g_ra, g_dec, protected_idx=[2, 3], other_idx=[0, 1],
                                    extra_pxscale=0.262, extra_pxscale_source="test", g_pxscale=g_px)
        self.assertTrue(r["ok"])
        self.assertEqual((r["n_overlap_protected"], r["n_overlap_other"]), (0, 1))

    def test_limited_mode_never_claims_absence_of_leakage(self):
        g_ra, g_dec, _ = self._scene()
        r = tm.check_cutout_overlap(np.array([200.0]), np.array([-40.0]), g_ra, g_dec, protected_idx=[0, 1, 2])
        self.assertEqual(r["mode"], "center_only_limited")
        self.assertTrue(r["ok"])
        self.assertTrue(r["message"].startswith("LIMITED center-match check only"))
        self.assertIn("does NOT show", r["message"])
        self.assertIn("Not evidence that leakage is absent", r["message"])
        for banned in ("proves", "no leakage", "leakage-free", "verified safe"):
            self.assertNotIn(banned, r["message"].lower())

    def test_limited_mode_still_catches_centre_matches(self):
        g_ra, g_dec, _ = self._scene()
        r = tm.check_cutout_overlap(np.array([10.0]), np.array([20.0 + 1 * ARC]), g_ra, g_dec, protected_idx=[0])
        self.assertFalse(r["ok"])
        self.assertEqual(r["n_overlap_protected"], 1)

    def test_footprint_mode_needs_both_pixel_scales(self):
        g_ra, g_dec, g_px = self._scene()
        only_extra = tm.check_cutout_overlap([10.0], [20.0], g_ra, g_dec, [0], extra_pxscale=0.262)
        only_gal = tm.check_cutout_overlap([10.0], [20.0], g_ra, g_dec, [0], g_pxscale=g_px)
        self.assertEqual(only_extra["mode"], "center_only_limited")
        self.assertEqual(only_gal["mode"], "center_only_limited")

    def test_bad_pixel_scale_rejected(self):
        g_ra, g_dec, g_px = self._scene()
        with self.assertRaises(ValueError):
            tm.check_cutout_overlap([10.0], [20.0], g_ra, g_dec, [0], extra_pxscale=0.0, g_pxscale=g_px)
        with self.assertRaises(ValueError):
            tm.check_cutout_overlap([10.0, 11.0], [20.0, 21.0], g_ra, g_dec, [0],
                                    extra_pxscale=np.array([0.26, 0.26, 0.26]), g_pxscale=g_px)

    def test_exact_duplicate_check_is_unchanged_and_still_runs_independently(self):
        train = make_train(30, 100, seed=10)
        x_val = rand_u8(5, seed=200)
        x_unl = train.x_unl.copy()
        x_unl[4] = x_val[1]
        with self.assertRaises(tm.LeakageError):
            tm.assert_no_leakage(tm.TrainingData(train.x_lab, train.y_lab, x_unl), x_val=x_val)


# --------------------------- finding 1: manifest, identity, resume guard --------------------------- #
def _read(path):
    with open(path) as f:
        return f.read()


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def make_identity(tmpdir, with_unlabeled=True, labels=None, tweak=None, split_seed=16):
    lab, unl = os.path.join(tmpdir, "g10.h5"), os.path.join(tmpdir, "unl.h5")
    _write(lab, b"labelled-file-bytes" * 100)
    if with_unlabeled:
        _write(unl, b"unlabelled-file-bytes" * 100)
    labels = np.arange(200) % 10 if labels is None else labels
    ix = tm.split_indices(labels, 0.1, 0.1, 0.125, split_seed)
    unl_idx = tm.unlabeled_pool_indices(ix["pool"], ix["lab"], False)
    if tweak:
        tweak(lab, unl)
    return tm.dataset_identity(lab, unl if with_unlabeled else None, labels, ix, unl_idx)


def make_manifest(identity, **overrides):
    cfg = tm.TrainConfig(**overrides.pop("cfg", {}))
    ds = tm.make_data_settings(split_seed=16, test_frac=0.1, val_frac=0.1, label_frac=0.125,
                               include_labelled_in_unlabeled=False, no_pool_unlabeled=False)
    ds.update(overrides.pop("data", {}))
    return tm.build_manifest(cfg, ds, identity, info={"overlap_check": {"mode": "x"}})


class TestFingerprints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_file_fingerprint_detects_same_size_content_change_and_size_change(self):
        p = os.path.join(self.tmp.name, "f.bin")
        _write(p, b"a" * 1000)
        base = tm.file_fingerprint(p, chunk=64)
        _write(p, b"a" * 999 + b"b")
        changed = tm.file_fingerprint(p, chunk=64)
        self.assertEqual(base["size"], changed["size"])
        self.assertNotEqual(base["sha256"], changed["sha256"])
        _write(p, b"a" * 1001)
        self.assertNotEqual(tm.file_fingerprint(p)["size"], base["size"])

    def test_chunk_size_does_not_change_the_hash(self):
        p = os.path.join(self.tmp.name, "f.bin")
        _write(p, os.urandom(5000))
        self.assertEqual(tm.file_fingerprint(p, chunk=7), tm.file_fingerprint(p, chunk=1 << 20))

    def test_array_fingerprint_depends_on_dtype_shape_and_values(self):
        a = np.arange(12, dtype=np.int64)
        self.assertEqual(tm.array_fingerprint(a), tm.array_fingerprint(a.copy()))
        self.assertNotEqual(tm.array_fingerprint(a), tm.array_fingerprint(a.astype(np.int32)))
        self.assertNotEqual(tm.array_fingerprint(a), tm.array_fingerprint(a.reshape(3, 4)))
        b = a.copy()
        b[3] = 99
        self.assertNotEqual(tm.array_fingerprint(a), tm.array_fingerprint(b))

    def test_identity_keys_and_sensitivity(self):
        base = make_identity(self.tmp.name)
        self.assertEqual(set(base), set(tm.IDENTITY_KEYS))
        self.assertEqual(base, make_identity(self.tmp.name))                          # deterministic
        self.assertNotEqual(base["labels"], make_identity(self.tmp.name, labels=(np.arange(200) + 1) % 10)["labels"])
        self.assertIsNone(make_identity(self.tmp.name, with_unlabeled=False)["unlabeled_file"])
        edited = make_identity(self.tmp.name, tweak=lambda lab, unl: _write(unl, b"EDITED" * 100))
        self.assertNotEqual(base["unlabeled_file"], edited["unlabeled_file"])

    def test_every_index_set_in_the_identity_reacts_to_a_different_split(self):
        a = make_identity(self.tmp.name, split_seed=16)
        b = make_identity(self.tmp.name, split_seed=17)
        for key in ("idx_pool", "idx_lab", "idx_val", "idx_test", "idx_unl_pool"):
            self.assertNotEqual(a[key], b[key], key)
        self.assertEqual(a["labeled_file"], b["labeled_file"])            # same files, different partition

    def test_index_identity_ignores_order_but_not_membership(self):
        labels = np.arange(200) % 10
        ix = tm.split_indices(labels, 0.1, 0.1, 0.125, 16)
        shuffled = {k: np.random.default_rng(0).permutation(v) for k, v in ix.items()}
        lab, unl = os.path.join(self.tmp.name, "g.h5"), os.path.join(self.tmp.name, "u.h5")
        _write(lab, b"x"), _write(unl, b"y")
        unl_idx = tm.unlabeled_pool_indices(ix["pool"], ix["lab"], False)
        a = tm.dataset_identity(lab, unl, labels, ix, unl_idx)
        b = tm.dataset_identity(lab, unl, labels, shuffled, np.random.default_rng(1).permutation(unl_idx))
        self.assertEqual(a, b)


class TestResumeManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.identity = make_identity(self.tmp.name)
        self.m = make_manifest(self.identity)
        self.path = os.path.join(self.tmp.name, "manifest.json")

    def refuses(self, new):
        with self.assertRaises(tm.ResumeError) as cm:
            tm.check_resume_compatible(self.m, new)
        return str(cm.exception)

    def test_identical_manifest_is_accepted_and_info_is_ignored(self):
        tm.check_resume_compatible(self.m, copy.deepcopy(self.m))
        other = copy.deepcopy(self.m)
        other["info"] = {"overlap_check": {"mode": "footprint"}}
        tm.check_resume_compatible(self.m, other)

    def test_manifest_covers_every_config_field_and_every_data_setting(self):
        self.assertEqual(set(self.m["config"]), {f.name for f in dataclasses.fields(tm.TrainConfig)})
        self.assertEqual(set(self.m["data"]), set(tm.DATA_SETTING_KEYS))
        self.assertEqual(set(self.m["identity"]), set(tm.IDENTITY_KEYS))

    def test_each_config_field_change_is_refused(self):
        for name in self.m["config"]:
            new = copy.deepcopy(self.m)
            new["config"][name] = "CHANGED"
            msg = self.refuses(new)
            self.assertIn("config." + name, msg)

    def test_each_data_setting_change_is_refused(self):
        for name in tm.DATA_SETTING_KEYS:
            new = copy.deepcopy(self.m)
            new["data"][name] = "CHANGED"
            self.assertIn("data." + name, self.refuses(new))

    def test_each_identity_item_change_is_refused(self):
        for name in tm.IDENTITY_KEYS:
            new = copy.deepcopy(self.m)
            new["identity"][name] = "CHANGED"
            self.assertIn("identity." + name, self.refuses(new))

    def test_named_review_items_via_real_builders(self):
        """split seed/fractions, label fraction, training seed, lr settings, augmentation, pool options,
        experiment, dataset files: each real change is refused."""
        cases = {
            "data.split_seed": make_manifest(self.identity, data={"split_seed": 17}),
            "data.test_frac": make_manifest(self.identity, data={"test_frac": 0.2}),
            "data.val_frac": make_manifest(self.identity, data={"val_frac": 0.2}),
            "data.label_frac": make_manifest(self.identity, data={"label_frac": 0.25}),
            "data.include_labelled_in_unlabeled": make_manifest(self.identity, data={"include_labelled_in_unlabeled": True}),
            "data.no_pool_unlabeled": make_manifest(self.identity, data={"no_pool_unlabeled": True}),
            "config.seed": make_manifest(self.identity, cfg={"seed": 1}),
            "config.lr": make_manifest(self.identity, cfg={"lr": 2e-4}),
            "config.min_lr": make_manifest(self.identity, cfg={"min_lr": 1e-7}),
            "config.decay": make_manifest(self.identity, cfg={"decay": 1.00001}),
            "config.aug_labeled": make_manifest(self.identity, cfg={"aug_labeled": False}),
            "config.aug_unlabeled": make_manifest(self.identity, cfg={"aug_unlabeled": True}),
            "config.experiment": make_manifest(self.identity, cfg={"experiment": "B", "lambda_rec": 1.0}),
            "config.sigma": make_manifest(self.identity, cfg={"sigma": 0.3}),
            "config.pseudo_threshold": make_manifest(self.identity, cfg={"pseudo_threshold": 0.9}),
            "config.pseudo_refresh": make_manifest(self.identity, cfg={"pseudo_refresh": "every_n_steps"}),
            "config.img_size": make_manifest(self.identity, cfg={"img_size": 48}),
        }
        for key, new in cases.items():
            self.assertIn(key, self.refuses(new))

    def test_changed_dataset_files_or_labels_are_refused(self):
        other_unl = make_identity(self.tmp.name, tweak=lambda lab, unl: _write(unl, b"OTHER" * 100))
        self.assertIn("identity.unlabeled_file", self.refuses(make_manifest(other_unl)))
        other_lab = make_identity(self.tmp.name, tweak=lambda lab, unl: _write(lab, b"OTHER" * 100))
        self.assertIn("identity.labeled_file", self.refuses(make_manifest(other_lab)))
        no_unl = make_identity(self.tmp.name, with_unlabeled=False)
        self.assertIn("identity.unlabeled_file", self.refuses(make_manifest(no_unl)))
        other_labels = make_identity(self.tmp.name, labels=(np.arange(200) + 3) % 10)
        self.assertIn("identity.labels", self.refuses(make_manifest(other_labels)))

    def test_missing_keys_in_saved_manifest_and_unknown_format_are_refused(self):
        old = copy.deepcopy(self.m)
        del old["data"]["split_seed"]
        with self.assertRaises(tm.ResumeError):
            tm.check_resume_compatible(old, self.m)
        for bad in ({"format": 1}, {}, None, "junk"):
            with self.assertRaises(tm.ResumeError):
                tm.check_resume_compatible(bad, self.m)

    def test_every_difference_is_listed(self):
        new = make_manifest(self.identity, cfg={"seed": 5, "sigma": 0.4}, data={"split_seed": 3})
        msg = self.refuses(new)
        for k in ("config.seed", "config.sigma", "data.split_seed"):
            self.assertIn(k, msg)

    def test_builders_reject_incomplete_inputs(self):
        with self.assertRaises(ValueError):
            tm.make_data_settings(split_seed=1)
        with self.assertRaises(ValueError):
            tm.build_manifest(tm.TrainConfig(), {"split_seed": 1}, self.identity)
        with self.assertRaises(ValueError):
            tm.build_manifest(tm.TrainConfig(), make_manifest(self.identity)["data"], {"labels": "x"})

    def test_prepare_fresh_writes_manifest(self):
        self.assertEqual(tm.prepare_run_manifest(False, self.path, self.m), "fresh")
        self.assertEqual(json.loads(_read(self.path)), self.m)

    def test_prepare_resume_accepts_same_and_never_overwrites_saved_manifest(self):
        tm.prepare_run_manifest(False, self.path, self.m)
        before = _read(self.path)
        same_but_new_info = copy.deepcopy(self.m)
        same_but_new_info["info"] = {"note": "different"}
        self.assertEqual(tm.prepare_run_manifest(True, self.path, same_but_new_info), "resume")
        self.assertEqual(_read(self.path), before)

    def test_prepare_resume_refuses_changed_settings_without_touching_file(self):
        tm.prepare_run_manifest(False, self.path, self.m)
        before = _read(self.path)
        with self.assertRaises(tm.ResumeError):
            tm.prepare_run_manifest(True, self.path, make_manifest(self.identity, cfg={"seed": 9}))
        self.assertEqual(_read(self.path), before)

    def test_prepare_resume_refuses_missing_or_corrupt_manifest(self):
        with self.assertRaises(tm.ResumeError):
            tm.prepare_run_manifest(True, self.path, self.m)                 # checkpoint but no manifest
        _write(self.path, b"{not json")
        with self.assertRaises(tm.ResumeError):
            tm.prepare_run_manifest(True, self.path, self.m)

    def test_stale_manifest_without_checkpoint_is_replaced(self):
        tm.prepare_run_manifest(False, self.path, self.m)
        new = make_manifest(self.identity, cfg={"seed": 4})
        self.assertEqual(tm.prepare_run_manifest(False, self.path, new), "fresh")
        self.assertEqual(json.loads(_read(self.path))["config"]["seed"], 4)


class TestSnapshotPositionHelper(unittest.TestCase):
    def test_expected_snapshot_step(self):
        self.assertEqual([tm.expected_snapshot_step(s, 3) for s in range(0, 8)], [0, 0, 0, 3, 3, 3, 6, 6])
        self.assertEqual(tm.expected_snapshot_step(10, 100), 0)
        self.assertEqual(tm.expected_snapshot_step(250, 100), 200)


# =========================================================================== #
# Part 2: TensorFlow
# =========================================================================== #
@unittest.skipUnless(HAVE_TF, "TensorFlow / generator_module not importable: Part 2 NOT RUN")
class TestPseudoLabelFiltering(unittest.TestCase):
    def test_softmax_argmax_and_threshold(self):
        tf = tm.tf
        logits = tf.constant([[10., 0., 0.], [0., 0., 0.], [3., 2.9, 0.], [0., 8., 0.]])
        probs, pred, conf, acc = tm.pseudo_label_from_logits(logits, 0.9)
        np.testing.assert_allclose(probs.numpy().sum(1), 1.0, rtol=1e-5)
        np.testing.assert_array_equal(pred.numpy(), [0, 0, 0, 1])
        np.testing.assert_array_equal(acc.numpy(), [True, False, False, True])

    def test_threshold_is_inclusive(self):
        logits = tm.tf.constant([[0., 0.]])                    # exactly 0.5 / 0.5
        self.assertTrue(bool(tm.pseudo_label_from_logits(logits, 0.5)[3].numpy()[0]))
        self.assertFalse(bool(tm.pseudo_label_from_logits(logits, 0.5001)[3].numpy()[0]))

    def test_threshold_extremes(self):
        logits = tm.tf.random.normal([20, 10], seed=1)
        self.assertEqual(int(tm.tf.reduce_sum(tm.tf.cast(tm.pseudo_label_from_logits(logits, 0.0)[3], tm.tf.int32))), 20)

    def test_counts(self):
        tf = tm.tf
        pred = tf.constant([0, 0, 2, 2, 2], tf.int64)
        acc = tf.constant([True, False, True, True, False])
        pc, ac = tm.pseudo_label_counts(pred, acc, 4)
        np.testing.assert_array_equal(pc.numpy(), [2, 0, 3, 0])
        np.testing.assert_array_equal(ac.numpy(), [1, 0, 2, 0])


@unittest.skipUnless(HAVE_TF, "TensorFlow / generator_module not importable: Part 2 NOT RUN")
class TestLossComposition(unittest.TestCase):
    def test_classifier_loss_uses_only_accepted_pseudo_labels(self):
        tf = tm.tf
        ce = lambda y, z: tf.keras.losses.sparse_categorical_crossentropy(y, z, from_logits=True).numpy()
        z_lab = tf.constant([[2., 0., 0.]])
        y_lab = tf.constant([0], tf.int64)
        z_unl = tf.constant([[0., 3., 0.], [0., 0., 4.], [1., 0., 0.]])
        pl = tf.constant([1, 2, 1], tf.int64)                  # third pseudo-label is wrong but rejected
        acc = tf.constant([True, True, False])
        total, sup, pseu = tm.classifier_loss(z_lab, y_lab, z_unl, pl, acc, 0.5)
        want_sup = ce(y_lab, z_lab).mean()
        want_pl = ce(pl[:2], z_unl[:2]).mean()
        self.assertAlmostEqual(float(sup), want_sup, places=5)
        self.assertAlmostEqual(float(pseu), want_pl, places=5)
        self.assertAlmostEqual(float(total), want_sup + 0.5 * want_pl, places=5)

    def test_nothing_accepted_reduces_to_baseline_supervised_loss(self):
        tf = tm.tf
        z_lab, y_lab = tf.constant([[1., 2., 3.]]), tf.constant([1], tf.int64)
        total, sup, pseu = tm.classifier_loss(z_lab, y_lab, tf.random.normal([20, 3]),
                                              tf.zeros([20], tf.int64), tf.zeros([20], tf.bool), 1.0)
        self.assertEqual(float(pseu), 0.0)
        self.assertAlmostEqual(float(total), float(sup), places=6)

    def test_A_never_calls_reconstruction_helper_B_does(self):
        tf = tm.tf
        clean, gen = tf.zeros((2, 4, 4, 3)), tf.fill((2, 4, 4, 3), 0.5)
        g_adv = tf.constant(1.25)
        calls = []
        real = tm.gm.reconstruction_l1

        def spy(a, b):
            calls.append(1)
            return real(a, b)
        with mock.patch.object(tm.gm, "reconstruction_l1", spy):
            tot, rec = tm.compose_generator_loss(g_adv, clean, gen, "A", None)
            self.assertEqual(calls, [])
            self.assertEqual(float(tot), 1.25)
            self.assertEqual(float(rec), 0.0)
            tot, rec = tm.compose_generator_loss(g_adv, clean, gen, "B", 4.0)
            self.assertEqual(len(calls), 1)
            self.assertAlmostEqual(float(rec), 0.5, places=6)
            self.assertAlmostEqual(float(tot), 1.25 + 4.0 * 0.5, places=6)


@unittest.skipUnless(HAVE_TF, "TensorFlow / generator_module not importable: Part 2 NOT RUN")
class TestTrainStep(unittest.TestCase):
    SIZE = 48

    @staticmethod
    def batch(seed, size=48, n_lab=1, n_unl=20):
        rng = np.random.default_rng(seed)
        return (rng.integers(0, 256, (n_lab, size, size, 3), dtype=np.uint8),
                np.array([3] * n_lab, np.int64),
                rng.integers(0, 256, (n_unl, size, size, 3), dtype=np.uint8))

    def test_one_step_counts_accept_all_and_pseudo_label_only_unlabelled(self):
        cfg = cfg_for(pseudo_threshold=0.0)
        tr = tm.Trainer(tm.build_models(cfg), cfg)
        out = tr.step(self.batch(0), 1)
        row = tr.collect_log_row(1, out)
        self.assertEqual(row["pl_seen"], 20)                   # 20 unlabelled, NOT 21
        self.assertEqual(row["pl_accepted"], 20)
        self.assertEqual(sum(row["pl_pred_counts"]), 20)
        self.assertEqual(sum(row["pl_accepted_counts"]), 20)
        self.assertEqual(row["pl_accept_rate"], 1.0)

    def test_threshold_one_accepts_nothing_at_init(self):
        cfg = cfg_for(pseudo_threshold=1.0)
        tr = tm.Trainer(tm.build_models(cfg), cfg)
        out = tr.step(self.batch(0), 1)
        row = tr.collect_log_row(1, out)
        self.assertEqual(row["pl_accepted"], 0)
        self.assertEqual(float(out["c_pseudo"]), 0.0)
        self.assertAlmostEqual(float(out["c_loss"]), float(out["c_sup"]), places=6)

    def test_window_resets_between_logs(self):
        cfg = cfg_for(pseudo_threshold=0.0)
        tr = tm.Trainer(tm.build_models(cfg), cfg)
        for s in (1, 2, 3):
            out = tr.step(self.batch(s), s)
        self.assertEqual(tr.collect_log_row(3, out)["pl_seen"], 60)
        out = tr.step(self.batch(4), 4)
        self.assertEqual(tr.collect_log_row(4, out)["pl_seen"], 20)

    def test_sequential_updates_keep_the_same_models_and_optimizers(self):
        cfg = cfg_for(pseudo_threshold=0.0)
        models = tm.build_models(cfg)
        tr = tm.Trainer(models, cfg)
        ids = {k: id(getattr(models, k)) for k in ("gen", "critic", "clf", "opt_c", "opt_d", "opt_g")}
        var_ids = None
        snaps = []
        for s in (1, 2, 3):
            before = [w.numpy().copy() for w in models.gen.trainable_variables[:2]]
            tr.step(self.batch(s), s)
            if var_ids is None:        # taken after step 1 so lazily-created Adam slots are included
                var_ids = {k: [id(v) for v in getattr(models, k).variables] for k in ("opt_c", "opt_d", "opt_g")}
            snaps.append(before)
            for name, opt in (("opt_c", models.opt_c), ("opt_d", models.opt_d), ("opt_g", models.opt_g)):
                self.assertEqual(int(opt.iterations.numpy()), s, name)       # state accumulated, not reset
        self.assertIs(tr.models, models)
        for k, v in ids.items():
            self.assertEqual(id(getattr(models, k)), v)
        for k, v in var_ids.items():
            self.assertEqual([id(x) for x in getattr(models, k).variables], v)   # no re-created slots
        self.assertFalse(np.allclose(snaps[0][0], snaps[2][0]))                  # weights actually moved

    def test_all_three_networks_update_each_step(self):
        cfg = cfg_for(pseudo_threshold=0.0)
        models = tm.build_models(cfg)
        tr = tm.Trainer(models, cfg)
        w0 = {k: [v.numpy().copy() for v in getattr(models, k).trainable_variables]
              for k in ("gen", "critic", "clf")}
        tr.step(self.batch(0), 1)
        for k, old in w0.items():
            new = getattr(models, k).trainable_variables
            self.assertTrue(any(not np.allclose(o, n.numpy()) for o, n in zip(old, new)), k)

    def test_experiment_A_vs_B_step_losses(self):
        calls = {"A": 0, "B": 0}
        real = tm.gm.reconstruction_l1
        for mode, lam in (("A", None), ("B", 3.0)):
            def spy(a, b, mode=mode):
                calls[mode] += 1
                return real(a, b)
            cfg = cfg_for(experiment=mode, lambda_rec=lam, pseudo_threshold=0.5)
            with mock.patch.object(tm.gm, "reconstruction_l1", spy):
                tr = tm.Trainer(tm.build_models(cfg), cfg)
                out = tr.step(self.batch(0), 1)                    # tracing happens here, inside the patch
            g_loss, g_adv, g_rec = (float(out[k]) for k in ("g_loss", "g_adv", "g_rec"))
            if mode == "A":
                self.assertEqual(g_rec, 0.0)
                self.assertAlmostEqual(g_loss, g_adv, places=5)
            else:
                self.assertGreater(g_rec, 0.0)
                self.assertAlmostEqual(g_loss, g_adv + 3.0 * g_rec, places=4)
        self.assertEqual(calls["A"], 0)
        self.assertGreaterEqual(calls["B"], 1)

    def test_noise_is_fresh_twice_per_step_with_configured_sigma(self):
        seen = []
        real = tm.gm.add_gaussian_noise

        def spy(clean, sigma, seed=None):
            seen.append((tuple(clean.shape), sigma, seed))
            return real(clean, sigma, seed)
        cfg = cfg_for(sigma=0.25)
        with mock.patch.object(tm.gm, "add_gaussian_noise", spy):
            tr = tm.Trainer(tm.build_models(cfg), cfg)
            tr.step(self.batch(0), 1)
        self.assertEqual(len(seen), 2)                             # critic step + generator step
        for shape, sigma, seed in seen:
            self.assertEqual(shape, (20, 48, 48, 3))
            self.assertEqual(sigma, 0.25)
            self.assertIsNone(seed)

    def test_snapshot_refresh_policy(self):
        cfg = cfg_for(pseudo_threshold=0.0, pseudo_refresh="every_n_steps", pseudo_refresh_every=2)
        models = tm.build_models(cfg)
        self.assertIsNot(models.pseudo_label_model, models.clf)
        tr = tm.Trainer(models, cfg)
        close = lambda a, b: all(np.allclose(x.numpy(), y.numpy()) for x, y in zip(a.variables, b.variables))
        self.assertTrue(close(models.clf, models.clf_snapshot))
        tr.step(self.batch(1), 1)
        self.assertFalse(close(models.clf, models.clf_snapshot))   # live classifier moved, snapshot frozen
        tr.step(self.batch(2), 2)
        self.assertTrue(close(models.clf, models.clf_snapshot))    # refreshed after step 2
        live = tm.build_models(cfg_for(pseudo_refresh="every_step"))
        self.assertIs(live.pseudo_label_model, live.clf)

    def test_run_training_loop_and_stop_policy(self):
        cfg = cfg_for(pseudo_threshold=0.0)
        models = tm.build_models(cfg)
        tr = tm.Trainer(models, cfg)
        rows = []

        class Src:
            def __init__(s):
                s.i = 0

            def next(s):
                s.i += 1
                if s.i > 3:
                    raise tm.StreamExhausted()
                return TestTrainStep.batch(s.i)
        status, last = tm.run_training(tr, Src(), 0, 10, 2, on_log=lambda s, r: rows.append(r))
        self.assertEqual((status, last), ("stream_exhausted", 3))
        self.assertEqual(int(models.opt_c.iterations.numpy()), 3)
        self.assertEqual(len(rows), 1)                             # logged once, at step 2
        self.assertEqual(rows[0]["pl_seen"], 40)


@unittest.skipUnless(HAVE_TF, "TensorFlow / generator_module not importable: Part 2 NOT RUN")
class TestSnapshotCheckpoint(unittest.TestCase):
    """finding 3: the every_n_steps snapshot and its refresh position survive save/restore."""

    def setUp(self):
        self.cfg = cfg_for(pseudo_threshold=0.0, pseudo_refresh="every_n_steps", pseudo_refresh_every=3)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _save(self, cfg, subdir, steps=(1, 2), saved_step=2, tamper_snapshot_step=None):
        models = tm.build_models(cfg)
        tr = tm.Trainer(models, cfg)
        for s in steps:
            tr.step(TestTrainStep.batch(s), s)
        step_var = tm.tf.Variable(saved_step, dtype=tm.tf.int64)
        ckpt = tm.make_checkpoint(models, step_var)
        mgr = tm.tf.train.CheckpointManager(ckpt, os.path.join(self.tmp.name, subdir), max_to_keep=1)
        if tamper_snapshot_step is not None:
            models.snapshot_step.assign(tamper_snapshot_step)
        tm.save_checkpoint(mgr, models)
        return models, mgr.latest_checkpoint

    def _fresh(self, cfg):
        models = tm.build_models(cfg)
        step_var = tm.tf.Variable(0, dtype=tm.tf.int64)
        return models, step_var, tm.make_checkpoint(models, step_var)

    def test_snapshot_and_refresh_position_survive_save_restore(self):
        tf = tm.tf
        m1, path = self._save(self.cfg, "ok")                       # 2 steps: snapshot still = step-0 weights
        self.assertEqual(int(m1.snapshot_step.numpy()), 0)
        x = tm.to_float(tf.constant(TestTrainStep.batch(9)[2]))
        snap_out = m1.clf_snapshot(x, training=False).numpy()
        live_out = m1.clf(x, training=False).numpy()
        self.assertFalse(np.allclose(snap_out, live_out))           # precondition: snapshot != live

        m2, step_var2, ckpt2 = self._fresh(self.cfg)
        self.assertFalse(np.allclose(m2.clf_snapshot(x, training=False).numpy(), snap_out))   # fresh models differ
        self.assertEqual(tm.restore_checkpoint(ckpt2, path, m2, self.cfg, step_var2), 2)

        self.assertEqual(int(m2.snapshot_step.numpy()), 0)          # refresh position restored
        np.testing.assert_allclose(m2.pseudo_label_model(x, training=False).numpy(), snap_out, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m2.clf(x, training=False).numpy(), live_out, rtol=1e-5, atol=1e-6)
        self.assertFalse(np.allclose(m2.pseudo_label_model(x, training=False).numpy(),
                                     m2.clf(x, training=False).numpy()))   # NOT silently re-synced to live

        tr2 = tm.Trainer(m2, self.cfg)
        tr2.step(TestTrainStep.batch(3), 3)                         # step 3 is a refresh step
        self.assertEqual(int(m2.snapshot_step.numpy()), 3)
        close = lambda a, b: all(np.allclose(p.numpy(), q.numpy()) for p, q in zip(a.variables, b.variables))
        self.assertTrue(close(m2.clf, m2.clf_snapshot))

    def test_refresh_position_after_a_refresh_is_restored(self):
        m1, path = self._save(self.cfg, "after", steps=(1, 2, 3, 4), saved_step=4)
        self.assertEqual(int(m1.snapshot_step.numpy()), 3)
        m2, step_var2, ckpt2 = self._fresh(self.cfg)
        self.assertEqual(tm.restore_checkpoint(ckpt2, path, m2, self.cfg, step_var2), 4)
        self.assertEqual(int(m2.snapshot_step.numpy()), 3)
        x = tm.to_float(tm.tf.constant(TestTrainStep.batch(9)[2]))
        np.testing.assert_allclose(m2.clf_snapshot(x, training=False).numpy(),
                                   m1.clf_snapshot(x, training=False).numpy(), rtol=1e-5, atol=1e-6)

    def test_inconsistent_refresh_position_is_refused(self):
        _, path = self._save(self.cfg, "bad", tamper_snapshot_step=1)    # step 2 with N=3 needs position 0
        m2, step_var2, ckpt2 = self._fresh(self.cfg)
        with self.assertRaises(tm.ResumeError):
            tm.restore_checkpoint(ckpt2, path, m2, self.cfg, step_var2)

    def test_checkpoint_without_snapshot_is_refused_not_resynced(self):
        live_cfg = cfg_for(pseudo_threshold=0.0, pseudo_refresh="every_step")
        _, path = self._save(live_cfg, "nosnap")                         # made in every_step mode
        m2, step_var2, ckpt2 = self._fresh(self.cfg)                     # now resumed as every_n_steps
        with self.assertRaises(tm.ResumeError) as cm:
            tm.restore_checkpoint(ckpt2, path, m2, self.cfg, step_var2)
        self.assertIn("no saved pseudo-label snapshot", str(cm.exception))

    def test_every_step_mode_needs_no_snapshot_state(self):
        live_cfg = cfg_for(pseudo_threshold=0.0, pseudo_refresh="every_step")
        m1, path = self._save(live_cfg, "live")
        self.assertIsNone(m1.clf_snapshot)
        m2, step_var2, ckpt2 = self._fresh(live_cfg)
        self.assertEqual(tm.restore_checkpoint(ckpt2, path, m2, live_cfg, step_var2), 2)
        self.assertIs(m2.pseudo_label_model, m2.clf)


if __name__ == "__main__":
    unittest.main()
