"""
Unit tests for data_audit.py (Member 3). Synthetic data only: no dataset download, no GPU, no TensorFlow.

    python -m unittest test_data_audit -v

Needs numpy, scipy, scikit-learn, h5py (h5py is preinstalled on Kaggle).
"""
import json
import os
import tempfile
import unittest

import h5py
import numpy as np

import data_audit as da
import training_module as tm

SETTINGS = tm.make_data_settings(split_seed=16, test_frac=0.1, val_frac=0.1, label_frac=0.125,
                                 include_labelled_in_unlabeled=False, no_pool_unlabeled=False)
ARCSEC = 1.0 / 3600.0


def make_h5(path, n, size=8, labels=None, radec=True, pxscale=True, seed=0, images=None):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f["images"] = images if images is not None else rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8)
        if labels is not None:
            f["ans"] = labels
        if radec:
            f["ra"] = rng.uniform(100, 200, n)
            f["dec"] = rng.uniform(0, 60, n)
        if pxscale:
            f["pxscale"] = np.full(n, 0.262)
    return path


def balanced_labels(n=400):
    return np.arange(n) % 10


class TestBaselineReproduction(unittest.TestCase):
    def test_split_reproduces_the_numbers_the_baseline_notebook_printed(self):
        """baseline_10000steps.ipynb printed: labeled 1773 | val 1774 | test 1774 | pool 14188 and these labelled
        class counts for the 17,736-image Galaxy10 file. split_indices must give the same."""
        cc = [1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873]          # printed by the baseline
        labels = np.random.default_rng(0).permutation(np.repeat(np.arange(10), cc))
        self.assertEqual(len(labels), 17736)
        ix = tm.split_indices(labels, 0.1, 0.1, 0.125, 16)
        self.assertEqual((len(ix["pool"]), len(ix["lab"]), len(ix["val"]), len(ix["test"])),
                         (14188, 1773, 1774, 1774))
        self.assertEqual(np.bincount(labels[ix["lab"]], minlength=10).tolist(),
                         [108, 185, 264, 203, 34, 204, 183, 263, 142, 187])


class TestInspect(unittest.TestCase):
    def test_small_cutout_file_is_described_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_h5(os.path.join(d, "x.h5"), 5, size=69, radec=False, pxscale=False)
            info = da.inspect_h5(p)
        self.assertEqual((info["n_images"], info["height"], info["width"]), (5, 69, 69))
        self.assertEqual(info["loader_crop_hw"], [37, 37])          # [32:224] on 69 px keeps 37 px
        self.assertFalse(info["loader_crop_is_192"])
        self.assertFalse(info["has_ra_dec"])
        self.assertFalse(info["has_pxscale"])
        self.assertEqual(len(info["file"]["sha256"]), 64)

    def test_full_size_file_with_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_h5(os.path.join(d, "g.h5"), 3, size=256, labels=[0, 1, 2])
            info = da.inspect_h5(p, fingerprint=False)
        self.assertTrue(info["loader_crop_is_192"])
        self.assertTrue(info["has_ra_dec"] and info["has_pxscale"] and info["has_labels"])
        self.assertEqual(info["class_counts"][:3], [1, 1, 1])

    def test_nan_in_coordinates_makes_metadata_unusable(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_h5(os.path.join(d, "g.h5"), 4, size=8)
            with h5py.File(p, "a") as f:
                ra = f["ra"][()]
                ra[1] = np.nan
                f["ra"][...] = ra
            info = da.inspect_h5(p, fingerprint=False)
        self.assertEqual(info["ra"]["n_nan"], 1)
        self.assertFalse(info["has_ra_dec"])

    def test_scan_hashes_match_training_module_and_blank_images_are_found(self):
        x = np.random.default_rng(1).integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
        x[2] = 7                                                      # constant image
        with tempfile.TemporaryDirectory() as d:
            p = make_h5(os.path.join(d, "g.h5"), 6, images=x)
            res = da.scan_images(p, chunk=4)                          # chunk smaller than N on purpose
        self.assertEqual({bytes.fromhex(h) for h in res["hashes"]}, tm.image_hashes(x))
        self.assertEqual(res["blank_idx"], [2])


class TestMembership(unittest.TestCase):
    def setUp(self):
        self.labels = balanced_labels()
        self.ix = tm.split_indices(self.labels, 0.1, 0.1, 0.125, 16)
        self.unl = tm.unlabeled_pool_indices(self.ix["pool"], self.ix["lab"], False)

    def test_real_split_is_clean_and_stratified(self):
        r = da.split_membership_report(self.labels, self.ix, self.unl, False)
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual(r["sizes"]["val"], 40)
        self.assertTrue(all(v == 0 for k, v in r["overlaps"].items() if k != "lab&unl_pool"))
        self.assertEqual(r["overlaps"]["lab&unl_pool"], 0)
        self.assertLess(max(r["max_class_share_deviation"].values()), 0.02)

    def test_val_image_moved_into_pool_is_caught(self):
        ix = {k: v.copy() for k, v in self.ix.items()}
        ix["pool"] = np.concatenate([ix["pool"], ix["val"][:1]])
        r = da.split_membership_report(self.labels, ix, self.unl, False)
        self.assertFalse(r["ok"])
        self.assertTrue(any("pool and val" in p for p in r["problems"]))

    def test_val_image_in_unlabelled_stream_is_caught(self):
        unl = np.concatenate([self.unl, self.ix["test"][:2]])
        r = da.split_membership_report(self.labels, self.ix, unl, False)
        self.assertFalse(r["ok"])
        self.assertTrue(any("unl_pool and test" in p for p in r["problems"]))

    def test_labelled_in_unlabelled_stream_follows_the_flag(self):
        unl_with = tm.unlabeled_pool_indices(self.ix["pool"], self.ix["lab"], True)
        self.assertTrue(da.split_membership_report(self.labels, self.ix, unl_with, True)["ok"])
        self.assertFalse(da.split_membership_report(self.labels, self.ix, unl_with, False)["ok"])
        self.assertFalse(da.split_membership_report(self.labels, self.ix, self.unl, True)["ok"])

    def test_missing_images_are_caught(self):
        ix = {k: v.copy() for k, v in self.ix.items()}
        ix["test"] = ix["test"][1:]
        self.assertFalse(da.split_membership_report(self.labels, ix, self.unl, False)["ok"])


class TestDuplicates(unittest.TestCase):
    def test_duplicate_across_splits_and_conflicting_labels(self):
        hashes = ["a", "b", "c", "a", "d", "e", "e"]
        labels = np.array([1, 2, 3, 4, 5, 6, 6])
        split_of = np.array(["train", "train", "val", "test", "train", "train", "train"], dtype=object)
        r = da.duplicate_groups(hashes, labels, split_of)
        self.assertEqual((r["n_duplicate_groups"], r["n_extra_copies"]), (2, 2))
        self.assertEqual(r["n_groups_across_splits"], 1)                # "a": train + test
        self.assertEqual(r["n_groups_touching_val_or_test"], 1)
        self.assertEqual(r["n_groups_with_conflicting_labels"], 1)      # "a": labels 1 vs 4; "e" agree

    def test_no_duplicates(self):
        r = da.duplicate_groups(["a", "b", "c"], np.array([0, 1, 2]), np.array(["train"] * 3, dtype=object))
        self.assertEqual((r["n_duplicate_groups"], r["n_groups_across_splits"]), (0, 0))


class TestSkyOverlap(unittest.TestCase):
    """pool = {0,1,2}; val = {3}; test = {4}. Galaxy 3 sits at ra=150, dec=30."""
    def setUp(self):
        self.ix = {"pool": np.array([0, 1, 2]), "val": np.array([3]), "test": np.array([4])}
        self.ra = np.array([150.0, 150.0, 150.0, 150.0, 120.0])
        self.dec = np.array([30.0 + 0 * ARCSEC, 30.0 + 80 * ARCSEC, 30.0 + 2000 * ARCSEC, 30.0, 10.0])
        self.px = 0.262

    def test_same_galaxy_twice_neighbour_and_far_galaxy(self):
        # galaxy 0: same place as val galaxy 3; galaxy 1: 80" away; galaxy 2: 2000" away
        r = da.galaxy10_sky_overlap(self.ra, self.dec, self.px, self.ix)
        self.assertEqual(r["n_pool_centre_within_val_test"], 1)
        # bound = (half_i + half_j) * sqrt(2): 256 px -> 94.8 arcsec (galaxy 1 flagged), 192 px -> 71.1 (not flagged)
        self.assertEqual(r["footprint_256"]["n_pool_galaxies_overlapping_val_test"], 2)
        self.assertEqual(r["footprint_192"]["n_pool_galaxies_overlapping_val_test"], 1)
        self.assertEqual(r["pool_indices_flagged_by_footprint"]["256"], [0, 1])
        self.assertEqual(r["pool_indices_flagged_by_footprint"]["192"], [0])

    def test_nothing_overlaps_when_far_apart(self):
        dec = self.dec.copy()
        dec[0], dec[1] = 5.0, 6.0
        r = da.galaxy10_sky_overlap(self.ra, dec, self.px, self.ix)
        self.assertEqual(r["n_pool_centre_within_val_test"], 0)
        self.assertEqual(r["footprint_256"]["n_pool_galaxies_overlapping_val_test"], 0)

    def test_nan_coordinates_are_refused(self):
        ra = self.ra.copy()
        ra[0] = np.nan
        with self.assertRaises(ValueError):
            da.galaxy10_sky_overlap(ra, self.dec, self.px, self.ix)


class TestCompareVersions(unittest.TestCase):
    def test_removed_modified_and_duplicate_copy_are_told_apart(self):
        # A: 10 images. B drops row 3 (gone), row 6 (a repeated copy of row 2), changes row 5's pixels.
        ha = ["h%d" % i for i in range(10)]
        ha[6] = ha[2]
        hb = [h for i, h in enumerate(ha) if i not in (3, 6)]
        hb[hb.index("h5")] = "h5_fixed"
        ra_a = np.arange(10.0) + 100
        dec_a = np.full(10, 20.0)
        keep = [i for i in range(10) if i not in (3, 6)]
        la = np.arange(10) % 3
        r = da.compare_galaxy10_versions(ha, hb, la, la[keep], (ra_a, dec_a), (ra_a[keep], dec_a[keep]))
        self.assertEqual((r["n_a"], r["n_b"]), (10, 8))
        self.assertEqual(r["unmatched_a_indices"], [3, 5, 6])
        self.assertEqual(r["unmatched_b_indices"], [4])
        self.assertEqual(r["n_removed_copy"], 1)                         # row 6 (same pixels as row 2 still in B)
        self.assertEqual(r["n_modified_same_position_different_pixels"], 1)   # row 5
        self.assertEqual(r["n_removed"], 1)                              # row 3
        self.assertEqual(r["removed_indices_in_a"], [3])
        self.assertTrue(r["b_keeps_a_order"])
        self.assertGreater(r["n_matched_rows_whose_index_differs"], 0)   # everything after row 3 shifted
        self.assertEqual(r["n_matched_with_changed_label"], 0)

    def test_identical_files_and_changed_labels(self):
        h = ["a", "b", "c"]
        r = da.compare_galaxy10_versions(h, list(h), np.array([0, 1, 2]), np.array([0, 2, 2]))
        self.assertEqual((r["n_unmatched_a"], r["n_unmatched_b"], r["n_matched_with_changed_label"]), (0, 0, 1))
        self.assertEqual(r["n_matched_rows_whose_index_differs"], 0)


class TestExternalCutouts(unittest.TestCase):
    def _g10(self, d, n=400):
        return make_h5(os.path.join(d, "g10.h5"), n, labels=balanced_labels(n), seed=3)

    def _extra(self, d, g10_path, ix, near_protected=False, size=256, metadata=True, pxscale=True, n=6):
        p = make_h5(os.path.join(d, "extra.h5"), n, size=size, radec=metadata, pxscale=metadata and pxscale, seed=9)
        if metadata:
            with h5py.File(g10_path) as g:
                gra, gdec = g["ra"][()], g["dec"][()]
            with h5py.File(p, "a") as f:
                ra, dec = f["ra"][()], f["dec"][()]
                ra[:], dec[:] = 10.0 + np.arange(n), -50.0                # far from every Galaxy10 galaxy
                if near_protected:
                    v = ix["val"][0]
                    ra[0], dec[0] = gra[v], gdec[v] + 30 * ARCSEC          # 30" from a validation galaxy
                f["ra"][...], f["dec"][...] = ra, dec
        return p

    def test_69px_cutouts_without_metadata_are_not_approved(self):
        with tempfile.TemporaryDirectory() as d:
            g10 = self._g10(d)
            labels = da.read_g10_arrays(g10)[0]
            ix = tm.split_indices(labels, 0.1, 0.1, 0.125, 16)
            extra = self._extra(d, g10, ix, size=69, metadata=False)
            r = da.check_external_cutouts(extra, g10, ix)
        self.assertEqual(r["verdict"]["status"], "NOT_APPROVED")
        text = " ".join(r["verdict"]["reasons"])
        self.assertIn("69x69", text)
        self.assertIn("37x37", text)
        self.assertIn("ra/dec", text)
        self.assertIn("pxscale", text)

    def test_clean_256px_cutouts_with_metadata_and_hashes_are_only_cleared_pending_signoff(self):
        with tempfile.TemporaryDirectory() as d:
            g10 = self._g10(d)
            labels = da.read_g10_arrays(g10)[0]
            ix = tm.split_indices(labels, 0.1, 0.1, 0.125, 16)
            extra = self._extra(d, g10, ix)
            r = da.check_external_cutouts(extra, g10, ix, da.scan_images(extra)["hashes"], da.scan_images(g10)["hashes"])
        self.assertEqual(r["overlap"]["mode"], "footprint")
        self.assertEqual(r["overlap"]["n_overlap_protected"], 0)
        self.assertEqual(r["verdict"], {"status": "CLEARED_PENDING_SIGNOFF", "reasons": []})

    def test_missing_duplicate_check_blocks_approval(self):
        with tempfile.TemporaryDirectory() as d:
            g10 = self._g10(d)
            ix = tm.split_indices(da.read_g10_arrays(g10)[0], 0.1, 0.1, 0.125, 16)
            r = da.check_external_cutouts(self._extra(d, g10, ix), g10, ix)       # no hashes given
        self.assertEqual(r["verdict"]["status"], "NOT_APPROVED")
        self.assertTrue(any("exact-duplicate" in x for x in r["verdict"]["reasons"]))

    def test_footprint_overlap_with_a_validation_galaxy_blocks_approval(self):
        with tempfile.TemporaryDirectory() as d:
            g10 = self._g10(d)
            ix = tm.split_indices(da.read_g10_arrays(g10)[0], 0.1, 0.1, 0.125, 16)
            extra = self._extra(d, g10, ix, near_protected=True)
            r = da.check_external_cutouts(extra, g10, ix, da.scan_images(extra)["hashes"], da.scan_images(g10)["hashes"])
        self.assertGreaterEqual(r["overlap"]["n_overlap_protected"], 1)
        self.assertEqual(r["verdict"]["status"], "NOT_APPROVED")

    def test_missing_pxscale_blocks_approval_even_with_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            g10 = self._g10(d)
            ix = tm.split_indices(da.read_g10_arrays(g10)[0], 0.1, 0.1, 0.125, 16)
            extra = self._extra(d, g10, ix, pxscale=False)
            r = da.check_external_cutouts(extra, g10, ix, da.scan_images(extra)["hashes"], da.scan_images(g10)["hashes"])
        self.assertEqual(r["overlap"]["mode"], "center_only_limited")
        self.assertEqual(r["verdict"]["status"], "NOT_APPROVED")

    def test_exact_duplicate_of_a_test_image_blocks_approval(self):
        with tempfile.TemporaryDirectory() as d:
            g10 = self._g10(d)
            ix = tm.split_indices(da.read_g10_arrays(g10)[0], 0.1, 0.1, 0.125, 16)
            extra = self._extra(d, g10, ix, size=8)               # 8 px images so they can equal Galaxy10's
            with h5py.File(g10) as g:
                img = g["images"][int(ix["test"][0])]
            with h5py.File(extra, "a") as f:
                x = f["images"][()]
                x[1] = img
                f["images"][...] = x
            info = da.inspect_h5(extra, fingerprint=False)
            n_dup = sum(1 for h in da.scan_images(extra)["hashes"]
                        if h in {da.scan_images(g10)["hashes"][i] for i in ix["test"]})
        self.assertEqual(n_dup, 1)
        v = da.external_cutout_verdict(info, None, n_dup)
        self.assertTrue(any("byte-identical" in r for r in v["reasons"]))


class TestArtifacts(unittest.TestCase):
    def setUp(self):
        self.labels = balanced_labels()
        self.ix = tm.split_indices(self.labels, 0.1, 0.1, 0.125, 16)
        self.unl = tm.unlabeled_pool_indices(self.ix["pool"], self.ix["lab"], False)
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.hashes = ["hash%d" % i for i in range(len(self.labels))]

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self):
        return da.write_split_artifacts(self.d, self.labels, self.ix, self.unl, SETTINGS,
                                        content_hashes=self.hashes, source_file="g10.h5")

    def test_round_trip_reproduces_exactly(self):
        paths = self._write()
        r = da.verify_split_artifacts(self.d, self.labels, SETTINGS)
        self.assertTrue(r["reproduced"], r["problems"])
        z = da.load_split_indices(self.d)
        for k in ("pool", "lab", "val", "test"):
            np.testing.assert_array_equal(z[k], self.ix[k])
        np.testing.assert_array_equal(z["unl_pool"], self.unl)
        with open(paths["manifest"]) as fh:
            m = json.load(fh)
        self.assertTrue(m["membership_ok"])
        self.assertEqual(m["data_settings"]["split_seed"], 16)
        self.assertIn("sklearn", m["versions"])

    def test_table_has_one_row_per_image_with_roles(self):
        import csv
        paths = self._write()
        with open(paths["table"]) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), len(self.labels))
        by = {k: [int(r["index"]) for r in rows if r["split"] == k] for k in ("train", "val", "test")}
        self.assertEqual(sorted(by["val"]), sorted(self.ix["val"].tolist()))
        self.assertEqual(sorted(by["test"]), sorted(self.ix["test"].tolist()))
        lab_rows = {int(r["index"]) for r in rows if r["in_labelled_stream"] == "1"}
        unl_rows = {int(r["index"]) for r in rows if r["in_unlabelled_stream"] == "1"}
        self.assertEqual(lab_rows, set(self.ix["lab"].tolist()))
        self.assertEqual(unl_rows, set(self.unl.tolist()))
        self.assertFalse(lab_rows & unl_rows)
        self.assertFalse(unl_rows & (set(by["val"]) | set(by["test"])))

    def test_changed_labels_settings_or_edited_files_are_detected(self):
        self._write()
        bad = self.labels.copy()
        bad[0] = (bad[0] + 1) % 10
        self.assertFalse(da.verify_split_artifacts(self.d, bad, SETTINGS)["reproduced"])
        other = dict(SETTINGS, split_seed=17)
        self.assertFalse(da.verify_split_artifacts(self.d, self.labels, other)["reproduced"])
        with open(os.path.join(self.d, "split_table.csv"), "a") as f:
            f.write("tampered\n")
        r = da.verify_split_artifacts(self.d, self.labels, SETTINGS)
        self.assertFalse(r["reproduced"])
        self.assertTrue(any("split_table.csv" in p for p in r["problems"]))

    def test_a_split_that_regenerates_differently_is_detected(self):
        self._write()
        z = dict(np.load(os.path.join(self.d, "split_indices.npz")))
        z["val"] = z["val"][::-1].copy()                          # same members, other order
        np.savez_compressed(os.path.join(self.d, "split_indices.npz"), **z)
        # the checksum of the npz no longer matches AND the order differs: both are reported
        r = da.verify_split_artifacts(self.d, self.labels, SETTINGS)
        self.assertFalse(r["reproduced"])
        self.assertTrue(any("same members" in p for p in r["problems"]))


class TestEndToEnd(unittest.TestCase):
    def test_audit_galaxy10_file_on_a_synthetic_file(self):
        with tempfile.TemporaryDirectory() as d:
            labels = balanced_labels(400)
            p = make_h5(os.path.join(d, "g10.h5"), 400, labels=labels, seed=5)
            report, state = da.audit_galaxy10_file(p, SETTINGS)
            json.dumps(report, default=da._jd)                    # must be JSON-safe
            self.assertTrue(report["membership"]["ok"])
            self.assertEqual(report["duplicates"]["n_duplicate_groups"], 0)
            self.assertEqual(report["blank_images"]["n"], 0)
            self.assertEqual(report["sky_overlap_inside_galaxy10"]["n_pool_centre_within_val_test"], 0)
            self.assertEqual(len(state["hashes"]), 400)
            lines = da.summarize_galaxy10_report(report)
            self.assertTrue(any(l.startswith("MEMBERSHIP  OK") for l in lines), lines)
            self.assertTrue(any(l.startswith("SKY") and "footprint 192px" in l for l in lines), lines)

    def test_file_without_sky_metadata_skips_the_sky_check_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_h5(os.path.join(d, "g10.h5"), 400, labels=balanced_labels(400), radec=False, pxscale=False)
            report, _ = da.audit_galaxy10_file(p, SETTINGS, with_hashes=False)
        self.assertIsInstance(report["sky_overlap_inside_galaxy10"], str)
        self.assertIn("not run", report["sky_overlap_inside_galaxy10"])
        self.assertTrue(any(l.startswith("SKY") and "not run" in l for l in da.summarize_galaxy10_report(report)))


if __name__ == "__main__":
    unittest.main()
