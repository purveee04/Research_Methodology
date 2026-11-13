"""
data_audit.py -- Member 3: data-integrity audit helpers (no training, no model code).

REUSES Member 2's training_module.py instead of re-implementing it:
    split_indices, unlabeled_pool_indices, make_data_settings, dataset_identity, file_fingerprint,
    array_fingerprint, footprint_overlap_pairs, positional_match_mask, check_cutout_overlap.
Nothing here changes how training works. Only numpy / scipy / scikit-learn are needed, plus h5py for the
functions that open HDF5 files. TensorFlow is NOT needed to run this module or its tests.

What this adds (things the existing guards do not do):
    1. inspect_h5 / scan_images        what is really inside an HDF5 file (shape, keys, NaNs, per-image hash)
    2. split_membership_report         explicit proof of train/val/test and labelled/unlabelled membership
    3. duplicate_groups                byte-identical images, and whether they straddle splits
    4. galaxy10_sky_overlap            neighbouring / repeated galaxies INSIDE Galaxy10 (train pool vs val/test)
    5. compare_galaxy10_versions       exactly which images differ between two Galaxy10 files (17,736 vs 17,675)
    6. check_external_cutouts          stricter go/no-go for decals_images.h5 (never trusts a command-line pixscale)
    7. write_split_artifacts / verify_split_artifacts
                                       saved index lists + labels + hashes, and proof they regenerate exactly

Image "ID": Galaxy10 has no ID column, so an image is identified by (row index in THAT file, blake2b-16 hash of its
raw bytes). Row indices change if a file version drops images; the hash does not.
"""
import csv
import datetime
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict

import numpy as np

import training_module as tm

LOADER_CROP = (32, 224)          # train_gc_swgan_i2i.load_h5_images keeps rows/cols [32:224] -> 192 px
CENTRE_ARCSEC = 3.0              # same centre-match radius as training_module.check_cutout_overlap
SPLIT_MANIFEST_FORMAT = 1


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % (o,))


def to_json(obj, **kw):
    return json.dumps(obj, indent=2, sort_keys=True, default=_jd, **kw)


# --------------------------------------------------------------------------- #
# 1. What is inside an HDF5 file
# --------------------------------------------------------------------------- #
def _stats(a):
    a = np.asarray(a, float).ravel()
    ok = a[~np.isnan(a)]
    return {"n": int(a.size), "n_nan": int(a.size - ok.size),
            "min": float(ok.min()) if ok.size else None, "max": float(ok.max()) if ok.size else None,
            "median": float(np.median(ok)) if ok.size else None}


def inspect_h5(path, image_key="images", label_key="ans", fingerprint=True, n_classes=10):
    """Shape / keys / metadata summary WITHOUT loading the images. fingerprint=True also hashes the file
    (SHA-256, tens of seconds for a multi-GB file)."""
    import h5py
    info = {"path": os.fspath(path), "size_bytes": os.path.getsize(path)}
    with h5py.File(path, "r") as f:
        info["keys"] = {k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)}
                        for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        if image_key not in info["keys"]:
            raise KeyError("%s has no %r dataset (keys: %s)" % (path, image_key, sorted(info["keys"])))
        n, h, w, c = f[image_key].shape
        info.update(n_images=int(n), height=int(h), width=int(w), channels=int(c))
        for k in ("ra", "dec", "pxscale", "redshift"):
            if k in info["keys"]:
                info[k] = _stats(f[k][()])
                info[k]["length_matches_images"] = bool(info[k]["n"] == n or (k == "pxscale" and info[k]["n"] == 1))
        info["has_labels"] = label_key in info["keys"]
        if info["has_labels"]:
            y = np.asarray(f[label_key][()]).astype(np.int64)
            info["label_range"] = [int(y.min()), int(y.max())]
            if y.min() >= 0:
                info["class_counts"] = np.bincount(y, minlength=max(n_classes, int(y.max()) + 1)).tolist()
    usable = lambda k: k in info and info[k]["length_matches_images"] and info[k]["n_nan"] == 0
    info["has_ra_dec"] = bool(usable("ra") and usable("dec"))
    info["has_pxscale"] = bool(usable("pxscale"))
    lo, hi = LOADER_CROP
    info["loader_crop_hw"] = [max(0, min(hi, h) - lo), max(0, min(hi, w) - lo)]
    info["loader_crop_is_192"] = info["loader_crop_hw"] == [hi - lo, hi - lo]
    if fingerprint:
        info["file"] = tm.file_fingerprint(path)
    return info


def scan_images(path, image_key="images", chunk=500):
    """One streamed pass: ordered list of per-image hashes (blake2b-16 of the raw uint8 bytes, hex; the same
    digest definition as training_module.image_hashes) and indices of blank (constant-value) images."""
    import h5py
    hashes, blank = [], []
    with h5py.File(path, "r") as f:
        ds = f[image_key]
        for s in range(0, ds.shape[0], chunk):
            x = np.ascontiguousarray(ds[s:s + chunk])
            flat = x.reshape(len(x), -1)
            blank += (s + np.flatnonzero(flat.min(1) == flat.max(1))).tolist()
            hashes += [hashlib.blake2b(x[i].tobytes(), digest_size=16).hexdigest() for i in range(len(x))]
    return {"hashes": hashes, "blank_idx": blank}


def read_g10_arrays(path):
    """labels, ra, dec, pxscale (None if absent) from a Galaxy10-style file."""
    import h5py
    with h5py.File(path, "r") as f:
        g = lambda k: np.array(f[k]) if k in f else None
        return np.asarray(f["ans"][()]).astype(np.int64), g("ra"), g("dec"), g("pxscale")


# --------------------------------------------------------------------------- #
# 2. Membership proof for the split
# --------------------------------------------------------------------------- #
def split_names(n, ix):
    """Array of 'train' / 'val' / 'test' per image index (object dtype)."""
    names = np.full(n, "unassigned", dtype=object)
    for k, nm in (("pool", "train"), ("val", "val"), ("test", "test")):
        names[np.asarray(ix[k])] = nm
    return names


def split_membership_report(labels, ix, unl_idx, include_labelled=False, n_classes=10):
    """Explicit checks on the index sets produced by training_module.split_indices /
    unlabeled_pool_indices. Returns {"ok", "problems", sizes, class_counts, overlaps, ...}."""
    labels = np.asarray(labels)
    n, problems = len(labels), []
    sets = {k: np.asarray(ix[k]) for k in ("pool", "lab", "val", "test")}
    sets["unl_pool"] = np.asarray(unl_idx)
    for k, a in sets.items():
        if a.size and (a.min() < 0 or a.max() >= n):
            problems.append("%s has indices outside [0, %d)" % (k, n))
        if len(np.unique(a)) != len(a):
            problems.append("%s contains repeated indices" % k)
    S = {k: set(a.tolist()) for k, a in sets.items()}
    overlaps = {}
    for a, b in (("pool", "val"), ("pool", "test"), ("val", "test"), ("lab", "val"), ("lab", "test"),
                 ("unl_pool", "val"), ("unl_pool", "test")):
        overlaps["%s&%s" % (a, b)] = len(S[a] & S[b])
        if overlaps["%s&%s" % (a, b)]:
            problems.append("%d image(s) are in both %s and %s" % (overlaps["%s&%s" % (a, b)], a, b))
    if not S["lab"] <= S["pool"]:
        problems.append("labelled set is not a subset of the train pool")
    if not S["unl_pool"] <= S["pool"]:
        problems.append("unlabelled pool is not a subset of the train pool")
    lab_in_unl = len(S["lab"] & S["unl_pool"])
    overlaps["lab&unl_pool"] = lab_in_unl
    if not include_labelled and lab_in_unl:
        problems.append("%d labelled image(s) are also in the unlabelled stream although "
                        "include_labelled_in_unlabeled=False" % lab_in_unl)
    if include_labelled and lab_in_unl != len(S["lab"]):
        problems.append("include_labelled_in_unlabeled=True but not all labelled images are in the unlabelled stream")
    if len(S["pool"] | S["val"] | S["test"]) != n:
        problems.append("pool + val + test do not cover all %d images exactly once" % n)
    out = {"n_images": n, "include_labelled_in_unlabeled": bool(include_labelled),
           "sizes": {k: int(len(a)) for k, a in sets.items()}, "overlaps": overlaps}
    if n and labels.min() >= 0 and labels.max() < n_classes:
        overall = np.bincount(labels, minlength=n_classes) / n
        out["class_counts"] = {k: np.bincount(labels[a], minlength=n_classes).tolist() for k, a in sets.items()}
        out["max_class_share_deviation"] = {
            k: float(np.abs(np.bincount(labels[a], minlength=n_classes) / len(a) - overall).max())
            for k, a in sets.items() if len(a)}
    else:
        problems.append("labels are outside [0, %d)" % n_classes)
    out["problems"], out["ok"] = problems, not problems
    return out


# --------------------------------------------------------------------------- #
# 3. Duplicates
# --------------------------------------------------------------------------- #
def duplicate_groups(hashes, labels=None, split_of=None, max_listed=50):
    """Groups of row indices whose images are byte-identical. split_of (array of split names) flags groups that
    straddle splits; labels flags groups whose copies carry different labels."""
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    dups = [g for g in groups.values() if len(g) > 1]
    out = {"n_duplicate_groups": len(dups), "n_extra_copies": int(sum(len(g) - 1 for g in dups)),
           "groups_listed": dups[:max_listed]}
    if split_of is not None:
        cross = [g for g in dups if len({split_of[i] for i in g}) > 1]
        out["n_groups_across_splits"] = len(cross)
        out["groups_across_splits_listed"] = cross[:max_listed]
        out["n_groups_touching_val_or_test"] = sum(1 for g in dups if {split_of[i] for i in g} & {"val", "test"})
    if labels is not None:
        out["n_groups_with_conflicting_labels"] = sum(1 for g in dups if len({int(labels[i]) for i in g}) > 1)
    return out


# --------------------------------------------------------------------------- #
# 4. Repeated / neighbouring galaxies inside Galaxy10 (train pool vs val/test)
# --------------------------------------------------------------------------- #
def galaxy10_sky_overlap(ra, dec, pxscale, ix, sizes=(256, 192), centre_arcsec=CENTRE_ARCSEC):
    """Does any TRAIN-POOL Galaxy10 galaxy sit on top of a validation/test galaxy?
      centre match : pool centre within centre_arcsec of a val/test centre (the same galaxy seen twice)
      footprint_S  : the S x S pixel squares may overlap (training_module.footprint_overlap_pairs; an UPPER
                     BOUND, orientation-independent). 256 = the whole stored image, 192 = what the model sees.
    Counts only; the caller decides what to do with them."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    px = np.broadcast_to(np.asarray(pxscale, float), ra.shape)
    pool, prot = np.asarray(ix["pool"]), np.concatenate([ix["val"], ix["test"]])
    used = np.concatenate([pool, prot])
    if np.isnan(ra[used]).any() or np.isnan(dec[used]).any() or np.isnan(px[used]).any():
        raise ValueError("ra/dec/pxscale contain NaN: sky checks cannot be trusted")
    if (px <= 0).any():
        raise ValueError("pxscale must be > 0")
    out = {"centre_arcsec": centre_arcsec,
           "n_pool_centre_within_val_test": int(
               tm.positional_match_mask(ra[pool], dec[pool], ra[prot], dec[prot], centre_arcsec).sum())}
    flagged = {}
    for s in sizes:
        half = 0.5 * s * px
        pairs = tm.footprint_overlap_pairs(ra[pool], dec[pool], half[pool], ra[prot], dec[prot], half[prot])
        rows = sorted({i for i, _ in pairs})
        out["footprint_%d" % s] = {"n_pool_galaxies_overlapping_val_test": len(rows), "n_pairs": len(pairs)}
        flagged[s] = [int(pool[i]) for i in rows]
        vt = tm.footprint_overlap_pairs(ra[ix["val"]], dec[ix["val"]], half[ix["val"]],
                                        ra[ix["test"]], dec[ix["test"]], half[ix["test"]])
        out["footprint_%d" % s]["n_val_test_pairs"] = len(vt)
    out["pool_indices_flagged_by_footprint"] = {str(s): v[:200] for s, v in flagged.items()}
    return out


# --------------------------------------------------------------------------- #
# 5. Two versions of Galaxy10 (the 17,736 vs 17,675 question)
# --------------------------------------------------------------------------- #
def compare_galaxy10_versions(hashes_a, hashes_b, labels_a=None, labels_b=None, radec_a=None, radec_b=None,
                              match_arcsec=1.0, max_listed=1000):
    """What does file A (e.g. the 17,736 original) have that file B (the 17,675 'repaired' file) lacks?
    Images are matched by pixel hash (k-th copy to k-th copy). Unmatched A rows are then classified using
    sky position when radec_* = (ra, dec) are given:
        removed_copy      the same pixels still exist in B (a duplicate copy was dropped)
        modified          not in B by pixels, but B has an unmatched row at the same sky position (<= match_arcsec)
        removed           neither
    """
    pos_b = defaultdict(list)
    for j, h in enumerate(hashes_b):
        pos_b[h].append(j)
    used, a2b, unmatched_a = Counter(), {}, []
    for i, h in enumerate(hashes_a):
        k = used[h]
        if h in pos_b and k < len(pos_b[h]):
            a2b[i] = pos_b[h][k]
            used[h] += 1
        else:
            unmatched_a.append(i)
    matched_b = set(a2b.values())
    unmatched_b = [j for j in range(len(hashes_b)) if j not in matched_b]
    order = [a2b[i] for i in sorted(a2b)]
    out = {"n_a": len(hashes_a), "n_b": len(hashes_b), "n_matched": len(a2b),
           "n_unmatched_a": len(unmatched_a), "n_unmatched_b": len(unmatched_b),
           "b_keeps_a_order": order == sorted(order),
           "n_matched_rows_whose_index_differs": int(sum(1 for i, j in a2b.items() if i != j)),
           "unmatched_a_indices": unmatched_a[:max_listed], "unmatched_b_indices": unmatched_b[:max_listed]}
    if labels_a is not None and labels_b is not None:
        out["n_matched_with_changed_label"] = int(sum(1 for i, j in a2b.items() if labels_a[i] != labels_b[j]))
        if unmatched_a:
            out["class_counts_of_unmatched_a"] = np.bincount(
                np.asarray(labels_a)[unmatched_a], minlength=10).tolist()
    removed_copy = [i for i in unmatched_a if hashes_a[i] in pos_b]
    rest = [i for i in unmatched_a if hashes_a[i] not in pos_b]
    out["n_removed_copy"] = len(removed_copy)
    if radec_a is not None and radec_b is not None and rest:
        ra_a, dec_a = np.asarray(radec_a[0], float), np.asarray(radec_a[1], float)
        ra_b, dec_b = np.asarray(radec_b[0], float), np.asarray(radec_b[1], float)
        r = np.asarray(rest)
        mod = tm.positional_match_mask(ra_a[r], dec_a[r], ra_b[unmatched_b], dec_b[unmatched_b], match_arcsec) \
            if unmatched_b else np.zeros(len(r), bool)
        near_kept = tm.positional_match_mask(ra_a[r], dec_a[r], ra_b, dec_b, CENTRE_ARCSEC)
        out["n_modified_same_position_different_pixels"] = int(mod.sum())
        out["n_removed"] = int((~mod).sum())
        out["n_removed_within_%garcsec_of_a_kept_galaxy" % CENTRE_ARCSEC] = int((near_kept & ~mod).sum())
        out["removed_indices_in_a"] = [int(v) for v in r[~mod]][:max_listed]
    else:
        out["n_removed_or_modified_position_unknown"] = len(rest)
    return out


# --------------------------------------------------------------------------- #
# 6. External cutouts (decals_images.h5): go / no-go
# --------------------------------------------------------------------------- #
def external_cutout_verdict(info, overlap=None, n_exact_duplicates=None):
    """Conservative gate. 'CLEARED_PENDING_SIGNOFF' needs EVERY condition below; one failure -> NOT_APPROVED.
    Differences from build_data's own guard: a pixel scale typed on the command line is NOT accepted (it is
    unverified), and the image size must suit the loader's [32:224] crop. Even when cleared, the evidence is
    metadata-based (it never looks at pixels), so the status is still 'pending team sign-off'."""
    reasons = []
    h, w = info["height"], info["width"]
    if h != w or not (224 <= h <= 256):
        reasons.append("images are %dx%d: the loader crop [32:224] would keep only %dx%d pixels (needs a square "
                       "image of 224-256 px; the footprint check assumes <= 256 px)"
                       % (h, w, info["loader_crop_hw"][0], info["loader_crop_hw"][1]))
    if not info.get("has_ra_dec"):
        reasons.append("no usable ra/dec arrays in the file (missing, wrong length, or NaN)")
    if not info.get("has_pxscale"):
        reasons.append("no usable pxscale in the file (a value typed on the command line cannot be verified)")
    if overlap is None:
        reasons.append("footprint overlap check could not run")
    else:
        if overlap.get("mode") != "footprint":
            reasons.append("overlap check ran in mode %r, not the full footprint check" % overlap.get("mode"))
        if not overlap.get("ok", False) or overlap.get("n_overlap_protected", 0) > 0:
            reasons.append("%s extra cutout(s) overlap a validation/test galaxy footprint"
                           % overlap.get("n_overlap_protected"))
    if n_exact_duplicates is None:
        reasons.append("exact-duplicate check against validation/test images was not run")
    elif n_exact_duplicates:
        reasons.append("%d extra cutout(s) are byte-identical to a validation/test image" % n_exact_duplicates)
    return {"status": "NOT_APPROVED" if reasons else "CLEARED_PENDING_SIGNOFF", "reasons": reasons}


def check_external_cutouts(extra_path, g10_path, ix, extra_hashes=None, g10_hashes=None):
    """Run the whole external-cutout gate. ix = training_module.split_indices(...) for THIS Galaxy10 file.
    extra_hashes / g10_hashes = scan_images(...)["hashes"] (optional but needed for the duplicate check)."""
    import h5py
    info = inspect_h5(extra_path)
    _, g_ra, g_dec, g_px = read_g10_arrays(g10_path)
    overlap = None
    if info["has_ra_dec"] and g_ra is not None and g_dec is not None:
        with h5py.File(extra_path, "r") as f:
            e_ra, e_dec = np.array(f["ra"]), np.array(f["dec"])
            e_px = np.array(f["pxscale"]) if info["has_pxscale"] else None
        overlap = tm.check_cutout_overlap(
            e_ra, e_dec, g_ra, g_dec, protected_idx=np.concatenate([ix["val"], ix["test"]]),
            other_idx=ix["pool"], extra_pxscale=e_px, extra_pxscale_source="read from the extra file",
            g_pxscale=g_px)
    n_dup = None
    if extra_hashes is not None and g10_hashes is not None:
        prot = {g10_hashes[i] for i in np.concatenate([ix["val"], ix["test"]])}
        n_dup = sum(1 for h in extra_hashes if h in prot)
    return {"file": info, "overlap": overlap, "n_exact_duplicates_of_val_test": n_dup,
            "verdict": external_cutout_verdict(info, overlap, n_dup)}


# --------------------------------------------------------------------------- #
# One call that audits a Galaxy10 file end to end
# --------------------------------------------------------------------------- #
def audit_galaxy10_file(path, data_settings, with_hashes=True, fingerprint=True):
    """Returns (report, state). report is JSON-safe. state keeps the big arrays for later steps:
    labels, ix, unl_idx, hashes, ra, dec, pxscale."""
    s = data_settings
    info = inspect_h5(path, fingerprint=fingerprint)
    labels, ra, dec, px = read_g10_arrays(path)
    ix = tm.split_indices(labels, s["test_frac"], s["val_frac"], s["label_frac"], s["split_seed"])
    unl = tm.unlabeled_pool_indices(ix["pool"], ix["lab"], s["include_labelled_in_unlabeled"])
    report = {"file": info, "data_settings": s,
              "membership": split_membership_report(labels, ix, unl, s["include_labelled_in_unlabeled"])}
    state = {"labels": labels, "ix": ix, "unl_idx": unl, "ra": ra, "dec": dec, "pxscale": px, "hashes": None}
    if with_hashes:
        scan = scan_images(path)
        state["hashes"] = scan["hashes"]
        report["blank_images"] = {"n": len(scan["blank_idx"]), "indices": scan["blank_idx"][:100]}
        report["duplicates"] = duplicate_groups(scan["hashes"], labels, split_names(len(labels), ix))
    if info["has_ra_dec"] and info["has_pxscale"]:
        report["sky_overlap_inside_galaxy10"] = galaxy10_sky_overlap(ra, dec, px, ix)
    else:
        report["sky_overlap_inside_galaxy10"] = "not run: ra/dec/pxscale missing or unusable"
    return report, state


def summarize_galaxy10_report(report):
    """Plain-language lines for the report returned by audit_galaxy10_file."""
    f, m = report["file"], report["membership"]
    sha = f.get("file", {}).get("sha256", "not computed")
    L = ["FILE        %d images of %dx%d, %d classes seen | sha256 %s" % (
        f["n_images"], f["height"], f["width"], len(f.get("class_counts", [])), sha[:16] + "..."),
         "SPLIT SIZES %s" % m["sizes"],
         "MEMBERSHIP  %s%s" % ("OK - train/val/test disjoint, labelled and unlabelled sets consistent"
                               if m["ok"] else "PROBLEMS: ", "" if m["ok"] else "; ".join(m["problems"]))]
    if "blank_images" in report:
        L.append("BLANK       %d blank (constant-value) images" % report["blank_images"]["n"])
    d = report.get("duplicates")
    if d:
        L.append("DUPLICATES  %d byte-identical groups (%d across splits, %d touching val/test, %d with conflicting labels)"
                 % (d["n_duplicate_groups"], d["n_groups_across_splits"], d["n_groups_touching_val_or_test"],
                    d["n_groups_with_conflicting_labels"]))
    o = report["sky_overlap_inside_galaxy10"]
    if isinstance(o, str):
        L.append("SKY         %s" % o)
    else:
        L.append("SKY         train-pool galaxies on top of a val/test galaxy: same centre (<=%g\") %d | "
                 "footprint 256px (upper bound) %d | footprint 192px (what the model sees, upper bound) %d" % (
                     o["centre_arcsec"], o["n_pool_centre_within_val_test"],
                     o["footprint_256"]["n_pool_galaxies_overlapping_val_test"],
                     o["footprint_192"]["n_pool_galaxies_overlapping_val_test"]))
    return L


# --------------------------------------------------------------------------- #
# 7. Saved, verifiable split artifacts
# --------------------------------------------------------------------------- #
def _versions():
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    try:
        import h5py
        v["h5py"] = h5py.__version__
    except ImportError:
        pass
    return v


def write_split_artifacts(out_dir, labels, ix, unl_idx, data_settings, identity=None, content_hashes=None,
                          ra=None, dec=None, source_file=None):
    """Writes, into out_dir:
         split_indices.npz     labels + the index arrays pool / lab / val / test / unl_pool  (the source of truth)
         split_table.csv       one row per image: index, label, split, stream membership, content hash, ra, dec
         split_manifest.json   seeds, fractions, sizes, class counts, hashes, library versions
    identity = training_module.dataset_identity(...) result (hashes of the HDF5 files and index sets)."""
    os.makedirs(out_dir, exist_ok=True)
    labels = np.asarray(labels).astype(np.int64)
    n = len(labels)
    unl = np.asarray(unl_idx)
    npz = os.path.join(out_dir, "split_indices.npz")
    np.savez_compressed(npz, labels=labels, pool=ix["pool"], lab=ix["lab"], val=ix["val"], test=ix["test"],
                        unl_pool=unl)
    names, in_lab, in_unl = split_names(n, ix), np.isin(np.arange(n), ix["lab"]), np.isin(np.arange(n), unl)
    table = os.path.join(out_dir, "split_table.csv")
    with open(table, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "split", "in_labelled_stream", "in_unlabelled_stream",
                    "content_hash", "ra", "dec"])
        for i in range(n):
            w.writerow([i, int(labels[i]), names[i], int(in_lab[i]), int(in_unl[i]),
                        content_hashes[i] if content_hashes is not None else "",
                        "%.7f" % ra[i] if ra is not None else "", "%.7f" % dec[i] if dec is not None else ""])
    rep = split_membership_report(labels, ix, unl, data_settings["include_labelled_in_unlabeled"])
    manifest = {"format": SPLIT_MANIFEST_FORMAT, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_file": source_file, "n_images": n, "data_settings": data_settings,
                "labels_sha256": tm.array_fingerprint(labels), "identity": identity,
                "sizes": rep["sizes"], "class_counts": rep.get("class_counts"),
                "membership_ok": rep["ok"], "membership_problems": rep["problems"],
                "has_content_hashes": content_hashes is not None, "versions": _versions(),
                "npz_sha256": tm.file_fingerprint(npz)["sha256"], "table_sha256": tm.file_fingerprint(table)["sha256"]}
    mpath = os.path.join(out_dir, "split_manifest.json")
    with open(mpath, "w") as f:
        f.write(to_json(manifest))
    return {"npz": npz, "table": table, "manifest": mpath}


def verify_split_artifacts(out_dir, labels, data_settings):
    """Re-derive the split with training_module.split_indices and compare with the saved arrays. Detects a
    changed label file, changed settings, or a scikit-learn version that now splits differently."""
    problems = []
    with open(os.path.join(out_dir, "split_manifest.json")) as f:
        m = json.load(f)
    z = np.load(os.path.join(out_dir, "split_indices.npz"))
    labels = np.asarray(labels).astype(np.int64)
    if tm.array_fingerprint(labels) != m["labels_sha256"]:
        problems.append("labels differ from the labels the split was saved with")
    if not np.array_equal(z["labels"], labels):
        problems.append("saved labels array differs from the labels given")
    if json.loads(json.dumps(data_settings, sort_keys=True)) != m["data_settings"]:
        problems.append("data settings differ from the saved ones")
    for key in ("npz_sha256", "table_sha256"):
        fn = "split_indices.npz" if key == "npz_sha256" else "split_table.csv"
        if tm.file_fingerprint(os.path.join(out_dir, fn))["sha256"] != m[key]:
            problems.append("%s was modified after it was written" % fn)
    s = data_settings
    ix = tm.split_indices(labels, s["test_frac"], s["val_frac"], s["label_frac"], s["split_seed"])
    unl = tm.unlabeled_pool_indices(ix["pool"], ix["lab"], s["include_labelled_in_unlabeled"])
    for k, new in (("pool", ix["pool"]), ("lab", ix["lab"]), ("val", ix["val"]), ("test", ix["test"]),
                   ("unl_pool", unl)):
        if not np.array_equal(z[k], new):
            same = set(z[k].tolist()) == set(np.asarray(new).tolist())
            problems.append("%s: regenerated split %s the saved one" % (
                k, "has the same members but a different order than" if same else "DIFFERS from"))
    now = _versions()
    drift = {k: (m["versions"].get(k), now.get(k)) for k in ("sklearn", "numpy") if m["versions"].get(k) != now.get(k)}
    return {"reproduced": not problems, "problems": problems, "library_version_changes": drift}


def load_split_indices(out_dir):
    """The saved index arrays as a dict(pool, lab, val, test, unl_pool, labels)."""
    z = np.load(os.path.join(out_dir, "split_indices.npz"))
    return {k: z[k] for k in z.files}
