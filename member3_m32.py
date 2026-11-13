"""Member 3 M3.2: conservative, reproducible leakage-aware Galaxy10 split.

No changes to original HDF5 or teammates' training files.
Workflow: build 256px circumscribed-footprint sky graph, keep connected components
in a single partition, minimally repair existing random split, and export indices.
This *over-approximates* overlapping square cutouts; it is not a duplicate classifier.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

EXPECTED_KEYS = ("pool", "val", "test", "lab", "unl_pool", "labels")
PARTS = ("pool", "val", "test")


def sha256_file(path, chunk=8*1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_prior_split(path):
    with np.load(path, allow_pickle=False) as obj:
        missing = set(EXPECTED_KEYS) - set(obj.files)
        if missing:
            raise ValueError(f"Missing original split arrays: {sorted(missing)}")
        return {key: np.asarray(obj[key], dtype=np.int64) for key in EXPECTED_KEYS}


def check_partition(indices, n):
    arrs = [np.asarray(indices[k], dtype=np.int64) for k in PARTS]
    for a in arrs:
        if len(np.unique(a)) != len(a) or (len(a) and (a.min() < 0 or a.max() >= n)):
            raise AssertionError("Duplicate or out-of-range indices")
    combined = np.concatenate(arrs)
    if len(combined) != n or len(np.unique(combined)) != n:
        raise AssertionError("Train, validation, test must partition all dataset indices")


def spherical_vectors(ra_deg, dec_deg):
    r = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    d = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    if len(r) != len(d) or not (np.isfinite(r).all() and np.isfinite(d).all()):
        raise ValueError("RA/DEC arrays must be equal-size and finite")
    if np.any((d < -np.pi/2) | (d > np.pi/2)):
        raise ValueError("DEC outside [-90,90]")
    return np.column_stack([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


def footprint_pairs(ra, dec, pxscale, image_width_px=256, extra_arcsec=0.0):
    """Return sorted (i,j,centre_separation_arcsec,limit_arcsec) pairs.

    Uses a circumscribed circle around each image's square footprint.
    This may flag nonoverlapping rectangles, but must not miss overlapping squares
    if the supplied centres, scales, and image dimensions are correct.
    """
    ra = np.asarray(ra)
    scale = np.asarray(pxscale, dtype=float)
    if len(scale) != len(ra) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Pixel scale must be finite positive arcsec/pixel for all rows")
    if image_width_px <= 0 or extra_arcsec < 0:
        raise ValueError("Invalid footprint arguments")
    xyz = spherical_vectors(ra, dec)
    radii = scale * image_width_px / math.sqrt(2.0) + extra_arcsec
    max_sum_arcsec = 2 * radii.max()
    chord_max = 2 * np.sin(np.deg2rad(max_sum_arcsec / 3600.0) / 2)
    candidate_pairs = cKDTree(xyz).query_pairs(r=chord_max, output_type="ndarray")
    pairs = []
    for i, j in candidate_pairs:
        distance = np.linalg.norm(xyz[i]-xyz[j])
        separation = np.rad2deg(2 * np.arcsin(np.clip(distance / 2, 0, 1))) * 3600
        threshold = radii[i] + radii[j]
        if separation <= threshold + 1e-8:
            pairs.append((int(i), int(j), float(separation), float(threshold)))
    pairs.sort(key=lambda t: (t[0],t[1]))
    return pairs


def components(n, pairs):
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    for i, j, *_ in pairs:
        ri, rj = find(i),find(j)
        if ri == rj: continue
        if rank[ri] < rank[rj]:ri,rj=rj,ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:rank[ri] += 1
    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    # Give each component an ordered, deterministic ID (minimum member index).
    minimum = {}
    for idx, root in enumerate(roots):
        if root not in minimum: minimum[root]=idx
    return np.asarray([minimum[root] for root in roots], dtype=np.int64)


def minimally_repair(prior, group_id):
    n = len(group_id)
    check_partition(prior, n)
    old_part = np.empty(n, dtype=np.int8)
    for index, name in enumerate(PARTS):
        old_part[prior[name]] = index
    assignment = np.empty(n, dtype=np.int8)
    groups = {}
    for idx, group in enumerate(group_id):
        groups.setdefault(int(group), []).append(idx)
    group_records=[]
    # Tie preference train > val > test; a simple documented minimal-change heuristic.
    for group, member_ids in sorted(groups.items()):
        counts=np.bincount(old_part[member_ids], minlength=3)
        chosen=int(np.argmax(counts))
        assignment[member_ids]=chosen
        if len(member_ids)>1:
            group_records.append({"group_id":group,"size":len(member_ids),"old_counts":dict(zip(PARTS,map(int,counts))),"assigned":PARTS[chosen]})
    result={key:np.where(assignment==idx)[0].astype(np.int64) for idx,key in enumerate(PARTS)}
    changed=np.where(assignment!=old_part)[0]
    result.update({"changed_indices":changed,"assignment":assignment, "group_records":group_records})
    return result


def allocate_labelled(prior, new_pool, labels, seed=16, label_frac=0.125):
    """Retain previous labelled IDs still in train; replenish per-class near 12.5%.

    Train-only labelled/unlabelled separation. Target is global 12.5% as in
    original prior, with per-class counts rounded from actual new train counts.
    """
    rng=np.random.default_rng(seed)
    labels=np.asarray(labels,dtype=np.int64)
    old_lab=set(map(int, prior["lab"]))
    new_lab=[]
    for cls in np.unique(labels):
        cls_pool=np.asarray([i for i in new_pool if labels[i]==cls],dtype=np.int64)
        cls_old_lab=np.asarray([i for i in cls_pool if int(i) in old_lab],dtype=np.int64)
        target=min(len(cls_pool), int(round(len(cls_pool)*label_frac)))
        if len(cls_old_lab)>target:
            cls_old_lab=np.sort(rng.choice(cls_old_lab,size=target,replace=False))
        if len(cls_old_lab)<target:
            selected=set(map(int, cls_old_lab))
            available=np.asarray([i for i in cls_pool if int(i) not in selected],dtype=np.int64)
            extra=rng.choice(available,size=target-len(cls_old_lab),replace=False)
            cls_old_lab=np.concatenate([cls_old_lab,extra])
        new_lab.extend(map(int,cls_old_lab))
    lab=np.asarray(sorted(new_lab),dtype=np.int64)
    unl=np.setdiff1d(new_pool,lab,assume_unique=True)
    return lab,unl


def validate_split(result, group_id, pairs, labels, expected_hash=None, actual_hash=None):
    n=len(group_id)
    check_partition(result,n)
    group_part={}
    for pidx,part in enumerate(PARTS):
        for i in result[part]:
            group=int(group_id[i]); prev=group_part.setdefault(group,pidx)
            if prev!=pidx: raise AssertionError(f"Group {group} crosses partitions")
    assignment=result["assignment"]
    for i,j,*_ in pairs:
        if assignment[i]!=assignment[j]:raise AssertionError("Sky-overlap edge crosses partitions")
    pool=set(map(int,result["pool"])); lab=set(map(int,result["lab"])); unl=set(map(int,result["unl_pool"]))
    if lab & unl or lab | unl != pool:
        raise AssertionError("Labelled/unlabelled do not exactly partition training pool")
    if expected_hash is not None and actual_hash != expected_hash:
        raise AssertionError("Dataset checksum mismatch")
    return {"status":"PASS_CONSERVATIVE_GEOMETRIC_GUARD", "n_pairs_checked":len(pairs),"n_components":len(set(map(int,group_id))),"n_multi_image_groups":int(sum(x["size"]>1 for x in result["group_records"])),"sizes":{k:len(result[k]) for k in ("pool","lab","unl_pool","val","test")},"class_counts":{k:np.bincount(labels[result[k]],minlength=10).tolist() for k in ("pool","lab","unl_pool","val","test")},"warning":"Geometric grouping is conservative, not manual confirmation of duplicates; external cutouts not included; prior checkpoints must not be used to evaluate this split."}


def build_corrected(h5_path, prior_npz, prior_report_json, out_dir, footprint_px=256, expected_sha256=None):
    import h5py
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    old=load_prior_split(prior_npz)
    audit=json.loads(Path(prior_report_json).read_text())
    actual_hash=sha256_file(h5_path)
    stored=audit["file"]["file"]["sha256"]
    expected=expected_sha256 or stored
    if actual_hash!=expected or expected!=stored:
        raise RuntimeError(f"Dataset SHA mismatch: actual={actual_hash}, expected={expected}, audit={stored}")
    with h5py.File(h5_path,"r") as f:
        ra=f["ra"][:];dec=f["dec"][:];pxscale=f["pxscale"][:]; labels=f["ans"][:].astype(np.int64)
    if len(labels)!=len(old["labels"]) or not np.array_equal(labels,old["labels"]):
        raise RuntimeError("Dataset label rows differ from original audited split")
    if footprint_px < 192:
        raise ValueError("Conservative guard must include full 192px crop")
    pairs=footprint_pairs(ra,dec,pxscale,footprint_px)
    group_id=components(len(labels),pairs)
    result=minimally_repair(old,group_id)
    result["lab"],result["unl_pool"]=allocate_labelled(old,result["pool"],labels,seed=int(audit["data_settings"]["split_seed"]),label_frac=float(audit["data_settings"]["label_frac"]))
    verification=validate_split(result,group_id,pairs,labels,expected,actual_hash)
    npz_path=out/"corrected_split_indices.npz"
    np.savez_compressed(npz_path,labels=labels,group_id=group_id,**{k:result[k] for k in ("pool","val","test","lab","unl_pool")})
    original_assigned=np.empty(len(labels),dtype=np.int8)
    for pidx,p in enumerate(PARTS):original_assigned[old[p]]=pidx
    with open(out/"overlap_review.csv","w",newline="") as fp:
        w=csv.writer(fp);w.writerow(["index_a","index_b","separation_arcsec","conservative_threshold_arcsec","label_a","label_b","previous_split_a","previous_split_b","corrected_split","manual_duplicate_confirmation"])
        for i,j,sep,limit in pairs:
            w.writerow([i,j,round(sep,6),round(limit,6),int(labels[i]),int(labels[j]),PARTS[original_assigned[i]],PARTS[original_assigned[j]],PARTS[result["assignment"][i]],"NOT_REVIEWED_GEOMETRIC_GUARD"])
    correction={"policy":"Keep all connected components of circumscribed-circle sky overlaps (256px, per-row pxscale) within one split; choose original majority assignment, tie pool>val>test. Relabel training-only samples to ~12.5% per class. Conservative proxy, not visually confirmed duplicates.","footprint_px":footprint_px,"dataset_sha256":actual_hash,"original_split_sha256":sha256_file(prior_npz),"original_audit_sha256":sha256_file(prior_report_json),"original_sizes":{p:len(old[p]) for p in ("pool","lab","unl_pool","val","test")},"corrected_sizes":verification["sizes"],"indices_changed_partition":result["changed_indices"].tolist(),"n_changed_partition":len(result["changed_indices"]),"n_pairs":len(pairs),"n_original_cross_partition_edges":sum(int(original_assigned[i]!=original_assigned[j]) for i,j,*_ in pairs),"n_group_components":verification["n_components"],"groups_multiple_images":result["group_records"],"blank_images_from_original_audit":audit.get("blank_images"),"unresolved":"Manual classification of geometric overlaps, one constant-value image, dataset version comparison, and external cutout provenance; require team signoff before final training.","created_utc":datetime.now(timezone.utc).isoformat(),"split_seed":audit["data_settings"]["split_seed"],"label_frac":audit["data_settings"]["label_frac"],"source":str(h5_path)}
    (out/"corrected_split_manifest.json").write_text(json.dumps(correction,indent=2))
    (out/"corrected_split_validation.json").write_text(json.dumps(verification,indent=2))
    (out/"TEAM_HANDOFF.txt").write_text("M3.2 CANDIDATE CORRECTED SPLIT - NOT AUTOMATICALLY APPROVED\n\n"+f"Dataset sha256: {actual_hash}\nOriginal split sha256: {correction['original_split_sha256']}\nCorrected split: corrected_split_indices.npz\nSizes: {verification['sizes']}\n{verification['status']}\n{verification['warning']}\n\nIMPORTANT: Member 2 must explicitly load and validate these indices before starting ANY training. Existing build_data/split_indices random generation is not automatically overridden by saving this file. BOTH A and B must train from scratch with this same corrected split.\n")
    return correction,verification

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--h5",required=True)
    p.add_argument("--prior-split",required=True)
    p.add_argument("--prior-report",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--footprint-px",type=int,default=256)
    a=p.parse_args()
    info,checks=build_corrected(a.h5,a.prior_split,a.prior_report,a.out,a.footprint_px)
    print(json.dumps({"dataset_hash":info["dataset_sha256"],"n_changed":info["n_changed_partition"],"n_pairs":info["n_pairs"],"sizes":checks["sizes"],"verification":checks["status"]},indent=2))
