#!/usr/bin/env python3
"""GC-SWGAN Member 3: audit-first, one-time evaluation of selected classifiers.

No calls to train_step(), optimizer.apply_gradients(), or test-based checkpoint selection.
Works with the EXACT code commit recorded in Member 2's handoff ZIP.
"""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse, csv, gc, hashlib, json, sys, zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import h5py

RUNS={'A':'full_A_96_seed0', 'B':'full_B_96_seed0'}
KNOWN_DATASET_SHA='a75906f8e3eae0150e3c9de48c7dd74faae3a823130e0030d27d6866a8ffbed4'
KNOWN_SPLIT_SHA='1ea636a1cd3df30fdd6b155b06896692198e141d536d28a60899f17955c1ede8'
ALLOWED_CONFIG_DIFFERENCES={'experiment','lambda_rec'}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_load(path):
    with open(path,encoding='utf-8') as f:return json.load(f)


def check(ok,msg):
    if not ok:raise RuntimeError('PREFLIGHT FAILED: '+msg)


def prepare_archive(archive,workspace):
    """Accept original zip or already-extracted root; do not overwrite the source."""
    archive=Path(archive)
    if archive.is_dir():
        root=archive
    else:
        check(archive.is_file(),f'Archive missing: {archive}')
        root=workspace/'extracted_member2'
        if not (root/'member2_run_code_commit.txt').is_file():
            root.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    target=(root/info.filename).resolve()
                    check(str(target).startswith(str(root.resolve())+os.sep), 'unsafe archive entry')
                z.extractall(root)
    return root


def preflight(repo,archive_root,h5_path):
    """File-only checks plus HDF5 SHA256 and exact corrected indices via team loader."""
    import pandas as pd
    import training_module as tm
    repo=Path(repo).resolve()
    archive_root=Path(archive_root).resolve()
    h5_path=Path(h5_path).resolve()
    check(h5_path.is_file(),f'Galaxy10 dataset missing: {h5_path}')
    commit_file=archive_root/'member2_run_code_commit.txt'
    check(commit_file.is_file(),'missing code commit provenance')
    commit=commit_file.read_text().strip()
    from subprocess import check_output
    actual_commit=check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    check(actual_commit==commit,f'Git checkout {actual_commit} != training commit {commit}. Checkout pinned commit in separate clone.')
    split_dir=repo/'artifacts/member3/corrected_split'
    paths={k:split_dir/v for k,v in [('npz','corrected_split_indices.npz'),('manifest','corrected_split_manifest.json'),('validation','corrected_split_validation.json')]}
    for k,p in paths.items():check(p.is_file(),f'Missing corrected split {k}: {p}')
    check(sha256(paths['npz'])==KNOWN_SPLIT_SHA,'corrected NPZ SHA mismatch')
    configs={};states={};logs={}; manifests={}
    for e,run in RUNS.items():
        out=archive_root/'outputs_i2i'/run
        ckpt=archive_root/'checkpoints_i2i'/run
        for p in (out/'args.json',out/'val.csv',out/'loss.csv',out/'split_audit.json',ckpt/'state.json',ckpt/'manifest.json',ckpt/'best_clf.index',ckpt/'best_clf.data-00000-of-00001'):
            check(p.is_file(),f'{e}: missing {p}')
        args=json_load(out/'args.json')
        state=json_load(ckpt/'state.json')
        manifest=json_load(ckpt/'manifest.json')
        split_audit=json_load(out/'split_audit.json')
        cfg=args['config']; configs[e]=cfg;states[e]=state;manifests[e]=manifest
        check(cfg['experiment']==e,f'{e}: wrong experiment in config')
        check(cfg['img_size']==96,f'{e}: unexpected resolution')
        check(cfg['seed']==0,f'{e}: unexpected seed')
        check(args['args']['skip_test'] is True,f'{e}: training evaluated test unexpectedly')
        check(manifest['data']['include_external_cutouts'] is False,f'{e}: external cutouts enabled')
        check(manifest['data']['split_source']=='member3_corrected_npz',f'{e}: wrong split source')
        fp=manifest['identity']['labeled_file']
        check(fp['sha256']==KNOWN_DATASET_SHA,f'{e}: dataset mismatch in run manifest')
        sp=manifest['info']['corrected_split']['split_npz']['sha256']
        check(sp==KNOWN_SPLIT_SHA,f'{e}: run used different split NPZ')
        check(split_audit['split_npz']['sha256']==KNOWN_SPLIT_SHA,f'{e}: split audit mismatch')
        check(split_audit['validation_status']=='PASS_CONSERVATIVE_GEOMETRIC_GUARD',f'{e}: split geometric guard not passed')
        val=pd.read_csv(out/'val.csv')
        loss=pd.read_csv(out/'loss.csv')
        check(len(val)==20 and int(val['step'].max())==10000,f'{e}: incomplete validation history')
        check(int(loss['step'].max())==10000,f'{e}: incomplete training loss history')
        check(val[['val_acc','val_macro_f1','val_balanced_acc']].notna().all().all(),f'{e}: missing validation metrics')
        check(np.isfinite(loss.select_dtypes(include='number').to_numpy()).all(),f'{e}: nonfinite loss logs')
        imax=val['val_macro_f1'].idxmax()
        step=int(val.loc[imax,'step']); f1=float(val.loc[imax,'val_macro_f1'])
        check(step==int(state['best_step']),f'{e}: best step differs from checkpoint state')
        check(abs(f1-float(state['best_val_macro_f1']))<1e-3,f'{e}: selected score differs from state')
        check((step==8500 if e=='A' else step==10000),f'{e}: unexpected selected step')
        logs[e]={'best_step':step,'best_validation_macro_f1':f1,'validation_rows':len(val),'loss_rows':len(loss)}
    for key in set(configs['A'])|set(configs['B']):
        if key not in ALLOWED_CONFIG_DIFFERENCES:check(configs['A'].get(key)==configs['B'].get(key),f'A/B mismatch for {key}')
    check(configs['A']['lambda_rec'] in (None,0), 'A should not have reconstruction weight')
    check(configs['B']['lambda_rec']==1.0,'B must have agreed reconstruction weight 1.0')
    for k in ('idx_test','idx_val','idx_lab','idx_pool','idx_unl_pool','labels'):
        check(manifests['A']['identity'][k]==manifests['B']['identity'][k],f'run identities disagree for {k}')
    # This checks the real 2.7 GB file sha, exact split membership, HDF5 labels, and disjoint group IDs.
    with h5py.File(h5_path,'r') as f:
        labels=np.asarray(f['ans'][:],dtype=np.int64)
        check(f['images'].shape==(17675,256,256,3),'Unexpected Galaxy10 image shape')
    ix,prov=tm.load_corrected_split(str(paths['npz']),str(paths['manifest']),str(paths['validation']),labels,str(h5_path))
    for key in ('pool','lab','unl_pool','val','test'):
        actual_identity=tm.array_fingerprint(np.sort(ix[key]))
        check(actual_identity==manifests['A']['identity']['idx_'+key],f'Corrected {key} indices differ from training identity')
    check(tm.array_fingerprint(labels)==manifests['A']['identity']['labels'],'Galaxy10 labels differ from training identity')
    check(len(ix['test'])==1731 and len(np.unique(ix['test']))==1731,'Wrong corrected test indices')
    check(not np.intersect1d(ix['test'],ix['pool']).size and not np.intersect1d(ix['test'],ix['val']).size,'Train/val/test overlap')
    check(not np.intersect1d(ix['val'],ix['pool']).size,'Training/validation overlap')
    check(set(ix['lab'].tolist()).isdisjoint(set(ix['test'].tolist())), 'Labeled train in test')
    check(set(ix['unl_pool'].tolist()).isdisjoint(set(ix['test'].tolist())), 'Unlabeled train in test')
    return {'training_code_commit':commit,'dataset_sha256':KNOWN_DATASET_SHA,'split_npz_sha256':KNOWN_SPLIT_SHA,
            'test_indices_count':len(ix['test']),'test_indices_sha256':sha256_indices(ix['test']),
            'run_evidence':logs,'checked_at_utc':datetime.now(timezone.utc).isoformat()},ix


def sha256_indices(a):
    a=np.asarray(a,dtype=np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_test_images(h5_path,indices,size=96):
    """Exactly match runner load_h5_images: crop 32:224, tf.image.resize(method='area'), round/clamp uint8.
    Only read test indices and leave the original dataset untouched.
    """
    import tensorflow as tf
    images=[]
    with h5py.File(h5_path,'r') as f:
        ds=f['images']
        for j in range(0,len(indices),64):
            cur=np.asarray(indices[j:j+64],dtype=np.int64)
            cropped=np.stack([ds[int(i),32:224,32:224,:] for i in cur])
            resized=tf.image.resize(cropped.astype(np.float32),(size,size),method='area').numpy()
            images.append(np.clip(np.rint(resized),0,255).astype(np.uint8))
    return np.concatenate(images,axis=0)


def run_test(repo,archive_root,h5_path,preflight_record,indices,out):
    import pandas as pd
    import tensorflow as tf
    import training_module as tm
    import train_gc_swgan_i2i as runner
    from sklearn.metrics import classification_report,confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    check(not (out/'FINAL_TEST_RESULTS.json').exists(),'Test results already exist: refusing repeat evaluation')
    check(not (out/'TEST_STARTED.json').exists(),'Test execution was previously started. Inspect outputs before rerunning.')
    # Save one-use intent before any held-out predictions; no tuning/repeated lookups.
    (out/'TEST_STARTED.json').write_text(json.dumps({'test_started_at_utc':datetime.now(timezone.utc).isoformat(),
                       'policy':'Single evaluation of frozen A/B selected checkpoints; no tuning'},indent=2))
    x=load_test_images(h5_path,indices['test'])
    with h5py.File(h5_path,'r') as f:y=np.asarray(f['ans'][:],dtype=np.int64)[indices['test']]
    result={'provenance':preflight_record,'models':{},'warnings':[
        'Single matched run seed (0), no multi-seed uncertainty interval.',
        '253 sky-footprint candidate pairs were grouped conservatively, not confirmed individually as duplicate galaxies.',
        'Pseudo-label acceptance was extremely low in both training runs.',
        'Test metrics were not used to choose checkpoints or hyperparameters.'
    ]}
    per_class_rows=[];summary=[]
    for exp,run in RUNS.items():
        tf.keras.backend.clear_session()
        trunk=tm.build_trunk(96)
        _,clf=tm.build_heads(trunk,96,10)
        prefix=str(Path(archive_root)/'checkpoints_i2i'/run/'best_clf')
        check(Path(prefix+'.index').is_file(),'Missing TF best_clf.index')
        check(Path(prefix+'.data-00000-of-00001').is_file(),'Missing TF best_clf.data file')
        clf.load_weights(prefix)
        probs=runner.predict_probs(clf,x,bs=64)
        check(probs.shape==(len(y),10),f'{exp}: wrong prediction shape')
        check(np.isfinite(probs).all(),f'{exp}: NaN/Inf prediction')
        check(np.allclose(probs.sum(axis=1),1,atol=1e-4),f'{exp}: invalid probability normalization')
        metrics=runner.full_metrics(probs,y)
        pred=np.argmax(probs,axis=1)
        cls=classification_report(y,pred,labels=list(range(10)),output_dict=True,zero_division=0)
        for c in range(10):
            row=cls[str(c)]
            per_class_rows.append({'experiment':exp,'class':c,'precision':row['precision'],
                   'recall':row['recall'],'f1':row['f1-score'],'support':int(row['support'])})
        cm=np.asarray(metrics['confusion_matrix'],dtype=np.int64)
        check(int(cm.sum())==1731, f'{exp}: confusion matrix count mismatch')
        result['models'][exp]={'run':run,'selected_validation_step':preflight_record['run_evidence'][exp]['best_step'],
                'selected_validation_macro_f1':preflight_record['run_evidence'][exp]['best_validation_macro_f1'],
                'checkpoint_sha256':{'index':sha256(prefix+'.index'),'data':sha256(prefix+'.data-00000-of-00001')},
                'test_metrics':metrics,'per_class':{str(k):cls[str(k)] for k in range(10)}}
        summary.append({'experiment':exp,'selected_val_step':preflight_record['run_evidence'][exp]['best_step'],
            'selected_val_macro_f1':preflight_record['run_evidence'][exp]['best_validation_macro_f1'],
            'test_accuracy':metrics['accuracy'],'test_macro_f1':metrics['macro_f1'],
            'test_balanced_accuracy':metrics['balanced_accuracy'],'test_weighted_f1':metrics['weighted_f1'],
            'test_ece':metrics['ece'],'test_brier':metrics['brier']})
        fig,ax=plt.subplots(figsize=(7,6))
        im=ax.imshow(cm,cmap='Blues')
        ax.set_title(f'Experiment {exp}: held-out test confusion matrix (n=1731)')
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
        ax.set_xticks(range(10));ax.set_yticks(range(10));fig.colorbar(im,ax=ax)
        for i in range(10):
            for j in range(10):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,
                        color=('white' if cm[i,j] > (cm.max()/2) else 'black'))
        fig.tight_layout();fig.savefig(out/f'test_confusion_{exp}.png',dpi=150);plt.close(fig)
        del clf,trunk,probs
        gc.collect()
    pd.DataFrame(summary).to_csv(out/'FINAL_TEST_COMPARISON.csv',index=False)
    pd.DataFrame(per_class_rows).to_csv(out/'FINAL_TEST_PER_CLASS.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    ax.bar(['A','B'],[result['models'][e]['test_metrics']['macro_f1'] for e in ('A','B')])
    ax.set_ylim(0,1);ax.set_ylabel('Held-out test macro-F1');ax.set_title('Final selected checkpoints — held-out test')
    fig.tight_layout();fig.savefig(out/'final_test_macro_f1_comparison.png',dpi=160);plt.close(fig)
    (out/'FINAL_TEST_RESULTS.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,help='Checkout of exact training code commit')
    p.add_argument('--archive',required=True,help='member2_full_AB_review.zip or extracted root')
    p.add_argument('--dataset',required=True,help='Galaxy10 HDF5 whose SHA256 matches run')
    p.add_argument('--output',default='/kaggle/working/member3_final_test_results')
    p.add_argument('--confirm-test-once',action='store_true',help='Authorizes one frozen final test inference for both checkpoints')
    a=p.parse_args()
    repo=Path(a.repo).resolve()
    sys.path.insert(0,str(repo))
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    arc=prepare_archive(a.archive,out)
    evidence,ix=preflight(repo,arc,a.dataset)
    (out/'PRETEST_VERIFICATION.json').write_text(json.dumps(evidence,indent=2))
    print('PREFLIGHT PASS',json.dumps(evidence,indent=2))
    if not a.confirm_test_once:
        print('NO TEST INFERENCE PERFORMED. Review PRETEST_VERIFICATION.json and rerun with --confirm-test-once to perform the one final test evaluation.')
        return
    r=run_test(repo,arc,a.dataset,evidence,ix,out)
    for ex in ('A','B'):
        m=r['models'][ex]['test_metrics']
        print(f'FINAL TEST {ex}: acc={m["accuracy"]:.4f}, macro_F1={m["macro_f1"]:.4f}, balanced_acc={m["balanced_accuracy"]:.4f}, weighted_F1={m["weighted_f1"]:.4f}')
    print('Saved final test artifacts at',out)


if __name__=='__main__':main()
