"""Member 3 M3.5: audit A/B experiment metadata and create VALIDATION-only reports.
No TensorFlow, data loading, model training or test-set inference.
The module refuses to claim a matched comparison if essential run evidence is absent.
"""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VALIDATION_FIELDS = {
    'step': ('step','iteration','global_step'),
    'accuracy': ('val_acc','val_accuracy','accuracy'),
    'macro_f1': ('val_macro_f1','macro_f1','val_f1_macro'),
    'balanced_accuracy': ('val_balanced_accuracy','val_bal_acc','balanced_accuracy'),
}
LOSS_FIELDS = {'step': ('step','iteration','global_step'),
    'c_loss': ('c_loss','c_total'), 'd_loss': ('d_loss','critic_loss'),
    'g_loss': ('g_loss','generator_loss'), 'g_adv': ('g_adv',), 'g_rec': ('g_rec',),
    'pl_accept_rate': ('pl_accept_rate','pseudo_accept_rate','pseudo_acceptance_rate')}
# Expected run settings: assert equality only when *both* manifests explicitly provide a key.
# Missing essentials block certified A/B fairness; never assume defaults are appropriate.
REQUIRED_MATCH_KEYS = ('dataset_sha256','split_npz_sha256','split_seed','run_seed',
    'img_size','steps','labeled_per_batch','unlabeled_per_batch',
    'sigma','pseudo_threshold','pseudo_refresh','pseudo_loss_weight',
    'stream_exhaustion','gp_weight','learning_rate')
EXEMPT_DIFF_KEYS=('experiment','lambda_rec','output_dir','run_name','name','run_dir')


def sha256_file(filename):
    d=hashlib.sha256()
    with open(filename,'rb') as f:
        for buf in iter(lambda:f.read(4*1024*1024), b''): d.update(buf)
    return d.hexdigest()


def load_json(filename):
    return json.loads(Path(filename).read_text(encoding='utf-8'))


def write_json(obj, filename):
    p=Path(filename);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    return p


def col(df, aliases, name):
    lower={str(c).lower():c for c in df.columns}
    for item in aliases[name]:
        if item.lower() in lower: return lower[item.lower()]
    raise ValueError('Missing %s column: expected one of %s; found %s' % (name,aliases[name],list(df.columns)))


def load_log(filename, fields):
    p=Path(filename)
    if not p.is_file(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    if df.empty: raise ValueError('Empty CSV: '+str(p))
    out=pd.DataFrame()
    for f in fields:
        source=col(df,VALIDATION_FIELDS if f in VALIDATION_FIELDS and fields is VALIDATION_FIELDS else LOSS_FIELDS,f)
        out[f]=pd.to_numeric(df[source],errors='raise')
        if not np.isfinite(out[f].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite %s in %s' %(f,p))
    if (out.step.diff().dropna() <= 0).any(): raise ValueError('Steps not strictly increasing in '+str(p))
    if fields is VALIDATION_FIELDS:
        for key in ('accuracy','macro_f1','balanced_accuracy'):
            if ((out[key]<0)|(out[key]>1)).any(): raise ValueError('Invalid metric range: '+key)
    return out


def load_validation(filename):
    return load_log(filename,VALIDATION_FIELDS)


def load_losses(filename):
    # Only require step plus core losses. Pseudo acceptance may not be present in older logs.
    p=Path(filename)
    if not p.is_file(): raise FileNotFoundError(p)
    raw=pd.read_csv(p)
    if raw.empty: raise ValueError('Empty loss CSV: '+str(p))
    d=pd.DataFrame()
    for k in ('step','c_loss','d_loss','g_loss','g_adv','g_rec'):
        d[k]=pd.to_numeric(raw[col(raw,LOSS_FIELDS,k)],errors='raise')
    try:
        d['pl_accept_rate']=pd.to_numeric(raw[col(raw,LOSS_FIELDS,'pl_accept_rate')],errors='raise')
    except ValueError as e:
        if 'Missing pl_accept_rate' not in str(e): raise
    if not np.isfinite(d.to_numpy(dtype=float)).all(): raise ValueError('Nonfinite loss log values')
    if (d.step.diff().dropna()<=0).any(): raise ValueError('Nonmonotonic steps')
    if 'pl_accept_rate' in d and ((d.pl_accept_rate<0)|(d.pl_accept_rate>1)).any():
        raise ValueError('Invalid pseudo-label acceptance rate')
    return d


def validate_loss_separation(losses, experiment, tolerance=1e-4):
    d=load_losses(losses) if isinstance(losses,(str,Path)) else losses
    if experiment not in ('A','B'): raise ValueError('Experiment must be A or B')
    if experiment=='A':
        wrong=(np.abs(d.g_rec)>tolerance)|(np.abs(d.g_loss-d.g_adv)>tolerance)
        return {'pass':not bool(wrong.any()), 'reason':'A must have g_rec=0 and g_loss=g_adv',
                'offending_rows':int(wrong.sum())}
    # B should include reconstruction but cannot determine lambda_rec without run config.
    return {'pass':None,'reason':'For B, confirm g_loss = g_adv + lambda_rec*g_rec with approved lambda_rec',
            'offending_rows':None}


def check_B_loss(losses, lambda_rec, tolerance=1e-4):
    if lambda_rec is None or not (float(lambda_rec)>0): raise ValueError('B requires lambda_rec>0')
    d=load_losses(losses) if isinstance(losses,(str,Path)) else losses
    wrong=np.abs(d.g_loss-(d.g_adv+float(lambda_rec)*d.g_rec))>tolerance
    return {'pass':not bool(wrong.any()),'reason':'B weighted reconstruction equation',
            'offending_rows':int(wrong.sum())}


def compare_settings(a,b):
    """User-provided normalized metadata. Missing keys -> NOT READY, not PASS."""
    if a.get('experiment')!='A' or b.get('experiment')!='B':
        raise ValueError('Must compare A with B')
    if a.get('lambda_rec') not in (None,0,0.0):
        raise ValueError('A may not have reconstruction weight')
    if b.get('lambda_rec') is None or float(b['lambda_rec'])<=0:
        raise ValueError('B needs positive lambda_rec')
    missing=[k for k in REQUIRED_MATCH_KEYS if k not in a or k not in b or a[k] is None or b[k] is None]
    unexpected={k:{'A':a.get(k,'<MISSING>'),'B':b.get(k,'<MISSING>')}
        for k in sorted(set(a)|set(b)) if a.get(k,'<MISSING>') != b.get(k,'<MISSING>')
        and k not in EXEMPT_DIFF_KEYS}
    # Additional conditions need explicit evidence, not inferred from a matching config.
    checklist=('dataset_hash_verified','approved_split_integrated','external_cutouts_excluded',
               'fresh_independent_start','team_split_approved')
    missing_evidence=[k for k in checklist if a.get(k) is not True or b.get(k) is not True]
    is_ready=not missing and not unexpected and not missing_evidence
    return {'ready_for_reportable_comparison':is_ready,
            'missing_match_fields':missing,'unexpected_differences':unexpected,
            'missing_true_evidence':missing_evidence,
            'approved_parameter_values_not_inferred':True}


def prepare_template(directory, split_sha256=None, dataset_sha256=None):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    base={k:None for k in REQUIRED_MATCH_KEYS}
    base['dataset_sha256']=dataset_sha256
    base['split_npz_sha256']=split_sha256
    base['labeled_per_batch']=1
    base['unlabeled_per_batch']=20
    for k in ('dataset_hash_verified','approved_split_integrated','external_cutouts_excluded',
              'fresh_independent_start','team_split_approved'):
        base[k]=False
    for e in ('A','B'):
        entry=dict(base,experiment=e,lambda_rec=None,run_name='experiment_'+e)
        write_json(entry,out/f'run_{e}_review_template.json')
    write_json({'status':'PENDING_RUNS',
      'note':'These are empty research result fields, not measured results.',
      'A':{},'B':{}},out/'comparison_pending.json')
    return out


def _plot_two(a,b,key,path,title):
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,df in [('A',a),('B',b)]:
        ax.plot(df.step,df[key],label=name)
    ax.set(title=title,xlabel='Training step',ylabel=key.replace('_',' ').title())
    ax.legend();ax.grid(alpha=0.25);fig.tight_layout()
    fig.savefig(path,dpi=160);plt.close(fig)


def generate_validation_report(val_a,val_b,loss_a,loss_b,output_dir,
                               config_a=None,config_b=None, allow_unapproved=False):
    """Real CSVs only. Validation results are clearly marked NON-FINAL unless approved and matched.
    No test metrics are read; model selection is not performed here.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    fairness=compare_settings(config_a,config_b) if config_a is not None and config_b is not None else {
        'ready_for_reportable_comparison':False,'reason':'Both run configuration/evidence JSON files are required'}
    if not fairness.get('ready_for_reportable_comparison') and not allow_unapproved:
        raise ValueError('A/B settings or signoff not verified. Refusing reportable comparison; use allow_unapproved=True for diagnostic plots only. '+str(fairness))
    va,vb=load_validation(val_a),load_validation(val_b)
    la,lb=load_losses(loss_a),load_losses(loss_b)
    sepA=validate_loss_separation(la,'A')
    sepB=check_B_loss(lb,config_b['lambda_rec']) if config_b is not None and config_b.get('lambda_rec') else {'pass':None,'reason':'B lambda_rec not provided'}
    record={'status':'REPORTABLE_VALIDATION_COMPARISON' if fairness.get('ready_for_reportable_comparison') and sepA['pass'] and sepB['pass'] else 'DIAGNOSTIC_ONLY_NOT_APPROVED',
       'fairness':fairness,'loss_equations':{'A':sepA,'B':sepB},
       'provenance':{'val_A':str(val_a),'val_B':str(val_b),'loss_A':str(loss_a),'loss_B':str(loss_b)},
       'selection_warning':'Best per-metric steps are descriptive; do not use test set to choose checkpoints.',
       'experiments':{}}
    rows=[]
    for name,df,loss in [('A',va,la),('B',vb,lb)]:
        data={}
        for k in ('accuracy','macro_f1','balanced_accuracy'):
            i=df[k].idxmax()
            data[k]={'best_validation':float(df.loc[i,k]),'step':int(df.loc[i,'step']),
                     'last_validation':float(df.iloc[-1][k])}
        data['last_step']=int(df.iloc[-1].step)
        if 'pl_accept_rate' in loss:
            data['last_logged_acceptance_rate']=float(loss.iloc[-1].pl_accept_rate)
        record['experiments'][name]=data
        rows.append({'experiment':name,'best_val_accuracy':data['accuracy']['best_validation'],
          'best_val_macro_f1':data['macro_f1']['best_validation'],
          'best_val_balanced_accuracy':data['balanced_accuracy']['best_validation'],
          'last_val_step':data['last_step'],'status':record['status']})
    for k in ('accuracy','macro_f1','balanced_accuracy'):
        _plot_two(va,vb,k,out/f'validation_{k}_A_vs_B.png','Validation comparison: '+k)
    for key in ('c_loss','d_loss','g_loss','g_adv','g_rec'):
        _plot_two(la,lb,key,out/f'training_{key}_A_vs_B.png','Logged training: '+key)
    if 'pl_accept_rate' in la and 'pl_accept_rate' in lb:
        _plot_two(la,lb,'pl_accept_rate',out/'pseudo_acceptance_A_vs_B.png','Pseudo-label acceptance (training)')
    pd.DataFrame(rows).to_csv(out/'validation_results.csv',index=False)
    write_json(record,out/'validation_report.json')
    return record
