"""Member 3 standalone evaluation/reporting helpers for GC-SWGAN.

Does not import or alter the training pipeline. Requires numpy, pandas,
scikit-learn and matplotlib. All labels are 0..9. Test data is never loaded here.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = np.arange(10, dtype=np.int64)


def evaluate_probabilities(y_true, probabilities, class_count=10):
    """Return metrics with a fixed class universe, including absent classes.

    probabilities: N x class_count, nonnegative finite row-normalized values.
    """
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('y_true must be one-dimensional integer class IDs')
    if p.ndim != 2 or p.shape != (len(y), class_count) or len(y) == 0:
        raise ValueError('probabilities must be nonempty N x class_count')
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('probabilities must be finite and nonnegative')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError('probability rows must sum to one')
    if np.any(y < 0) or np.any(y >= class_count):
        raise ValueError('true class ID out of range')
    pred = np.argmax(p, axis=1)
    classes = np.arange(class_count)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0)
    present = support > 0
    # Standard sklearn macro-F1 with explicit labels averages across all 10 classes.
    # Also report macro-F1 over classes present in y_true, to avoid ambiguity.
    return {
        'n': int(len(y)), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1_all_10': float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
        'macro_f1_present_classes': float(f1[present].mean()) if present.any() else None,
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'support': support.astype(int).tolist(),
        'confusion_matrix': confusion_matrix(y, pred, labels=classes).tolist(),
        'predicted_class': pred.astype(int).tolist(),
    }


def save_json(value, file):
    p = Path(file); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    return str(p)


def plot_confusion_matrix(cm, destination, normalize=False, title=None):
    cm = np.asarray(cm)
    if cm.shape != (10, 10) or np.any(cm < 0) or not np.all(np.isfinite(cm)):
        raise ValueError('Expected nonnegative 10 x 10 matrix')
    z = cm.astype(float)
    if normalize:
        totals = z.sum(axis=1, keepdims=True)
        z = np.divide(z, totals, out=np.zeros_like(z), where=totals != 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(z)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_title(title or ('Normalized confusion matrix' if normalize else 'Confusion matrix'))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p=Path(destination); p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=180);plt.close(fig)
    return str(p)


# Various versions of the runner may use different column names. Fail explicitly
# when not recognized rather than silently drawing an empty chart.
ALIASES = {
    'step': ('step','global_step','iteration','iter'),
    'accuracy': ('val_acc','val_accuracy','accuracy','acc'),
    'macro_f1': ('val_macro_f1','macro_f1','val_f1_macro','f1_macro'),
    'balanced_accuracy': ('val_balanced_accuracy','val_bal_acc','balanced_accuracy','balanced_acc'),
    'acceptance_rate': ('pseudo_accept_rate','pseudo_acceptance_rate','pseudo_rate','accepted_rate','pseudo_accepted_fraction'),
    'accepted': ('pseudo_accepted','n_pseudo_accepted','accepted_pseudo','accepted'),
    'considered': ('pseudo_considered','n_unlabeled','unlabeled_count','pseudo_total','n_pseudo_candidates'),
    'critic_loss': ('d_loss','critic_loss','loss_d'),
    'generator_loss': ('g_loss','generator_loss','loss_g'),
    'classifier_loss': ('c_total','classifier_loss','c_loss'),
    'reconstruction_loss': ('g_rec','reconstruction_loss','loss_rec'),
}

def find_column(df, concept, *, required=True):
    names = ALIASES.get(concept, (concept,))
    lower = {str(c).lower(): c for c in df.columns}
    for alias in names:
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise KeyError(f'No column for {concept}; available: {list(df.columns)}. Add an alias only after checking the real CSV.')
    return None


def read_log(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f'Empty log: {p}')
    return df


def plot_log_curve(csv_file, concept, destination, *, title=None):
    df = read_log(csv_file)
    x = find_column(df,'step'); y=find_column(df,concept)
    vals = pd.to_numeric(df[y], errors='coerce')
    steps = pd.to_numeric(df[x], errors='coerce')
    if not (np.isfinite(vals).any() and np.isfinite(steps).any()):
        raise ValueError(f'No numeric values for {concept}')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(steps, vals,marker='.',markersize=3)
    ax.set(xlabel='Training step',ylabel=concept.replace('_',' ').title(),title=title or concept.replace('_',' ').title())
    ax.grid(alpha=.25);fig.tight_layout()
    p=Path(destination);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=180);plt.close(fig)
    return str(p)


def pseudo_acceptance_frame(csv_file):
    df=read_log(csv_file)
    s=find_column(df,'step')
    rate=find_column(df,'acceptance_rate',required=False)
    if rate is not None:
        values=pd.to_numeric(df[rate],errors='coerce')
    else:
        a=find_column(df,'accepted');b=find_column(df,'considered')
        aa=pd.to_numeric(df[a],errors='coerce');bb=pd.to_numeric(df[b],errors='coerce')
        values=aa.div(bb.replace(0,np.nan)).fillna(0)
    if (values.dropna()<0).any() or (values.dropna()>1).any():
        raise ValueError('Acceptance rates must lie in [0,1]; check whether CSV is percentages')
    return pd.DataFrame({'step':pd.to_numeric(df[s]),'acceptance_rate':values})


def plot_pseudo_acceptance(csv_file,destination):
    data=pseudo_acceptance_frame(csv_file)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(data['step'],data['acceptance_rate']);ax.set_ylim(0,1)
    ax.set(xlabel='Training step',ylabel='Accepted fraction',title='Pseudo-label acceptance (training only)')
    ax.grid(alpha=.25);fig.tight_layout()
    p=Path(destination);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=180);plt.close(fig);return str(p)


def compare_validation_runs(a_val_csv,b_val_csv,output_dir):
    """Compare VAL only. Does not choose a checkpoint or read test data."""
    a,b=read_log(a_val_csv),read_log(b_val_csv)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    summary={'source':'validation only; not final test metrics','experiments':{}}
    for tag,df in [('A',a),('B',b)]:
        s=find_column(df,'step')
        summary['experiments'][tag]={}
        for concept in ('accuracy','macro_f1','balanced_accuracy'):
            col=find_column(df,concept,required=False)
            if col is not None:
                v=pd.to_numeric(df[col],errors='coerce')
                if v.notna().any():
                    j=v.idxmax()
                    summary['experiments'][tag][concept]={'best':float(v.loc[j]),'step_of_best':int(df.loc[j,s])}
    for concept in ('accuracy','macro_f1','balanced_accuracy'):
        fig,ax=plt.subplots(figsize=(8,4.5));n=0
        for tag,df in [('A',a),('B',b)]:
            y=find_column(df,concept,required=False)
            if y is not None:
                ax.plot(df[find_column(df,'step')],df[y],label=tag);n+=1
        if n:
            ax.set(xlabel='Training step',ylabel=concept.replace('_',' ').title(),title='Validation comparison: '+concept)
            ax.legend();ax.grid(alpha=.25);fig.tight_layout()
            fig.savefig(out/f'compare_{concept}.png',dpi=180)
        plt.close(fig)
    save_json(summary,out/'validation_comparison.json')
    return summary


def fair_config_diff(a,b,allowed=('experiment','lambda_rec','name','output_dir','run_dir')):
    """Compare already saved run configs; never assume two runs are matched."""
    a,b=dict(a),dict(b)
    differences={k:{'A':a.get(k,'<MISSING>'),'B':b.get(k,'<MISSING>')}
                 for k in sorted(set(a)|set(b)) if a.get(k,'<MISSING>')!=b.get(k,'<MISSING>')}
    return {'allowed_differences':{k:v for k,v in differences.items() if k in allowed},
            'unexpected_differences':{k:v for k,v in differences.items() if k not in allowed},
            'matched_except_allowed':not any(k not in allowed for k in differences)}
