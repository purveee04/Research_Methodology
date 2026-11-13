"""Traceable *static* source inspection, NOT proof of runtime behavior.
Runs against the files physically present in Kaggle's cloned repository.
"""
import ast
import json
from pathlib import Path

QUESTIONS = {
 'critic_gradient_penalty':('training_module.py',('gradient_penalty','gp_weight','lambda_gp')),
 'pseudo_confidence_filter':('training_module.py',('pseudo','confidence','threshold')),
 'generator_AB_loss':('training_module.py',('lambda_rec','reconstruction','experiment')),
 'persistent_optimizer':('training_module.py',('optimizer','apply_gradients')),
 'training_validation_boundary':('train_gc_swgan_i2i.py',('build_data','split_indices','skip_test')),
 'full_metrics':('train_gc_swgan_i2i.py',('full_metrics','predict_probs')),
}

def audit_sources(repo):
    root=Path(repo)
    output={'repository':str(root),'notice':'Text/AST presence is NOT verified correct execution; inspect highlighted source and run TensorFlow tests.', 'checks':{}}
    for category,(filename,tokens) in QUESTIONS.items():
        f=root/filename
        if not f.is_file():
            output['checks'][category]={'status':'NOT_VERIFIED','reason':f'Missing {filename}'};continue
        source=f.read_text(encoding='utf8')
        try: tree=ast.parse(source)
        except SyntaxError as e:
            output['checks'][category]={'status':'NOT_VERIFIED','reason':f'Syntax error: {e}'};continue
        funcs=[]
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                funcs.append({'name':n.name,'line':n.lineno})
        lines=source.splitlines()
        occurrences={t:[i+1 for i,line in enumerate(lines) if t.lower() in line.lower()][:18] for t in tokens}
        output['checks'][category]={
            'status':'SOURCE_PRESENT_REVIEW_REQUIRED' if any(occurrences.values()) else 'NOT_VERIFIED',
            'file':filename,'token_line_numbers':occurrences,
            'related_definitions':[f for f in funcs if any(t.lower() in f['name'].lower() for t in tokens)][:25],
        }
    return output

if __name__=='__main__':
    import sys
    repo=sys.argv[1] if len(sys.argv)>1 else '.'
    print(json.dumps(audit_sources(repo),indent=2))
