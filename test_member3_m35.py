import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from member3_m35 import *

class TestM35(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name)
        self.a={k:1 for k in REQUIRED_MATCH_KEYS}
        self.a.update(experiment='A',lambda_rec=None)
        self.b=dict(self.a,experiment='B',lambda_rec=2)
        for d in (self.a,self.b):
            for key in ('dataset_hash_verified','approved_split_integrated','external_cutouts_excluded','fresh_independent_start','team_split_approved'): d[key]=True
        for e in ('A','B'):
            pd.DataFrame({'step':[10,20], 'val_acc':[.2,.3], 'val_macro_f1':[.1,.2], 'val_balanced_accuracy':[.2,.25]}).to_csv(self.p/f'val_{e}.csv',index=False)
            pd.DataFrame({'step':[10,20],'c_loss':[2,1.8],'d_loss':[4,3],'g_adv':[-1,-1],
                'g_rec':[0,0] if e=='A' else [.1,.2],
                'g_loss':[-1,-1] if e=='A' else [-.8,-.6],
                'pl_accept_rate':[.0,.5]}).to_csv(self.p/f'loss_{e}.csv',index=False)
    def tearDown(self):self.t.cleanup()
    def test_fair_config(self):self.assertTrue(compare_settings(self.a,self.b)['ready_for_reportable_comparison'])
    def test_changed_settings_block(self):
        self.b['sigma']=2;self.assertFalse(compare_settings(self.a,self.b)['ready_for_reportable_comparison'])
    def test_missing_review_blocks(self):
        self.b['team_split_approved']=False;self.assertFalse(compare_settings(self.a,self.b)['ready_for_reportable_comparison'])
    def test_missing_config_fields_block(self):
        del self.a['dataset_sha256'];self.assertFalse(compare_settings(self.a,self.b)['ready_for_reportable_comparison'])
    def test_A_loss(self):self.assertTrue(validate_loss_separation(self.p/'loss_A.csv','A')['pass'])
    def test_B_loss(self):self.assertTrue(check_B_loss(self.p/'loss_B.csv',2)['pass'])
    def test_loss_detects_wrong_A(self):
        d=load_losses(self.p/'loss_A.csv');d.loc[0,'g_rec']=1
        self.assertFalse(validate_loss_separation(d,'A')['pass'])
    def test_report_generation(self):
        r=generate_validation_report(*(self.p/f'{kind}_{e}.csv' for kind,e in [('val','A'),('val','B'),('loss','A'),('loss','B')]),self.p/'report',self.a,self.b)
        self.assertEqual(r['status'],'REPORTABLE_VALIDATION_COMPARISON')
        self.assertTrue((self.p/'report/validation_results.csv').exists())
    def test_report_refuses_unapproved(self):
        with self.assertRaises(ValueError):
            generate_validation_report(*(self.p/f'{kind}_{e}.csv' for kind,e in [('val','A'),('val','B'),('loss','A'),('loss','B')]),self.p/'report')
    def test_diagnostic_only(self):
        r=generate_validation_report(*(self.p/f'{kind}_{e}.csv' for kind,e in [('val','A'),('val','B'),('loss','A'),('loss','B')]),self.p/'report',allow_unapproved=True)
        self.assertEqual(r['status'],'DIAGNOSTIC_ONLY_NOT_APPROVED')
    def test_template(self):
        prepare_template(self.p/'draft',dataset_sha256='abc')
        d=load_json(self.p/'draft/run_B_review_template.json')
        self.assertIsNone(d['lambda_rec']);self.assertFalse(d['team_split_approved'])
    def test_invalid_log(self):
        pd.DataFrame({'step':[2,1], 'val_acc':[.2,.3], 'val_macro_f1':[.1,.2], 'val_balanced_accuracy':[.2,.25]}).to_csv(self.p/'bad.csv',index=False)
        with self.assertRaises(ValueError):load_validation(self.p/'bad.csv')

if __name__=='__main__':unittest.main()
