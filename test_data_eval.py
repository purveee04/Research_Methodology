"""CPU-only synthetic tests for Member 3 evaluation functions."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import data_eval as de

class TestEvaluation(unittest.TestCase):
    def test_perfect_all_classes(self):
        y=np.arange(10);p=np.eye(10)
        z=de.evaluate_probabilities(y,p)
        self.assertEqual(z['accuracy'],1)
        self.assertEqual(z['macro_f1_all_10'],1)
        self.assertEqual(z['balanced_accuracy'],1)
        self.assertEqual(np.diag(z['confusion_matrix']).sum(),10)
        self.assertEqual(z['support'],[1]*10)

    def test_missed_class(self):
        y=np.arange(10);p=np.eye(10);p[9]=p[0]
        z=de.evaluate_probabilities(y,p)
        self.assertAlmostEqual(z['accuracy'],.9)
        self.assertEqual(z['per_class_recall'][9],0.)
        self.assertEqual(z['confusion_matrix'][9][0],1)

    def test_absent_classes(self):
        y=np.array([0,0,1,1]);p=np.eye(10)[y]
        z=de.evaluate_probabilities(y,p)
        self.assertEqual(len(z['per_class_recall']),10)
        self.assertEqual(z['support'][9],0)
        self.assertAlmostEqual(z['macro_f1_all_10'],.2)
        self.assertEqual(z['macro_f1_present_classes'],1.)
        self.assertEqual(z['balanced_accuracy'],1.)

    def test_reject_bad_shapes(self):
        for p in (np.zeros((2,9)),np.zeros((1,10)),np.full((2,10),np.nan)):
            with self.assertRaises(ValueError): de.evaluate_probabilities(np.array([0,1]),p)

    def test_reject_invalid_probs(self):
        with self.assertRaises(ValueError): de.evaluate_probabilities(np.array([0]),np.ones((1,10)))
        with self.assertRaises(ValueError): de.evaluate_probabilities(np.array([10]),np.eye(10)[[0]])

    def test_plot_confusion_and_save_json(self):
        with tempfile.TemporaryDirectory() as d:
            cm=np.eye(10,dtype=int)
            fn=de.plot_confusion_matrix(cm,Path(d)/'cm.png',normalize=True)
            self.assertTrue(Path(fn).stat().st_size>1000)
            fn=de.save_json({'x':np.int64(3)},Path(d)/'data.json')
            self.assertEqual(json.loads(Path(fn).read_text())['x'],3)

    def test_log_curves(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'val.csv'
            pd.DataFrame({'step':[1,2,3],'val_acc':[.1,.2,.3],'val_macro_f1':[.01,.04,.08]}).to_csv(p,index=False)
            fn=de.plot_log_curve(p,'accuracy',Path(d)/'acc.png')
            self.assertTrue(Path(fn).exists())
            with self.assertRaises(KeyError):de.plot_log_curve(p,'critic_loss',Path(d)/'bad.png')

    def test_acceptance_rate_from_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'loss.csv'
            pd.DataFrame({'step':[1,2,3], 'pseudo_accepted':[0,10,20], 'pseudo_considered':[20,20,20]}).to_csv(p,index=False)
            frame=de.pseudo_acceptance_frame(p)
            np.testing.assert_allclose(frame.acceptance_rate,[0,.5,1.])
            self.assertTrue(Path(de.plot_pseudo_acceptance(p,Path(d)/'pseudo.png')).exists())

    def test_compare_validation_and_configs(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.csv';b=Path(d)/'b.csv'
            pd.DataFrame({'step':[1,2],'val_acc':[.1,.4],'val_macro_f1':[.08,.3]}).to_csv(a,index=False)
            pd.DataFrame({'step':[1,2],'val_acc':[.2,.3],'val_macro_f1':[.09,.2]}).to_csv(b,index=False)
            summary=de.compare_validation_runs(a,b,Path(d)/'results')
            self.assertEqual(summary['experiments']['A']['accuracy']['best'],.4)
            self.assertTrue((Path(d)/'results'/'compare_accuracy.png').exists())
            self.assertTrue(de.fair_config_diff({'seed':16,'experiment':'A'},{'seed':16,'experiment':'B'})['matched_except_allowed'])
            self.assertFalse(de.fair_config_diff({'seed':16},{'seed':0})['matched_except_allowed'])

if __name__=='__main__': unittest.main()
