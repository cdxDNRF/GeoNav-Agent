"""Threshold selection support, abstention and correct public-state filtering."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.candidate_reliability import calibrate,threshold_metrics,winners


class ReliabilityTests(unittest.TestCase):
    def bank(self):
        # For current12, up7 is correct; a distant target0 is not adjacent.
        return [dict(area=f'a{i}',current=12,target=7 if j<20 else 0,label=0 if j<20 else 4) for i in range(5) for j in range(40)]
    def test_precision_screen_rejects_high_wrong_scores_and_selects_lowest_supported_threshold(self):
        b=self.bank();p=np.array([[.98,.002,.002,.002,.014] if r['label']==0 else [.92,.01,.01,.01,.05] for r in b])
        x=calibrate(p,b);self.assertEqual(x['threshold'],.95);self.assertEqual(x['grid'][5]['metrics']['accepted'],100)
        self.assertEqual(x['grid'][5]['metrics']['accepted_precision'],1);self.assertFalse(x['grid'][4]['passed'])
    def test_no_valid_threshold_is_explicit_all_abstain(self):
        b=self.bank();p=np.tile([.98,.002,.002,.002,.014],(len(b),1));x=calibrate(p,b)
        self.assertIsNone(x['threshold']);self.assertTrue(x['all_abstain'])
        m=threshold_metrics(p,b,None);self.assertEqual(m['accepted'],0);self.assertIsNone(m['accepted_precision']);self.assertEqual(m['precision_interval']['zero_accept_resamples'],2000)
    def test_scalar_gate_preserves_legal_top_abstention_without_reranking(self):
        b=[dict(area='a',current=2,target=7,label=2)];p=np.array([[.7,.01,.27,.01,.01]])
        self.assertEqual(threshold_metrics(p,b,.5)['accepted'],0)
    def test_unaccepted_sources_remain_bootstrap_units(self):
        b=self.bank();p=np.array([[.98,.002,.002,.002,.014] if r['area']=='a0' else [.01,.01,.01,.01,.96] for r in b]);m=threshold_metrics(p,b,.95)
        self.assertEqual(m['precision_interval']['source_count'],5);self.assertEqual(m['sources_with_acceptances'],1);self.assertGreater(m['precision_interval']['zero_accept_resamples'],0)
    def test_four_view_choice_uses_pixels_and_raw_joint_probabilities(self):
        b=[dict(area='a',current=12,target=7,label=0)];p=np.array([[[.49,0,0,0,.51],[.4,0,0,0,.6],[.3,0,0,0,.7],[.2,0,0,0,.8]]]);payloads={a:{('a',7):str(a).encode()} for a in (0,90,180,270)}
        indices,values=winners(p,payloads,b);self.assertEqual(indices.tolist(),[0]);self.assertEqual(values[0,0],.49);self.assertEqual(threshold_metrics(values,b,.5)['accepted'],0)


if __name__=='__main__':unittest.main()
