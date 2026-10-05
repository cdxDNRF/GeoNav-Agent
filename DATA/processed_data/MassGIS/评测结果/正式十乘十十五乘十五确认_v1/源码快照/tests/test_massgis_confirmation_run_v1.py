"""Execution-denominator, pairing, recovery and public-transition checks."""
from pathlib import Path
from unittest.mock import patch
import json
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import massgis_confirmation_run_v1 as r
from eval import audit_massgis_confirmation_run_v1 as audit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_massgis_confirmation_protocol_v1 import ConfirmTests,AreaCompatibilityTests


def row(area='a',seed=0,ep='e',won=False,sg=300,steps=20,accepted=0,hits=0):
    return dict(area=area,seed=seed,episode_id=ep,success=won,sg_m=0 if won else sg,
        valid_travel_m=steps*300,steps=steps,revisits=0,out_of_bounds=0,
        evaluation_diagnostics=[dict(cue_accepted=True,accepted_true_hit=(i<hits)) for i in range(accepted)])


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=(r.ROOT/'选题报告相关').resolve()
        cls.temp=Path(tempfile.mkdtemp(prefix='MassGIS正式执行测试临时_',dir=cls.parent)).resolve()
    @classmethod
    def tearDownClass(cls):
        if cls.temp.parent!=cls.parent or not cls.temp.name.startswith('MassGIS正式执行测试临时_'):raise ValueError('Unsafe cleanup')
        shutil.rmtree(cls.temp)
    def test_all_terminal_SG_denominator_and_move_separate(self):
        rows=[row(won=True,steps=4),row(ep='f',sg=1200)]
        m=r.metrics(rows);self.assertEqual((m['SR'],m['SG_m'],m['movement_m']),(.5,600,3600))
        audit.equal(m,audit.metric(rows))
    def test_actual_acceptance_vs_budget_rates(self):
        m=r.metrics([row(accepted=3,hits=2)])
        self.assertAlmostEqual(m['accepted_precision'],2/3);self.assertEqual(m['false_cue_action_rate'],.05)
        self.assertNotEqual(m['accepted_precision'],1-m['false_cue_action_rate'])
    def test_no_accepted_cue_precision_is_unavailable(self):
        self.assertIsNone(r.metrics([row()])['accepted_precision'])
        self.assertEqual(r.metrics([row()])['false_cue_action_rate'],0)
    def test_empty_cohort_rejected(self):
        with self.assertRaises(ValueError):r.metrics([])
    def test_unpaired_episode_rejected(self):
        with self.assertRaises(ValueError):r.compare([row(ep='x')],[row(ep='y')])
    def test_bootstrap_whole_regions_not_weights(self):
        c=[];b=[];regions=[]
        for i in range(10):
            area=f'a{i}';regions.append(dict(area=area))
            for s in range(3):
                for j in range(10):
                    b.append(row(area,s,str(j),won=j<5))
                    c.append(row(area,s,str(j),won=j<(6 if s!=1 else 4)))
        with patch.object(r,'manifest',return_value={'regions':regions}):
            comp=r.compare(c,b);other=audit.comparison(c,b)
        audit.equal(comp,other)
        for x in comp['region95_SR_difference']:self.assertAlmostEqual(x,1/30)
        self.assertEqual(sum(x>0 for x in comp['seed_SR_differences']),2)
        self.assertEqual((comp['restored'],comp['harmed']),(20,10))
    def test_improvement_does_not_hide_stratum_loss(self):
        c=dict(SR_difference=.04,SG_m_difference=-100,seed_SR_differences=[.04,.03,.05],region95_SR_difference=[.01,.07])
        strata={'short':dict(SR_difference=-.03,SG_m_difference=0),'long':dict(SR_difference=.11,SG_m_difference=-200)}
        g=r.main_gates(c,strata);self.assertTrue(g['SR_gain']);self.assertFalse(g['stratum_SR_protection'])
    def test_SG_loss_rejects_positive_SR(self):
        c=dict(SR_difference=.06,SG_m_difference=10,seed_SR_differences=[.06]*3,region95_SR_difference=[.01,.09])
        self.assertFalse(r.main_gates(c,{'one':c})['SG_no_worse'])
    def test_incomplete_file_is_not_retried(self):
        p=self.temp/'partial.jsonl';p.write_text('{}\n',encoding='utf-8')
        with self.assertRaises(ValueError):r.completed_job(p,{'binding':1},1)
        self.assertEqual(p.read_text(encoding='utf-8'),'{}\n')
    def test_complete_seal_requires_input_binding(self):
        p=self.temp/'sealed.jsonl';p.write_text('{}\n',encoding='utf-8');binding={'fixed_input':'one'}
        r.write(p.with_suffix('.seal.json'),dict(binding=binding,records=1,sha256=r.sha(p)))
        self.assertTrue(r.completed_job(p,binding,1))
        with self.assertRaises(ValueError):r.completed_job(p,{'fixed_input':'changed'},1)
        p.write_text('{"altered":true}\n',encoding='utf-8')
        with self.assertRaises(ValueError):r.completed_job(p,binding,1)
    def test_wrong_target_diagnostic_is_not_true_success(self):
        cells=[dict(crosses_source_seam=False,pieces=[dict(sheet_id='synthetic')]) for _ in range(100)]
        d=r.diagnostic(0,10,1,dict(action='right',cue_action='right'),20,10,cells)
        self.assertFalse(d['accepted_true_hit']);self.assertTrue(d['accepted_given_hit'])
        self.assertEqual(d['true_distance'],1)
    def test_last_step_diagnostic_preserves_opportunity(self):
        cells=[dict(crosses_source_seam=False,pieces=[dict(sheet_id='synthetic')]) for _ in range(100)]
        d=r.diagnostic(0,1,1,dict(action='right',cue_action='right'),1,10,cells)
        self.assertTrue(d['accepted_true_hit']);self.assertEqual(d['remaining'],1)


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (ExecutionTests,ConfirmTests,AreaCompatibilityTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    r.write(r.TEST,dict(successful=result.wasSuccessful(),tests_run=result.testsRun,new_execution_tests=12,
        frozen_native_and_legacy_tests=30,formal_model_forwards=0,formal_navigation_records=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
