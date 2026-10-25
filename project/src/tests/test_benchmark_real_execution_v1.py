"""CPU guards for actual execution freeze/closure and M0 regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC=Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from tools.prepare_benchmark_real_v1 import training_gate,engineering_gate,make_configuration,ROOT,BATCH,save
from tools.verify_benchmark_preparation_v1 import local_closure,fake_assets
from tools.benchmark_assets_v1 import EvaluatorAssets,payload_identity
from eval.benchmark_engineering_v1 import compare_m0
from eval.benchmark_runner_v2 import hashed


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='测试临时_',dir=BATCH/'核验')
        self.folder=Path(self.directory.name).resolve()
        if (BATCH/'核验').resolve() not in self.folder.parents:raise ValueError('test cleanup outside registered parent')

    def tearDown(self):
        if (BATCH/'核验').resolve() not in self.folder.parents:raise ValueError('test cleanup outside registered parent')
        self.directory.cleanup()

    def test_partial_training_and_missing_seed_rejected(self):
        p=self.folder/'audit.json'
        save(p,dict(status='passed',training_completed=True,completed_seeds=[0,1]))
        with self.assertRaisesRegex(ValueError,'three-seed'):training_gate(p)

    def test_complete_training_gate(self):
        p=self.folder/'audit.json';d=dict(status='passed',training_completed=True,completed_seeds=[0,1,2]);save(p,d)
        self.assertEqual(training_gate(p),d)

    def test_engineering_count_and_error_gate(self):
        d=dict(status='passed',m0_episodes=3000,geo_episodes=48,action_mismatches=0,input_mismatches=0,reset_mismatches=0)
        engineering_gate(d)
        for field,bad in [('m0_episodes',2999),('geo_episodes',47),('action_mismatches',1),('status','failed')]:
            copy=dict(d);copy[field]=bad
            with self.assertRaises(ValueError):engineering_gate(copy)

    def test_exclusive_write_cannot_replace_frozen_output(self):
        p=self.folder/'new.json';save(p,{'first':1})
        with self.assertRaises(FileExistsError):save(p,{'second':2})
        self.assertEqual(json.loads(p.read_text('utf-8')),{'first':1})

    def test_local_closure_includes_lazy_loaders_without_forward(self):
        d=local_closure([SRC/'tools/execute_benchmark_real_v1.py'])
        self.assertIn('project/src/agents/benchmark_local_policy_v1.py',d)
        self.assertIn('project/src/train/geoexplorer_training_v1.py',d)
        self.assertIn('project/src/eval/audit_benchmark_runner_v2.py',d)

    def fixture(self):
        old=dict(episode_id='one',success=True,sg=0,steps=1,termination='goal_reached',
            trajectory=[dict(step=0,patch_id=0,action=None,out_of_bounds=False,revisited=False),
                        dict(step=1,patch_id=1,action='right',out_of_bounds=False,revisited=False)],
            decisions=[dict(action='right',explorer_action='right',public_position=[0,0],public_visited=[0],remaining_budget=10,
                explorer_features_sha256='e',current_image_sha256='c',target_image_sha256='t',cue_features_sha256='h',
                cue_action=None,reason='not_adjacent',explorer_logits=[0.,1.,0.,0.],probabilities=[0.,0.,0.,0.,1.])])
        row=dict(episode_id='one',success=True,sg_cells=0,termination='goal_reached',
                 steps=[dict(action='right',next_position=1,out_of_bounds=False,revisited=False)])
        return old,row,deepcopy(old['decisions'])

    def test_exact_m0_regression_and_wrong_action_detected(self):
        a,b,c=self.fixture();t=dict(logit_tolerance=1e-5,probability_tolerance=5e-5)
        self.assertFalse(any(compare_m0(a,b,c,'Full',t).values()))
        b['steps'][0]['action']='down'
        self.assertTrue(compare_m0(a,b,c,'Full',t)['action_errors'])

    def test_disabled_erases_target_head_without_false_input_regression(self):
        a,b,c=self.fixture();c[0]['target_image_sha256']='none';c[0]['cue_features_sha256']='mean';c[0]['probabilities']=[.2]*5
        self.assertFalse(any(compare_m0(a,b,c,'Disabled',dict(logit_tolerance=1e-5,probability_tolerance=5e-5)).values()))

    def test_nonfinite_and_global_input_drift_rejected(self):
        a,b,c=self.fixture();c[0]['explorer_logits'][0]=float('nan');c[0]['explorer_features_sha256']='bad'
        diff=compare_m0(a,b,c,'Full',dict(logit_tolerance=1e-5,probability_tolerance=5e-5))
        self.assertTrue(diff['numeric_errors']);self.assertTrue(diff['input_errors'])

    def test_external_configuration_is_deterministic_and_covers_all_payloads(self):
        plan,cells,means=fake_assets()
        evidence=dict(queue_sha256=hashed(plan.record()),assets={a:{str(i):payload_identity(p) for i,p in v.items()} for a,v in cells.items()})
        data=EvaluatorAssets(plan,cells,means,evidence)
        regs=[dict(method=m,seed=seed,artifacts={'project/local_policy_default.json':'a'*64}) for m in ('M0','GeoExplorer') for seed in (0,1,2)]
        args=(data,regs,{'project/src/eval/benchmark_runner_v2.py':'b'*64},dict(path='README.md',sha256='c'*64))
        one=make_configuration(*args);two=make_configuration(*args)
        self.assertEqual(hashed(one),hashed(two));self.assertEqual(len(one['registrations']),6)
        self.assertEqual(one['assets'],evidence['assets'])


if __name__=='__main__':unittest.main()
