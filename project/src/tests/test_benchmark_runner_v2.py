"""CPU fake-policy contract tests; no real loader calls or model forwards."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from agents.benchmark_contract_v1 import FeatureBundle, PolicyRequest, ScriptedPolicy
from agents.benchmark_local_policy_v1 import GlobalOnlyPolicy, M0PublicPolicy, SCHEMA
from env.environment import Observation, StepInfo
from eval.benchmark_runner_v1 import Payload, Task, FitMean
from eval.benchmark_runner_v2 import RealPlan, RegisteredPolicy, evaluate, run_episode, hashed, metrics

ROOT = SRC.parents[1]
TEMP = ROOT / '项目导航/共同真实评测准备_v1/核验'


def feature(value=1):
    return FeatureBundle(tuple((name, (float(value),) * width) for name, width in SCHEMA))


def assets():
    return {'area': {i: Payload(f'pixel-{i}'.encode(), feature(i + 1)) for i in range(25)}}


class FakeNavigator:
    def __init__(self):
        import numpy as np
        self.hm = np.zeros(512, np.float32); self.lm = np.zeros((4, 768), np.float32)
        self.pm = np.zeros((4, 3, 64, 3), np.float32); self.reset()
    def reset(self): self.profiles = {}; self.calls = []
    def act(self, observation, current, current_local, target, target_local):
        self.calls.append((observation, current.copy(), target.copy(), target_local.copy(), self.condition))
        return {'action': 'right'}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP)
        self.folder = Path(self.directory.name); self.bank = assets()
        self.tasks = (Task('a', 'area', 0, 1, 24, 'C1'), Task('b', 'area', 5, 6, 24, 'C1'))
        self.plan = RealPlan(self.tasks, 'synthetic_interface')
        weight = self.folder / 'fake-weight'; weight.write_bytes(b'fake-interface-artifact')
        source = Path(__file__)
        self.artifacts = ((weight, sha256(weight.read_bytes()).hexdigest()),)
        self.sources = ((source, sha256(source.read_bytes()).hexdigest()),)
        self.mean = FitMean(Payload(b'fit-mean-placeholder', feature(17)), 'a' * 64)

    def tearDown(self): self.directory.cleanup()

    def reg(self, method, factory=None, seed=0):
        return RegisteredPolicy(method, seed, factory or (lambda: ScriptedPolicy(('right',))),
              self.artifacts, self.sources, self.mean, 'a' * 64)

    def test_same_queue_four_controls_and_reset_factory_once(self):
        made = []; policies = []
        class Counting(ScriptedPolicy):
            def reset(self):
                super().reset(); self.resets = getattr(self, 'resets', 0) + 1
        def factory():
            made.append(True); p = Counting(('right',)); policies.append(p); return p
        summary = evaluate(self.plan, self.bank, [self.reg('M0', factory), self.reg('GeoExplorer', factory)],
                           hashed(self.plan.record()), self.folder / 'out')
        self.assertEqual(len(made), 2); self.assertTrue(all(p.resets == 9 for p in policies))
        self.assertEqual(len(summary['arms']), 8); self.assertFalse(summary['scientific_sr'])
        self.assertTrue(all(a['metrics']['planned'] == 2 and a['metrics']['successes'] == 2 for a in summary['arms']))
        self.assertEqual(summary['source_file_groups'], 1)

    def test_wrong_target_does_not_change_evaluation_truth(self):
        t = self.tasks[0]
        row = run_episode(t, self.plan, self.bank['area'], self.bank['area'][24], ScriptedPolicy(('right',)))
        self.assertTrue(row['success']); self.assertEqual(row['final_position'], 1)

    def test_last_step_arrival_precedes_budget_failure(self):
        task = Task('last', 'area', 0, 6, 24, 'C2')
        policy = ScriptedPolicy(('down', 'up') * 4 + ('down', 'right'))
        row = run_episode(task, self.plan, self.bank['area'], self.bank['area'][6], policy)
        self.assertTrue(row['success']); self.assertEqual(len(row['steps']), 10); self.assertEqual(row['sg_cells'], 0)

    def test_out_of_bounds_spends_budget_and_no_movement(self):
        row = run_episode(self.tasks[0], self.plan, self.bank['area'], None, ScriptedPolicy(('up',)))
        self.assertFalse(row['success']); self.assertEqual(len(row['steps']), 10)
        self.assertEqual(row['movement_cells'], 0); self.assertTrue(all(s['out_of_bounds'] for s in row['steps']))
        self.assertEqual(row['steps'][-1]['public']['remaining_budget'], 1)

    def test_disabled_erases_target_and_only_public_state_received(self):
        seen = []
        class Spy:
            def reset(self): pass
            def decide(self, request):
                seen.append(request); return 'right'
        row = run_episode(self.tasks[0], self.plan, self.bank['area'], None, Spy())
        request = seen[0]; self.assertTrue(row['success'])
        self.assertIsNone(request.target_features); self.assertEqual(request.observation.target_image, b'')
        for name in ('goal', 'area', 'stratum', 'assets', 'episode_id'):
            self.assertFalse(hasattr(request, name)); self.assertFalse(hasattr(request.observation, name))
        self.assertNotIn('values', row['steps'][0]['public']['current_features'])

    def test_partial_training_cannot_enter_real_queue(self):
        tasks = tuple(Task(f'{area}-{distance}-{j}', f'area{area}', 0, (distance - 4) * 5 + 4,
                     1, f'C{distance}') for area in range(10) for distance in range(4, 9) for j in range(5))
        plan = RealPlan(tasks, 'known_retrospective_Masa_test250'); plan.validate()
        gate = self.folder / 'partial.json'; gate.write_text(json.dumps(dict(status='passed', training_completed=False,
                                                              completed_seeds=[0, 1])))
        with self.assertRaisesRegex(ValueError, 'all three'):
            evaluate(plan, {}, [], hashed(plan.record()), self.folder / 'real', (gate, sha256(gate.read_bytes()).hexdigest()))
        self.assertFalse((self.folder / 'real').exists())

    def test_queue_and_source_drift_rejected_before_output(self):
        regs = [self.reg('M0'), self.reg('GeoExplorer')]
        with self.assertRaisesRegex(ValueError, 'queue drift'):
            evaluate(self.plan, self.bank, regs, 'f' * 64, self.folder / 'out')
        bad = replace(regs[0], sources=((Path(__file__), 'f' * 64),))
        with self.assertRaisesRegex(ValueError, 'source drift'):
            evaluate(self.plan, self.bank, [bad, regs[1]], hashed(self.plan.record()), self.folder / 'out')

    def test_model_factory_error_preserves_planned_denominator(self):
        def rejected(): raise ValueError('sensitive exception message should not be written')
        summary = evaluate(self.plan, self.bank, [self.reg('M0', rejected), self.reg('GeoExplorer')],
                           hashed(self.plan.record()), self.folder / 'out')
        failed = [a for a in summary['arms'] if a['method'] == 'M0']
        self.assertTrue(all(a['metrics']['sr'] == 0 and a['metrics']['policy_errors'] == 2 for a in failed))
        self.assertTrue(all(a['metrics']['sg_cells_all_planned'] is None for a in failed))
        self.assertNotIn('sensitive exception', (self.folder / 'out/M0_s0_Full.jsonl').read_text())

    def test_methods_require_matching_seed_sets(self):
        with self.assertRaisesRegex(ValueError, 'seed cohorts'):
            evaluate(self.plan, self.bank, [self.reg('M0'), self.reg('GeoExplorer', seed=1)],
                     hashed(self.plan.record()), self.folder / 'out')

    def test_no_unverified_meters(self):
        with self.assertRaisesRegex(ValueError, 'physical scale'): replace(self.plan, cell_size_m=300).validate()
        with self.assertRaises(ValueError): metrics([], 2)

    def test_global_projection_excludes_local_profile_channels(self):
        seen = []
        class Inner:
            def reset(self): pass
            def decide(self, request): seen.append(request); return 'right'
        wrapper = GlobalOnlyPolicy(Inner())
        obs = Observation(b'current', b'target', (0, 0), 5, 10, (0,))
        wrapper.decide(PolicyRequest(obs, feature(), feature(2), True, None))
        self.assertEqual(seen[0].current_features.schema, (('global', 512),))
        self.assertEqual(seen[0].observation, obs)

    def test_m0_disabled_uses_fit_constants_and_resets(self):
        nav = FakeNavigator(); policy = M0PublicPolicy(nav)
        obs = Observation(b'current', b'', (0, 0), 5, 10, (0,))
        self.assertEqual(policy.decide(PolicyRequest(obs, feature(1), None, False, None)), 'right')
        self.assertEqual(nav.calls[-1][-1], 'Baseline'); self.assertFalse(nav.calls[-1][2].any())
        with self.assertRaisesRegex(ValueError, 'reset required'):
            policy.decide(PolicyRequest(obs, feature(1), None, False, None))
        policy.reset(); self.assertFalse(nav.calls)

    def test_m0_mean_features_and_target_constancy(self):
        nav = FakeNavigator(); policy = M0PublicPolicy(nav)
        obs = Observation(b'current', b'mean', (0, 0), 5, 10, (0,))
        policy.decide(PolicyRequest(obs, feature(1), feature(17), True, None))
        self.assertTrue((nav.calls[-1][2] == 17).all()); self.assertEqual(nav.calls[-1][-1], 'CueFull')
        with self.assertRaisesRegex(ValueError, 'target changed'):
            policy.decide(PolicyRequest(obs, feature(1), feature(18), True, StepInfo(False, False, 1)))


if __name__ == '__main__': unittest.main()
