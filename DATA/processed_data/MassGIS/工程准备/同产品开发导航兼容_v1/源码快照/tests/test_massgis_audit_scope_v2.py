"""Rule/agent scope and OOB accounting must remain distinct."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.audit_massgis_navigation_compat_v2 import enforce_motion_scope,move
from eval import massgis_navigation_compat_v1 as m

class Scope(unittest.TestCase):
    def test_unchanged_neural_requirement(self):
        enforce_motion_scope(False,0)
        with self.assertRaises(AssertionError):enforce_motion_scope(False,1)
    def test_rule_boundary_action_is_charged_not_relocated(self):
        self.assertEqual(move(0,'up',10),(0,True));enforce_motion_scope(True,1)
        env=m.NativeAreaEnv(m.root(10));s=m.spec(10)
        ep=m.NativeEpisode('rule_boundary','dev','img_6100',0,1,1,s.budget,10,s.name,'MassGIS2005/DATA005_known_full4')
        env.reset(ep);obs,done,info=env.step('up')
        self.assertEqual((obs.position,obs.remaining_budget),((0,0),19));self.assertTrue(info.out_of_bounds);self.assertFalse(done)
        env.step('right');r=env.evaluator_result()
        self.assertEqual((r['steps'],r['out_of_bounds'],r['valid_travel_m'],r['sg_m']),(2,1,300,0))
    def test_rule_rows_remain_in_planned_denominator(self):
        rows=m.rows(m.OUT/'规则基线/g10_FixedRegion.jsonl')
        self.assertEqual(len(rows),75);self.assertEqual(sum(r['out_of_bounds'] for r in rows),3)
    def test_invalid_counters_rejected(self):
        for count in (-1,True,1.0):
            with self.assertRaises(ValueError):enforce_motion_scope(True,count)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Scope))
    m.write(m.QA/'审计适配_v2测试.json',dict(successful=result.wasSuccessful(),tests_run=result.testsRun,neural_oob_requirement_unchanged=True,rule_denominator_unchanged=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
