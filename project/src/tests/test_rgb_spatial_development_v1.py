"""Pure/small-state preflight. No backbone loads/forwards or optimizer steps."""
import copy
import math
import unittest
from project.src.eval.rgb_spatial_metrics_v1 import measure,summarize,paired_interval
from project.src.eval.rgb_spatial_development_v1 import advance,io_ledger,new_execution_guard
from project.src.train.rgb_spatial_training_v1 import finite_gradient_status
from project.src.data.rgb_spatial_supervision_protocol_v1 import calibrate


def bank():
    return [dict(case_id=str(i),split='fit',query='a',relation=k,raw_sensitivity_relation=k,camera_xy=[0,0] if i<2 else None,
                 raw_camera_xy=[0,0] if i<2 else None,mpp=1) for i,k in enumerate(('inside','outside','unrelated'))]


class Tests(unittest.TestCase):
    def test_joint_requires_acceptance_and5m_localization(self):
        c=bank();o=[dict(case_id=str(i),score=.9 if i==0 else .1,point=[10,0]) for i in range(3)]
        m,_,_=measure(c,o,.5);self.assertEqual(m['macro_joint_exact'],[2,3])
        o[0]['point']=[3,4];m,_,_=measure(c,o,.5);self.assertEqual(m['macro_joint'],1)

    def test_raw_sensitivity_never_rewrites_primary(self):
        c=bank();c[0]['raw_sensitivity_relation']='outside';c[1]['raw_sensitivity_relation']='inside';c[1]['raw_camera_xy']=[10,0]
        original=copy.deepcopy(c);o=[dict(case_id=str(i),score=.9 if i==0 else .1,point=[0,0] if i==0 else [10,0]) for i in range(3)]
        p,_,_=measure(c,o,.5);r,_,_=measure(c,o,.5,raw=True)
        self.assertEqual(p['macro_joint'],1);self.assertEqual(r['macro_joint_exact'],[1,3]);self.assertEqual(c,original)

    def test_planned_denominator_duplicate_or_missing_rejected(self):
        c=bank()
        with self.assertRaises(ValueError):measure(c,[dict(case_id='0',score=.1,point=[0,0])],.5)

    def test_fit_only_calibration_and_exact_tie(self):
        c=bank();o=[dict(case_id=str(i),score=.5,point=[10,0]) for i in range(3)]
        self.assertEqual(calibrate(c,o)['fit_macro_joint_exact'],[2,3]);self.assertGreater(calibrate(c,o)['tau'],.5)
        c[0]['split']='development'
        with self.assertRaises(ValueError):calibrate(c,o)

    def test_full_result_matrix_required(self):
        with self.assertRaises(ValueError):summarize(dict(folds=[]),[],[])

    def test_counters_block_excess_and_negative_before_call(self):
        c={};advance(c,'encoding',72,72)
        with self.assertRaises(ValueError):advance(c,'encoding',1,72)
        with self.assertRaises(ValueError):advance(c,'encoding',-1,72)
        self.assertEqual(c['encoding'],72)

    def test_io_addendum_explicit_and_run_never_blindly_restarted(self):
        self.assertEqual(io_ledger(3)['cumulative_decodes'],9)
        self.assertEqual(io_ledger(3)['cumulative_encodings'],80)
        with self.assertRaises(ValueError):io_ledger(0)
        for f in ('首正式执行绑定.json','执行开始.json'):
            with self.assertRaises(ValueError):new_execution_guard({f})

    def test_finite_stationary_gradients_allowed_nonfinite_rejected(self):
        self.assertEqual(finite_gradient_status(dict(Wg=0,Wa=1,b_empty=0))['zeros'],['Wg','b_empty'])
        with self.assertRaises(ValueError):finite_gradient_status(dict(Wg=float('nan'),Wa=1,b_empty=1))
        with self.assertRaises(ValueError):finite_gradient_status(dict(Wg=1,Wa=1))

    def test_source_paired_interval_deterministic_and_no_pseudoreplication(self):
        self.assertEqual(paired_interval([0,0,0]),[0,0])
        self.assertEqual(paired_interval([-.1,0,.1]),paired_interval([-.1,0,.1]))


if __name__=='__main__':unittest.main()
