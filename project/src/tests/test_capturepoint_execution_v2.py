"""Synthetic math/queue/control tests; no encoder loading, images or development scores."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from agents.capturepoint_target_head_v1 import TargetHead
from eval.capturepoint_execution_scoring_v1 import scores_for,rank_all,summarize
from data.mmgag_views_v1 import normalize_features
from tools.run_capturepoint_execution_v2 import read,QUEUE,execute,role_signature
class ExecutionBoundaries(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(13);self.identity={'U':np.zeros((512,8),np.float32),'V':self.rng.normal(0,.02,(8,512)).astype(np.float32)}
    def test_target_only_identity_and_rejects_bank_metadata(self):
        h=TargetHead(self.identity);g=normalize_features(self.rng.normal(size=512));np.testing.assert_allclose(h(g),g,atol=1e-12)
        for invalid in [np.zeros(512),np.zeros((25,512)),{'goal':4},np.full(512,np.nan)]:
            with self.assertRaises(ValueError):h(invalid)
        with self.assertRaises(TypeError):h(g,goal=0)
    def test_disabled_never_calls_head(self):
        def forbidden(*args):raise AssertionError('Head called on disabled target')
        np.testing.assert_array_equal(scores_for({},'Disabled',None,None,forbidden,None),np.zeros(25))
    def test_score_cannot_use_hidden_goal_and_mean_is_fit_only(self):
        g=normalize_features(self.rng.normal(size=(14,512)));a=normalize_features(self.rng.normal(size=(1134,512)));mean=normalize_features(g[:9].mean(0))
        t=read(QUEUE)['development_tasks'][0];other=dict(t,goal_index_evaluator_only=(t['goal_index_evaluator_only']+1)%25)
        h=TargetHead(self.identity)
        np.testing.assert_array_equal(scores_for(t,'Full',g,a,h,mean),scores_for(other,'Full',g,a,h,mean))
        changed=g.copy();changed[9:]=normalize_features(self.rng.normal(size=(5,512)))
        np.testing.assert_array_equal(scores_for(t,'Mean',g,a,h,mean),scores_for(t,'Mean',changed,a,h,normalize_features(changed[:9].mean(0))))
    def test_synthetic1250_and_no_disabled_duplicate(self):
        q=read(QUEUE);g=self.rng.normal(size=(14,512));a=self.rng.normal(size=(1134,512))
        states={arm:copy.deepcopy(self.identity) for arm in ['Identity','SwissRank8Frozen','MMRank8Final']}
        rows=rank_all(q,g,a,states);self.assertEqual(len(rows),1250);self.assertEqual(sum(r['condition']=='Disabled' for r in rows),125)
        summary=summarize(rows);self.assertEqual(len(summary['contrasts']),15)
        self.assertTrue(all(v['top1']['net']==0 for k,v in summary['contrasts'].items() if k.startswith('MM_vs_')))
    def test_checkpoint_gradient_resume_uses_old_frozen_math(self):
        from train.weak_scene_alignment_v1 import engineering_checks
        report=engineering_checks();self.assertTrue(report['passed']);self.assertTrue(report['resume_parameters_and_optimizer_bitwise_equal'])
    def test_event_keyword_and_optimizer_logging_interfaces(self):
        from unittest.mock import mock_open,patch
        from tools.run_capturepoint_execution_v2 import event
        import json
        with patch.object(Path,'open',mock_open()) as file:
            event(Path('not_written'),'encoder_attempt',kind='ground',indices=[0])
            written=file().write.call_args.args[0]
            self.assertEqual(json.loads(written),{'event':'encoder_attempt','kind':'ground','indices':[0]})
            event(Path('not_written'),'optimizer_saved',step=1,loss=1.)
            self.assertEqual(json.loads(file().write.call_args.args[0])['step'],1)

    def test_full_run_requires_separate_authorization(self):
        with self.assertRaises(AssertionError):execute(None)
    def test_source_role_change_is_detected_without_state_noise(self):
        first={'records':[{'source_id':'x','group':'g','role':'fit_preparation','representative':True}],'training_started':False}
        other=copy.deepcopy(first);other['training_started']=True
        self.assertEqual(role_signature(first),role_signature(other))
        other['records'][0]['role']='confirmation_reserved'
        self.assertNotEqual(role_signature(first),role_signature(other))
if __name__=='__main__':unittest.main()
