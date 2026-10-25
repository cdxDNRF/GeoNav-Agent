"""CV-019 leakage poison, immutable prefix, real gradient/resume and score permission tests."""
import copy
import json
import unittest
from unittest.mock import mock_open,patch
import numpy as np
from train.capturepoint_coverage_runtime_v1 import reuse_plan,assemble,fit_tensors,synthetic_checks
from eval.capturepoint_coverage_scoring_v1 import rank_candidate,summarize
from data.capturepoint_coverage_protocol_v1 import CANDIDATE


class CoverageExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def queue(self):
        q={'ground_encoder_inputs':[{'sha256':f'g{i}'} for i in range(17)],
            'aerial_encoder_inputs':[{'sha256':f'a{i}'} for i in range(1134)],'training_queries':[],
            'development_tasks':[{'pair_id':f'p{i}','group_evaluator_only':f'dev{i//25}','ground_feature_index':9+i//25,
                'wrong_ground_feature_index':9+(i//25+1)%5,'aerial_feature_indices':list(range((9+i//25)*81,(9+i//25)*81+25)),
                'goal_index_evaluator_only':i%25} for i in range(125)]}
        q['aerial_encoder_inputs'] += [{'sha256':f'a{i}' if i<35 else f'new{i}'} for i in range(243)]
        for i in range(12):
            gi=i if i<9 else i+5;start=i*81 if i<9 else 1134+(i-9)*81
            q['training_queries'].append({'ground_feature_index':gi,'group_evaluator_only':f'fit{i if i<9 else i-9}',
                'aerial_feature_indices':list(range(start,start+81)),'positive_candidate_index_evaluator_only':40})
        return q

    def arrays(self):
        r=np.random.default_rng(19);return r.normal(size=(17,512)).astype(np.float32),r.normal(size=(1377,512)).astype(np.float32)

    def rows(self):
        q=self.queue();g,a=self.arrays();weights={'U':np.zeros((512,8)),'V':np.zeros((8,512))}
        new=rank_candidate(q,g,a,weights)
        old=[]
        for arm in ('Identity','SwissRank8Frozen','MMRank8Final'):
            for r in new:x=copy.deepcopy(r);x['arm']=arm;old.append(x)
        for r in new[::3]:x=copy.deepcopy(r);x['arm']='SharedDisabled';x['condition']='Disabled';old.append(x)
        return old,new

    def test_sha_cache_reuse_and_old_prefix(self):
        q=self.queue();p=reuse_plan(q);self.assertEqual((p['old_aerial_rows_reused'],len(p['new_aerial_items']),p['new_unique_encoder_inputs'],p['new_encoder_calls']),(35,208,211,27))
        g,a=self.arrays();new=np.full((208,512),3,np.float32);gg,aa=assemble(g[:14],a[:1134],g[14:],new,p)
        np.testing.assert_array_equal(gg[:14],g[:14]);np.testing.assert_array_equal(aa[:1134],a[:1134]);np.testing.assert_array_equal(aa[1134:1169],a[:35])

    def test_fit_gather_ignores_poisoned_development(self):
        q=self.queue();g,a=self.arrays();g[9:14]=np.nan;a[729:1134]=np.nan
        gg,aa,pp,b=fit_tensors(g,a,q);self.assertTrue(np.isfinite(gg).all() and np.isfinite(aa).all());self.assertEqual(gg.shape,(12,512));self.assertEqual(b['ground_indices'],list(range(9))+[14,15,16])

    def test_development_index_cannot_be_training_query(self):
        q=self.queue();g,a=self.arrays();q['training_queries'][9]['ground_feature_index']=9
        with self.assertRaises(ValueError):fit_tensors(g,a,q)

    def test_real_sampled_gradient_and_resume(self):
        r=synthetic_checks();self.assertTrue(r['passed'] and r['sampled_resume_bitwise_equal'] and r['optimizer_only8192_head']);self.assertEqual(r['real_training_updates'],0)

    def test_hidden_answer_does_not_change_scores(self):
        q=self.queue();g,a=self.arrays();w={'U':np.zeros((512,8)),'V':np.zeros((8,512))};before=rank_candidate(q,g,a,w)
        q['development_tasks'][0]['goal_index_evaluator_only']=24;after=rank_candidate(q,g,a,w)
        for x,y in zip(before,after):self.assertEqual(x['scores'],y['scores']);self.assertEqual(x['ranking'],y['ranking'])

    def test_planned1625_and_six_source_contrasts(self):
        old,new=self.rows();s=summarize(old,new);self.assertEqual(s['planned_records'],1625);self.assertEqual(len(s['contrasts']),6);self.assertEqual(s['source_units'],5)
        self.assertEqual(s['counts'][CANDIDATE]['Full'],s['counts']['MMRank8Final']['Full'])

    def test_missing_or_duplicate_records_rejected(self):
        old,new=self.rows()
        with self.assertRaises(ValueError):summarize(old,new[:-1])
        new[-1]=new[0]
        with self.assertRaises(ValueError):summarize(old,new)

    def test_event_kind_keyword_is_not_a_parameter_collision(self):
        from tools.run_capturepoint_coverage_v1 import event
        fake=mock_open()
        with patch('pathlib.Path.open',fake):event('encoder_saved',kind='ground',indices=[0])
        value=json.loads(fake().write.call_args[0][0]);self.assertEqual(value['event'],'encoder_saved');self.assertEqual(value['kind'],'ground')


if __name__=='__main__':unittest.main()
