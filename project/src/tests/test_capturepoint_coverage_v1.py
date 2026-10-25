"""CV-018 batch budget, source isolation, prefix preservation and gate edges."""
import copy
import unittest
from collections import Counter
from data.capturepoint_coverage_protocol_v1 import CANDIDATE,sample_indices,build_queue,coverage_gate


class CoverageBoundaries(unittest.TestCase):
    def test_original9_sampler_is_exact_fixed_batch(self):
        for t in range(500):self.assertEqual(sample_indices(9,t),list(range(9)))

    def test_12_sampler_keeps4500_uses_and375_each(self):
        batches=[sample_indices(12,t) for t in range(500)]
        self.assertTrue(all(len(set(b))==9 for b in batches))
        self.assertEqual(Counter(i for b in batches for i in b),Counter({i:375 for i in range(12)}))

    def test_sampler_boundaries(self):
        for n,t in [(10,0),(12,-1),(12,500),(True,0),(9,True)]:
            with self.assertRaises(ValueError):sample_indices(n,t)

    def queue(self):
        old={'ground_encoder_inputs':[{'path':f'g{i}'} for i in range(14)],'aerial_encoder_inputs':[{'path':f'a{i}'} for i in range(1134)],
            'training_queries':[{'source_id_evaluator_only':f's{i}','group_evaluator_only':f'fit{i}'} for i in range(9)],
            'development_tasks':[{'pair_id':f'p{i}','ground_feature_index':9+i//25,'goal_index_evaluator_only':i%25} for i in range(125)]}
        extras=[{'source_id':f'x{i}','group':f'fit{i}','role':'fit_preparation','ground':{'path':f'newg{i}'},
            'tiles':[{'path':f'new{i}/sat_{j//9+1}_{j%9+1}.jpg','sha256':f'{i}-{j}'} for j in range(81)]} for i in range(3)]
        return old,extras

    def test_old_feature_prefix_and_development_unchanged(self):
        old,extra=self.queue();before=copy.deepcopy(old);q=build_queue(old,extra)
        self.assertEqual(old,before);self.assertEqual(q['development_tasks'],old['development_tasks'])
        self.assertEqual(q['ground_encoder_inputs'][:14],old['ground_encoder_inputs']);self.assertEqual(q['aerial_encoder_inputs'][:1134],old['aerial_encoder_inputs'])
        self.assertEqual([t['ground_feature_index'] for t in q['training_queries'][9:]],[14,15,16]);self.assertEqual(q['mean_ground_feature_indices'],list(range(9)))

    def test_never_expand_from_dev_or_confirmation(self):
        for role,group in [('development','fit0'),('confirmation_reserved','fit0'),('fit_preparation','dev0')]:
            old,extras=self.queue();extras[0].update(role=role,group=group)
            with self.assertRaises(ValueError):build_queue(old,extras)

    def groups(self):
        return {f'g{i}':{'planned':25,CANDIDATE:{'Full':v,'Mean':0,'Wrong':0},'Identity':{'Full':0},'SwissRank8Frozen':{'Full':0},'MMRank8Final':{'Full':0},'Disabled':0} for i,v in enumerate([7,7,6,6,6])}

    def test_full32_integer_boundary_and_required_completion(self):
        s=self.groups();self.assertTrue(coverage_gate(s,complete=True,engineering_and_audit=True)['passed'])
        self.assertFalse(coverage_gate(s)['passed']);s['g0'][CANDIDATE]['Full']-=1
        self.assertFalse(coverage_gate(s,complete=True,engineering_and_audit=True)['gates']['Full_top1_at_least32'])

    def test_target_control_and_mm9_effect_boundaries(self):
        s=self.groups()
        for i,v in enumerate([4,3,3,3,3]):s[f'g{i}'][CANDIDATE]['Mean']=v
        self.assertTrue(coverage_gate(s,complete=True,engineering_and_audit=True)['passed'])
        s['g0'][CANDIDATE]['Mean']+=1;self.assertFalse(coverage_gate(s,complete=True,engineering_and_audit=True)['gates']['Full_minus_Mean_at_least16'])
        s=self.groups()
        for i,v in enumerate([6,6,5,5,5]):s[f'g{i}']['MMRank8Final']['Full']=v
        self.assertTrue(coverage_gate(s,complete=True,engineering_and_audit=True)['passed'])
        s['g0']['MMRank8Final']['Full']+=1;self.assertFalse(coverage_gate(s,complete=True,engineering_and_audit=True)['gates']['Full_minus_MM9_at_least5'])

    def test_denominator_and_count_type_guard(self):
        s=self.groups();s['g0']['planned']=24
        with self.assertRaises(ValueError):coverage_gate(s)
        s=self.groups();s['g0'][CANDIDATE]['Full']=True
        with self.assertRaises(ValueError):coverage_gate(s)


if __name__=='__main__':unittest.main()
