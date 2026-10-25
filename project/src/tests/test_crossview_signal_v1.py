"""Synthetic tests only: no project feature/weight/raw-image reads."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.crossview_signal_v1 import unit,order,screen,ARMS,SOURCES,scene_scores,regroup

def synthetic_windows():
    aerial=[{'path':'unused'} for _ in range(1134)];tasks=[];rows=[]
    for g,s in enumerate(SOURCES):
        start=729+81*g
        for i in range(81):aerial[start+i]={'path':f'synthetic_{g}/sat_{i//9+1}_{i%9+1}.jpg'}
        for r in range(5):
            for c in range(5):
                pid=f'synthetic_{g}_{r}_{c}';indices=[start+(r+y)*9+c+x for y in range(5) for x in range(5)]
                truth=(4-r)*5+4-c;group='group_'+str(g)
                tasks.append({'pair_id':pid,'source_id_evaluator_only':s,'group_evaluator_only':group,
                    'ground_feature_index':9+g,'wrong_ground_feature_index':9+(g+1)%5,
                    'aerial_feature_indices':indices,'goal_index_evaluator_only':truth})
                for arm,cond in [(a,k) for a in ARMS for k in ['Full','Mean','Wrong']]+[('SharedDisabled','Disabled')]:
                    scores=[float(i-start) for i in indices] if cond!='Disabled' else [0.]*25
                    rows.append({'pair_id':pid,'group':group,'arm':arm,'condition':cond,'scores':scores,'truth_evaluator_only':truth})
    return {'development_tasks':tasks,'aerial_encoder_inputs':aerial},rows

class Identity:
    def __init__(self,_):pass
    def __call__(self,q):return q.copy()

class SignalTests(unittest.TestCase):
    def test_full_overlap_reassembly_and_proxy_mapping(self):
        q,rows=synthetic_windows();v=regroup(q,rows)
        self.assertEqual(v['stored_score_slots'],4860);self.assertEqual(len(v['cases']),60)
        for case in v['cases']:
            self.assertEqual(case['true_rank'],41);self.assertEqual(case['best_cell_distance'],8)
    def test_missing_duplicate_or_conflicting_window_rejected(self):
        q,rows=synthetic_windows()
        for broken in [rows[:-1],rows+[copy.deepcopy(rows[0])]]:
            with self.assertRaises(ValueError):regroup(q,broken)
        broken=copy.deepcopy(rows);broken[13]['scores'][0]+=1
        with self.assertRaises(ValueError):regroup(q,broken)
    def test_wrong_mapping_and_true_cell_drift_rejected(self):
        q,rows=synthetic_windows();q['development_tasks'][0]['wrong_ground_feature_index']=9
        with self.assertRaises(ValueError):regroup(q,rows)
        q,rows=synthetic_windows();q['development_tasks'][0]['goal_index_evaluator_only']=0
        with self.assertRaises(ValueError):regroup(q,rows)
    def test_unit_rejects_invalid(self):
        for v in [np.zeros(512),np.full(512,np.nan)]:
            with self.assertRaises(ValueError):unit(v)
    def test_tie_is_deterministic_and_full_permutation(self):
        for n in [5,81]:
            self.assertEqual(order([0]*n,'fixed'),order([0]*n,'fixed'))
            self.assertEqual(sorted(order([0]*n,'fixed')),list(range(n)))
    def test_scene_uses_fixed_mean_and_wrong_row_reuse(self):
        rng=np.random.default_rng(4);g=rng.normal(size=(14,512));a=rng.normal(size=(1134,512))
        sources=[{'source':s,'indices':list(range(i*81,(i+1)*81))} for i,s in enumerate(SOURCES)]
        r=scene_scores(g,a,{x:{} for x in ARMS},sources,Identity)
        self.assertEqual(r['frozen_head_projections'],24);self.assertEqual(r['new_scene_scores'],120)
        for arm in ARMS:
            p=r['arms'][arm]['per_source']
            for i in range(5):self.assertEqual(p[i]['conditions']['Wrong']['scores'],p[(i+1)%5]['conditions']['Full']['scores'])
            self.assertTrue(all(p[0]['conditions']['Mean']['scores']==x['conditions']['Mean']['scores'] for x in p))
    def test_extra_dev_or_metadata_fields_do_not_change_scores(self):
        rng=np.random.default_rng(2);g=rng.normal(size=(14,512));a=rng.normal(size=(1134,512))
        sources=[{'source':s,'indices':list(range(i*81,(i+1)*81))} for i,s in enumerate(SOURCES)]
        baseline=scene_scores(g,a,{x:{} for x in ARMS},sources,Identity)
        changed=copy.deepcopy(sources)
        for x in changed:x['hidden_goal']=999;x['distance']=123
        self.assertEqual(baseline,scene_scores(g,a,{x:{} for x in ARMS},changed,Identity))
    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):scene_scores(np.ones((17,512)),np.ones((1134,512)),{},[],Identity)
    def test_screen_requires_same_three_sources_above_both(self):
        cells=[];scene={'arms':{}}
        for arm in ARMS:
            per=[]
            for i,s in enumerate(SOURCES):
                for c in ['Full','Mean','Wrong']:
                    rank=3 if c=='Full' else 4 if (i<3 if c=='Mean' else i>=2) else 2
                    cells.append({'source':s,'arm':arm,'condition':c,'true_rank':rank,'top1':False,'top5':True,'best_within_one_cell_diagnostic_only':False})
                per.append({'conditions':{c:{'top1':i<3 if c=='Full' else i==4} for c in ['Full','Mean','Wrong']}})
            scene['arms'][arm]={'per_source':per}
        r=screen({'cases':cells},scene)
        self.assertFalse(r['direction_signal_present'])
        for v in r['arms'].values():self.assertEqual(len(v['cell_rank_better_than_both']),1)

if __name__=='__main__':unittest.main()
