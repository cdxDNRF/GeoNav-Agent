import copy
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.src.data.sample4geo_protocol_v1 import ROOT,DEV,build_manifest
from project.src.eval.sample4geo_controls_v1 import unit,rank_case,paired_delta,evaluate
from project.src.tools.run_sample4geo_controls_v1 import FeatureCache,feature_digest,check_resume


def synthetic_evaluation():
    inputs=[];sources=[];vectors={};old={'sources':[],'cases':[]}
    def add(v):
        idx=len(inputs);inputs.append({'id':idx});vectors[idx]=v;return idx
    for i in range(9):
        v=np.eye(1,1024,10+i,dtype=np.float32)[0]
        sources.append({'source_id':f'fit{i}','group':f'fit{i}','ground_id':add(v),'role':'fit'})
    coords=[[r,c] for r in range(9) for c in range(9)]
    for i in range(5):
        query=np.eye(1,1024,i,dtype=np.float32)[0]
        other=np.eye(1,1024,100+i,dtype=np.float32)[0]
        gid=add(query);ids=[add(query if j==40 else other) for j in range(81)]
        s={'source_id':DEV[i],'group':DEV[i],'ground_id':gid,'tile_ids':ids,'truth_index_evaluator_only':40,'tile_coords':coords,'role':'development'}
        sources.append(s);old['sources'].append({'source':DEV[i],'truth_index':40,'coords':coords})
        for c in ('Full','Mean','Wrong'):
            old['cases'].append({'source':DEV[i],'arm':'Identity','condition':c,'scores':[0.0]*81})
    return {'inputs':inputs,'sources':sources},vectors,old


class ControlsTests(unittest.TestCase):
    def test_cosine_and_tie_without_truth_priority(self):
        np.testing.assert_allclose(unit([3,4]),[.6,.8],atol=1e-15)
        r=rank_case([1,1,0],1)
        self.assertEqual(r['rank'],2);self.assertEqual(r['rank_best_tie'],1);self.assertEqual(r['rank_worst_tie'],2)
        self.assertEqual(rank_case([0]*81,40)['rank'],41)

    def test_invalid_features_and_scores(self):
        for x in ([0,0],[float('nan'),1]):
            with self.assertRaises(ValueError):unit(x)
        with self.assertRaises(ValueError):rank_case([0],1)
        with self.assertRaises(ValueError):feature_digest(np.ones(1024,dtype=np.float64))

    def test_complete_controls_and_gates(self):
        p,v,o=synthetic_evaluation();r=evaluate(p,v,o)
        self.assertTrue(r['summary']['screen']['all_effect_gates_passed'])
        self.assertEqual(r['summary']['screen']['grid']['Full']['top1_hits'],5)
        self.assertEqual(r['summary']['screen']['grid']['Mean']['top1_hits'],0)
        self.assertEqual(r['summary']['screen']['scene']['Wrong']['top1_hits'],0)
        self.assertEqual(len(r['grid_cases']),20)
        self.assertEqual(r['summary']['planned_sources'],5)

    def test_mean_uses_fit_only_and_wrong_is_cyclic(self):
        p,v,o=synthetic_evaluation();r=evaluate(p,v,o)
        expected=unit(np.mean([unit(v[s['ground_id']]) for s in p['sources'][:9]],axis=0))
        np.testing.assert_allclose(r['summary']['fit_mean'],expected)
        for i,sid in enumerate(DEV):
            case=next(x for x in r['grid_cases'] if x['source']==sid and x['condition']=='Wrong')
            self.assertEqual(case['wrong_source_evaluator_only'],DEV[(i+1)%5])
            self.assertNotEqual(case['wrong_source_evaluator_only'],sid)

    def test_missing_input_and_old_label_mismatch_rejected(self):
        p,v,o=synthetic_evaluation();del v[0]
        with self.assertRaises(ValueError):evaluate(p,v,o)
        p,v,o=synthetic_evaluation();o['sources'][0]['truth_index']=0
        with self.assertRaises(ValueError):evaluate(p,v,o)

    def test_recovery_damage_and_group_bootstrap(self):
        rows=[]
        for i in range(5):
            for c,r in [('Full',1 if i==0 else 2),('Mean',2 if i==0 else 1)]:
                x=rank_case([0,1] if r==2 else [1,0],0)
                rows.append(dict(x,source=f's{i}',group=f'g{i}',condition=c))
        r=paired_delta(rows,'Mean','grid')
        self.assertEqual(r['recovered_top1'],1);self.assertEqual(r['damaged_top1'],4)
        self.assertEqual(r['independent_units'],5);self.assertEqual(r['bootstrap_draws'],3125)
        self.assertAlmostEqual(r['source_bootstrap']['top1_delta']['mean'],-.6)

    def test_transactional_cache_recovery_and_ambiguous_stop(self):
        base=ROOT/'DATA/processed_data/MM-GAG/单表征目标对照开发_v1'
        with tempfile.TemporaryDirectory(prefix='test_tmp-',dir=base) as tmp:
            folder=Path(tmp).resolve();self.assertTrue(folder.is_relative_to(base.resolve()))
            path=folder/'features.sqlite';c=FeatureCache(path)
            v=np.ones(1024,dtype=np.float32);digest=c.put(1,v)
            with self.assertRaises(sqlite3.IntegrityError):c.put(1,v)
            np.testing.assert_array_equal(c.get(1),v)
            self.assertEqual(digest,hashlib.sha256(v.tobytes()).hexdigest())
            self.assertEqual(check_resume([{'id':0},{'id':1}],c,[{'phase':'before_forward','input_id':1}],0),[1])
            self.assertEqual(check_resume([{'id':0},{'id':1}],c,[{'phase':'before_forward','input_id':1},{'phase':'forward_complete','input_id':1}],0),[])
            with self.assertRaises(ValueError):check_resume([{'id':0},{'id':1},{'id':2}],c,[{'phase':'before_forward','input_id':2}],0)
            c.close();c=FeatureCache(path,readonly=True);np.testing.assert_array_equal(c.get(1),v);c.close()

    def test_manifest_rejects_confirmation_and_candidate_swap(self):
        # Metadata-only synthetic inventory with full legacy index correspondence.
        pre={'sources':[]};ledger={'records':[]};old={'sources':[]};queue={'aerial_encoder_inputs':[],'ground_encoder_inputs':[]}
        for i in range(14):
            fit=i<9;sid=f'fit{i}' if fit else DEV[i-9]
            raw={'path':f'g{i}','bytes':1,'sha256':f'g{i}'}
            tiles=[{'path':f'{sid}/sat_{r}_{c}.jpg','bytes':1,'sha256':f'{sid}-{r}-{c}'} for r in range(1,10) for c in range(1,10)]
            pre['sources'].append({'source_id':sid,'group':sid,'role':'fit' if fit else 'development','ground':raw,'tiles':tiles})
            ledger['records'].append({'source_id':sid,'group':sid,'role':'fit_preparation' if fit else 'development','representative':True})
            queue['ground_encoder_inputs'].append(raw);start=len(queue['aerial_encoder_inputs']);queue['aerial_encoder_inputs']+=tiles
            if not fit:old['sources'].append({'source':sid,'ground_index':i,'indices':list(range(start,start+81)),
                                            'coords':[[r,c] for r in range(9) for c in range(9)],'truth_index':40})
        ins,sources=build_manifest(pre,ledger,old,queue);self.assertEqual(len(ins),419)
        bad=copy.deepcopy(ledger);bad['records'][-1]['role']='confirmation_reserved'
        with self.assertRaises(ValueError):build_manifest(pre,bad,old,queue)
        bad=copy.deepcopy(queue);bad['aerial_encoder_inputs'][-1]=bad['aerial_encoder_inputs'][-2]
        with self.assertRaises(ValueError):build_manifest(pre,ledger,old,bad)


if __name__=='__main__':unittest.main()
