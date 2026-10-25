"""CV-009 in-memory boundary tests, NumPy/statistics only."""
import ast
import hashlib
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from eval.weak_transfer_diagnosis_v1 import exact_source_bootstrap,check_rank_row,sample_structure,feature_support,weight_statistics,SOURCES


class DiagnosticBoundaries(unittest.TestCase):
    def test_bootstrap_uses_five_sources(self):
        out=exact_source_bootstrap([.2]*5)
        self.assertEqual(out['resamples'],3125);self.assertEqual(out['source_units'],5)
        np.testing.assert_allclose(out['percentile95'],[.2,.2],atol=1e-14)
        with self.assertRaises(ValueError):exact_source_bootstrap([0]*125)

    def test_saved_rank_must_match_scores_and_goal(self):
        scores=list(range(25));pair='synthetic';order=list(reversed(range(25)))
        r={'scores':scores,'pair_id':pair,'truth_evaluator_only':24,'ranking':order,'true_rank':1,'top1':True,'top5':True}
        check_rank_row(r)
        with self.assertRaises(AssertionError):check_rank_row(dict(r,true_rank=2))
        with self.assertRaises(AssertionError):check_rank_row(dict(r,scores=[float('nan')]*25))

    def test_25_shifted_windows_are_one_query(self):
        bank={'queries':[{'scene':'x','role':'development'}],'cells':[{'scene':'x'}]*81}
        records=[{'scene':'x','positive_pixel_sha256':'same'} for _ in range(25)]
        tasks=[]
        for variant in range(25):
            r,c=divmod(variant,5);cells=[(y-r,x-c) for y in range(5) for x in range(5)]
            tasks.append({'dataset':'weak_dev','group':'x','candidates':cells})
        out=sample_structure(bank,tasks,records)['per_source']['x']
        self.assertEqual((out['unique_ground_queries'],out['all25_common_cells'],out['union_cells']),(1,1,81))

    def test_support_excludes_fit_query_itself(self):
        rng=np.random.default_rng(9);g=rng.normal(size=(10,512));mm=rng.normal(size=(2,512))
        names=[f'f{i}' for i in range(9)]
        bank={'queries':[{'scene':name,'role':'fit'} for name in names]+[{'scene':'held','role':'development'}]}
        out=feature_support(g,mm,bank,names,[{'pair_id':'m0','group_id':'g0'},{'pair_id':'m1','group_id':'g1'}])
        self.assertEqual(out['fit_affine_rank'],8)
        for r in out['per_query']:
            if r['role']=='fit':self.assertNotEqual(r['query'],r['nearest_fit_query'])
        self.assertEqual(out['head_projection_calls'],0)

    def test_weight_spectrum_is_parameter_only(self):
        rng=np.random.default_rng(2);u=rng.normal(size=(512,8));v=rng.normal(size=(8,512))
        log={'updates':500,'unique_fit_queries':9,'supervision_uses':4500,'log':[{'loss_before_update':4.}],'final_fit_loss':.1}
        stats=weight_statistics({'U':np.zeros_like(u),'V':v},{'U':u,'V':v},log)
        self.assertTrue(all(x==0 for x in stats['Identity']['residual_operator_singular_values']))
        self.assertAlmostEqual(stats['Rank8Final']['residual_operator_frobenius'],np.linalg.norm(u@v),places=9)

    def test_static_no_model_image_or_external_runtime_imports(self):
        forbidden={'torch','transformers','safetensors','PIL','requests','httpx','subprocess','importlib','socket'}
        for name in SOURCES:
            tree=ast.parse((ROOT/'project/src'/name).read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):self.assertFalse(any(a.name.split('.')[0] in forbidden for a in node.names))
                if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],forbidden)


if __name__=='__main__':unittest.main()
