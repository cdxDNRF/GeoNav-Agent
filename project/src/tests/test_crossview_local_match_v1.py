"""CV-003 hand-computed algebra, public input boundary and regression-stop checks."""
import hashlib,inspect,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from agents.crossview_local_match_v1 import symmetric_maxsim,stored_unit_tokens,unit_tokens,project_last_layer,check_global_replay,rank_cells
class MatchingTests(unittest.TestCase):
    def test_hand_computed_two_direction_score(self):
        a=np.eye(2);b=np.array([[1,0],[1,0]])
        self.assertAlmostEqual(symmetric_maxsim(a,b),.75)
        self.assertAlmostEqual(symmetric_maxsim(b,a),.75)
    def test_single_token_reduces_to_cosine(self):
        self.assertAlmostEqual(symmetric_maxsim([[3,4]],[[1,0]]),.6)
    def test_permutation_invariance_and_self_match(self):
        rng=np.random.default_rng(41);a=rng.normal(size=(49,512));b=rng.normal(size=(49,512))
        self.assertAlmostEqual(symmetric_maxsim(a,b),symmetric_maxsim(a[::-1],b[::-1]),places=12)
        self.assertAlmostEqual(symmetric_maxsim(a,a),1,places=12)
    def test_invalid_or_whole_map_inputs_rejected(self):
        for x in [np.zeros((49,512)),np.full((49,512),np.nan),np.ones((25,49,512))]:
            with self.assertRaises(ValueError):symmetric_maxsim(x,np.ones((49,512)))
        with self.assertRaises(ValueError):symmetric_maxsim(np.ones((49,3)),np.ones((49,4)))
    def test_quantized_cache_is_reloaded_precision(self):
        a=np.array([[3,4],[8,15]],np.float32);q=stored_unit_tokens(a)
        self.assertEqual(q.dtype,np.float16)
        reference=q.astype(np.float64);reference/=np.sqrt((reference*reference).sum(axis=1,keepdims=True))
        self.assertTrue(np.array_equal(unit_tokens(q),reference))
        with self.assertRaises(ValueError):stored_unit_tokens(np.zeros((1,49,512)))
    def test_cls_excluded_and_existing_projection_only(self):
        class Vision:
            @staticmethod
            def post_layernorm(x):return x
        class Model:
            vision_model=Vision()
            @staticmethod
            def visual_projection(x):return x[...,:512]
        h=np.ones((2,50,768),np.float32);h[:,0]=9
        cls,patch=project_last_layer(Model(),h)
        self.assertEqual(patch.shape,(2,49,512));self.assertTrue((patch==1).all());self.assertTrue((cls==9).all())
        with self.assertRaises(ValueError):project_last_layer(Model(),h[:,:49])
    def test_global_mismatch_stops_before_scoring(self):
        self.assertEqual(check_global_replay(np.zeros((2,512)),np.zeros((2,512))),0)
        with self.assertRaises(ValueError):check_global_replay(np.full((2,512),2e-5),np.zeros((2,512)))
        with self.assertRaises(ValueError):check_global_replay([[np.nan]],[[0]])
    def test_ties_do_not_use_truth_or_position_bias(self):
        expected=sorted(range(25),key=lambda j:hashlib.sha256(f'CV-002/tie/v1:cvp_00:{j}'.encode()).hexdigest())
        self.assertEqual(rank_cells(np.zeros(25),'cvp_00'),expected)
        self.assertEqual(list(inspect.signature(symmetric_maxsim).parameters),['current_tokens','target_tokens'])
        with self.assertRaises(ValueError):rank_cells(np.zeros(24),'cvp_00')
if __name__=='__main__':unittest.main()
