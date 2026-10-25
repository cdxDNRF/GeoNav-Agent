"""In-memory tests: no directories, data, model, navigation or networking."""
import ast
from pathlib import Path
import unittest

from project.src.eval import diagnose_ground_spatial_evidence_v1 as d
from project.src.eval import audit_ground_spatial_evidence_v1 as audit


class Tests(unittest.TestCase):
    def test_gradient_orientation_and_ties(self):
        good=[-d.dist(i,4,3) for i in range(9)]
        self.assertEqual(d.edge_counts(good,4,3)['correct'],12)
        self.assertEqual(d.edge_counts([-x for x in good],4,3)['incorrect'],12)
        self.assertEqual(d.edge_counts([0.]*9,4,3)['half_tie_accuracy'],.5)

    def test_local_coordinates_not_flat_index(self):
        self.assertEqual(d.dist(2,3,3),3)
        self.assertNotIn(3,d.adj(2,3))
        with self.assertRaises(ValueError):d.edge_counts(list(range(9)),4,3,[(2,3)])

    def test_rank_counts_revisits_and_collinearity(self):
        self.assertEqual(d.rank_geometry([14,14],5),0)
        self.assertEqual(d.rank_geometry([14,13,12],5),1)
        self.assertEqual(d.rank_geometry([14,13,9],5),2)
        self.assertEqual(d.rank_geometry([0,6,12],5),1)

    def test_source_bootstrap_not_edge_bootstrap(self):
        self.assertEqual(d.source_interval([0.,0.,0.,0.,0.]),[0.,0.])
        self.assertEqual(d.source_interval([.1]*5),[.1,.1])
        self.assertEqual(d.quantile([0,10],.25),2.5)

    def test_different_equations_agree_on_adversarial_geometry(self):
        for cells in ([14],[14,13,12],[0,6,12],[0,1,5],[1,1,2,8]):
            self.assertEqual(d.rank_geometry(cells,5),audit.geometric_rank(cells,5))
        for scores in ([0.]*9,list(range(9)),[-d.dist(i,4,3) for i in range(9)]):
            self.assertEqual(d.edge_counts(scores,4,3),audit.scalar_edges(scores,4,3))
        for got,want in zip(d.source_interval([-1.,1.]),audit.multinomial_interval([-1,1])):
            self.assertAlmostEqual(got,want,12)

    def test_invalid_scores_and_empty_observed_edges(self):
        with self.assertRaises(ValueError):d.edge_counts([float('nan')]*9,4,3)
        with self.assertRaises(ValueError):d.edge_counts([0.]*9,9,3)
        self.assertIsNone(d.edge_counts([0.]*9,4,3,[])['half_tie_accuracy'])

    def test_saved_terminal_is_not_an_observation(self):
        events=[dict(position=1,next_position=2),dict(position=2,next_position=5)]
        self.assertEqual(d.observed_route_edges(events),[(1,2)])
        self.assertEqual(d.observed_route_edges([dict(position=1,next_position=2)]),[])

    def test_window_mapping_and_axis_signs(self):
        info=dict(indices=list(range(100,181)))
        ids=[100+20+r*9+c for r in range(5) for c in range(5)]
        task=dict(aerial_feature_indices=ids,goal_index_evaluator_only=12)
        self.assertEqual(d.window_mapping(task,info)[12],40)
        with self.assertRaises(ValueError):d.window_mapping(dict(task,aerial_feature_indices=ids[::-1]),info)
        with self.assertRaises(ValueError):d.window_mapping(dict(task,goal_index_evaluator_only=13),info)

    def test_inputs_do_not_include_unconsumed_assets(self):
        self.assertFalse(any('/raw_data/' in p or '/models/' in p or p.endswith(('.npy','.npz','.pt')) for p in d.INPUTS.values()))
        self.assertEqual(d.frozen_scope()['resources']['navigation'],0)
        self.assertEqual(d.frozen_scope()['resources']['encoder_forward'],0)

    def test_zero_model_and_policy_execution_dependencies(self):
        tree=ast.parse(Path(d.__file__).read_text(encoding='utf-8'))
        imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.extend(a.name.split('.')[0] for a in n.names)
            if isinstance(n,ast.ImportFrom):imports.append(n.module.split('.')[0])
        self.assertTrue(set(imports)<={'hashlib','itertools','json','math','pathlib','sys','time'})
        forbidden={'observe','forward','predict','execute_ground_local_loop_v2','GroundLocalLoop'}
        called={getattr(n.func,'attr',getattr(n.func,'id','')) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(called&forbidden)


if __name__=='__main__':unittest.main()
