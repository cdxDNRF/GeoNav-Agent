"""Memory-only boundary tests; no model construction or source image reads."""
import ast
import unittest
from pathlib import Path
from project.src.agents import ccvpe_runtime_v1 as rt

ROOT = Path(__file__).resolve().parents[3]


class Tests(unittest.TestCase):
    def source(self):
        return (ROOT/'project/.runtime/ccvpe_engineering_v1/upstream_models.py').read_text(encoding='utf-8-sig')

    def test_only_reviewed_definitions(self):
        tree = rt.selected_tree(self.source())
        self.assertEqual({n.name for n in tree.body}, rt.DEFINITIONS)
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree)))

    def test_no_pretrained_factory(self):
        t = ast.unparse(rt.selected_tree(self.source()))
        self.assertNotIn('from_pretrained',t)
        self.assertEqual(t.count('EfficientNet.from_name('),2)

    def test_original_full_expression_preserved(self):
        source = self.source()
        original = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CVM_KITTI')
        original = next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        fn = rt.controlled_tree(source).body[0]
        fn.name='forward';fn.args.args.pop();fn.args.defaults.pop()
        count=0
        for n in ast.walk(fn):
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.IfExp):
                self.assertEqual(n.value.test.id,'disabled')
                self.assertEqual(ast.unparse(n.value.body),'torch.zeros_like(sat_map_norm)')
                n.value=n.value.orelse;count+=1
        self.assertEqual(count,6)
        self.assertEqual(ast.dump(fn),ast.dump(original))

    def test_goal_interface_no_truth(self):
        fn=rt.controlled_tree(self.source()).body[0]
        self.assertEqual([a.arg for a in fn.args.args],['self','grd','sat','disabled'])
        self.assertIsNone(fn.args.kwarg)

    def test_all_matching_assignments_guarded(self):
        fn=rt.controlled_tree(self.source())
        scores=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='matching_score' for t in n.targets)]
        self.assertEqual(len(scores),6)
        self.assertTrue(all(isinstance(n.value,ast.IfExp) for n in scores))

    def test_wrong_route_and_invalid_controls(self):
        a,b=object(),object()
        self.assertIs(rt.route_ground('Wrong',a,b),b)
        for arm in ('Full','Disabled','Mean'):self.assertIs(rt.route_ground(arm,a,b),a)
        with self.assertRaises(ValueError):rt.route_ground('Wrong',a)
        with self.assertRaises(ValueError):rt.route_ground('unknown',a)

    def test_mean_requires_all_levels_before_model_call(self):
        for means in (None,[],[1]*5):
            with self.assertRaises(ValueError):rt.run_control(None,None,None,None,'Mean',means=means)

    def test_resource_boundary(self):
        from project.src.tools.ccvpe_engineering_v1 import guard_violation,LIMIT
        self.assertFalse(guard_violation(LIMIT,1800))
        self.assertTrue(guard_violation(LIMIT+1,0))
        self.assertTrue(guard_violation(1,1800.001))

    def test_ground_letterbox_and_local_unknown(self):
        import torch
        from PIL import Image
        g=rt.prepare_rgb(Image.new('RGB',(962,378),(255,0,0)),True)
        self.assertEqual(tuple(g.shape),(1,3,256,1024))
        self.assertTrue(torch.equal(g[:,:,:,:180],torch.zeros_like(g[:,:,:,:180])))
        a=rt.prepare_local_rgb(Image.new('RGB',(125,125),(255,0,0)),12)
        self.assertEqual(tuple(a.shape),(1,3,512,512))
        self.assertEqual(float(a[:,:,:150,:].abs().max()),0)
        self.assertGreater(float(a[:,:,220:290,220:290].abs().max()),0)
        with self.assertRaises(ValueError):rt.prepare_local_rgb(Image.new('RGB',(625,625)),12)
        with self.assertRaises(ValueError):rt.prepare_local_rgb(Image.new('RGB',(125,125)),25)


if __name__=='__main__':unittest.main()
