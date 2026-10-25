"""CV-010 metadata-only grouping/leakage/geometry boundary tests."""
import ast
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_partition_v1 import distance,components,group_graph,propose,window_refs,tile_registry


class MetadataPlanning(unittest.TestCase):
    def test_identical_and_dateline_coordinates(self):
        self.assertEqual(distance([0,0],[0,0]),0.)
        self.assertLess(distance([0,179.999],[0,-179.999]),250)
        with self.assertRaises(ValueError):distance([91,0],[0,0])

    def test_transitive_components_cannot_split_chain(self):
        self.assertEqual(components(['a','b','c','d'],[('a','b'),('b','c')]),[['a','b','c'],['d']])

    def test_shared_bytes_merge_far_apart_sources(self):
        s=[{'id':'a','coordinate':[0,0]},{'id':'b','coordinate':[60,60]}]
        p={'a':{(0,0):{'sha256':'same'}},'b':{(0,0):{'sha256':'same'}}}
        groups,_,_=group_graph(s,p,{'a':'g1','b':'g2'})
        self.assertEqual(groups,[['a','b']])

    def test_duplicate_ground_merges_even_without_shared_aerial(self):
        s=[{'id':'a','coordinate':[0,0]},{'id':'b','coordinate':[60,60]}]
        p={'a':{(0,0):{'sha256':'p1'}},'b':{(0,0):{'sha256':'p2'}}}
        self.assertEqual(group_graph(s,p,{'a':'same','b':'same'})[0],[['a','b']])

    def test_proposal_deterministic_and_nonactivating(self):
        groups=[[f's{i}'] for i in range(20)];hashes={f's{i}':f'h{i}' for i in range(20)}
        a=propose(groups,hashes);b=propose(list(reversed(groups)),hashes)
        self.assertEqual(a,b);self.assertEqual([sum(g['proposed_role']==r for g in a) for r in ['fit','development','confirmation']],[9,5,6])
        self.assertTrue(all(g['current_role']=='reserved' and not g['activated'] for g in a))
        with self.assertRaises(ValueError):propose(groups+[groups[0]],hashes)

    def test_center_label_all25_both_axis_conventions(self):
        tiles={(r,c):{'position':(r,c)} for r in range(11) for c in range(11)}
        for goal in range(25):
            for axis in ['row','column']:self.assertEqual(window_refs(tiles,goal,axis)[goal],tiles[5,5])
        with self.assertRaises(ValueError):window_refs(tiles,25)

    def test_complete_numeric_grid_required(self):
        source={'aerial_directories':['d']};raw=[{'path':f'd/sat_{r}_{c}.jpg'} for r in range(11) for c in range(11)]
        self.assertEqual(len(tile_registry(source,raw)),121)
        with self.assertRaises(ValueError):tile_registry(source,raw[:-1])

    def test_planner_blocks_raw_byte_reads_and_inference_imports(self):
        from tools.plan_capturepoint_sources_v1 import sha,SOURCES
        with self.assertRaises(ValueError):sha(ROOT/'DATA/raw_data/MM-GAG/any.jpg')
        forbidden={'torch','transformers','safetensors','PIL','numpy','requests','httpx','subprocess','socket','importlib'}
        for name in SOURCES:
            for node in ast.walk(ast.parse((ROOT/'project/src'/name).read_text(encoding='utf-8-sig'))):
                if isinstance(node,ast.Import):self.assertFalse(any(a.name.split('.')[0] in forbidden for a in node.names))
                if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],forbidden)


if __name__=='__main__':unittest.main()
