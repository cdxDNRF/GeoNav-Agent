"""Boundary checks for dataset identity, coordinate rejection and goal-crop geometry."""
import importlib.util,unittest,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'project/src/tools/mmgag_crossview_inventory_v1.py'
spec=importlib.util.spec_from_file_location('mmgag_static',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AdmissionTests(unittest.TestCase):
    def test_heic_jpg_identity_and_collisions_preserved(self):
        self.assertEqual(m.key('IMG_1.HEIC'),m.key('img_1.jpg'))
        out=m.coordinates([{'filename':'IMG_1.HEIC','lat':'1','lon':'2'},{'filename':'img_1.jpg','lat':'3','lon':'4'}])
        self.assertEqual(len(out['img_1']),2)
    def test_nonfinite_or_outside_coordinate_rejected(self):
        out=m.coordinates([{'filename':f'{i}.jpg','lat':lat,'lon':lon} for i,(lat,lon) in enumerate([('nan','2'),('91','2'),('1','181'),('inf','0'),('x','1')])])
        self.assertTrue(all(not v[0]['valid'] for v in out.values()))
    def test_numeric_grid_excludes_stitch(self):
        names=[f'sat_{r}_{c}.jpg' for r in range(11) for c in range(11)]+['stitched_image.jpg','stitched_image copy.jpg']
        result=m.grid_issues(names);self.assertEqual(result['patch_files'],121);self.assertFalse(result['missing'])
        self.assertEqual(m.patch_position('sat_0_10.jpg'),(0,10))
    def test_alias_duplicate_missing_and_overflow(self):
        result=m.grid_issues(['sat_0_1.jpg','sat_00_1.jpg','sat_11_0.jpg'])
        self.assertEqual(result['duplicate_indices'],[(0,1)]);self.assertEqual(result['outside'],[(11,0)]);self.assertEqual(len(result['missing']),120)
    def test_goal_not_fixed_public_center(self):
        public=set()
        for idx in range(25):
            crop=m.goal_window(idx);r,c=divmod(idx,5);self.assertEqual((crop['top']+r,crop['left']+c),(5,5));public.add((r,c))
        self.assertEqual(len(public),25)
        with self.assertRaises(ValueError):m.goal_window(25)
    def test_geographic_transitive_grouping_and_hash_overlap(self):
        s=[{'id':str(i),'coordinate':[0,x]} for i,x in enumerate([0,.02,.04,30])]
        hashes={'0':{'a'},'1':{'b'},'2':{'c'},'3':{'c'}}
        g=m.grouped_sources(s,hashes);self.assertEqual(g['groups'],[['0','1','2','3']])
    def test_no_model_or_network_imports(self):
        for file in [P,ROOT/'project/src/eval/audit_mmgag_crossview_inventory_v1.py']:
            tree=ast.parse(file.read_text(encoding='utf-8'));modules=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):modules.extend(a.name for a in n.names)
                if isinstance(n,ast.ImportFrom):modules.append(n.module or '')
            self.assertFalse(any(x.split('.')[0] in ['torch','transformers','requests','httpx','urllib','agents','train'] for x in modules))
if __name__=='__main__':unittest.main()
