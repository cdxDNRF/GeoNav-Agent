"""Input-path, repeated-source grouping and protected-role boundary checks."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.crossview_training_inventory_v1 import safe_path,name_parts,components,role_capacity
class InventoryTests(unittest.TestCase):
    def test_metadata_path_cannot_escape_dataset(self):
        base=Path(__file__).resolve().parent
        self.assertEqual(safe_path(base,'nested/photo.jpg'),base/'nested/photo.jpg')
        for t in ['../outside.jpg',str(base.parent/'absolute.jpg')]:
            with self.assertRaises(ValueError):safe_path(base,t)
    def test_named_variants_are_not_independent_scenes(self):
        parsed=[name_parts(f'Lake_A_{i:02d}.jpg') for i in range(25)]
        self.assertEqual({x[0] for x in parsed},{'Lake_A'});self.assertEqual({x[1] for x in parsed},set(range(25)))
    def test_invalid_suffix_is_not_silently_a_goal_label(self):
        for t in ['Lake_25.jpg','Lake_99.jpg','Lake_1.jpg','Lake.jpg']:
            with self.assertRaises(ValueError):name_parts(t)
    def test_shared_pixels_merge_transitively(self):
        self.assertEqual(components(['a','b','c','d'],[('a','b'),('b','c'),('a','b')]),[['a','b','c'],['d']])
    def test_assigned_development_or_reserved_cannot_become_train(self):
        x=role_capacity([['a'],['b'],['c']],[0],[1,2]);self.assertEqual(x['unassigned_training_groups'],0)
        with self.assertRaises(ValueError):role_capacity([['a'],['b']],[0],[0,1])
    def test_empty_and_disconnected_groups_preserved(self):
        self.assertEqual(components([],[]),[]);self.assertEqual(components(['z','a'],[]),[['a'],['z']])
if __name__=='__main__':unittest.main()
