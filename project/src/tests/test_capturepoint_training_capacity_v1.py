"""Pair cardinality and storage/role boundaries, without any real image access."""
import unittest
from eval.capturepoint_training_capacity_v1 import capacity,load,entry,ROOT,distance


class CapacityBoundaries(unittest.TestCase):
    def test_same_photo_repeated_windows_do_not_create_photos(self):
        def r(i,g,c):return {'source_id':str(i),'group':'one','ground':{'sha256':g},'center_proxy':{'sha256':c}}
        x=capacity([r(0,'a','x'),r(1,'a','y'),r(2,'b','y')])
        self.assertEqual(x,{'source_records':3,'source_groups':1,'distinct_ground_bytes':2,'distinct_positive_bytes':2,'distinct_byte_pairs':3})

    def test_duplicate_source_not_new_sample(self):
        r={'source_id':'same','group':'one','ground':{'sha256':'a'},'center_proxy':{'sha256':'b'}}
        with self.assertRaises(ValueError):capacity([r,r])

    def test_raw_feature_and_model_paths_rejected(self):
        for p in [ROOT/'DATA/raw_data/MM-GAG/a.json',ROOT/'models/a.json',ROOT/'DATA/processed_data/MM-GAG/a.npy']:
            with self.assertRaises(ValueError):load(p)
        for p in [ROOT/'DATA/raw_data/MM-GAG/a.jpg',ROOT/'models/a.bin']:
            with self.assertRaises(ValueError):entry(p)

    def test_geographic_distance_keeps_meters_not_grid(self):
        self.assertEqual(distance([0,0],[0,0]),0)
        self.assertAlmostEqual(distance([0,0],[0,1]),111195.0802335329,places=5)


if __name__=='__main__':unittest.main()
