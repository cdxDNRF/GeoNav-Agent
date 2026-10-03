"""Negative labels and class exposure are invariant under target rotations."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.rotation_negative import negative_angles


class RotationNegativeTests(unittest.TestCase):
    def test_positives_are_never_rotated_or_relabelled(self):
        labels=np.array([0,4,1,4,2,4,3,4],np.int64);copy=labels.copy()
        for epoch in range(16):self.assertTrue((negative_angles(labels,epoch)[labels<4]==0).all())
        np.testing.assert_array_equal(labels,copy)
    def test_half_of_negatives_rotate_with_equal_three_angle_exposure(self):
        labels=np.array([0,1,2,3]+[4]*12,np.int64)
        for epoch in range(16):
            ids=negative_angles(labels,epoch);self.assertEqual(np.bincount(ids[labels==4],minlength=4).tolist(),[6,2,2,2])
    def test_each_negative_sees_three_native_and_three_rotated_views_per_cycle(self):
        labels=np.array([4]*6,np.int64);schedule=np.stack([negative_angles(labels,e) for e in range(6)])
        for column in schedule.T:self.assertEqual(np.bincount(column,minlength=4).tolist(),[3,1,1,1])
    def test_actual_budget_preserves_all_positive_and_negative_counts(self):
        labels=np.array([0]*1740+[1]*1740+[2]*1740+[3]*1740+[4]*45240,np.int64)
        total=np.zeros(4,np.int64)
        for e in range(16):total+=np.bincount(negative_angles(labels,e),minlength=4)
        self.assertEqual(total.tolist(),[473280,120640,120640,120640]);self.assertEqual(len(labels)*16,835200)
    def test_bad_labels_and_epoch_are_rejected(self):
        for labels,epoch in [(np.array([5]),0),(np.array([4]),16),(np.array([4.]),0),(np.array([[4]]),0)]:
            with self.assertRaises(ValueError):negative_angles(labels,epoch)


if __name__=='__main__':unittest.main()
