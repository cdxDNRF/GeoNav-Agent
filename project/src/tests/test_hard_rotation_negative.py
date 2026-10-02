import unittest
import numpy as np
from data.hard_rotation_negative import hard_angles
from data.rotation_negative import negative_angles


class HardRotationTests(unittest.TestCase):
    def test_labels_slots_and_quotas(self):
        labels=np.tile([0,4,1,4,2,4,3,4,4,4],2);scores=np.linspace(0,1,36).reshape(12,3);before=labels.copy()
        for epoch in range(16):
            mined=hard_angles(labels,epoch,scores);base=negative_angles(labels,epoch)
            np.testing.assert_array_equal(mined>0,base>0);np.testing.assert_array_equal(np.bincount(mined,minlength=4),[14,2,2,2]);self.assertTrue((mined[labels<4]==0).all())
        np.testing.assert_array_equal(labels,before)

    def test_ties_are_fixed_by_rank_then_small_angle(self):
        labels=np.full(12,4);mined=hard_angles(labels,0,np.full((12,3),.8))
        np.testing.assert_array_equal(mined,[0,0,0,1,1,2,0,0,0,2,3,3])

    def test_hardest_selected_slot_gets_first_choice(self):
        labels=np.full(6,4);scores=np.zeros((6,3));scores[3]=[.7,.6,.5];scores[4]=[.8,.4,.3];scores[5]=[.9,.2,.1]
        np.testing.assert_array_equal(hard_angles(labels,0,scores),[0,0,0,3,2,1])

    def test_full_budget_counts(self):
        labels=np.concatenate((np.zeros(6960,np.int64),np.full(45240,4)));scores=np.zeros((45240,3));total=np.zeros(4,np.int64)
        for epoch in range(16):total+=np.bincount(hard_angles(labels,epoch,scores),minlength=4)
        np.testing.assert_array_equal(total,[473280,120640,120640,120640])

    def test_rejects_invalid_score_bank(self):
        labels=np.full(6,4)
        for scores in (np.zeros((5,3)),np.full((6,3),np.nan),np.full((6,3),1.1),np.full((6,3),-.1)):
            with self.assertRaises(ValueError):hard_angles(labels,0,scores)
        with self.assertRaises(ValueError):hard_angles(np.full(5,4),0,np.zeros((5,3)))

    def test_does_not_modify_scores(self):
        labels=np.full(12,4);scores=np.random.default_rng(7).random((12,3)).astype(np.float32);before=scores.copy()
        hard_angles(labels,7,scores);np.testing.assert_array_equal(scores,before)


if __name__=='__main__':unittest.main()
