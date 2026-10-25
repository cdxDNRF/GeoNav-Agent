import copy
import unittest
import numpy as np
from eval.compile_geoexplorer_evidence_close_v1 import ROOT,INPUTS,read,extract_evidence,coordinate_vectors,geometry_report,progress_actions
from eval.audit_geoexplorer_evidence_close_v1 import analytic_classes,independent_progress


class EvidenceCloseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.documents={k:read(ROOT/p) for k,p in INPUTS.items() if p.endswith('.json')}

    def test_original_denominators(self):
        out=extract_evidence(self.documents)
        self.assertEqual(sum(r['planned'] for r in out['navigation']['metrics']),6000)
        self.assertEqual(out['diagnostics']['context']['active_positions'],622887)

    def test_reject_changed_plan(self):
        d=copy.deepcopy(self.documents);d['comparison']['aggregate_metrics'][0]['planned']=749
        with self.assertRaises(ValueError):extract_evidence(d)

    def test_reject_duplicate_arm(self):
        d=copy.deepcopy(self.documents);d['comparison']['aggregate_metrics'][1]=copy.deepcopy(d['comparison']['aggregate_metrics'][0])
        with self.assertRaises(ValueError):extract_evidence(d)

    def test_reject_missing_audit(self):
        d=copy.deepcopy(self.documents);d['training_audit']['status']='incomplete'
        with self.assertRaises(ValueError):extract_evidence(d)

    def test_no_reopening_factor(self):
        d=copy.deepcopy(self.documents);d['gradient']['loss_weight_candidate_screen_passed']=True
        with self.assertRaises(ValueError):extract_evidence(d)

    def test_full_circle_is_near_not_byte_identical(self):
        v=coordinate_vectors(5,2,np.float32)
        self.assertFalse(np.array_equal(v[0],v[4]))
        self.assertLess(np.linalg.norm(v[0]-v[4]),1e-6)
        self.assertEqual(sorted(geometry_report(5,2,1e-6)['classes']),analytic_classes(5,False))

    def test_half_circle_distinguishes_boundaries(self):
        row=geometry_report(5,1,1e-6)
        self.assertEqual(row['class_count'],25);self.assertEqual(row['near_pair_count'],0)
        self.assertGreater(row['minimum_distinct_cell_l2'],.7)

    def test_direction_label_is_offline_only(self):
        self.assertEqual(progress_actions(0,2,5),{1})
        self.assertEqual(progress_actions(4,2,5),{3})
        for c in range(25):
            for g in range(25):self.assertEqual(progress_actions(c,g,5),set(independent_progress(c,g,5)))

    def test_invalid_geometry(self):
        for grid in [1,0,-1,5.0,True]:
            with self.assertRaises(ValueError):coordinate_vectors(grid,2,np.float32)
        with self.assertRaises(ValueError):coordinate_vectors(5,3,np.float32)


if __name__=='__main__':unittest.main()
