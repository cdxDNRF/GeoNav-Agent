"""Synthetic checks, no CCVPE forward, downloads or real image reads."""
import copy
import math
import unittest
from project.src.eval import ccvpe_screen_v1 as run
from project.src.data import spatial_perception_protocol_v1 as p


class Tests(unittest.TestCase):
    def cases(self,fold):return run.read(run.DESIGN)['cases'][fold]

    def test_roi_partition_both_scales(self):
        for size in (256,512):
            points=set()
            for cell in range(25):
                ys,xs=run.roi_indices(size,cell)
                for y in ys:
                    for x in xs:
                        self.assertNotIn((y,x),points);points.add((y,x))
                        r,c=divmod(cell,5)
                        self.assertTrue(c*125<=(x+.5)*625/size<(c+1)*125)
                        self.assertTrue(r*125<=(y+.5)*625/size<(r+1)*125)
            self.assertEqual(len(points),size*size)

    def test_peak_and_score_ignore_unvisited(self):
        import torch
        h=torch.zeros((1,1,512,512));m=torch.zeros((1,16,256,256))
        hy,hx=run.roi_indices(512,12);my,mx=run.roi_indices(256,12)
        h[0,0,0,0]=99;h[0,0,hy[0]+7,hx[0]+5]=1
        m[0,0,0,0]=1;m[0,3,my[-1],mx[-1]]=.75
        r,hh,mm=run.extract_visible([None,h,None,m],12)
        self.assertEqual(r['score'],.75)
        self.assertEqual(r['predicted_xy_native'],[(hx[0]+5.5)*625/512,(hy[0]+7.5)*625/512])

    def test_row_major_tie_and_invalid_output(self):
        import torch
        h=torch.zeros((1,1,512,512));m=torch.zeros((1,16,256,256))
        r,_,_=run.extract_visible([None,h,None,m],24)
        self.assertEqual(r['heat_peak_offset_yx'],[0,0])
        h[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):run.extract_visible([None,h,None,m],24)
        with self.assertRaises(ValueError):run.roi_indices(512,True)

    def test_calibration_identity_and_split(self):
        cases=self.cases('calibration');rows=[dict(case_id=c['case_id'],arm='Full',score=.2) for c in cases]
        rec=run.calibration_records(cases,rows)
        self.assertEqual(len(rec),20)
        self.assertEqual(p.calibrate_threshold(rec)['tau'],math.nextafter(.2,math.inf))
        with self.assertRaises(ValueError):run.calibration_records(cases,rows[:-1]+rows[:1])
        rows[0]['arm']='Wrong'
        with self.assertRaises(ValueError):run.calibration_records(cases,rows)
        rec[0]['fold']='development_screen'
        with self.assertRaises(ValueError):p.calibrate_threshold(rec)

    def test_planned_denominator_and_goal_control_gate(self):
        cases=self.cases('development_screen');rows=[]
        for c in cases:
            e=c['evaluator_only'];b=c['current_crop_bounds'];xy=e['query_xy_native'] if e['relation']=='inside' else [(b[0]+b[2])/2,(b[1]+b[3])/2]
            for arm in p.ARMS:rows.append(dict(case_id=c['case_id'],arm=arm,score=.9 if e['relation']=='inside' and arm=='Full' else -.5,predicted_xy_native=xy))
        s=run.summarize(cases,rows,.5)
        self.assertTrue(s['perception_screen_passed'])
        self.assertEqual(s['metrics']['Full']['positive_accepted_and_within5m'],2)
        self.assertEqual(s['contrasts']['Wrong']['by_class']['inside']['net'],2)
        with self.assertRaises(ValueError):run.summarize(cases,rows[:-1],.5)
        rows[0]['predicted_xy_native']=[cases[0]['current_crop_bounds'][0]+.5,cases[0]['current_crop_bounds'][1]+.5]
        self.assertFalse(run.summarize(cases,rows,.5)['perception_screen_passed'])

    def test_model_payload_no_answers(self):
        payload=dict(given_target_rgb=None,current_crop_rgb=None,public_cell=12,grid_size=5,parent_size=[625,625])
        self.assertTrue(p.valid_model_fields(payload))
        for field in ('heading','camera_xy','source_id','parent_rgb','goal_index'):
            with self.assertRaises(ValueError):p.valid_model_fields(dict(payload,**{field:None}))


if __name__=='__main__':unittest.main()
