"""Synthetic trajectory cases for diagnostic boundary and attribution errors."""
from copy import deepcopy
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.continued_failure_diagnosis import analyze,pair,classification

def fixture(cells,goal,budget=None,arm='M0'):
    k=5;budget=budget or len(cells)-1
    distance=lambda c:abs(c//k-goal//k)+abs(c%k-goal%k)
    names={-5:'up',1:'right',5:'down',-1:'left'};order=('up','right','down','left')
    tr=[dict(step=0,patch_id=cells[0],action=None,out_of_bounds=False,revisited=False)];ds=[]
    for i,(a,b) in enumerate(zip(cells,cells[1:])):
        action=names[b-a];logits=[0.0]*4;logits[order.index(action)]=1
        tr.append(dict(step=i+1,patch_id=b,action=action,out_of_bounds=False,revisited=b in cells[:i+1]))
        ds.append(dict(step=i+1,remaining_budget=budget-i,public_position=list(divmod(a,k)),public_visited=cells[:i+1],
            action=action,explorer_action=action,explorer_logits=logits,cue_action=None,reason='not_adjacent',probabilities=[0,0,0,0,1],
            current_image_sha256=str(a),target_image_sha256=str(goal),cue_features_sha256=str((a,goal))))
    revisits=sum(x['revisited'] for x in tr);success=cells[-1]==goal
    ep=dict(episode_id='synthetic',area='img_40',source_tile='synthetic/source',grid_size=k,budget=budget,goal=goal,start=cells[0],dist=distance(cells[0]))
    row=dict(episode_id='synthetic',area='img_40',source='synthetic/source',condition='CueFull',status='completed',steps=len(ds),
        success=success,sg=distance(cells[-1]),distance=ep['dist'],termination='goal_reached' if success else 'budget_exhausted',
        trajectory=tr,decisions=ds,revisits=revisits,repeat_visit_rate=revisits/len(ds),out_of_bounds=0,arm=arm,local_checkpoint_seed=0)
    return row,ep

class DiagnosisTests(unittest.TestCase):
    def test_last_budget_adjacency_is_not_actionable(self):
        row,ep=fixture([0,5,10,15,20],21)
        result=analyze(row,ep)
        self.assertEqual(result['failure_class'],'adjacent_only_at_terminal');self.assertEqual(result['opportunities'],[])
    def test_budget_positive_opportunity_can_be_missed(self):
        row,ep=fixture([0,5,10,15,20],1)
        result=analyze(row,ep)
        self.assertEqual(result['failure_class'],'missed_actionable_adjacency')
        self.assertEqual(result['opportunities'][0]['remaining'],4)
        self.assertEqual(result['opportunities'][0]['true_direction'],'right')
    def test_near_and_never_near_remain_distinct(self):
        a,e=fixture([0,5,10,15],17);b,f=fixture([0,5,10,15],24)
        self.assertEqual(analyze(a,e)['failure_class'],'near_without_adjacency')
        self.assertEqual(analyze(b,f)['failure_class'],'never_within_two')
    def test_last_step_success_takes_precedence(self):
        r,e=fixture([2,1,0],0)
        self.assertEqual(analyze(r,e)['failure_class'],'success')
    def test_fresh_alternative_requires_a_real_revisit(self):
        r,e=fixture([0,5,0,1],24);d=analyze(r,e)
        self.assertEqual(d['counts']['avoidable_revisits'],1);self.assertEqual(d['counts']['immediate_reverse'],1)
        self.assertFalse(d['public_steps'][0]['avoidable_revisit']);self.assertTrue(d['public_steps'][1]['avoidable_revisit'])
    def test_replay_rejects_teleport_and_budget_corruption(self):
        r,e=fixture([0,5,10],24);bad=deepcopy(r);bad['trajectory'][1]['patch_id']=6
        with self.assertRaises(AssertionError):analyze(bad,e)
        bad=deepcopy(r);bad['decisions'][0]['remaining_budget']=0
        with self.assertRaises(AssertionError):analyze(bad,e)
    def test_first_divergence_requires_same_visible_inputs(self):
        a,e=fixture([0,5,10],24);b,f=fixture([0,1,2],24,arm='Continue5')
        x=analyze(a,e);y=analyze(b,f);z=pair(x,y)
        self.assertEqual(z['outcome'],'both_fail');self.assertEqual(z['first_divergence']['state_step'],0)
        y['public_steps'][0]['target_image_sha256']='different'
        with self.assertRaises(AssertionError):pair(x,y)
    def test_success_cannot_continue_after_reaching_goal(self):
        r,e=fixture([0,1,2],1)
        with self.assertRaises(AssertionError):analyze(r,e)

if __name__=='__main__':unittest.main()
