"""CV-012 synthetic gate and task-boundary tests; no model or raw images."""
import copy,sys,unittest,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_alignment_protocol_v1 import gate,build_queues
def counts(full=None,mean=None,swiss=None):
    full=full or [15]*5;mean=mean or [5]*5;swiss=swiss or [10]*5
    return {str(i):{'planned':25,'Identity':{'Full':1,'Mean':0,'Wrong':0},'SwissRank8Frozen':{'Full':swiss[i],'Mean':0,'Wrong':0},
        'MMRank8Final':{'Full':full[i],'Mean':mean[i],'Wrong':0},'Disabled':0} for i in range(5)}
class ProtocolBoundaries(unittest.TestCase):
    def check(self,c):return gate(c,complete=True,engineering_and_audit=True)
    def test_passing_synthetic_development_is_not_execution(self):self.assertTrue(self.check(counts())['passed'])
    def test_absolute32_rounding_boundary(self):
        self.assertFalse(self.check(counts([6,6,6,6,7],[0]*5,[1]*5))['gates']['Full_Top1_at_least32_of125'])
        self.assertTrue(self.check(counts([6,6,6,7,7],[0]*5,[1]*5))['passed'])
    def test_control16_net_boundary(self):
        self.assertFalse(self.check(counts([10]*5,[7]*5,[1]*5))['gates']['Full_minus_Mean_at_least16'])
        self.assertTrue(self.check(counts([10]*5,[7,7,7,7,6],[1]*5))['passed'])
    def test_swiss_baseline_is_required(self):
        self.assertTrue(self.check(counts([10]*5,[0]*5,[9]*5))['passed'])
        self.assertFalse(self.check(counts([10]*5,[0]*5,[10,9,9,9,9]))['gates']['Full_minus_SwissFrozen_at_least5'])
    def test_two_sources_cannot_carry_acceptance(self):
        c=counts([25,25,0,0,0],[0]*5,[0]*5)
        for p in c.values():p['Identity']['Full']=0
        self.assertFalse(self.check(c)['passed']);self.assertEqual(self.check(c)['source_positive_against_controls'],2)
    def test_missing_jobs_or_audit_do_not_pass(self):
        self.assertFalse(gate(counts(),complete=False,engineering_and_audit=True)['passed'])
        self.assertFalse(gate(counts(),complete=True,engineering_and_audit=False)['passed'])
        with self.assertRaises(ValueError):gate(counts(),complete='false',engineering_and_audit=True)
    def test_invalid_denominator_and_noninteger_counts_rejected(self):
        c=counts();c['0']['planned']=24
        with self.assertRaises(ValueError):self.check(c)
        c=counts();c['0']['Disabled']=True
        with self.assertRaises(ValueError):self.check(c)
    def test_all_windows_hide_goal_in_opaque_id(self):
        sources=[]
        for i in range(14):sources.append({'source_id':str(i),'group':f'g{i}','role':'fit' if i<9 else 'development',
            'ground':{'path':f'ground{i}'},'tiles':[{'path':f'a{i}/sat_{r}_{c}.jpg','sha256':f'{i}_{r}_{c}'} for r in range(1,10) for c in range(1,10)]})
        q=build_queues(sources);self.assertEqual(q['planned_rankings'],1250)
        for t in q['development_tasks']:
            self.assertNotIn('goal',t['pair_id']);self.assertEqual(t['aerial_feature_indices'][t['goal_index_evaluator_only']],t['ground_feature_index']*81+40)
            self.assertNotEqual(t['ground_feature_index'],t['wrong_ground_feature_index'])
        sources[0]['tiles'][40]['sha256']=sources[0]['tiles'][0]['sha256']
        with self.assertRaises(ValueError):build_queues(sources)
    def test_raw_access_blocked_and_no_inference_imports(self):
        from tools.freeze_capturepoint_alignment_v1 import sha,SOURCES
        with self.assertRaises(ValueError):sha(ROOT/'DATA/raw_data/MM-GAG/any.jpg')
        for source in SOURCES:
            for n in ast.walk(ast.parse((ROOT/'project/src'/source).read_text(encoding='utf-8-sig'))):
                if isinstance(n,ast.Import):self.assertFalse(any(a.name.split('.')[0] in {'torch','transformers','PIL','numpy','subprocess','requests','httpx'} for a in n.names))
                if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],{'torch','transformers','PIL','numpy'})
if __name__=='__main__':unittest.main()
