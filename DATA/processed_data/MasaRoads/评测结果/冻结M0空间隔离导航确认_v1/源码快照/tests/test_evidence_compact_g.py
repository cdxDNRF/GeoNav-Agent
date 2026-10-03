import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.collaboration_pilot import digest_payload
from agents.evidence_budget import EvidenceLedger,paths_for
from eval.evidence_compact_g import (ROOT,OLD,OUT,read,lines,load_inputs,episode_run,option_action,sample_for,observe_public_ledger)
from env.scaled_grid import ScaledEpisode
from env.environment import Observation


class ClosedLoopTests(unittest.TestCase):
    def test_accepted_cue_interrupts_remaining_option(self):
        action,queue,interrupted,control=option_action({'cue_action':'up','action':'up'},['left'])
        self.assertEqual((action,queue,interrupted,control),('up',[],True,'accepted_cue'))

    def test_option_uses_actual_next_move_then_returns_to_base(self):
        base={'cue_action':None,'action':'up'}
        action,queue,interrupted,_=option_action(base,['left','down'])
        self.assertEqual((action,queue,interrupted),('left',['down'],False))
        action,queue,_,_=option_action(base,queue);self.assertEqual((action,queue),('down',[]))
        self.assertEqual(option_action(base,queue)[0],'up')

    def test_cloud_history_is_only_distinct_actual_prior_observations(self):
        obs=Observation(b'current',b'target',(0,2),10,18,(0,1,2))
        ledger=EvidenceLedger(10,20)
        for pos,remaining,visited,pixel,action in [((0,0),20,(0,),b'a',None),((0,1),19,(0,1),b'b','right'),((0,2),18,(0,1,2),b'current','right')]:
            frame=Observation(pixel,b'target',pos,10,remaining,visited)
            observe_public_ledger(ledger,frame,{'cue_action':None},action)
        base={'action':'down','explorer_action':'down','cue_action':None,'probabilities':[0,0,0,0,1]}
        sample=sample_for(obs,base,ledger,[{'cell':0,'pixels':b'a'},{'cell':1,'pixels':b'b'}],paths_for(obs.position,obs.visited,10,18,'down'))
        self.assertEqual(sample['images'],[('target',b'target'),('current',b'current'),('history_0',b'a'),('history_1',b'b')])
        self.assertNotIn('goal',sample['state'])

    def test_frozen_M0_all_20_routes_exactly_reproduce_old_S4(self):
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        banks,means=load_inputs();old={r['episode_id']:r for r in lines(OLD/'Edge_s0/导航_CueFull_轨迹.jsonl')}
        for record in read(OUT/'总体任务.json'):
            ep=ScaledEpisode(**record);row=episode_run(ep,'M0',None,None,banks,means)
            previous=old[ep.episode_id]
            self.assertEqual(row['trajectory'],previous['trajectory'])
            self.assertEqual(row['success'],previous['success']);self.assertEqual(row['sg'],previous['sg'])
            self.assertEqual([d['base'] for d in row['decisions']],previous['decisions'])


if __name__=='__main__':unittest.main()
