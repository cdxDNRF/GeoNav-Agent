import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.local_ledger_option import LocalLedgerOption
from env.environment import Observation
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.local_ledger_expansion import load_inputs,make_agent,row_run,read,lines,OLD,DATA,PREVIOUS


class LocalLedgerTests(unittest.TestCase):
    def test_two_moves_never_overrides_accepted_cue(self):
        ctrl=LocalLedgerOption();obs=Observation(b'a',b't',(0,0),10,20,(0,))
        ctrl.decide(obs,{'cue_action':None,'action':'right','explorer_action':'right'})
        ctrl.pending=['down'];obs=Observation(b'b',b't',(0,1),10,19,(0,1))
        result=ctrl.decide(obs,{'cue_action':'right','action':'right','explorer_action':'down'})
        self.assertEqual(result['action'],'right');self.assertTrue(result['option_interrupted']);self.assertEqual(result['pending_after'],[])

    def test_reset_removes_history_option_and_intervention(self):
        ctrl=LocalLedgerOption();ctrl.used=True;ctrl.pending=['left'];ctrl.previous='down'
        ctrl.reset();self.assertFalse(ctrl.used);self.assertEqual(ctrl.pending,[]);self.assertIsNone(ctrl.previous)
        # A ledger snapshot is defined only after its initial observation.
        self.assertEqual(ctrl.ledger.visited,[]);self.assertEqual(ctrl.ledger.nodes,{})

    def test_one_intervention_and_legal_exploration_without_target_direction(self):
        ctrl=LocalLedgerOption();visited=[];position=(0,0);current=b'a'
        for step in range(4):
            cell=position[0]*10+position[1];visited.append(cell)
            obs=Observation(b'a' if cell==0 else b'b',b't',position,10,20-step,tuple(visited))
            base={'cue_action':None,'action':'right' if position==(0,0) else 'left','explorer_action':'right'}
            result=ctrl.decide(obs,base)
            if step==2:
                self.assertIsNotNone(result['interaction']);self.assertTrue(ctrl.used)
            if step==3:self.assertIsNone(result['interaction'])
            moves={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)};dr,dc=moves[result['action']]
            position=(position[0]+dr,position[1]+dc);self.assertTrue(0<=position[0]<10 and 0<=position[1]<10)

    def test_controller_reproduces_all_original_20_M1_routes(self):
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        banks,means=load_inputs();agent=make_agent(0,means,'CueFull');env=ScaledGridEnv(DATA);wrong=read(OLD/'错误目标计划.json')
        actual={r['episode_id']:r for r in lines(PREVIOUS/'阶段G/导航轨迹.jsonl') if r['arm']=='M1'}
        for task in read(PREVIOUS/'总体任务.json'):
            ep=ScaledEpisode(**task);row=row_run(agent,LocalLedgerOption(True),ep,env,banks,wrong)
            previous=actual[ep.episode_id];self.assertEqual(row['trajectory'],previous['trajectory'])
            self.assertEqual([r['base'] for r in row['decisions']],[r['base'] for r in previous['decisions']])
            self.assertEqual([r['action'] for r in row['decisions']],[r['action'] for r in previous['decisions']])


if __name__=='__main__':unittest.main()
