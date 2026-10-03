import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from agents.cloud_gateway import GatewayConfig,GatewayTransport,GATEWAY,MODELS,registered_options
from agents.vlm import APIConfig,VLMPolicy
from agents.protected_ledger_option import ProtectedLedgerOption
from env.environment import Observation


class GatewayTests(unittest.TestCase):
    def test_registry_and_no_default_change(self):
        self.assertFalse(registered_options()['default_changed'])
        for model in MODELS:self.assertEqual(GatewayConfig(model).model,model)

    def test_http_exception_is_exact_loopback_only(self):
        for url in ['http://remote.example/v1', 'http://localhost:8790/v1', GATEWAY+'/', GATEWAY+'?x=1', 'http://u:p@127.0.0.1:8790/v1']:
            with self.assertRaises(ValueError):GatewayConfig(MODELS[0],base_url=url)
        with self.assertRaises(ValueError):APIConfig(GATEWAY,MODELS[0],'dummy')

    def test_no_redirect_or_unregistered_route(self):
        seen=[]
        def respond(request):
            seen.append(str(request.url));return httpx.Response(302,headers={'location':'https://example.com/'})
        transport=GatewayTransport(GatewayConfig(MODELS[0]),httpx.MockTransport(respond))
        try:
            response=transport.request('GET',GATEWAY+'/models');self.assertEqual(response.status_code,302)
            self.assertEqual(len(seen),1)
            with self.assertRaises(ValueError):transport.request('GET',GATEWAY+'/models?x=1')
            with self.assertRaises(ValueError):transport.request('POST','http://example.com/v1/chat/completions')
        finally:transport.close()

    def test_existing_strict_action_parser_works_with_new_option(self):
        transport=GatewayTransport(GatewayConfig(MODELS[0]),httpx.MockTransport(lambda request:httpx.Response(200,json={
            'model':MODELS[0],'choices':[{'message':{'content':'{"action":"right"}'},'finish_reason':'stop'}]})))
        try:
            action,record=VLMPolicy(GatewayConfig(MODELS[0]),transport).act(Observation(b'a',b'b',(0,0),5,10,(0,)))
            self.assertEqual(action,'right');self.assertEqual(record['status'],'ok')
        finally:transport.close()


class ProtectionTests(unittest.TestCase):
    def primed(self,pending):
        ctrl=ProtectedLedgerOption();obs=Observation(b'a',b't',(1,1),10,20,(11,))
        ctrl.decide(obs,dict(cue_action=None,action='right',explorer_action='right'))
        ctrl.used=True;ctrl.pending=pending
        return ctrl,Observation(b'b',b't',(1,2),10,19,(11,12))

    def test_fresh_original_veto_cancels_option_and_tracks_actual_action(self):
        ctrl,obs=self.primed(['down','left'])
        r=ctrl.decide(obs,dict(cue_action=None,action='right',explorer_action='right'))
        self.assertTrue(r['protection_veto']);self.assertEqual(r['action'],'right');self.assertEqual(ctrl.previous,'right')
        self.assertEqual(ctrl.pending,[]);self.assertTrue(ctrl.used)

    def test_original_revisit_can_be_overridden(self):
        ctrl,obs=self.primed(['down'])
        r=ctrl.decide(obs,dict(cue_action=None,action='left',explorer_action='left'))
        self.assertFalse(r['protection_veto']);self.assertEqual(r['action'],'down')

    def test_accepted_cue_keeps_priority_even_if_pointing_to_known_cell(self):
        ctrl,obs=self.primed(['down'])
        r=ctrl.decide(obs,dict(cue_action='left',action='left',explorer_action='right'))
        self.assertEqual(r['action'],'left');self.assertEqual(r['control'],'accepted_cue');self.assertFalse(r['protection_veto'])
        self.assertEqual(ctrl.pending,[])

    def test_reset_and_shared_move_do_not_veto(self):
        ctrl,obs=self.primed(['right','down'])
        r=ctrl.decide(obs,dict(cue_action=None,action='right',explorer_action='right'))
        self.assertFalse(r['protection_veto']);self.assertEqual(ctrl.pending,['down'])
        ctrl.reset();self.assertFalse(ctrl.used);self.assertIsNone(ctrl.previous)


if __name__=='__main__':unittest.main()
