import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from project.src.webapp.cloud_v2 import CloudConfig, CloudPolicy, payload_for
    from project.src.webapp.batch_v2 import summarize
except ImportError:
    CloudConfig = CloudPolicy = payload_for = summarize = None


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(CloudConfig, 'API模块尚未实现')
        self.obs=SimpleNamespace(position=(0,0),grid_size=5,remaining_budget=10,visited=(0,),
            legal_actions=('right','down'),current_image=b'current',target_image=b'target')
        self.base=dict(action='right',explorer_logits=[0,1,0,0],probabilities=[0,1,0,0,0])

    def test_url_and_key_boundaries(self):
        config=CloudConfig('http://127.0.0.1:8790/v1','test','')
        self.assertNotIn('api_key',config.public())
        for url in ['http://example.com/v1','https://user:secret@example.com/v1','https://example.com/v1?key=secret']:
            with self.assertRaises(ValueError):CloudConfig(url,'test','')

    def test_payload_only_has_public_pair_and_state(self):
        data,audit=payload_for(self.obs,CloudConfig('https://example.com/v1','test',''))
        content=data['messages'][1]['content']
        self.assertEqual(sum(x['type']=='image_url' for x in content),2)
        self.assertNotIn('goal',str(audit));self.assertNotIn('source',str(audit));self.assertNotIn('episode_id',str(audit))

    def test_valid_reply_and_usage(self):
        def transport(config,payload):return {'choices':[{'message':{'content':'{"action":"down","evidence":"visible road"}'},'finish_reason':'stop'}],'usage':{'total_tokens':30}}
        p=CloudPolicy(CloudConfig('https://example.com/v1','test',''),transport=transport,max_requests=2)
        row=p.act(self.obs,self.base)
        self.assertEqual(row['action'],'down');self.assertFalse(row['cloud']['fallback'])
        self.assertEqual(row['cloud']['usage']['total_tokens'],30)

    def test_invalid_reply_fallback_then_circuit_breaker(self):
        p=CloudPolicy(CloudConfig('https://example.com/v1','test',''),transport=lambda *_:{},max_requests=4)
        for _ in range(3):self.assertTrue(p.act(self.obs,self.base)['cloud']['fallback'])
        with self.assertRaises(ValueError):p.act(self.obs,self.base)
        self.assertEqual(p.requests,3)

    def test_request_cap_and_private_error_are_not_echoed(self):
        token='sk-'+'X'*40
        def bad(*_):raise RuntimeError(token)
        p=CloudPolicy(CloudConfig('https://example.com/v1','test',token),transport=bad,max_requests=1)
        row=p.act(self.obs,self.base);self.assertNotIn(token,str(row))
        with self.assertRaises(ValueError):p.act(self.obs,self.base)

    def test_truncated_duplicate_and_illegal_actions_rejected(self):
        for text,reason in [('{"action":"right","action":"down"}','stop'),('{"action":"left"}','stop'),('{"action":"down"}','length')]:
            def transport(*_,t=text,r=reason):return {'choices':[{'message':{'content':t},'finish_reason':r}]}
            p=CloudPolicy(CloudConfig('https://example.com/v1','test',''),transport=transport)
            self.assertTrue(p.act(self.obs,self.base)['cloud']['fallback'])

    def test_planned_denominator_and_incomplete_sg(self):
        a={'status':'completed','result':{'success':True,'sg_m':0,'valid_travel_m':300},'decisions':[]}
        b={'status':'cancelled','result':None,'decisions':[]}
        r=summarize([a,b],4)
        self.assertEqual(r['sr_planned'],.25);self.assertEqual(r['completed'],1)
        self.assertIsNone(r['sg_m']);self.assertFalse(r['complete'])


if __name__=='__main__':unittest.main()
