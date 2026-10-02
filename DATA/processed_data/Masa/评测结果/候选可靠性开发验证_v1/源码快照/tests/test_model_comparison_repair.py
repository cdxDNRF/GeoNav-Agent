import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.curl_transport import CurlTransport
from agents.vlm import APIConfig, VLMPolicy
from agents.governor import COARSE_REGIONS
from env.environment import Observation
from eval.model_comparison_repair import NonThinkingTransport


class RepairTests(unittest.TestCase):
    def setUp(self):
        NonThinkingTransport.receipts = []

    def test_option_is_added_to_actual_payload_without_mutating_original(self):
        payload = dict(model='deepseek-flash', max_tokens=1024, messages=[])
        with patch.object(CurlTransport, 'request', return_value=httpx.Response(200, json={})) as sent:
            NonThinkingTransport(60).request('POST', 'https://api.deepseek.com/v1/chat/completions', {}, json=payload)
        self.assertNotIn('thinking', payload)
        self.assertEqual(sent.call_args.kwargs['json']['thinking'], {'type': 'disabled'})
        self.assertEqual(len(NonThinkingTransport.receipts), 1)
        self.assertNotIn('messages', NonThinkingTransport.receipts[0])

    def test_repair_refuses_other_provider_and_model(self):
        with self.assertRaises(ValueError):
            NonThinkingTransport(60).request('POST', 'https://ollama.com/v1/chat/completions', {}, json={'model': 'gemma4:31b'})
        with self.assertRaises(ValueError):
            NonThinkingTransport(60).request('POST', 'https://api.deepseek.com/v1/chat/completions', {}, json={'model': 'other'})

    def test_existing_G_policy_uses_repair_transport_with_valid_response(self):
        content = json.dumps(dict(ranked_regions=list(COARSE_REGIONS), ranked_actions=['up', 'right', 'down', 'left']))
        response = httpx.Response(200, json=dict(model='deepseek-flash', choices=[dict(message={'content': content}, finish_reason='stop')]))
        with patch.object(CurlTransport, 'request', return_value=response) as sent:
            client = VLMPolicy(APIConfig('https://api.deepseek.com/v1', 'deepseek-flash', 'fake-secret', max_tokens=1024), NonThinkingTransport(60))
            regions, ranked, record = client.rank_hierarchical(Observation(b'current', b'target', (0, 0), 5, 10, (0,)))
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(sent.call_args.kwargs['json']['thinking']['type'], 'disabled')
        self.assertEqual(tuple(regions), COARSE_REGIONS)
        self.assertEqual(len(ranked), 4)
        self.assertNotIn('fake-secret', json.dumps(NonThinkingTransport.receipts))


if __name__ == '__main__':
    unittest.main()
