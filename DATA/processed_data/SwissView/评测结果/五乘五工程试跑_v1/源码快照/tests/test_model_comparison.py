"""Provider isolation and actual runner failure semantics, without network calls."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.governor import COARSE_REGIONS
from agents.vlm import APIConfig
from data.make_episodes import MASA
from env.episode import Episode
from eval.model_comparison import load_candidate, paired_results
from eval.s2_suite import run_job, read_lines
from eval.validation_suite import select_validation


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = select_validation([Episode.from_dict(r) for r in read_lines(MASA / '任务清单_v2/episodes_val.jsonl')])

    def test_explicit_provider_ignores_foreign_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.env'
            path.write_text('VLM_BASE_URL=https://example.invalid/v1\nVLM_MODEL=vision-a\nVLM_API_KEY=secret-a\n', encoding='utf-8')
            with patch.dict(os.environ, {'VLM_BASE_URL': 'https://other.invalid', 'VLM_MODEL': 'wrong', 'VLM_API_KEY': 'secret-b'}):
                result = load_candidate(path, 'https://example.invalid/v1', 'vision-a')
            self.assertEqual(result.api_key, 'secret-a')
            self.assertEqual(result.max_tokens, 1024)
            self.assertEqual(result.timeout, 60)
            with self.assertRaises(ValueError):
                load_candidate(path, 'https://other.invalid', 'vision-a')

    def run_mock(self, handler, count):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        (output / '预登记.json').write_text('{}', encoding='utf-8')
        client = httpx.Client(transport=httpx.MockTransport(handler))
        api = APIConfig('https://example.invalid/v1', 'vision-a', 'secret-a', max_tokens=1024)
        stop = Event()
        with patch('eval.s2_suite.CurlTransport', return_value=client):
            result = run_job(MASA, output, self.episodes[:count], {}, dict(job_id='Mock', arm='G', round=0), api, stop)
        return result, read_lines(output / 'Mock/任务结果.jsonl'), read_lines(output / 'Mock/API调用.jsonl'), stop

    def test_auth_failure_stops_only_provider_with_all_terminal_records(self):
        result, records, calls, stop = self.run_mock(lambda req: httpx.Response(401, json={'error': {'code': 'invalid'}}), 4)
        self.assertEqual([r['completion'] for r in records], ['api_error', 'not_run', 'not_run', 'not_run'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['metrics']['planned'], 4)
        self.assertEqual(result['metrics']['sr_gate'], 0)
        self.assertEqual(result['metrics']['sg_gate'], 8)
        self.assertTrue(stop.is_set())
        self.assertFalse(Event().is_set())

    def test_truncation_has_no_fallback_or_retry_and_three_failure_circuit(self):
        body = dict(model='vision-a', choices=[dict(message=dict(content=''), finish_reason='length')])
        result, records, calls, stop = self.run_mock(lambda req: httpx.Response(200, json=body), 4)
        self.assertEqual([r['completion'] for r in records], ['api_error'] * 3 + ['not_run'])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(not r['actions'] for r in records))
        self.assertTrue(stop.is_set())
        self.assertTrue(result['audit_passed'])

    def test_real_G_runner_and_parser_replay_under_mock_api(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload['model'], 'vision-a')
            self.assertEqual(payload['max_tokens'], 1024)
            self.assertEqual(sum(c['type'] == 'image_url' for c in payload['messages'][1]['content']), 2)
            content = json.dumps(dict(ranked_regions=list(COARSE_REGIONS), ranked_actions=['up', 'right', 'down', 'left']))
            return httpx.Response(200, json=dict(model='vision-a', choices=[dict(message=dict(content=content), finish_reason='stop')]))
        result, records, calls, stop = self.run_mock(handler, 2)
        self.assertEqual(result['metrics']['completed'], 2)
        self.assertTrue(result['audit_passed'])
        self.assertEqual(len(calls), sum(len(r['actions']) for r in records))
        self.assertFalse(stop.is_set())

    def test_incomplete_pair_never_counted_as_navigation_loss(self):
        episodes = self.episodes[:2]
        a = [dict(episode_id=e.episode_id, completion='completed', evaluation={'success': True}) for e in episodes]
        b = [dict(episode_id=episodes[0].episode_id, completion='api_error', evaluation=None),
             dict(episode_id=episodes[1].episode_id, completion='completed', evaluation={'success': False})]
        result = paired_results(episodes, {'Gemma': a, 'DeepSeek': b})
        self.assertEqual(result['counts']['incomplete_pair'], 1)
        self.assertEqual(result['counts']['gemma_only'], 1)
        self.assertEqual(result['common_completed'], 1)


if __name__ == '__main__':
    unittest.main()
