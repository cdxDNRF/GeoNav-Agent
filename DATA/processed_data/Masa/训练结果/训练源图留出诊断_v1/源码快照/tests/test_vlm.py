"""API适配离线测试：全部Mock，不发云端请求、不加载真实Key。"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.vlm import (APIConfig, APITrialError, VLMPolicy, observation_request,
                         ranked_observation_request, parse_ranked_actions,
                         confirmation_ranked_observation_request,
                         parse_ranked_confirmation, hierarchical_observation_request,
                         parse_hierarchical_ranked)
from agents.governor import COARSE_REGIONS
from env.environment import Observation
from eval.pilot import aggregate, call_statistics, select_pilot, run_policy
from eval.replay_responses import replay
from env.episode import Episode


class VLMTests(unittest.TestCase):
    def setUp(self):
        self.config = APIConfig("https://example.invalid/v1", "test-vision", "secret-for-test")
        self.obs = Observation(b"current", b"target", (0, 0), 5, 10, (0,))

    def policy(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return VLMPolicy(self.config, client)

    def ranked_response(self, content='{"ranked_actions":["right","up","left","down"]}', reason="stop"):
        return {"model": "test-vision", "choices": [{"message": {"content": content},
                    "finish_reason": reason}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 7, "total_tokens": 307}}

    def confirmation_response(self, content='{"ranked_actions":["right","up","left","down"],"target_evidence":"low"}', reason="stop"):
        return {"model": "test-vision", "choices": [{"message": {"content": content},
                    "finish_reason": reason}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 9, "total_tokens": 309}}

    def hierarchical_response(self, content=None, reason="stop"):
        content = content or json.dumps({"ranked_regions": list(COARSE_REGIONS),
                                         "ranked_actions": ["right", "up", "left", "down"]})
        return {"model": "test-vision", "choices": [{"message": {"content": content},
                    "finish_reason": reason}],
                "usage": {"prompt_tokens": 350, "completion_tokens": 20, "total_tokens": 370}}

    def response(self, content='{"action":"right"}', reason="stop"): 
        return {"model": "test-vision", "choices": [{"message": {"content": content},
                    "finish_reason": reason}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 7, "total_tokens": 307}}

    def test_request_contains_two_images_only_public_state(self):
        payload, audit = observation_request(self.obs, self.config)
        content = payload["messages"][1]["content"]
        self.assertEqual(sum(x["type"] == "image_url" for x in content), 2)
        self.assertEqual(set(audit["public_state"]), {"position_row_col", "grid_size", "remaining_budget", "visited_cell_ids", "legal_actions"})
        self.assertNotIn("secret-for-test", json.dumps(payload))
        self.assertNotIn("data:image", json.dumps(audit))
        self.assertNotIn("secret-for-test", repr(self.config))

    def test_ranked_request_has_same_two_images_and_new_schema(self):
        payload, audit = ranked_observation_request(self.obs, self.config)
        self.assertEqual(sum(x["type"] == "image_url" for x in payload["messages"][1]["content"]), 2)
        self.assertEqual(audit["output_schema"], "ranked_actions")
        self.assertEqual(audit["prompt_version"], "aerial-pair-ranked-actions-v1")
        self.assertNotIn("secret-for-test", json.dumps(payload))

    def test_ranked_output_is_exact_permutation(self):
        p = self.policy(lambda req: httpx.Response(200, json=self.ranked_response()))
        ranked, record = p.rank(self.obs)
        self.assertEqual(ranked, ("right", "up", "left", "down"))
        self.assertEqual(record["output_schema"], "ranked_actions")
        self.assertEqual(record["ranked_actions"], list(ranked))

    def test_ranked_parser_rejects_missing_duplicate_invalid_extra(self):
        for content in ('{"ranked_actions":["up","right"]}',
                        '{"ranked_actions":["up","up","left","down"]}',
                        '{"ranked_actions":["up","right","left","stop"]}',
                        '{"ranked_actions":["up","right","left","down"],"score":1}',
                        'explanation {"ranked_actions":["up","right","left","down"]}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_ranked_actions(content, self.obs.legal_actions, "stop")

    def test_ranked_fence_and_truncation(self):
        ranked, normalized = parse_ranked_actions('```json\n{"ranked_actions":["left","up","right","down"]}\n```', self.obs.legal_actions, "stop")
        self.assertTrue(normalized)
        self.assertEqual(ranked[0], "left")
        with self.assertRaises(ValueError):
            parse_ranked_actions('{"ranked_actions":["up","right","down","left"]}', self.obs.legal_actions, "length")

    def test_confirmation_request_and_parser(self):
        payload, audit = confirmation_ranked_observation_request(self.obs, self.config)
        self.assertEqual(sum(x["type"] == "image_url" for x in payload["messages"][1]["content"]), 2)
        self.assertEqual(audit["output_schema"], "ranked_actions_with_target_evidence")
        self.assertEqual(audit["prompt_version"], "aerial-pair-ranked-confirmation-v1")
        ranked, evidence, normalized = parse_ranked_confirmation(
            '{"ranked_actions":["left","up","right","down"],"target_evidence":"high"}',
            self.obs.legal_actions, "stop")
        self.assertEqual(ranked[0], "left")
        self.assertEqual(evidence, "high")
        self.assertFalse(normalized)

    def test_confirmation_parser_rejects_invalid_evidence_or_extra_key(self):
        for content in (
            '{"ranked_actions":["up","right","left","down"],"target_evidence":"certain"}',
            '{"ranked_actions":["up","right","left","down"],"target_evidence":"high","score":1}',
            '{"ranked_actions":["up","right"],"target_evidence":"low"}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_ranked_confirmation(content, self.obs.legal_actions, "stop")

    def test_confirmation_policy_logs_evidence(self):
        p = self.policy(lambda req: httpx.Response(200, json=self.confirmation_response(
            '{"ranked_actions":["left","up","right","down"],"target_evidence":"high"}')))
        ranked, evidence, record = p.rank_with_confirmation(self.obs)
        self.assertEqual(ranked[0], "left")
        self.assertEqual(evidence, "high")
        self.assertEqual(record["output_schema"], "ranked_actions_with_target_evidence")
        self.assertEqual(record["target_evidence"], "high")

    def test_hierarchical_request_and_parser(self):
        payload, audit = hierarchical_observation_request(self.obs, self.config)
        self.assertEqual(sum(x["type"] == "image_url" for x in payload["messages"][1]["content"]), 2)
        self.assertEqual(audit["output_schema"], "hierarchical_ranked_regions_actions")
        self.assertEqual(audit["prompt_version"], "aerial-pair-hierarchical-search-v1")
        regions, ranked, normalized = parse_hierarchical_ranked(
            json.dumps({"ranked_regions": list(reversed(COARSE_REGIONS)),
                        "ranked_actions": ["left", "up", "right", "down"]}),
            self.obs.legal_actions, COARSE_REGIONS, "stop")
        self.assertEqual(regions[0], "r2c2")
        self.assertEqual(ranked[0], "left")
        self.assertFalse(normalized)

    def test_hierarchical_parser_rejects_invalid_permutations(self):
        for content in (
            json.dumps({"ranked_regions": list(COARSE_REGIONS[:-1]),
                        "ranked_actions": ["up", "right", "left", "down"]}),
            json.dumps({"ranked_regions": list(COARSE_REGIONS),
                        "ranked_actions": ["up", "up", "left", "down"]}),
            json.dumps({"ranked_regions": list(COARSE_REGIONS),
                        "ranked_actions": ["up", "right", "left", "down"], "extra": 1}),
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_hierarchical_ranked(content, self.obs.legal_actions, COARSE_REGIONS, "stop")

    def test_hierarchical_policy_logs_regions_and_actions(self):
        p = self.policy(lambda req: httpx.Response(200, json=self.hierarchical_response(
            json.dumps({"ranked_regions": list(reversed(COARSE_REGIONS)),
                        "ranked_actions": ["left", "up", "right", "down"]}))))
        regions, ranked, record = p.rank_hierarchical(self.obs)
        self.assertEqual(regions[0], "r2c2")
        self.assertEqual(ranked[0], "left")
        self.assertEqual(record["output_schema"], "hierarchical_ranked_regions_actions")
        self.assertEqual(record["ranked_regions"], list(regions))

    def test_valid_output_and_usage(self):
        def handler(request):
            self.assertEqual(request.headers["Authorization"], "Bearer secret-for-test")
            self.assertEqual(str(request.url), "https://example.invalid/v1/chat/completions")
            return httpx.Response(200, json=self.response())
        action, record = self.policy(handler).act(self.obs)
        self.assertEqual(action, "right")
        self.assertEqual(record["usage"]["total_tokens"], 307)
        self.assertEqual(record["status"], "ok")

    def test_refuses_invalid_action_and_no_retry(self):
        for content in ('{"action":"stop"}', '{"action":"right","goal":4}',
                        'explanation\n{"action":"up"}', '{"action":"up","action":"right"}', None):
            count = []
            def handler(request):
                count.append(1)
                return httpx.Response(200, json=self.response(content))
            with self.subTest(content=content), self.assertRaises(APITrialError):
                self.policy(handler).act(self.obs)
            self.assertEqual(len(count), 1)

    def test_complete_json_fence_is_normalized_without_action_repair(self):
        p = self.policy(lambda req: httpx.Response(200, json=self.response('```json\n{"action": "left"}\n```')))
        action, record = p.act(self.obs)
        self.assertEqual(action, "left")
        self.assertEqual(record["format_normalization"], "markdown_fence")
        p = self.policy(lambda req: httpx.Response(200, json=self.response('```json\n{"action": "stop"}\n```')))
        with self.assertRaises(APITrialError):
            p.act(self.obs)

    def test_truncated_output_is_failure(self):
        p = self.policy(lambda req: httpx.Response(200, json=self.response(reason="length")))
        with self.assertRaises(APITrialError):
            p.act(self.obs)

    def test_timeout_record_has_no_secret(self):
        def handler(request):
            raise httpx.ReadTimeout("secret-for-test", request=request)
        with self.assertRaises(APITrialError) as cm:
            self.policy(handler).act(self.obs)
        self.assertNotIn("secret-for-test", json.dumps(cm.exception.record))
        self.assertEqual(cm.exception.record["status"], "transport_error")

    def test_http_error_redacted(self):
        p = self.policy(lambda req: httpx.Response(401, json={"error": {"message": "bad secret-for-test"}}))
        with self.assertRaises(APITrialError) as cm:
            p.act(self.obs)
        self.assertNotIn("secret-for-test", json.dumps(cm.exception.record))
        self.assertIn("[REDACTED]", cm.exception.record["error_message"])

    def test_malformed_response_schema(self):
        for data in ([], {"choices": []}, {"choices": [None]}, {"choices": [{"message": []}]}):
            with self.subTest(data=data), self.assertRaises(APITrialError):
                self.policy(lambda req: httpx.Response(200, json=data)).act(self.obs)

    def test_reasoning_not_logged(self):
        data = self.response()
        data["choices"][0]["message"]["reasoning"] = "hidden internal reasoning"
        _, record = self.policy(lambda req: httpx.Response(200, json=data)).act(self.obs)
        self.assertTrue(record["reasoning_present"])
        self.assertNotIn("hidden internal reasoning", json.dumps(record))

    def test_missing_usage_is_not_zero(self):
        result = call_statistics([{"status": "ok", "usage": None, "latency_seconds": 2}])
        self.assertIsNone(result["reported_token_totals"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["usage_reported_requests"], 0)

    def test_api_failure_not_disguised_as_navigation_failure(self):
        result = aggregate([{"completion": "api_error", "evaluation": None},
                            {"completion": "not_run", "evaluation": None}])
        self.assertEqual(result["api_errors"], 1)
        self.assertEqual(result["not_run"], 1)
        self.assertIsNone(result["metrics_completed_only"])

    def test_selection_independent_of_order(self):
        episodes = [Episode("a", "val", "img_0", 0, 4, 4),
                    Episode("b", "val", "img_0", 0, 18, 6),
                    Episode("c", "val", "img_0", 0, 4, 4)]
        self.assertEqual(select_pilot(episodes), select_pilot(list(reversed(episodes))))

    def test_runner_halts_api_error_without_fallback_move(self):
        class FakeEnv:
            done = False
            steps = 0
            def reset(inner, ep):
                return self.obs
            def step(inner, action):
                inner.steps += 1
                raise AssertionError("no fallback movement allowed")
        ep = Episode("a", "val", "img_0", 0, 4, 4)
        env = FakeEnv()
        calls = []
        def failing(obs):
            raise APITrialError("http_error", {"status": "http_error", "http_status": 429})
        result = run_policy(env, ep, "vlm", failing, lambda call, eid: calls.append(call))
        self.assertEqual(result["completion"], "api_error")
        self.assertEqual(env.steps, 0)
        self.assertIsNone(result["evaluation"])
        self.assertEqual(len(calls), 1)

    def test_rejects_credential_in_url_and_insecure_url(self):
        for url in ("http://example.invalid/v1", "https://key:password@example.invalid/v1"):
            with self.assertRaises(ValueError):
                APIConfig(url, "model", "key")

    def test_offline_replay_supports_ranked_response(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "calls.jsonl"
            output = Path(tmp) / "replay.json"
            source.write_text(json.dumps({"call_index": 1, "status": "ok", "output_schema": "ranked_actions",
                                          "raw_content": '{"ranked_actions":["right","up","left","down"]}',
                                          "finish_reason": "stop"}) + "\n", encoding="utf-8")
            summary = replay(source, output)
            self.assertEqual(summary["valid_responses"], 1)
            self.assertEqual(summary["results"][0]["ranked_actions"][0], "right")

    def test_models_probe(self):
        p = self.policy(lambda req: httpx.Response(200, json={"data": [{"id": "test-vision"}]}))
        self.assertTrue(p.probe()["requested_model_listed"])


if __name__ == "__main__":
    unittest.main()
