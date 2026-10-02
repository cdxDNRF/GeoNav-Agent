"""Offline S2 tests; synthetic maps and mocked model responses only."""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import sys
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_foundation import Fixture
from agents.governor import COARSE_REGIONS, SpatialMemoryGovernor
from agents.vlm import APIConfig, APITrialError, hierarchical_observation_request, ranked_observation_request
from env.environment import Observation
from eval import s2_suite as s2
from eval.s2_metrics import summarize, acceptance


class FakeClient:
    def __init__(self, config, transport):
        self.config = config
        self.client = self

    def close(self):
        pass

    def rank_hierarchical(self, obs):
        _, audit = hierarchical_observation_request(obs, self.config)
        ranks = ("right", "down", "left", "up")
        result = {"ranked_actions": list(ranks), "ranked_regions": list(COARSE_REGIONS)}
        return COARSE_REGIONS, ranks, dict(audit, **result, status="ok", finish_reason="stop", raw_content=json.dumps(result), latency_seconds=.01)

    def rank(self, obs):
        _, audit = ranked_observation_request(obs, self.config)
        ranks = ("right", "down", "left", "up")
        result = {"ranked_actions": list(ranks)}
        return ranks, dict(audit, **result, status="ok", finish_reason="stop", raw_content=json.dumps(result), latency_seconds=.01)


class S2Tests(Fixture):
    def setUp(self):
        super().setUp()
        self.output = self.root / "s2"
        self.output.mkdir()
        (self.output / "预登记.json").write_text('{}', encoding="utf-8")
        self.api = APIConfig("https://example.invalid/v1", "mock-model", "test-only-secret")

    def run_mock(self, arm, episodes=None):
        job = {"job_id": f"{arm}_r0", "arm": arm, "round": 0}
        with patch.object(s2, "VLMPolicy", FakeClient):
            return s2.run_job(self.root, self.output, episodes or [self.ep()], {}, job, self.api, threading.Event())

    def test_gray_is_fixed_and_does_not_mutate_original_or_environment(self):
        obs = self.env().reset(self.ep())
        masked = s2.agent_observation("G_gray_target", obs)
        self.assertNotEqual(masked.target_image, obs.target_image)
        self.assertEqual(masked.current_image, obs.current_image)
        self.assertEqual(masked.target_image, s2.gray_image())
        self.assertIs(s2.agent_observation("G", obs), obs)
        with Image.open(BytesIO(masked.target_image)) as image:
            self.assertEqual(image.getextrema(), ((128, 128), (128, 128), (128, 128)))
        self.assertEqual(masked.position, obs.position)

    def test_no_region_ignores_regions_with_same_ranked_schema(self):
        obs = self.env().reset(self.ep())
        ranked = ("right", "down", "left", "up")
        a = s2.decision_for("G_no_region", obs, ranked, COARSE_REGIONS)
        b = s2.decision_for("G_no_region", obs, ranked, tuple(reversed(COARSE_REGIONS)))
        self.assertEqual(a, b)
        self.assertEqual(a, SpatialMemoryGovernor().choose(obs, ranked))

    def test_missing_and_api_error_keep_planned_denominators(self):
        ep = self.ep()
        eps = [replace(ep, episode_id=str(i)) for i in range(3)]
        rows = [{"episode_id": "0", "completion": "completed", "evaluation": {"success": True, "sg": 0}, "actions": []},
                {"episode_id": "1", "completion": "api_error", "evaluation": None, "actions": []}]
        result = summarize(eps, rows)
        self.assertEqual(result["q"], 1/3)
        self.assertEqual(result["sr_gate"], 1/3)
        self.assertEqual(result["sr_nav"], 1)
        self.assertAlmostEqual(result["sg_gate"], 16/3)
        self.assertFalse(result["distance_complete"])

    def test_duplicate_and_unknown_records_rejected(self):
        row = {"episode_id": "not-a-task"}
        with self.assertRaises(ValueError):
            summarize([self.ep()], [row])
        with self.assertRaises(ValueError):
            summarize([self.ep()], [row, row])

    def test_registration_plan_has_exactly_17_jobs(self):
        jobs = s2.planned_jobs()
        self.assertEqual(len(jobs), 17)
        self.assertEqual(len({j["job_id"] for j in jobs}), 17)
        self.assertEqual(sum(j["arm"] in s2.VLM_ARMS for j in jobs), 12)

    def test_gate_cannot_pass_incomplete_suite(self):
        self.assertEqual(acceptance({})["status"], "pending")

    def test_gate_requires_real_visual_gain_not_only_high_absolute_SR(self):
        jobs = {}
        for job in s2.planned_jobs():
            sr = .70 if job["arm"] in ("G", "G_gray_target") else .50
            metrics = {"planned": 100, "recorded": 100, "q": 1,
                       "distance_complete": True, "sr_macro": sr, "sg_macro": 1,
                       "repeat_rate": .01,
                       "by_distance": {str(d): {"sr_gate": sr} for d in range(4, 9)}}
            jobs[job["job_id"]] = {"audit_passed": True, "metrics": metrics}
        gate = acceptance(jobs)
        self.assertEqual(gate["status"], "failed")
        self.assertFalse(gate["checks"]["G_vs_G_gray_target_gain_ge_5pp"])
        for r in range(3):
            jobs[f"G_gray_target_r{r}"]["metrics"]["sr_macro"] = .55
        self.assertTrue(acceptance(jobs)["passed"])

    def test_offline_rules_reproduce_and_existing_job_never_reexecutes(self):
        for arm in ("Frontier", "FixedRegion", "Random"):
            first = self.run_mock(arm)
            self.assertTrue(first["audit_passed"])
            self.assertEqual(first["api"]["attempts"], 0)
            second = self.run_mock(arm)
            self.assertEqual(first, second)

    def test_all_online_arms_audit_with_mocked_responses(self):
        for arm in s2.VLM_ARMS:
            result = self.run_mock(arm)
            self.assertTrue(result["audit_passed"])
            self.assertEqual(result["metrics"]["q"], 1)
            calls = s2.read_lines(self.output / f"{arm}_r0/API调用.jsonl")
            self.assertGreater(len(calls), 0)
            if arm == "G_gray_target":
                self.assertTrue(all(c["target_image_sha256"] == sha256(s2.gray_image()).hexdigest() for c in calls))

    def test_started_episode_is_not_called_again_after_interruption(self):
        ep = self.ep()
        directory = self.output / "G_r0"
        directory.mkdir()
        s2._append(directory / "请求意图.jsonl", {"episode_id": ep.episode_id, "step": 1, "attempt": 1, "request_index": 1}, "")
        result = self.run_mock("G")
        self.assertEqual(result["metrics"]["completed"], 0)
        self.assertEqual(result["metrics"]["sg_gate"], 8)
        self.assertEqual(result["api"]["attempts"], 1)
        self.assertEqual(result["api"]["responses"], 0)

    def test_auth_failure_stops_remaining_tasks_without_random_fallback(self):
        ep = self.ep()
        episodes = [replace(ep, episode_id=f"ep{i}") for i in range(2)]
        class AuthFail(FakeClient):
            def rank_hierarchical(self, obs):
                raise APITrialError("http_error", {"status": "http_error", "http_status": 401})
        job = {"job_id": "G_r0", "arm": "G", "round": 0}
        with patch.object(s2, "VLMPolicy", AuthFail):
            result = s2.run_job(self.root, self.output, episodes, {}, job, self.api, threading.Event())
        self.assertEqual(result["metrics"]["terminal_states"], {"api_error": 1, "not_run": 1})
        self.assertEqual(result["api"]["attempts"], 1)
        self.assertEqual(result["metrics"]["executed_actions"], 0)


if __name__ == "__main__":
    unittest.main()
