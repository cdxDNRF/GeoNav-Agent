"""三项验证的选题、规则策略和轨迹指标测试，无云端调用。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.curl_transport import CurlTransport
from agents.exploration import FrontierPolicy
from env.environment import Observation
from env.episode import Episode
from eval.diagnostics import analyze_positions
from eval.validation_suite import select_validation, transient
from agents.vlm import APITrialError


class ValidationTests(unittest.TestCase):
    def obs(self, current, visited):
        return Observation(b"current", b"target", divmod(current, 5), 5, 10, tuple(visited))

    def test_rule_prefers_unvisited_and_ignores_target(self):
        obs = self.obs(12, [12, 7])
        policy = FrontierPolicy()
        self.assertEqual(policy.act(obs), "right")
        self.assertEqual(policy.act(replace(obs, target_image=b"different")), "right")

    def test_rule_backtracks_when_all_neighbors_visited(self):
        obs = self.obs(0, [0, 1, 5])
        self.assertEqual(FrontierPolicy().act(obs), "right")
        obs = self.obs(1, [0, 1, 5, 0, 1])
        self.assertEqual(FrontierPolicy().act(obs), "right")

    def test_rule_boundary_and_full_grid(self):
        policy = FrontierPolicy()
        for node in range(25):
            action = policy.act(self.obs(node, list(range(25))))
            row, col = divmod(node, 5)
            dr, dc = {"up":(-1,0), "right":(0,1), "down":(1,0), "left":(0,-1)}[action]
            self.assertTrue(0 <= row+dr < 5 and 0 <= col+dc < 5)

    def test_old_loop_diagnosis(self):
        result = analyze_positions([5,6,11,12,7,12,7,12,7,12,7], 13)
        self.assertEqual(result["unique_cells_including_start"], 5)
        self.assertEqual(result["revisit_steps"], 6)
        self.assertEqual(result["minimum_distance"], 1)
        self.assertEqual(result["first_adjacent_step"], 3)
        self.assertTrue(result["two_cell_cycle_present"])
        self.assertEqual(result["longest_alternating_run_moves"], 7)
        self.assertEqual(result["departures_from_adjacent"], 4)

    def test_backtrack_not_automatically_sustained_cycle(self):
        result = analyze_positions([0,1,0,5,10], 24)
        self.assertEqual(result["immediate_reversal_windows_aba"], 1)
        self.assertFalse(result["two_cell_cycle_present"])
        result = analyze_positions([0,0,0,0], 24)
        self.assertFalse(result["two_cell_cycle_present"])
        self.assertEqual(result["repeat_visit_rate"], 1)

    def test_adjacent_success_vs_departure(self):
        result = analyze_positions([10,11,12,13], 13)
        self.assertFalse(result["adjacent_but_not_reached"])
        self.assertEqual(result["minimum_distance"], 0)
        self.assertEqual(result["departures_from_adjacent"], 0)

    def test_balanced_selection_independent_of_input_order(self):
        episodes = []
        for area in range(4):
            for dist in range(4,9):
                for i in range(5):
                    episodes.append(Episode(f"{area}-{dist}-{i}", "val", f"img_{area}", 0, 24, dist))
        selected = select_validation(episodes)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected, select_validation(list(reversed(episodes))))
        self.assertEqual(len({(e.area,e.dist) for e in selected}), 20)
        with self.assertRaises(ValueError):
            select_validation([replace(e,split="test") for e in episodes])

    def test_curl_credentials_only_stdin_and_no_insecure_switch(self):
        with patch('agents.curl_transport.shutil.which', return_value='curl'), patch('agents.curl_transport.subprocess.run') as run:
            run.return_value = SimpleNamespace(returncode=0, stdout=b'{"ok":true}\n__STATUS__200', stderr=b'')
            client = CurlTransport()
            response = client.request('POST','https://example.invalid/v1', {'Authorization':'Bearer FAKE_SECRET'},json={'image':'value'})
            self.assertEqual(response.status_code,200)
            argv = run.call_args.args[0]
            self.assertNotIn('FAKE_SECRET', str(argv))
            self.assertEqual(argv, ['curl','--disable','--config','-'])
            self.assertIn(b'FAKE_SECRET',run.call_args.kwargs['input'])
            self.assertNotIn(b'insecure',run.call_args.kwargs['input'])
            self.assertNotIn(b'location',run.call_args.kwargs['input'])

    def test_curl_failure_does_not_expose_stderr(self):
        with patch('agents.curl_transport.shutil.which', return_value='curl'), patch('agents.curl_transport.subprocess.run') as run:
            run.return_value = SimpleNamespace(returncode=35, stdout=b'', stderr=b'FAKE_SECRET')
            with self.assertRaises(httpx.ConnectError) as cm:
                CurlTransport().request('GET','https://example.invalid', {})
            self.assertNotIn('FAKE_SECRET',str(cm.exception))

    def test_retry_only_transient_failure(self):
        self.assertTrue(transient(APITrialError("transport_error", {})))
        self.assertTrue(transient(APITrialError("http_error", {"http_status":429})))
        self.assertFalse(transient(APITrialError("invalid_action", {"http_status":200})))
        self.assertFalse(transient(APITrialError("http_error", {"http_status":401})))


if __name__ == "__main__":
    unittest.main()
