"""Replay-only boundary tests, using synthetic files, not historical evaluators."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from project.src.webapp.replay_v1 import ReplayStore, digest, safe_path, validate_record
from project.src.webapp.server_v1 import make_server

ROOT = Path(__file__).resolve().parents[3]
TEMP = ROOT / "平台/主动探索演示_v1/核验"


def fixture():
    task = {"start": 0, "goal": 2, "grid_size": 3, "budget": 2, "episode_id": "toy", "area": "img_toy", "protocol": "toy"}
    record = {"episode_id": "toy", "success": True, "status": "completed", "steps": 2, "sg": 0, "sg_m": 0,
        "cell_size_m": 300, "valid_travel_m": 600, "revisits": 0, "out_of_bounds": 0, "policy": "M0", "seed": 0,
        "termination": "goal_reached", "condition": "CueFull", "area": "img_toy", "protocol": "toy", "stratum": "short", "distance": 2,
        "trajectory": [{"step": 0, "patch_id": 0, "action": None, "out_of_bounds": False},
            {"step": 1, "patch_id": 1, "action": "right", "out_of_bounds": False},
            {"step": 2, "patch_id": 2, "action": "right", "out_of_bounds": False}],
        "decisions": [{"step": 1, "public_position": [0, 0], "public_visited": [0], "remaining_budget": 2, "action": "right", "explorer_action": "right", "explorer_logits": [0, 1, 0, 0], "probabilities": [0, 1, 0, 0, 0]},
            {"step": 2, "public_position": [0, 1], "public_visited": [0, 1], "remaining_budget": 1, "action": "right", "explorer_action": "right", "explorer_logits": [0, 1, 0, 0], "probabilities": [0, 1, 0, 0, 0]}]}
    return task, record


class ReplayTests(unittest.TestCase):
    def setUp(self):
        TEMP.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="临时_", dir=TEMP)
        self.root = Path(self.tmp.name)
        self.assertTrue(self.root.resolve().is_relative_to(TEMP.resolve()))
        self.store = ReplayStore.__new__(ReplayStore)
        self.store.root = self.root
        self.store.bindings, self.store.entries, self.store.images = {}, {}, {}
        self.store.catalogs = {"toy": {"id": "toy", "report": "toy.md"}}
        self.task, self.record = fixture()
        self.raw = (json.dumps(self.record) + "\n").encode()
        (self.root / "log.jsonl").write_bytes(self.raw)
        self.store._index_log("toy", "log.jsonl", digest(self.raw), {"toy": self.task}, "M0", 0, 1, 2)
        self.eid = "toy:M0:0:toy"
        for cell in range(3):
            raw = f"fake-jpeg-{cell}".encode()
            (self.root / f"{cell}.jpg").write_bytes(raw)
            self.store.images[("toy", "img_toy", cell)] = (f"{cell}.jpg", digest(raw))

    def tearDown(self):
        self.assertTrue(Path(self.tmp.name).resolve().is_relative_to(TEMP.resolve()))
        self.tmp.cleanup()

    def test_last_budget_step_success_and_meters(self):
        validate_record(self.record, self.task)
        frame = self.store.frame(self.eid, 2)
        self.assertEqual(frame["result"], {"success": True, "sg_m": 0, "termination": "goal_reached"})
        self.assertEqual(frame["movement_m"], 600)
        self.assertEqual(frame["remaining"], 0)

    def test_cursor_has_no_future_observation_or_truth(self):
        frame = self.store.frame(self.eid, 0)
        self.assertEqual(frame["visited"], [0])
        self.assertEqual(len(frame["route"]), 1)
        self.assertNotIn("diagnostic", frame)
        self.assertNotIn("result", frame)
        self.assertEqual(frame["decision"]["public_position"], [0, 0])
        self.assertEqual(self.store.frame(self.eid, 1, True)["diagnostic"]["distance_cells"], 1)

    def test_unvisited_image_denied_goal_allowed(self):
        with self.assertRaises(ValueError):
            self.store.image(self.eid, 0, "1")
        self.assertEqual(self.store.image(self.eid, 0, "goal")[0], b"fake-jpeg-2")
        self.assertEqual(self.store.image(self.eid, 1, "1")[0], b"fake-jpeg-1")

    def test_image_and_log_mutation_rejected(self):
        (self.root / "0.jpg").write_bytes(b"altered")
        with self.assertRaises(ValueError):
            self.store.image(self.eid, 0, "0")
        (self.root / "log.jsonl").write_bytes(b"x" + self.raw[1:])
        with self.assertRaises(ValueError):
            self.store.frame(self.eid, 0)

    def test_seal_or_counts_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.store.bound_bytes("log.jsonl", "0" * 64)
        with self.assertRaises(ValueError):
            self.store._index_log("toy", "log.jsonl", digest(self.raw), {"toy": self.task}, "M0", 0, 2, 2)

    def test_path_and_cursor_boundaries(self):
        for path in ("../.env", str(self.root / "0.jpg"), ""):
            with self.assertRaises(ValueError):
                safe_path(self.root, path)
        for step in (-1, 3):
            with self.assertRaises(ValueError):
                self.store.frame(self.eid, step)

    def test_public_state_alignment_is_verified(self):
        for field, value in (("public_position", [1, 1]), ("public_visited", [0, 2]), ("remaining_budget", 1)):
            bad = copy.deepcopy(self.record)
            bad["decisions"][0][field] = value
            with self.assertRaises(ValueError):
                validate_record(bad, self.task)

    def test_bad_action_sg_and_travel_rejected(self):
        for field, value in (("sg_m", 300), ("valid_travel_m", 300), ("success", False)):
            bad = copy.deepcopy(self.record)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_record(bad, self.task)
        bad = copy.deepcopy(self.record); bad["trajectory"][1]["patch_id"] = 3
        with self.assertRaises(ValueError):
            validate_record(bad, self.task)

    def test_out_of_bounds_consumes_budget_and_no_travel(self):
        task = {**self.task, "budget": 1}
        record = copy.deepcopy(self.record)
        record.update(success=False, steps=1, sg=2, sg_m=600, valid_travel_m=0, revisits=1, out_of_bounds=1, termination="budget_exhausted")
        record["trajectory"] = [record["trajectory"][0], {"step": 1, "patch_id": 0, "action": "up", "out_of_bounds": True}]
        record["decisions"] = [{**record["decisions"][0], "remaining_budget": 1, "action": "up"}]
        validate_record(record, task)

    def test_export_label_and_origin_http_restrictions(self):
        server = make_server(self.store, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            from urllib.parse import urlencode
            payload = json.load(urlopen(address + "/api/export?" + urlencode({"id": self.eid})))
            self.assertFalse(payload["new_navigation"])
            self.assertEqual(payload["saved_record"]["steps"], 2)
            for path in ("/.env", "/../../README.md", "/api/run"):
                with self.assertRaises(HTTPError) as context:
                    urlopen(address + path)
                self.assertEqual(context.exception.code, 404)
            with self.assertRaises(HTTPError) as context:
                urlopen(Request(address + "/api/catalog", headers={"Host": "attacker.example"}))
            self.assertEqual(context.exception.code, 403)
            with self.assertRaises(HTTPError) as context:
                urlopen(Request(address + "/api/run", method="POST", data=b"{}"))
            self.assertEqual(context.exception.code, 501)
        finally:
            server.shutdown(); server.server_close(); thread.join(2)


if __name__ == "__main__":
    unittest.main()
