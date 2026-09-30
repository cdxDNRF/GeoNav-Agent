"""用独立合成地图测试规则；不修改真实原始数据或既有试卷。"""
import csv
from dataclasses import asdict, FrozenInstanceError
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_episodes import generate, pairs_at_distance
from env.environment import GridWorldEnv, Observation
from env.episode import ACTIONS, Episode, PROTOCOL, inspect_area, load_episodes, manhattan
from eval.evaluate import evaluate, metrics, verify_task_file


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for split in ("train", "val", "test"):
            folder = self.root / "patches" / split / "img_0"
            folder.mkdir(parents=True)
            rows.append({"split": split, "img_id": "img_0", "source_tile": f"{split}.png"})
            for i in range(25):
                image = Image.new("RGB", (300, 300), (i * 7, 30, 80))
                exif = Image.Exif()
                exif[270] = f"SECRET_goal_patch_{i}"
                image.save(folder / f"patch_{i}.jpg", exif=exif)
        with (self.root / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["split", "img_id", "source_tile"])
            writer.writeheader()
            writer.writerows(rows)

    def ep(self, start=0, goal=24, budget=10):
        return Episode("example", "test", "img_0", start, goal, manhattan(start, goal), budget)

    def env(self):
        return GridWorldEnv(self.root)


class EnvironmentTests(Fixture):
    def test_agent_payload_has_no_answers_or_filenames(self):
        env = self.env()
        obs = env.reset(self.ep())
        self.assertEqual(set(asdict(obs)), {"current_image", "target_image", "position", "grid_size",
                                           "remaining_budget", "visited", "legal_actions", "media_type"})
        for payload in (obs.current_image, obs.target_image):
            self.assertIsInstance(payload, bytes)
            self.assertNotIn(b"SECRET", payload)
            with Image.open(BytesIO(payload)) as image:
                self.assertEqual(image.size, (300, 300))
                self.assertFalse(image.getexif())
                self.assertFalse(image.info)
        _, _, info = env.step("right")
        self.assertEqual(set(asdict(info)), {"out_of_bounds", "revisited", "step_count"})
        self.assertNotIn("goal", str(asdict(info)))

    def test_four_actions_every_cell(self):
        env = self.env()
        for start in range(25):
            for action, (dr, dc) in ACTIONS.items():
                env.reset(self.ep(start, (start + 12) % 25))
                row, col = divmod(start, 5)
                outside = not (0 <= row + dr < 5 and 0 <= col + dc < 5)
                obs, _, info = env.step(action)
                self.assertEqual(obs.position, (row, col) if outside else (row + dr, col + dc))
                self.assertEqual(info.out_of_bounds, outside)
                self.assertEqual(obs.remaining_budget, 9)

    def test_boundary_consumes_step_and_counts_repeat(self):
        env = self.env()
        env.reset(self.ep(budget=1))
        obs, done, info = env.step("up")
        self.assertEqual(obs.position, (0, 0))
        self.assertTrue(done and info.out_of_bounds and info.revisited)
        result = env.evaluator_result()
        self.assertFalse(result["success"])
        self.assertEqual(result["repeat_visit_rate"], 1)
        self.assertEqual(result["sg"], 8)

    def test_last_step_success_beats_budget(self):
        env = self.env()
        env.reset(self.ep(goal=1, budget=1))
        _, done, _ = env.step("right")
        result = env.evaluator_result()
        self.assertTrue(done and result["success"])
        self.assertEqual(result["termination"], "goal_reached")
        self.assertEqual(result["sg"], 0)

    def test_oracle_all_ordered_pairs(self):
        env = self.env()
        for start in range(25):
            for goal in range(25):
                if start == goal:
                    continue
                distance = manhattan(start, goal)
                env.reset(self.ep(start, goal, distance))
                row, col = divmod(start, 5)
                gr, gc = divmod(goal, 5)
                actions = (["down" if gr > row else "up"] * abs(gr - row)
                           + ["right" if gc > col else "left"] * abs(gc - col))
                for action in actions:
                    env.step(action)
                self.assertTrue(env.done)
                result = env.evaluator_result()
                self.assertEqual((result["success"], result["sg"], result["steps"], result["revisits"]),
                                 (True, 0, distance, 0))

    def test_invalid_action_does_not_mutate(self):
        env = self.env()
        env.reset(self.ep())
        for action in ("stop", "diagonal", None, 1):
            with self.assertRaises(ValueError):
                env.step(action)
        obs, _, info = env.step("right")
        self.assertEqual((info.step_count, obs.remaining_budget), (1, 9))

    def test_lifecycle_and_reset_isolation(self):
        env = self.env()
        with self.assertRaises(RuntimeError):
            env.step("up")
        with self.assertRaises(RuntimeError):
            env.evaluator_result()
        env.reset(self.ep(budget=1))
        with self.assertRaises(RuntimeError):
            env.evaluator_result()
        env.step("up")
        with self.assertRaises(RuntimeError):
            env.step("right")
        obs = env.reset(self.ep(start=12, goal=24))
        self.assertEqual(obs.visited, (12,))
        self.assertEqual(obs.remaining_budget, 10)
        with self.assertRaises(FrozenInstanceError):
            obs.position = (0, 0)

    def test_result_is_detached(self):
        env = self.env()
        env.reset(self.ep(budget=1))
        env.step("up")
        result = env.evaluator_result()
        result["trajectory"][0]["patch_id"] = 999
        self.assertEqual(env.evaluator_result()["trajectory"][0]["patch_id"], 0)

    def test_bad_episode_rejected(self):
        base = asdict(self.ep())
        for changes in ({"start": -1}, {"goal": 25}, {"goal": 0, "dist": 0},
                        {"budget": 0}, {"budget": True}, {"dist": 1}, {"grid_size": 10},
                        {"split": "bad"}, {"area": "../img_0"}, {"goal_rc": [1, 1]},
                        {"protocol": "unknown"}, {"hidden": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.env().reset(base | changes)

    def test_modified_image_cannot_silently_use_cache(self):
        env = self.env()
        ep = self.ep()
        env.reset(ep)
        path = self.root / "patches/test/img_0/patch_0.jpg"
        Image.new("RGB", (300, 300), "red").save(path)
        with self.assertRaises(ValueError):
            env.reset(ep)

    def test_missing_and_wrong_size_patch(self):
        folder = self.root / "patches/test/img_0"
        Image.new("RGB", (150, 150)).save(folder / "patch_0.jpg")
        with self.assertRaises(ValueError):
            self.env().reset(self.ep())
        (folder / "patch_0.jpg").unlink()
        with self.assertRaises(ValueError):
            inspect_area(self.root, "test", "img_0")


class GenerationTests(Fixture):
    def test_distance_pool_counts(self):
        self.assertEqual({d: len(pairs_at_distance(5, d)) for d in range(4, 9)},
                         {4: 120, 5: 80, 6: 40, 7: 16, 8: 4})

    def test_split_order_and_single_split_identical(self):
        one, two, three = [self.root / name for name in ("one", "two", "three")]
        generate(self.root, one, ["val", "test", "train"])
        generate(self.root, two, ["train", "test", "val"])
        generate(self.root, three, ["test"])
        for name in ("episodes_test.jsonl", "manifest_test.json"):
            self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())
            self.assertEqual((one / name).read_bytes(), (three / name).read_bytes())
        generate(self.root, one, ["test", "val", "train"])

    def test_counts_distances_and_duplicate_sampling(self):
        out = self.root / "tasks"
        generate(self.root, out, ["test"])
        episodes, manifest = verify_task_file(self.root, out / "episodes_test.jsonl")
        self.assertEqual(len(episodes), 25)
        self.assertEqual(manifest["area_count"], 1)
        self.assertGreaterEqual(manifest["duplicate_draws_per_distance"]["8"], 1)
        for ep in episodes:
            self.assertEqual(ep.dist, manhattan(ep.start, ep.goal))

    def test_refuses_changed_output_without_partial_write(self):
        out = self.root / "tasks"
        generate(self.root, out, ["test"])
        previous = (out / "episodes_test.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            generate(self.root, out, ["val", "test"], seed=99)
        self.assertEqual(previous, (out / "episodes_test.jsonl").read_bytes())
        self.assertFalse((out / "episodes_val.jsonl").exists())

    def test_bad_generation_args(self):
        for kwargs in ({"splits": []}, {"splits": ["test", "test"]}, {"budget": 0},
                       {"n_per_dist": 0}, {"seed": -1}, {"splits": ["unknown"]}):
            with self.assertRaises(ValueError):
                generate(self.root, self.root / "bad", **kwargs)

    def test_episode_tamper_detected(self):
        out = self.root / "tasks"
        generate(self.root, out, ["test"])
        path = out / "episodes_test.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"budget": 10', b'"budget": 11', 1))
        with self.assertRaises(ValueError):
            verify_task_file(self.root, path)

    def test_patch_tamper_detected(self):
        out = self.root / "tasks"
        generate(self.root, out, ["test"])
        Image.new("RGB", (300, 300), "red").save(self.root / "patches/test/img_0/patch_0.jpg")
        with self.assertRaises(ValueError):
            verify_task_file(self.root, out / "episodes_test.jsonl")

    def test_split_substitution_rejected(self):
        out = self.root / "tasks"
        generate(self.root, out, ["val", "test"])
        path = out / "episodes_test.jsonl"
        path.write_bytes((out / "episodes_val.jsonl").read_bytes())
        with self.assertRaises(ValueError):
            verify_task_file(self.root, path, "test")
        with self.assertRaises(ValueError):
            evaluate(self.root, out, ["test"], [0], self.root / "results")

    def test_manifest_code_and_rules_tampering_rejected(self):
        out = self.root / "tasks"
        generate(self.root, out, ["test"])
        path = out / "manifest_test.json"
        original = path.read_text(encoding="utf-8")
        for field, value in (("generator_sha256", "bad"), ("validator_sha256", "bad"),
                             ("boundary", "wrap"), ("termination", "stop_only"),
                             ("distances", [4, 5]), ("sampling", "without_replacement")):
            changed = json.loads(original)
            changed[field] = value
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_task_file(self.root, out / "episodes_test.jsonl")
        path.write_text(original, encoding="utf-8")

    def test_duplicate_episode_ids_rejected(self):
        path = self.root / "duplicate.jsonl"
        line = json.dumps(asdict(self.ep())) + "\n"
        path.write_text(line * 2, encoding="utf-8")
        with self.assertRaises(ValueError):
            load_episodes(path)


class EvaluationTests(Fixture):
    def test_metric_denominators(self):
        records = [{"success": True, "sg": 0, "steps": 2, "revisits": 0,
                    "repeat_visit_rate": 0, "out_of_bounds": 0},
                   {"success": False, "sg": 4, "steps": 4, "revisits": 2,
                    "repeat_visit_rate": .5, "out_of_bounds": 1}]
        result = metrics(records)
        self.assertEqual(result["sr"], .5)
        self.assertEqual(result["mean_sg_all_episodes"], 2)
        self.assertAlmostEqual(result["repeat_visit_rate_micro"], 2 / 6)
        self.assertEqual(result["repeat_visit_rate_macro"], .25)
        with self.assertRaises(ValueError):
            metrics([])

    def test_random_evaluation_reproducible_and_logs_complete(self):
        tasks = self.root / "tasks"
        generate(self.root, tasks, ["val", "test"])
        out = self.root / "result"
        summary = evaluate(self.root, tasks, ["val", "test"], [0, 1], out)
        again = evaluate(self.root, tasks, ["test", "val"], [1, 0], out)
        self.assertEqual(summary, again)
        lines = (out / "随机基线_轨迹.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 100)
        for line in lines:
            result = json.loads(line)
            self.assertEqual(len(result["trajectory"]), result["steps"] + 1)
            self.assertLessEqual(result["steps"], result["budget"])
            final = result["trajectory"][-1]["patch_id"]
            self.assertEqual(result["sg"], manhattan(final, result["goal"]))
            self.assertEqual(result["success"], result["sg"] == 0)
        digest = sha256((out / "随机基线_轨迹.jsonl").read_bytes()).hexdigest()
        self.assertEqual(summary["trajectory_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
