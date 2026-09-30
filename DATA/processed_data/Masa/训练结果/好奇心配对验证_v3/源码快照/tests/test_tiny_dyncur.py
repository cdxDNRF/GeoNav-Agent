import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.episode import Episode
from train.dyncur_tiny import (EmbeddingStore, FEATURE_DIM, TrainConfig, TinyPolicy,
                               _transition_reward, curiosity_gate, policy_features)


class TinyDynCurTests(unittest.TestCase):
    def test_embedding_store_validates_object_dict(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.npy"
            np.save(path, {"img_0": np.zeros((25, 512), dtype=np.float32)})
            store = EmbeddingStore(path)
            self.assertEqual(store.patch("img_0", 0).shape, (512,))
            self.assertEqual(store.patch("img_0", 0).dtype, np.float32)

    def test_policy_features_do_not_contain_goal_fields(self):
        x = policy_features(np.zeros(512, np.float32), np.ones(512, np.float32), (1, 2), 7, (7, 8))
        self.assertEqual(x.shape, (FEATURE_DIM,))
        self.assertNotIn("goal", json.dumps(x.tolist()))

    def test_pbrs_reward_is_training_side_and_progress_has_positive_sign(self):
        ep = Episode("e", "train", "img_0", 0, 24, 8, 10)
        cfg = TrainConfig("ppo_pbrs", 0, 1, 1, 1e-3, .5, .99, 0.0, .1, "cpu")
        self.assertGreater(_transition_reward(ep, (0, 0), (1, 0), False, cfg), 0)
        self.assertGreater(_transition_reward(ep, (3, 4), (4, 4), True, cfg), 0)

    def test_curiosity_gate_is_larger_far_from_target(self):
        cfg = TrainConfig("ppo_pbrs_curiosity", 0, 1, 1, 1e-3, .5, .99, .1, .1, "cpu")
        self.assertGreater(curiosity_gate(Episode("e", "train", "img_0", 0, 24, 8, 10), (0, 0), cfg),
                           curiosity_gate(Episode("e", "train", "img_0", 0, 24, 8, 10), (3, 4), cfg))

    def test_model_has_small_parameter_budget(self):
        model = TinyPolicy()
        self.assertLess(sum(p.numel() for p in model.parameters()), 2_000_000)
        self.assertEqual(model.step(__import__('torch').zeros(2, FEATURE_DIM))[0].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
