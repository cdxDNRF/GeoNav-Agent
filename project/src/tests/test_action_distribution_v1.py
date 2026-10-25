"""CPU probability/gradient contracts; no project weights, training or CUDA calls."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
from agents.action_distribution_v1 import masked_distribution, evaluate_distribution
from eval.geoexplorer_distribution_diagnosis_v1 import kind, move, uniform_ratios, summarize_log, source_contract


def original_sampling_function():
    source = ROOT / 'project/src/train/geoexplorer_training_v1.py'
    tree = ast.parse(source.read_text('utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'masked_distribution')
    namespace = {'torch': torch, 'move': move}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['masked_distribution']


class DistributionTests(unittest.TestCase):
    def test_original_acting_distribution_unchanged_for_all_cells(self):
        original = original_sampling_function()
        vectors = torch.softmax(torch.arange(100, dtype=torch.float32).reshape(25, 4) / 17, -1)
        candidate = masked_distribution(vectors, tuple(range(25)))
        for cell in range(25):
            policy = SimpleNamespace(actor=lambda state: vectors[cell:cell + 1])
            expected = original(policy, torch.zeros(1, 512), cell).probs[0]
            torch.testing.assert_close(candidate.probs[cell], expected, atol=0, rtol=0)

    def test_identical_parameters_ratio_one_for_all_actions(self):
        probabilities = torch.softmax(torch.tensor([[.7, -.2, 1.3, -.8]], dtype=torch.float64), -1)
        original = original_sampling_function()
        for cell in range(25):
            old = original(SimpleNamespace(actor=lambda state: probabilities), torch.zeros(1, 512), cell)
            for action in range(4):
                actions = torch.tensor([action], dtype=torch.long)
                current, _ = evaluate_distribution(probabilities, [cell], actions)
                torch.testing.assert_close(torch.exp(current - old.log_prob(actions)), torch.ones(1, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_original_ratio_mismatch_is_visible_at_boundary(self):
        p = torch.full((1, 4), .25, dtype=torch.float64)
        old = masked_distribution(p, [0])
        action = torch.tensor([1])
        ratio = torch.exp(torch.distributions.Categorical(p).log_prob(action) - old.log_prob(action))
        self.assertLess(float(ratio), .8)
        interior = masked_distribution(p, [12])
        torch.testing.assert_close(interior.probs, p)

    def test_out_of_bounds_epsilon_is_preserved(self):
        p = torch.full((1, 4), .25, dtype=torch.float64)
        dist = masked_distribution(p, [0])
        self.assertGreater(float(dist.probs[0, 0]), 0)
        logs, entropy = evaluate_distribution(p, [0], torch.tensor([0]))
        self.assertTrue(torch.isfinite(logs).all() and torch.isfinite(entropy).all())

    def test_entropy_and_gradients_follow_sampling_distribution(self):
        logits = torch.tensor([[.1, .2, .3, .4]], requires_grad=True)
        p = torch.softmax(logits, -1)
        logs, entropy = evaluate_distribution(p, [0], torch.tensor([1]))
        reference = masked_distribution(p, [0])
        torch.testing.assert_close(entropy, reference.entropy())
        grad = torch.autograd.grad((logs + .01 * entropy).sum(), logits)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_negative_advantage_legacy_clip_has_flat_gradient(self):
        logits = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        p = torch.softmax(logits, -1)
        actions = torch.tensor([1])
        old_log = masked_distribution(p.detach(), [0]).log_prob(actions)
        legacy_log = torch.distributions.Categorical(p).log_prob(actions)
        legacy_ratio = torch.exp(legacy_log - old_log)
        legacy_loss = -torch.minimum(-legacy_ratio, -legacy_ratio.clamp(.8, 1.2)).mean()
        legacy_grad = torch.autograd.grad(legacy_loss, logits, retain_graph=True)[0]
        self.assertEqual(float(legacy_grad.abs().sum()), 0)
        new_log, _ = evaluate_distribution(p, [0], actions)
        ratio = torch.exp(new_log - old_log)
        new_loss = -torch.minimum(-ratio, -ratio.clamp(.8, 1.2)).mean()
        self.assertGreater(float(torch.autograd.grad(new_loss, logits)[0].abs().sum()), 0)

    def test_rejects_malformed_public_inputs(self):
        p = torch.full((1, 4), .25)
        for cells, grid, epsilon in (([], 5, 1e-6), ([True], 5, 1e-6), ([25], 5, 1e-6), ([0], 1, 1e-6), ([0], 5, 0)):
            with self.assertRaises(ValueError):masked_distribution(p, cells, grid, epsilon)
        for bad in (torch.full((1, 4), float('nan')), torch.zeros(1, 4), torch.tensor([[.5, .5, .1, -.1]])):
            with self.assertRaises(ValueError):masked_distribution(bad, [0])
        with self.assertRaises(ValueError):evaluate_distribution(p, [0], torch.tensor([4]))
        with self.assertRaises(ValueError):evaluate_distribution(p, [0], torch.tensor([1.0]))

    def test_grid_geometries_and_formula_examples(self):
        self.assertEqual([kind(c) for c in (0, 2, 12)], ['corner', 'edge', 'interior'])
        self.assertEqual(move(0, 0), 0)
        self.assertEqual(move(0, 1), 1)
        rows = uniform_ratios()
        self.assertEqual(len(rows), 25)
        self.assertGreater(rows[0]['original_same_parameter_ratios'][0], 100000)
        self.assertEqual(rows[0]['consistent_same_parameter_ratios'], [1.] * 4)
        p = torch.full((2, 4), .25)
        self.assertEqual(masked_distribution(p, [0, 99], grid=10).probs.shape, (2, 4))

    def test_bound_upstream_contract(self):
        self.assertTrue(all(source_contract(ROOT).values()))


if __name__ == '__main__':unittest.main()
