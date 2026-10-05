"""EVAL-003诊断逻辑边界测试：合成轨迹验证机会分类、终局邻接、误触发反事实与独立重算。"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from eval.diagnose_massgis_mistrigger_v1 import (  # noqa: E402
    DELTAS, manhattan, opportunity_facts, independent_recount, verify_trajectory_files)

def synth_record(steps, start, goal, budget, k, success=None):
    """构造合成轨迹记录。steps为(action, cue_action或None)序列。"""
    pos = start
    trajectory = [{'step': 0, 'patch_id': start, 'action': None, 'out_of_bounds': False, 'revisited': False}]
    decisions = []
    remaining = budget
    for i, (action, cue) in enumerate(steps, 1):
        dr, dc = DELTAS[action]
        y, x = divmod(pos, k)
        ny, nx = y + dr, x + dc
        oob = not (0 <= ny < k and 0 <= nx < k)
        if not oob:
            pos = ny * k + nx
        remaining -= 1
        trajectory.append({'step': i, 'patch_id': pos, 'action': action,
                           'out_of_bounds': oob, 'revisited': False})
        decisions.append({'step': i, 'public_position': [pos // k, pos % k],
                          'public_visited': [0], 'remaining_budget': remaining + 1,
                          'explorer_action': action, 'explorer_logits': [0.0] * 4,
                          'action': cue or action, 'cue_action': cue, 'reason': 'x',
                          'probabilities': [0.2] * 5,
                          'current_image_sha256': 'a', 'target_image_sha256': 'b',
                          'explorer_features_sha256': 'c', 'cue_features_sha256': 'd'})
    success = (pos == goal) if success is None else success
    return dict(episode_id='synth', success=success, termination='x', sg=manhattan(pos, goal, k),
                sg_m=manhattan(pos, goal, k) * 300, steps=len(steps), revisits=0,
                repeat_visit_rate=0.0, out_of_bounds=0, trajectory=trajectory,
                decisions=decisions, evaluation_diagnostics=[{}] * len(decisions),
                status='completed', grid_size=k)


class OpportunityClassification(unittest.TestCase):
    def test_success_has_exactly_one_actionable_opportunity(self):
        # 5x5: start 0 -> goal 1: 一步即邻接并接受right进入目标
        task = dict(episode_id='synth', grid_size=5, goal=1, budget=10, start=0, dist=1)
        rec = synth_record([('right', 'right')], 0, 1, 10, 5)
        fact = opportunity_facts(rec, task)
        self.assertEqual(fact['opportunities'], 1)
        self.assertTrue(fact['had_opportunity'])
        self.assertTrue(rec['success'])
        self.assertFalse(fact['terminal_adjacent_only'])
        first = fact['first_opportunity']
        self.assertTrue(first['cue_accepted'])
        self.assertTrue(first['accepted_direction_correct'])
        self.assertFalse(first['accepted_direction_wrong'])
        self.assertEqual(first['remaining_budget'], 10)

    def test_terminal_adjacency_is_not_opportunity(self):
        # 0->2两步恰好耗尽预算：终局2与goal 3相邻，但之后无决策步，不构成可行动机会
        task = dict(episode_id='synth', grid_size=5, goal=3, budget=2, start=0, dist=3)
        rec = synth_record([('right', None), ('right', None)], 0, 3, 2, 5)
        self.assertEqual(rec['sg'], 1)  # 终局邻接但未到达
        self.assertFalse(rec['success'])
        fact = opportunity_facts(rec, task)
        self.assertEqual(fact['opportunities'], 0)
        self.assertFalse(fact['had_opportunity'])
        self.assertTrue(fact['terminal_adjacent_only'])
        self.assertEqual(fact['min_distance'], 1)

    def test_adjacent_without_budget_is_not_opportunity(self):
        # 预算为1时到达邻接格：执行最后一动作前remaining=1>=1算机会；本例构造预算0情形
        # start 0, budget 1, first action right到1（goal=2的邻接），但remaining在动作前=1：
        # 执行动作前位于0（与2不邻接），因此无机会。
        task = dict(episode_id='synth', grid_size=5, goal=2, budget=1, start=0, dist=2)
        rec = synth_record([('right', None)], 0, 2, 1, 5)
        fact = opportunity_facts(rec, task)
        self.assertEqual(fact['opportunities'], 0)
        self.assertEqual(fact['min_distance'], 1)
        # 注意：终格1与goal 2邻接，但预算耗尽 -> terminal_adjacent_only
        self.assertTrue(fact['terminal_adjacent_only'])

    def test_wrong_direction_accept_flagged(self):
        # goal=1在上方(0,1)，当前6(1,1)与其邻接：接受left到5是错误方向（up才是正确）
        task = dict(episode_id='synth', grid_size=5, goal=1, budget=10, start=6, dist=2)
        rec = synth_record([('left', 'left')], 6, 1, 10, 5)
        fact = opportunity_facts(rec, task)
        self.assertEqual(fact['opportunities'], 1)
        first = fact['first_opportunity']
        self.assertTrue(first['cue_accepted'])
        self.assertFalse(first['accepted_direction_correct'])
        self.assertTrue(first['accepted_direction_wrong'])

    def test_had_opportunity_but_failed_is_distinguishable(self):
        # 邻接格上弃权并越界原地耗步，最终未到达：机会>0且失败（真实数据中该类为0）
        task = dict(episode_id='synth', grid_size=5, goal=1, budget=3, start=0, dist=1)
        rec = synth_record([('up', None), ('up', None), ('up', None)], 0, 1, 3, 5)
        self.assertFalse(rec['success'])
        fact = opportunity_facts(rec, task)
        self.assertEqual(fact['opportunities'], 3)
        self.assertTrue(fact['had_opportunity'])
        self.assertFalse(fact['terminal_adjacent_only'])
        first = fact['first_opportunity']
        self.assertFalse(first['cue_accepted'])
        self.assertIsNone(first['accepted_direction_correct'])

    def test_boundary_stay_counts_toward_terminal(self):
        # 试图越界：留在原地并耗一步；原地格0与goal 1邻接，两步都在机会位
        task = dict(episode_id='synth', grid_size=5, goal=1, budget=3, start=0, dist=1)
        rec = synth_record([('up', None), ('right', None)], 0, 1, 3, 5)
        self.assertTrue(rec['success'])
        # 执行right前位于0，与1邻接且remaining=2>=1 -> 机会（越界那步之前也邻接且预算3>=1）
        fact = opportunity_facts(rec, task)
        self.assertEqual(fact['opportunities'], 2)
        self.assertEqual(rec['trajectory'][1]['patch_id'], 0)  # 原地
        self.assertTrue(rec['trajectory'][1]['out_of_bounds'])


class IndependentRecount(unittest.TestCase):
    def test_rebuild_matches_record(self):
        records = [synth_record([('right', None), ('down', 'down')], 0, 6, 10, 5)]
        jobs = {(5, 'M0', 0, 'img_7005'): records}
        plans = {5: {'synth': dict(episode_id='synth', grid_size=5, goal=6, budget=10, start=0, dist=2)}}
        cross = independent_recount(jobs, plans)
        self.assertEqual(cross['episodes'], 1)
        self.assertEqual(cross['actionable_opportunities'], 1)  # 第二步前位于1，与6邻接
        self.assertEqual(cross['accepted_at_opp_correct'], 1)
        self.assertEqual(cross['fail_terminal_adjacent'], 0)

    def test_rebuild_asserts_sg_consistency(self):
        records = [synth_record([('right', None)], 0, 3, 10, 5)]
        records[0]['sg'] = 99  # 人为错误
        jobs = {(5, 'M0', 0, 'img_7005'): records}
        plans = {5: {'synth': dict(episode_id='synth', grid_size=5, goal=3, budget=10, start=0, dist=3)}}
        with self.assertRaises(AssertionError):
            independent_recount(jobs, plans)


class ProductionInputGuards(unittest.TestCase):
    def test_verify_trajectory_files_checks_seal(self):
        # 真实输入的快速子集校验：一个文件读取并验证成功
        jobs = verify_trajectory_files()
        self.assertEqual(sum(len(v) for v in jobs.values()), 10500)


if __name__ == '__main__':
    unittest.main(verbosity=2)
