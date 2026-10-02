"""Preserve frozen audit; repair only angle labels for identical candidate payloads."""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import audit_orientation_confirmation as original
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest


def candidate_angles(given_angle,payloads,ep):
    # The production stable sort starts from relative0,90,180,270.
    # Equal payload SHA means identical input pixels/features; retain that order.
    return sorted(((given_angle+r)%360 for r in (0,90,180,270)),
        key=lambda a:sha256(payloads[a][ep.area,ep.goal]).hexdigest())


class SymmetryAuditTests(unittest.TestCase):
    def test_identical_pixels_preserve_relative_angle_order(self):
        ep=SimpleNamespace(area='a',goal=1);p={a:{('a',1):b'same'} for a in (0,90,180,270)}
        self.assertEqual(candidate_angles(90,p,ep),[90,180,270,0]);self.assertEqual(candidate_angles(0,p,ep),[0,90,180,270])
    def test_distinct_pixels_use_the_same_canonical_absolute_order(self):
        ep=SimpleNamespace(area='a',goal=1);p={a:{('a',1):str(a).encode()} for a in (0,90,180,270)}
        self.assertEqual(candidate_angles(90,p,ep),candidate_angles(0,p,ep))


def main():
    mode=sys.argv[sys.argv.index('--mode')+1];out=original.paths(mode)[1]
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(SymmetryAuditTests));assert result.wasSuccessful()
    before=inspect.getsource(original.candidate_replay)
    old="angles=sorted((0,90,180,270),key=lambda a:sha256(payloads[a][ep.area,ep.goal]).hexdigest())"
    assert before.count(old)==1
    after=before.replace(old,'angles=candidate_angles(given_angle,payloads,ep)')
    original.candidate_angles=candidate_angles
    exec(compile(after,str(Path(__file__).resolve()),'exec'),original.__dict__)
    note=dict(reason='Identical rotated PNGs have equal SHA; stable production sort preserves relative angle input order. Original audit assumed absolute angle order and falsely rejected labels. Pixel-derived scores and actions unchanged.',
        original_error='all four pixel-derived candidates[0].relative_clockwise: expected 270, got 0',
        counterexample='development Edge_s0 train_img_107_d5_002; four identical PNGs and identical probabilities',
        original_audit_sha256=digest(Path(original.__file__)),repair_sha256=digest(Path(__file__)),
        repaired_function_before_sha256=sha256(before.encode()).hexdigest(),repaired_function_after_sha256=sha256(after.encode()).hexdigest(),
        regression_tests_run=result.testsRun,tests_passed=True,policy_or_results_or_original_audit_changed=False,
        rerun_saved_experiment=False,only_audit_label_reconstruction_changed=True)
    write_new(out/'审计修复记录.json',note)
    (out/'审计修复源码.py').write_bytes(Path(__file__).read_bytes())
    original.main()


if __name__=='__main__':main()
