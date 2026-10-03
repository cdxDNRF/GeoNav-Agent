"""Audit-only timing schema repair, layered over the sealed rule-reset v2.

Elapsed wall time cannot be reconstructed from trajectories. Validate that
single field as finite and positive, then compare every remaining summary field
with the unchanged independent equations. All models, gates and outputs stay
frozen. Previous failures and both prior auditors remain byte-identical.
"""
from pathlib import Path
import argparse
import math
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import audit_spatial_area_confirmation_v2 as reset_revision

core = reset_revision.core
ROOT, OUT = core.ROOT, core.OUT
RECEIPT = OUT / '审计修订登记_v3.json'
SEAL = OUT / '审计修订封存_v3.json'
original_same = core.same_values


def same_values(left, right, atol=2e-10, rtol=2e-10):
    """Only top-level navigation elapsed time is not a replayable statistic."""
    required = {'stage', 'arms', 'effects', 'target_checks', 'planned_total', 'primary_SR'}
    if (isinstance(left, dict) and isinstance(right, dict) and required <= left.keys()
            and required <= right.keys() and left.get('stage') in core.STAGES
            and left.keys() - right.keys() == {'seconds'} and not right.keys() - left.keys()):
        seconds = left['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            return False
        left = {key: value for key, value in left.items() if key != 'seconds'}
    return original_same(left, right, atol, rtol)


def prepare():
    reset_revision.check_revision()
    core.protocol.check_bindings(core.read(OUT / '预登记.json'))
    if RECEIPT.exists() or SEAL.exists() or (OUT / '工程接口/独立复核.json').exists():
        raise ValueError('exclusive audit-only v3 registration required')
    difference = core.read(OUT / '工程接口/汇总差异诊断_v2.json')
    core.need(difference['exact_difference_count'] == 1
              and difference['difference_count_after_metadata_exclusion'] == 0
              and difference['excluded_metadata_keys'] == ['seconds']
              and difference['first_differences'][0]['path'] == '$.seconds'
              and difference['trajectory_records'] == 4200
              and difference['stored_summary_sha256'] == core.digest(OUT / '工程接口/对照汇总.json'),
              'audit v3 may repair only the verified elapsed-time schema discrepancy')
    sources = [Path(__file__), ROOT / 'project/src/tests/test_spatial_area_confirmation_audit_v3.py']
    for source in tuple(sources):
        snapshot = OUT / '源码快照' / source.relative_to(ROOT / 'project/src')
        with snapshot.open('xb') as handle:
            handle.write(source.read_bytes())
        sources.append(snapshot)
    inputs = [reset_revision.RECEIPT, reset_revision.RECEIPT_SEAL,
              OUT / '工程接口/汇总差异诊断_v2.json', OUT / '工程接口/原独立复核失败_v2.json',
              OUT / '工程接口/对照汇总.json', OUT / '预登记.json', OUT / '预登记封存.json']
    core.write_exclusive(RECEIPT, dict(revision=3, audit_only=True,
        reason='only nonreproducible navigation-summary seconds field; validate finite positive and compare all statistics',
        neural_rule_actions_or_results_changed=False, model_threshold_or_gate_changed=False,
        original_navigation_records_reused=4200, original_probe_records_reused=4320,
        source_sha256={p.relative_to(ROOT).as_posix(): core.digest(p) for p in sources},
        input_sha256={p.relative_to(ROOT).as_posix(): core.digest(p) for p in inputs},
        prior_revision_receipt_sha256=core.digest(reset_revision.RECEIPT),
        difference_diagnostic_sha256=core.digest(OUT / '工程接口/汇总差异诊断_v2.json')))
    core.write_exclusive(SEAL, dict(receipt_sha256=core.digest(RECEIPT)))


def check_revision():
    reset_revision.check_revision()
    receipt = core.read(RECEIPT)
    if core.digest(RECEIPT) != core.read(SEAL)['receipt_sha256']:
        raise ValueError('timing audit revision seal drift')
    for field in ('source_sha256', 'input_sha256'):
        for name, expected in receipt[field].items():
            if core.digest(ROOT / name) != expected:
                raise ValueError('audit-only v3 binding drift: ' + name)


def audited_write(path, value):
    if path.name == '独立复核.json':
        check_revision()
        value = dict(value, audit_revision=3, audit_only_rule_reset_correction=True,
                     audit_only_timing_metadata_correction=True,
                     audit_revision_receipt_sha256=core.digest(RECEIPT),
                     reset_revision_receipt_sha256=core.digest(reset_revision.RECEIPT),
                     original_frozen_auditor_sha256=core.digest(reset_revision.ORIGINAL))
    reset_revision.original_write(path, value)


def run(stage):
    check_revision()
    core.replay_episode = reset_revision.replay_episode
    core.same_values = same_values
    core.write_exclusive = audited_write
    core.run_audit(stage)
    check_revision()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--stage', choices=tuple(core.STAGES), default='engineering')
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run(args.stage)
