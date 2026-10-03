"""Versioned audit-only correction; the original frozen auditor stays intact.

The original rule replay accidentally resets a nonexistent neural agent. A
stateless reset shim makes that call a no-op only for rule rows. Every original
pixel, model-equation, trajectory, metric and probe check is still executed.
No navigation result, model, criterion or source file is rewritten.
"""
from pathlib import Path
import argparse
import importlib.util
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / 'project/src/eval/audit_spatial_area_confirmation.py'
spec = importlib.util.spec_from_file_location('eval._spatial_audit_v2_runtime', ORIGINAL)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
OUT = core.OUT
RECEIPT = OUT / '审计修订登记_v2.json'
RECEIPT_SEAL = OUT / '审计修订封存_v2.json'
original_replay = core.replay_episode
original_write = core.write_exclusive


class RuleWithoutMemory:
    def reset(self):
        """Rules use only current public observations; there is no RNN state."""


def replay_episode(stage, seed, condition, ep, saved, agent, env, banks, means, wrong, provenance, strata, region):
    if seed is None:
        if agent is not None or condition not in core.RULES:
            raise ValueError('stateless original rule replay required')
        agent = RuleWithoutMemory()
    return original_replay(stage, seed, condition, ep, saved, agent, env, banks, means, wrong, provenance, strata, region)


def prepare():
    core.protocol.check_bindings(core.read(OUT / '预登记.json'))
    if RECEIPT.exists() or RECEIPT_SEAL.exists() or (OUT / '工程接口/独立复核.json').exists():
        raise ValueError('exclusive audit revision receipt required')
    sources = [Path(__file__), ORIGINAL, ROOT / 'project/src/tests/test_spatial_area_confirmation_audit_v2.py']
    for path in (sources[0], sources[2]):
        snapshot = OUT / '源码快照' / path.relative_to(ROOT / 'project/src')
        with snapshot.open('xb') as handle:
            handle.write(path.read_bytes())
        sources.append(snapshot)
    inputs = [p for p in (OUT / '工程接口').rglob('*') if p.is_file()]
    inputs.extend([OUT / '预登记.json', OUT / '预登记封存.json', OUT / '工程接口/原独立复核失败_v1.json'])
    original_write(RECEIPT, dict(revision=2, audit_only=True,
        reason='frozen v1 rule branch calls reset on None; stateless reset shim only',
        neural_policy_or_rule_actions_changed=False, model_threshold_or_gate_changed=False,
        navigation_records_reused=4200, probe_records_reused=4320, confirmation_not_started=True,
        source_sha256={p.relative_to(ROOT).as_posix(): core.digest(p) for p in sources},
        original_engineering_sha256={p.relative_to(ROOT).as_posix(): core.digest(p) for p in inputs}))
    original_write(RECEIPT_SEAL, dict(receipt_sha256=core.digest(RECEIPT)))


def check_revision():
    receipt = core.read(RECEIPT)
    if core.digest(RECEIPT) != core.read(RECEIPT_SEAL)['receipt_sha256']:
        raise ValueError('audit revision seal drift')
    for field in ('source_sha256', 'original_engineering_sha256'):
        for name, expected in receipt[field].items():
            if core.digest(ROOT / name) != expected:
                raise ValueError('audit-only revision binding changed: ' + name)


def audited_write(path, value):
    if path.name == '独立复核.json':
        check_revision()
        value = dict(value, audit_revision=2, audit_only_rule_reset_correction=True,
                     audit_revision_receipt_sha256=core.digest(RECEIPT),
                     original_frozen_auditor_sha256=core.digest(ORIGINAL))
    original_write(path, value)


def run(stage):
    check_revision()
    core.replay_episode = replay_episode
    core.write_exclusive = audited_write
    core.run_audit(stage)
    check_revision()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--stage', choices=tuple(core.STAGES), default='engineering')
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run(args.stage)
