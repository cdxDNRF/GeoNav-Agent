"""Restore a missing image_payload import without modifying frozen auditor/results."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import audit_scaled_edge_confirmation as original
from env.environment import image_payload
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest


def main():
    out=original.OUT;record=out/'审计修复记录.json'
    if record.exists():raise ValueError('immutable repair record exists')
    source=Path(__file__);name=source.relative_to(original.SRC).as_posix()
    copy=out/'源码快照'/name
    if copy.exists():raise ValueError('repair snapshot already exists')
    copy.write_bytes(source.read_bytes())
    write_new(out/'原始审计异常.json',dict(error_type='NameError',missing_name='image_payload',
        stage='Independent raw-image reconstruction: first source/cell, prior to model/action replay.',
        original_source_sha256=digest(Path(original.__file__)),training_changed=False,evaluation_results_changed=False))
    write_new(record,dict(issue='Frozen formal auditor omitted env.environment.image_payload import.',
        correction='Bind the existing metadata-stripping image_payload helper in the frozen auditor namespace; all hashes/features/actions/gates remain checked.',
        original_auditor_sha256=digest(Path(original.__file__)),repair_source_sha256={name:digest(source)},
        summary_before_sha256=digest(out/'对照汇总.json'),registration_sha256=digest(out/'预登记.json'),
        training_changed=False,evaluation_changed=False,threshold_changed=False,task_changed=False,
        frozen_original_source_changed=False,verification='Resume full image reconstruction,7500 neural episodes,1000 rules,43200 probe predictions and all source intervals.'))
    original.image_payload=image_payload
    original.main()


if __name__=='__main__':main()
