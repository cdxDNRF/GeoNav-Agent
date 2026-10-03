"""Correct inherited audit totals for two rotation-bearing arms; frozen experiment unchanged."""
import json
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from eval.audit_hard_rotation_training import OUT,registration,fit_images,train_replay,diagnostic,read,digest,write_new


def main():
    if (OUT/'独立复核.json').exists():raise ValueError('immutable audit exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=registration();assert reg['mining_fit_only'] and reg['mining_pairs']==45240 and reg['mining_predictions']==407160;values,fit_info=fit_images(reg);train_replay(reg,values);diag_info=diagnostic(reg)
    resources=[read(OUT/p['folder']/'训练资源.json') for p in reg['plans']]
    for r in resources:
        assert r['view_counts']==[473280,120640,120640,120640] and r['optimizer_steps']==1632 and r['pair_uses']==835200 and r['positive_pair_uses']==111360
    audit=dict(status='passed',same_execution_agent_separate_algorithm=True,all_six_training_replays_exact=True,uniform_heads_reproduced_exact=True,
        models=6,optimizer_steps=sum(r['optimizer_steps'] for r in resources),pair_uses=sum(r['pair_uses'] for r in resources),positive_pair_uses=sum(r['positive_pair_uses'] for r in resources),
        rotated_negative_uses=sum(sum(r['view_counts'][1:]) for r in resources),rotated_negative_uses_by_arm={arm:sum(sum(read(OUT/p['folder']/'训练资源.json')['view_counts'][1:]) for p in reg['plans'] if p['arm']==arm) for arm in reg['arms']},
        diagnostic_predictions=316800,diagnostic_precision_intervals=6,mining_predictions_recomputed=407160,mining_epoch_tables_recomputed=96,fit_only_mining_verified=True,
        fit_images=fit_info,diagnostic_images=diag_info,default_unchanged=True,encoder_and_explorer_unchanged=True,
        audit_receipt_adjustment=dict(reason='both Uniform and HardNeg rotate negatives; inherited receipt counted one arm',experiment_and_frozen_source_unchanged=True,script=Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]).as_posix(),script_sha256=digest(Path(__file__))),
        artifacts_sha256={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()})
    assert audit['rotated_negative_uses']==2171520
    write_new(OUT/'独立复核.json',audit);receipt=dict(status='completed_and_audited',quality_and_replay_passed=True,allow_development=True,training_factor_effectiveness_not_yet_established=True,
        default_changed=False,default_sha256=reg['default_sha256'],audit_sha256=digest(OUT/'独立复核.json'),heads_sha256=read(OUT/'全部训练结束.json')['heads_sha256'],cloud_calls=0,new_SwissView_evaluations=0,
        audit_script_sha256=digest(Path(__file__)))
    write_new(OUT/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
