"""Correct only the declared total:20+20+3*3*20=220; never alter navigation or SR/SG."""
from datetime import datetime,timezone
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.evidence_compact_g import G,OUT,ROOT,SRC,read,write,digest,lines


def register():
    declaration=read(G/'预登记.json');tasks=read(OUT/'总体任务.json');assert len(tasks)==20
    assert declaration['planned_records']==200 and declaration['API_repeats']==3
    code=[SRC/'eval/evidence_compact_count_correction.py',SRC/'eval/audit_evidence_compact_g_count_correction.py']
    reg=dict(utc=datetime.now(timezone.utc).isoformat(),reason='clerical arithmetic; M0 20 + M1 20 + three cloud arms x three repeats x20 =220',
        original_declared_total=200,correct_total=220,cohort_sha256=digest(OUT/'总体任务.json'),
        original_G_registration_sha256=digest(G/'预登记.json'),new_requests=0,new_training_steps=0,
        methods_tasks_repeats_permissions_and_budget_unchanged=True,SR_SG_per_arm_denominators_unchanged=True,
        original_registration_and_execution_source_preserved=True,
        source_sha256={p.relative_to(SRC).as_posix():digest(p) for p in code})
    for f in code:
        dst=OUT/'源码快照'/f.relative_to(SRC);dst.write_bytes(f.read_bytes())
    write(G/'计数勘误预登记.json',reg);print(reg)


def correct():
    reg=read(G/'计数勘误预登记.json');original=read(G/'对照汇总.json');rows=list(lines(G/'导航轨迹.jsonl'))
    assert len(rows)==220 and original['planned_records']==200
    assert {a:sum(r['arm']==a for r in rows) for a in ('M0','M1','M2','M3','M4')}=={'M0':20,'M1':20,'M2':60,'M3':60,'M4':60}
    obj=read(G/'对照汇总.json');check=obj['candidate_checks'];del check['all_200_terminal_records']
    check['all_220_terminal_records']=all(r['status']=='completed' for r in rows)
    obj['planned_records']=220;obj['candidate_numeric_passed']=all(check.values())
    obj['count_correction']=dict(registration_sha256=digest(G/'计数勘误预登记.json'),
        original_summary_sha256=digest(G/'对照汇总.json'),changed_fields=['planned_records','candidate_checks.total_terminal_count','candidate_numeric_passed'],
        metrics_effects_recovery_resources_unchanged=True)
    assert all(obj[k]==original[k] for k in ('arms','effects','recovery','cloud_requests','combined_requests'))
    write(G/'对照汇总_220条计数勘误.json',obj);print({'correct_total':220,'metrics_unchanged':True,'candidate_passed':obj['candidate_numeric_passed']})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--correct',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.correct:correct()
    else:p.error('register or correct')
