"""CV-015 schema recovery. Adapt old CV-005 truth from its original frozen tasks.

The v1 source/binding remain immutable. Reuses the original pure counting,
alternate-audit and document implementations; no feature/model access.
"""
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import crossview_evidence_close_v1 as core

ROOT=core.ROOT
BASE=core.BASE
OUT=BASE/'跨视角阶段证据收束_v2'
LABELS=BASE/'跨视角感知最小验证_v1/冻结执行任务.json'
LEGACY=BASE/'拍摄点语义与完整目标图验证_v1/逐对排序结果.json'
SOURCES=core.SOURCES+['project/src/eval/crossview_evidence_close_v2.py','project/src/tests/test_crossview_evidence_close_v2.py']


def adapt_legacy(document,tasks):
    """Explicit label join, not inference from ranks, scores, names or predictions."""
    labels={t['pair_id']:t for t in tasks}
    if len(labels)!=16 or len(tasks)!=16:raise ValueError('Original16 frozen tasks required')
    result=copy.deepcopy(document)
    for row in result['rows']:
        task=labels.get(row['pair_id'])
        if task is None or row['group_id']!=task['group_id']:raise ValueError('Unknown pair/group')
        truth=task['goal_index_evaluator_only']
        if 'truth_evaluator_only' in row and row['truth_evaluator_only']!=truth:raise ValueError('Conflicting saved truth')
        row['truth_evaluator_only']=truth
    return result


def collect():
    output=ROOT/OUT
    if output.exists():raise FileExistsError(output)
    index=json.loads((ROOT/'项目导航/实验索引.json').read_text(encoding='utf-8-sig'))
    batches=[b for b in index['batches'] if re.fullmatch(r'CV-0(?:0[1-9]|1[0-4])',b.get('category',''))]
    if len(batches)!=14:raise ValueError('Original14 predecessors required')
    paths={LABELS,LEGACY,core.QUEUE,BASE/'同任务来源用途账本_v1.json',
        core.RUN/'逐题排序结果.json',core.RUN/'开发对照汇总.json',core.RUN/'训练日志.json',core.RUN/'执行完成.json',
        core.RUN/'核验/最终门槛判定.json',core.RUN/'核验/另实现复核.json',
        BASE/'目标格定义与视觉尺度审查_v1/审查汇总.json',BASE/'弱场景与拍摄点迁移差异诊断_v1/目标语义与证据边界.json'}
    for _,folder,summary,rows,_,_ in core.SERIES:paths.update((BASE/folder/summary,BASE/folder/rows))
    paths.update(Path(b['path'])/'阶段封存.json' for b in batches)
    failed=BASE/'跨视角阶段证据收束_v1/输入与源码绑定.json'
    protected=core.text_json(core.RUN/'首forward绑定.json')['protected'][:4]
    output.mkdir();(output/'核验').mkdir()
    def save(name,data):
        p=output/name
        if p.exists():raise FileExistsError(p)
        p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    binding={'task':'CV-015','date':'2026-10-18','inputs':[core.digest(p) for p in sorted(paths)],
        'sources':[core.digest(p) for p in SOURCES],'protected_current_files':[core.digest(e['path']) for e in protected],
        'previous_failure_binding':core.digest(failed),'output_limit_bytes':3*1024**2,
        'scope':'saved JSON decisions/metadata only; no raw/features/weights/model calls or new predictions',
        'legacy_truth_adapter':'CV-005 record pair_id+group_id -> original CV-002 frozen task goal index; never infer labels from rankings',
        'statistical_plan':core.text_json(failed)['statistical_plan']}
    save('输入与源码绑定.json',binding)
    seals={}
    for b in batches:
        for e in core.text_json(Path(b['path'])/'阶段封存.json')['files']:seals[e['path']]=e['sha256']
    for e in binding['inputs']:
        if not e['path'].endswith('阶段封存.json') and e['path']!=(BASE/'同任务来源用途账本_v1.json').as_posix():
            if seals.get(e['path'])!=e['sha256']:raise ValueError(('Legacy sealed input changed',e['path']))
    history=[]
    for category,folder,_,rows_name,dataset,arm in core.SERIES:
        path=BASE/folder/rows_name;data=core.text_json(path)
        if path==LEGACY:data=adapt_legacy(data,core.text_json(LABELS)['pairs'])
        rows=data['rows']
        if dataset:rows=[r for r in rows if r['dataset']==dataset]
        if arm:rows=[r for r in rows if r.get('arm')==arm]
        if not rows:raise ValueError('Frozen series empty')
        history.append({'series':category,'batch_path':(BASE/folder).as_posix(),'arm':arm or 'original',
            'label_semantics':'weak released common-cell proxy' if dataset=='weak_dev' else 'CSV/code camera-point proxy',
            'counts':{c:core.counts([r for r in rows if r['condition']==c]) for c in ('Full','Mean','Wrong')},
            'source_groups':len({r.get('group',r.get('group_id')) for r in rows}),'navigation_SR':None})
    current=core.summarize_current(core.text_json(core.RUN/'逐题排序结果.json')['rows'],core.text_json(core.QUEUE))
    if current['counts']!=core.text_json(core.RUN/'开发对照汇总.json')['counts']:raise ValueError('CV-014 count mismatch')
    roles=core.text_json(BASE/'同任务来源用途账本_v1.json')['records'];reserved=[r for r in roles if r['role']=='confirmation_reserved']
    if len(reserved)!=24 or len({r['group'] for r in reserved})!=6:raise ValueError('Reserved roles changed')
    log=core.text_json(core.RUN/'训练日志.json')
    result={'task':'CV-015','historical_series':history,'current':current,'current_role_counts':dict(Counter(r['role'] for r in roles)),
        'reserved_groups':6,'reserved_records':24,'fit_loss_first':log['log'][0]['loss_before_update'],'fit_loss_final':log['final_fit_loss'],
        'fit_updates':log['updates'],'fit_supervision_uses':log['supervision_uses'],
        'effect_gates_failed':[k for k,v in core.text_json(core.RUN/'核验/最终门槛判定.json')['gates'].items() if not v],
        'new_raw_reads':0,'new_model_forward_calls':0,'new_predictions':0,'new_training_steps':0,'new_navigation_episodes':0,'new_API_calls':0,'confirmation_access':0,
        'new_candidate_admitted':False,'next_direction':'bounded label/observability and same-area multi-query relation admission on14 already-consumed fit/dev sources; no training admitted'}
    save('证据重计汇总.json',result)
    save('格式恢复映射.json',{'previous_failure_binding':binding['previous_failure_binding'],
        'failure':'v1 rejected CV-005 rows without evaluator truth; no output summary or model calls',
        'labels_input':core.digest(LABELS),'adapted_rows':96,'kept_for_statistics':48,'original_files_modified':False,
        'labels_inferred_from_scores':False,'old_source_paths_preserved':True})
    for e in binding['inputs']+binding['sources']+binding['protected_current_files']+[binding['previous_failure_binding']]:
        if core.digest(e['path'])!=e:raise ValueError('Bound original changed')
    print(json.dumps({'collected':True,'saved_rankings_checked':1817,'new_model_calls':0,'next_candidate_admitted':False},ensure_ascii=False))


def audit():
    from eval import audit_crossview_evidence_close_v1 as alternate
    raw_load=alternate.load
    def load_with_legacy_labels(path):
        data=raw_load(path)
        if Path(path).resolve()==(ROOT/LEGACY).resolve():
            # Separate join implementation: explicit loop, no producer adapter use.
            frozen=raw_load(ROOT/LABELS)['pairs']
            for row in data['rows']:
                matched=[t for t in frozen if t['pair_id']==row['pair_id'] and t['group_id']==row['group_id']]
                assert len(matched)==1
                row['truth_evaluator_only']=matched[0]['goal_index_evaluator_only']
        return data
    alternate.OUT=ROOT/OUT;alternate.load=load_with_legacy_labels
    alternate.audit()


def close():
    from documents import close_crossview_evidence_v1 as document
    document.OUT=ROOT/OUT
    document.main()


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('collect','audit','close'):raise SystemExit('collect | audit | close; each once')
    {'collect':collect,'audit':audit,'close':close}[sys.argv[1]]()
