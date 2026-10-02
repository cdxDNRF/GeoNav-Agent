"""Frozen 20-task pilot: one call, same-agent reflection, independent verifier."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.collaboration_pilot import (MODES, MAX_CALLS, MAX_REVIEWS, PREFLIGHT_PROMPT, PROMPTS,
    PilotAPI, CollaborationLayer, canonical, digest_payload, image_part)
from agents.vlm import APIConfig, _parse_json_object
from agents.curl_transport import CurlTransport
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.precomputed_frozen_edge import PrecomputedFrozenEdgeNavigator
from env.episode import Episode
from env.environment import GridWorldEnv, image_payload
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
MASA = ROOT / 'DATA/processed_data/Masa'
OLD = MASA / '训练结果/边缘连续性可信线索对照_v1'
CACHE = MASA / '目标方向候选_v1/开发数据/方向视图'
OUT = MASA / '评测结果/多Agent同预算小验证_v1'
DOC = ROOT / '选题报告相关/多Agent同预算小验证方案_v1.md'
TESTS = ROOT / '选题报告相关/多Agent同预算小验证测试记录_v1.json'
DEFAULT = ROOT / 'project/local_policy_default.json'
REPEATS = (0, 1, 2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def select_tasks(bank):
    areas = sorted({x['area'] for x in bank}, key=lambda a: sha256(('coop-pilot-v1:' + a).encode()).hexdigest())[:20]
    selected = []
    for i, area in enumerate(areas):
        choices = [r for r in bank if r['area'] == area and r['dist'] == 4+i % 5]
        if len(choices) != 1:
            raise ValueError('parent bank must have one task per source/distance')
        selected.append(choices[0])
    if len(selected) != 20 or Counter(x['dist'] for x in selected) != {d: 4 for d in range(4, 9)}:
        raise ValueError('pilot balance')
    return selected


def config():
    raw = APIConfig.load()
    if raw.model != 'gemma4:31b' or raw.base_url.rstrip('/') != 'https://ollama.com/v1':
        raise ValueError('pilot freezes existing Gemma provider/model')
    return APIConfig(raw.base_url, raw.model, raw.api_key, timeout=60, max_tokens=1024)


def prepare():
    if OUT.exists():
        raise ValueError('immutable pilot exists; use --run-frozen, never recreate')
    tests = read(TESTS)
    if not tests['successful']:
        raise ValueError('pre-run tests required')
    bank = read(OLD / '导航任务.json')
    split = read(OLD / '源图划分.json')
    selected = select_tasks(bank)
    areas = {x['area'] for x in selected}
    if not areas.issubset(split['held']) or areas.intersection(split['original_split']['fit']):
        raise ValueError('must use known held development sources')
    cfg = config()
    OUT.mkdir(parents=True)
    for name, source in [('冻结方案.md', DOC), ('冻结默认配置.json', DEFAULT), ('测试记录.json', TESTS),
                         ('任务要求与论文框架对应说明.md', ROOT/'选题报告相关/任务要求与论文框架对应说明_v1.md')]:
        (OUT/name).write_bytes(source.read_bytes())
    write_new(OUT/'导航任务.json', selected)
    references = [OLD/'导航任务.json', OLD/'源图划分.json', DEFAULT]
    defaults = read(DEFAULT)
    references += [ROOT/r['path'] for r in defaults['checkpoints'] + defaults['cue_heads']]
    references += [ROOT/r['path'] for r in defaults['means'].values()]
    references += [CACHE/f'R0_{n}.npz' for n in ('全局特征', '局部特征', '边缘profile')]
    preflight_areas = sorted(split['head_fit'])[:3]
    input_areas = areas.union(preflight_areas)
    references += [MASA/f'patches/train/{a}/patch_{j}.jpg' for a in sorted(input_areas) for j in range(25)]
    files = [p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','data','env','eval','train','tests')]
    for p in files:
        dst = OUT/'源码快照'/p.relative_to(SRC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    registration = dict(version='cooperation-pilot-v1', utc=datetime.now(timezone.utc).isoformat(),
        episodes=20, source_files=20, distinct_routes=20, known_development=True,
        selected_by='sha256(coop-pilot-v1:area), first20; distance4+i%5; never read performance',
        arms=['Edge']+list(MODES), api_repeats=list(REPEATS), local_checkpoint_seed=0,
        local_reference_repeated=False, grid_size=5, budget=10, threshold=.5,
        model=cfg.public(), maximum_chat_requests=MAX_CALLS, preflight_requests=6,
        maximum_navigation_requests=600, maximum_completion_tokens=MAX_CALLS*1024,
        maximum_images_per_request=4, maximum_reviews_per_episode=MAX_REVIEWS,
        maximum_wall_seconds=2700, automatic_retries=0, new_training_steps=0,
        preflight=dict(areas=preflight_areas, engineering_required_valid=6, minimum_identity_correct=4,
                       scope='pixel identity only; does not establish navigation/adjacency understanding'),
        trigger='accepted legal unvisited cue != explorer; at most2 reviews',
        coordination='supported and matching actions for2call arms, otherwise frozenEdge; errors fallback',
        histories='two actual prior observations; episode-local independent role states and prior messages',
        request_orders='per task and repeat cyclically rotate three LLM arms to limit temporal provider confounding',
        stop='HTTP401/403/429 or3consecutive errors or45min or606calls; no retries/paid top-up',
        cost='existing account; hard request/output-token/image cap; dollar unit price not verified, no invented estimate',
        candidate_gate=dict(SR_gain=.02, positive_API_repeats=2, SG_no_worse=True,
                            all_planned_episodes_required=True, all_review_responses_valid=True),
        bootstrap=dict(seed=4121, resamples=4000, unit='source file, paired mean acrossAPIrepeats'),
        default_sha256=digest(DEFAULT), sources_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in references},
        source_sha256={p.relative_to(SRC).as_posix():digest(p) for p in files},
        frozen_files_sha256={p.name:digest(p) for p in OUT.iterdir() if p.is_file()},
        new_unseen_source_files=0, true_goal_input_allowed=False, cloud_started=False)
    write_new(OUT/'预登记.json', registration)
    print(json.dumps(dict(prepared=True, tasks=20, sources=20, max_requests=MAX_CALLS, model=cfg.model)), flush=True)


def load_banks():
    values=[]
    for name in ('全局特征', '局部特征', '边缘profile'):
        with np.load(CACHE/f'R0_{name}.npz') as f:
            values.append({a:f[a] for a in f.files})
    return tuple(values)


def load_local():
    ag = load_frozen_edge_default(read(OUT/'冻结默认配置.json'), ROOT, seed=0, device='cpu')
    return PrecomputedFrozenEdgeNavigator(ag.explorer, ag.head, 'cpu', ag.em, ag.hm, ag.lm, ag.pm, .5)


def local_decision(agent, obs, ep, banks):
    g, l, p = banks
    cell=obs.position[0]*5+obs.position[1]
    return agent.act_with_profiles(obs,g[ep.area][cell],l[ep.area][cell],g[ep.area][ep.goal],l[ep.area][ep.goal],p[ep.area][cell],p[ep.area][ep.goal])


def run_episode(ep, arm, repeat, api, banks):
    ag=load_local();ag.reset()
    layer=None if arm=='Edge' else CollaborationLayer(arm, api)
    env=GridWorldEnv(MASA);obs=env.reset(ep);decisions=[]
    while not env.done:
        base=local_decision(ag,obs,ep,banks)
        interaction=None if layer is None else layer.act(obs,base,dict(arm=arm,repeat=repeat,episode_id=ep.episode_id,step=len(decisions)+1))
        action=base['action'] if interaction is None else interaction['action']
        decisions.append(dict(base=base,interaction=interaction,action=action))
        obs,_,info=env.step(action)
        if info.out_of_bounds:
            raise ValueError('illegal executed action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,arm=arm,api_repeat=repeat,
                local_checkpoint_seed=0,status='completed',decisions=decisions)


def preflight(api, reg):
    rows=[]
    for i, area in enumerate(reg['preflight']['areas']):
        cell=i*7
        current=image_payload(MASA/f'patches/train/{area}/patch_{cell}.jpg')
        for same in (True,False):
            target=image_payload(MASA/f'patches/train/{area}/patch_{cell if same else (cell+1)%25}.jpg')
            content=image_part('target',target)+image_part('current',current)
            payload=dict(model=api.config.model,messages=[dict(role='system',content=PREFLIGHT_PROMPT),dict(role='user',content=content)],
                         temperature=0,max_tokens=api.config.max_tokens,response_format={'type':'json_object'},stream=False)
            audit=dict(role='preflight',image_count=2,image_labels=['target','current'],
                image_sha256=[sha256(target).hexdigest(),sha256(current).hexdigest()],request_sha256=digest_payload(payload),
                system_prompt_sha256=sha256(PREFLIGHT_PROMPT.encode()).hexdigest())
            def parser(text, finish):
                obj, norm=_parse_json_object(text,finish)
                if not isinstance(obj,dict) or set(obj)!={'same_pixels'} or type(obj['same_pixels']) is not bool:
                    raise ValueError('invalid preflight')
                return obj,norm
            if api.circuit_reason:
                rows.append(dict(index=len(rows),expected=same,status='not_run',correct=False,area=area,cell=cell))
                continue
            answer, record=api.exchange(payload,audit,dict(kind='preflight',index=len(rows)),parser)
            rows.append(dict(index=len(rows),expected=same,parsed=answer,correct=answer is not None and answer['same_pixels']==same,
                status=record['status'],call_id=record['call_id'],area=area,cell=cell))
    passed=sum(r['status']=='ok' for r in rows)==6 and sum(r['correct'] for r in rows)>=4
    write_new(OUT/'工程视觉诊断.json',dict(passed=passed,valid=sum(r['status']=='ok' for r in rows),correct=sum(r['correct'] for r in rows),
        tasks=rows,scope='identity/schema diagnosis, not calibrated directional target evidence'))
    if not passed:
        api.circuit_reason=api.circuit_reason or 'preflight_failed'
    print(json.dumps(dict(preflight_passed=passed,valid=sum(r['status']=='ok' for r in rows),correct=sum(r['correct'] for r in rows))),flush=True)


def append(row):
    with (OUT/'导航轨迹.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')


def skipped(ep,arm,repeat,reason):
    return dict(episode_id=ep.episode_id,area=ep.area,distance=ep.dist,arm=arm,api_repeat=repeat,
        local_checkpoint_seed=0,status='not_run',not_run_reason=reason,success=False,sg=8,steps=0,
        revisits=0,repeat_visit_rate=0,out_of_bounds=0,trajectory=[],decisions=[],termination='not_run')


def metrics(rows):
    steps=sum(r['steps'] for r in rows)
    completed=sum(r['status']=='completed' for r in rows)
    return dict(planned=len(rows),completed=completed,Q_nav=completed/len(rows),successes=sum(r['success'] for r in rows),
        SR_gate=sum(r['success'] for r in rows)/len(rows),SG_gate=statistics.mean(r['sg'] for r in rows),
        mean_steps=statistics.mean(r['steps'] for r in rows),revisit_rate=sum(r['revisits'] for r in rows)/steps if steps else 0)


def summary():
    reg=read(OUT/'预登记.json');rows=read_lines(OUT/'导航轨迹.jsonl')
    calls=read_lines(OUT/'API响应.jsonl') if (OUT/'API响应.jsonl').exists() else []
    runs={}
    for arm in reg['arms']:
        repeats=[None] if arm=='Edge' else REPEATS
        runs[arm]={str(r):metrics([x for x in rows if x['arm']==arm and x['api_repeat']==r]) for r in repeats}
    average={arm:{k:statistics.mean(v[k] for v in runs[arm].values()) for k in ('SR_gate','SG_gate','Q_nav')}
             for arm in reg['arms']}
    paired=[];sgpaired=[]
    for repeat in REPEATS:
        a=runs['TwoAgent'][str(repeat)];b=runs['SingleReflect'][str(repeat)]
        paired.append(a['SR_gate']-b['SR_gate']);sgpaired.append(a['SG_gate']-b['SG_gate'])
    sources=sorted({r['area'] for r in rows});sr_by_source=[];sg_by_source=[]
    for area in sources:
        coop=[r for r in rows if r['arm']=='TwoAgent' and r['area']==area]
        ref=[r for r in rows if r['arm']=='SingleReflect' and r['area']==area]
        sr_by_source.append(statistics.mean(r['success'] for r in coop)-statistics.mean(r['success'] for r in ref))
        sg_by_source.append(statistics.mean(r['sg'] for r in coop)-statistics.mean(r['sg'] for r in ref))
    rng=np.random.default_rng(reg['bootstrap']['seed']);samples=rng.integers(0,len(sources),(reg['bootstrap']['resamples'],len(sources)))
    interval=lambda arr:np.quantile(np.asarray(arr)[samples].mean(1),[.025,.975]).tolist()
    checks=dict(SR_gain_2pp=statistics.mean(paired)>=.02-1e-12,two_positive_API_repeats=sum(x>1e-12 for x in paired)>=2,
        SG_no_worse=statistics.mean(sgpaired)<=1e-12,complete=all(x['Q_nav']==1 for x in average.values()),
        valid_cloud_responses=all(c['status']=='ok' for c in calls),preflight=read(OUT/'工程视觉诊断.json')['passed'])
    resources={}
    for arm in MODES:
        armcalls=[c for c in calls if c['tag'].get('arm')==arm]
        rr=[r for r in rows if r['arm']==arm]
        resources[arm]=dict(calls=len(armcalls),valid=sum(c['status']=='ok' for c in armcalls),
            input_tokens=sum((c.get('usage') or {}).get('prompt_tokens',0) or 0 for c in armcalls),
            output_tokens=sum((c.get('usage') or {}).get('completion_tokens',0) or 0 for c in armcalls),
            images=sum(c['image_count'] for c in armcalls),
            median_latency_seconds=statistics.median(c['latency_seconds'] for c in armcalls) if armcalls else None,
            review_rounds=sum(bool(d['interaction']['call_ids']) for r in rr for d in r['decisions']),
            changed_actions=sum(d['action']!=d['base']['action'] for r in rr for d in r['decisions']),
            disagreement_or_abstention=sum(d['interaction']['coordination_reason']=='disagreement_or_abstention_fallback' for r in rr for d in r['decisions']))
    result=dict(runs=runs,averages=average,paired_SR_by_API_repeat=paired,paired_SG_by_API_repeat=sgpaired,
        source_SR_interval95=interval(sr_by_source),source_SG_interval95=interval(sg_by_source),
        source_effects=[dict(area=a,SR_gain=s,SG_change=g) for a,s,g in zip(sources,sr_by_source,sg_by_source)],
        checks=checks,candidate_numeric_passed=all(checks.values()),audit_pending=True,default_changed=False,
        training_steps=0,total_requests=len(calls),resources=resources,
        independent_generalization_proven=False,target_mechanism_proven=False,
        per_distance={a:{str(d):metrics([r for r in rows if r['arm']==a and r['distance']==d]) for d in range(4,9)} for a in reg['arms']})
    write_new(OUT/'对照汇总.json',result)
    return result


def run_frozen(confirmed):
    if not confirmed:
        raise ValueError('cloud transmission must be authorized via CLI')
    if (OUT/'启动记录.json').exists():
        raise ValueError('batch already started; never rerun/cherry-pick cloud answers')
    reg=read(OUT/'预登记.json');cfg=config()
    if cfg.public()!=reg['model']:
        raise ValueError('cloud configuration changed after freeze')
    for name, expected in reg['sources_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('frozen input changed')
    for name, expected in reg['source_sha256'].items():
        if digest(SRC/name)!=expected:raise ValueError('frozen code changed')
    for name, expected in reg['frozen_files_sha256'].items():
        if digest(OUT/name)!=expected:raise ValueError('frozen documents changed')
    write_new(OUT/'启动记录.json',dict(utc=datetime.now(timezone.utc).isoformat(),registration_sha256=digest(OUT/'预登记.json')))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    banks=load_banks();eps=[Episode.from_dict(x) for x in read(OUT/'导航任务.json')]
    api=PilotAPI(cfg,CurlTransport(cfg.timeout),OUT)
    preflight(api,reg);start=time.monotonic()
    for ep in eps:
        append(run_episode(ep,'Edge',None,api,banks))
    for repeat in REPEATS:
        for i,ep in enumerate(eps):
            order=list(MODES);offset=(repeat+i)%len(order);order=order[offset:]+order[:offset]
            for arm in order:
                if time.monotonic()-start>=reg['maximum_wall_seconds']:
                    api.circuit_reason='wall_time_cap'
                if api.total_calls>=MAX_CALLS:
                    api.circuit_reason='request_cap'
                row=skipped(ep,arm,repeat,api.circuit_reason) if api.circuit_reason else run_episode(ep,arm,repeat,api,banks)
                append(row)
            status=dict(status='running',API_repeat=repeat,completed_tasks_in_repeat=i+1,chat_requests=api.total_calls,circuit=api.circuit_reason)
            (OUT/'执行状态.json').write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8')
            print(json.dumps(status,ensure_ascii=False),flush=True)
    result=summary()
    (OUT/'执行状态.json').write_text(json.dumps(dict(status='completed',records=200,chat_requests=api.total_calls,
        circuit=api.circuit_reason,candidate_numeric_passed=result['candidate_numeric_passed']),ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(completed=True,averages=result['averages'],candidate_numeric_passed=result['candidate_numeric_passed']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--run-frozen',action='store_true');p.add_argument('--confirm-cloud-transmission',action='store_true');args=p.parse_args()
    if args.prepare_only:prepare()
    elif args.run_frozen:run_frozen(args.confirm_cloud_transmission)
    else:p.error('choose --prepare-only or --run-frozen')
