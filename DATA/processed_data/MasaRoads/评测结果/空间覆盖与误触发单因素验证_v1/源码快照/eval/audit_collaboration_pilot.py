"""Post-run public-input, role-state, response and checkpoint replay; no API calls."""
from collections import Counter
from hashlib import sha256
import json
import statistics
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.collaboration_pilot import request_for, parse_recommendation, boundary_legal, PROMPTS, PREFLIGHT_PROMPT, digest_payload, image_part
from agents.vlm import APIConfig, _parse_json_object
from env.episode import Episode
from env.environment import GridWorldEnv, image_payload
from eval.collaboration_pilot import ROOT,SRC,MASA,OUT,DEFAULT,load_banks,load_local,local_decision,read,read_lines,select_tasks,OLD
from train.dyncur_tiny import digest
from train.curiosity_controlled import write_new


def check(condition, label):
    if not condition:
        raise ValueError('audit failed: '+label)


def parsed_reply(record, legal, labels):
    if record['status']!='ok':
        check(record.get('parsed') is None,'failed API output never used')
        return None
    obj,norm=parse_recommendation(record['raw_content'],record['finish_reason'],legal,labels)
    check(obj==record['parsed'],'saved final reply strict parse')
    check(record['format_normalization']==('markdown_fence' if norm else 'none'),'normalization')
    return obj


def independent_coordination(arm, base, items):
    if any(x is None for x in items):return base,'cloud_error_fallback'
    if arm=='SingleOnce':
        return (items[0]['action'],'single_supported') if items[0]['target_evidence']=='supported' else (base,'single_abstention_fallback')
    if items[0]['action']==items[1]['action'] and items[0]['target_evidence']==items[1]['target_evidence']=='supported':
        return items[0]['action'],'joint_supported'
    return base,'disagreement_or_abstention_fallback'


def independent_metrics(rows):
    n=len(rows);steps=sum(r['steps'] for r in rows)
    return dict(planned=n,completed=sum(r['status']=='completed' for r in rows),Q_nav=sum(r['status']=='completed' for r in rows)/n,
        successes=sum(bool(r['success']) for r in rows),SR_gate=sum(bool(r['success']) for r in rows)/n,
        SG_gate=sum(r['sg'] for r in rows)/n,mean_steps=sum(r['steps'] for r in rows)/n,
        revisit_rate=sum(r['revisits'] for r in rows)/steps if steps else 0)


def audit():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=read(OUT/'预登记.json');tasks=read(OUT/'导航任务.json');summary=read(OUT/'对照汇总.json')
    for name,h in reg['sources_sha256'].items():check(digest(ROOT/name)==h,'input '+name)
    for name,h in reg['source_sha256'].items():
        check(digest(SRC/name)==h,'live source '+name)
        check(digest(OUT/'源码快照'/name)==h,'snapshot source '+name)
    for name,h in reg['frozen_files_sha256'].items():check(digest(OUT/name)==h,'frozen '+name)
    check(digest(DEFAULT)==reg['default_sha256'],'original default retained')
    check(tasks==select_tasks(read(OLD/'导航任务.json')),'score-free selection')
    check(len({t['area'] for t in tasks})==20 and Counter(t['dist'] for t in tasks)=={d:4 for d in range(4,9)},'task/source balance')
    check(read(OUT/'启动记录.json')['registration_sha256']==digest(OUT/'预登记.json'),'freeze before start')
    eps={t['episode_id']:Episode.from_dict(t) for t in tasks};banks=load_banks()
    rows=read_lines(OUT/'导航轨迹.jsonl');intents=read_lines(OUT/'请求意图.jsonl');responses=read_lines(OUT/'API响应.jsonl')
    check(len(rows)==200,'planned terminal records')
    check(len(intents)==len(responses)<=606,'intent-response count and totalcap')
    check([x['call_id'] for x in responses]==list(range(1,len(responses)+1)),'no duplicated/retried requests')
    replies={x['call_id']:x for x in responses};consumed=set()
    for i,r in zip(intents,responses):
        check(all(r[k]==v for k,v in i.items()),'request audit included in response')
        check(not any(k in r for k in ('reasoning','reasoning_content','api_key')),'no secrets/hidden reasoning')
    cfg=APIConfig(reg['model']['base_url'],reg['model']['model'],'not-used',reg['model']['timeout_seconds'],reg['model']['max_tokens'])
    preflight=read(OUT/'工程视觉诊断.json')
    for d in preflight['tasks']:
        if d['status']=='not_run':continue
        r=replies[d['call_id']];consumed.add(r['call_id'])
        cur=image_payload(MASA/f'patches/train/{d["area"]}/patch_{d["cell"]}.jpg')
        tgt=image_payload(MASA/f'patches/train/{d["area"]}/patch_{d["cell"] if d["expected"] else (d["cell"]+1)%25}.jpg')
        payload=dict(model=cfg.model,messages=[dict(role='system',content=PREFLIGHT_PROMPT),dict(role='user',content=image_part('target',tgt)+image_part('current',cur))],
                     temperature=0,max_tokens=cfg.max_tokens,response_format={'type':'json_object'},stream=False)
        check(digest_payload(payload)==r['request_sha256'],'preflight pixel input')
        if r['status']=='ok':
            obj,_=_parse_json_object(r['raw_content'],r['finish_reason'])
            check(set(obj)=={'same_pixels'} and type(obj['same_pixels']) is bool and obj==d['parsed'],'preflight parse')
            check(d['correct']==(obj['same_pixels']==d['expected']),'preflight identity metric')
    check(preflight['passed']==(preflight['valid']==6 and preflight['correct']>=4),'preflight gate')
    action_count=0;role_counts=Counter();seen=set();intervention_diagnostics=[]
    baseline={r['episode_id']:r for r in rows if r['arm']=='Edge'}
    for row in rows:
        key=(row['episode_id'],row['arm'],row['api_repeat']);check(key not in seen,'no duplicated task/arm/repeat');seen.add(key)
        check(row['local_checkpoint_seed']==0,'fixed local model, APIrepeats are notseeds')
        ep=eps[row['episode_id']]
        if row['status']=='not_run':
            check(row['success'] is False and row['sg']==8 and not row['trajectory'] and not row['decisions'],'missing conservative penalty')
            continue
        ag=load_local();ag.reset();env=GridWorldEnv(MASA);obs=env.reset(ep)
        memories={'planner':[],'verifier':[],'single':[]};frames=[];reviews=0
        for step,d in enumerate(row['decisions'],1):
            check(not env.done,'no actions after terminal')
            base=local_decision(ag,obs,ep,banks)
            check(base==d['base'],'full local checkpoint decision replay')
            check(d['action'] in boundary_legal(obs),'public legal action')
            interaction=d['interaction'];arm=row['arm']
            if arm=='Edge':
                check(interaction is None and d['action']==base['action'],'unaltered strong reference')
            else:
                trigger=reviews<2 and base['cue_action'] is not None and base['cue_action']!=base['explorer_action']
                check(trigger==interaction['triggered'],'same public conflict trigger')
                check(memories==interaction['role_memories_before'],'episode-local role messages')
                ids=interaction['call_ids'];items=[]
                if ids:
                    reviews+=1
                    check(interaction['review_index']==reviews,'review index')
                    check(len(ids)<=(1 if arm=='SingleOnce' else 2),'per-review requestcap')
                    for j,call_id in enumerate(ids):
                        r=replies[call_id];check(call_id not in consumed,'one request used once');consumed.add(call_id)
                        role='planner' if j==0 else 'verifier' if arm=='TwoAgent' else 'reflector'
                        memory=memories['planner' if arm=='TwoAgent' else 'single'] if j==0 else memories['verifier' if arm=='TwoAgent' else 'single']
                        proposal=(items[0] or {'invalid':True}) if role=='reflector' else None
                        _,expected=request_for(obs,base,frames,memory,role,cfg,proposal)
                        check(all(r[k]==v for k,v in expected.items()),'exact public request and permissions')
                        check(r['tag']==dict(arm=arm,repeat=row['api_repeat'],episode_id=ep.episode_id,step=step),'request stage tag')
                        if role=='verifier':check('current_proposal' not in r['public_input'],'blind current verification')
                        items.append(parsed_reply(r,boundary_legal(obs),r['image_labels']));role_counts[role]+=1
                    if arm!='SingleOnce' and len(items)==1:items.append(None)
                    check(items==interaction['recommendations'],'final messages agree with rawresponses')
                    action,reason=independent_coordination(arm,base['action'],items)
                    check((action,reason)==(d['action'],interaction['coordination_reason']),'independent arbitration')
                    shared=dict(position=list(obs.position),remaining_budget=obs.remaining_budget,planner=items[0],
                                checker=items[1] if len(items)==2 else None,executed_action=action)
                    if arm=='TwoAgent':
                        memories['planner'].append(dict(self_role='planner',exchange=shared));memories['verifier'].append(dict(self_role='verifier',exchange=shared))
                    else:memories['single'].append(dict(self_role='single',exchange=shared))
                else:
                    check(d['action']==base['action'] and interaction['coordination_reason'] in ('no_review','cloud_circuit_fallback'),'non-review/fallback original action')
                frames.append(dict(pixels=obs.current_image,position=list(obs.position),remaining_budget=obs.remaining_budget));frames=frames[-2:]
            obs,_,info=env.step(d['action']);check(not info.out_of_bounds,'no boundary mistakes');action_count+=1
        check(env.done,'terminal completion')
        check(all(row[k]==v for k,v in env.evaluator_result().items()),'trajectory/state/metrics terminal replay')
        if row['arm']!='Edge':
            ref=baseline[row['episode_id']]
            intervention_diagnostics.append(dict(episode_id=row['episode_id'],area=row['area'],arm=row['arm'],api_repeat=row['api_repeat'],
                corrected_failure=not ref['success'] and row['success'],damaged_success=ref['success'] and not row['success'],
                changed_actions=sum(d['action']!=d['base']['action'] for d in row['decisions'])))
    check(consumed==set(replies),'allcloud requests accounted including engineering')
    expected_keys={(ep.episode_id,'Edge',None) for ep in eps.values()}
    expected_keys|={(ep.episode_id,a,r) for ep in eps.values() for a in ('SingleOnce','SingleReflect','TwoAgent') for r in (0,1,2)}
    check(seen==expected_keys,'complete planned key set')
    for arm,runs in summary['runs'].items():
        for rep,m in runs.items():
            subset=[r for r in rows if r['arm']==arm and str(r['api_repeat'])==rep]
            expected=independent_metrics(subset)
            check(all(abs(expected[k]-m[k])<1e-12 for k in expected),'independent full metrics')
    sr=[];sg=[]
    for rep in (0,1,2):
        a=summary['runs']['TwoAgent'][str(rep)];b=summary['runs']['SingleReflect'][str(rep)]
        sr.append(a['SR_gate']-b['SR_gate']);sg.append(a['SG_gate']-b['SG_gate'])
    check(sr==summary['paired_SR_by_API_repeat'] and sg==summary['paired_SG_by_API_repeat'],'paired repetitions')
    source_sr=[];source_sg=[]
    for area in sorted({ep.area for ep in eps.values()}):
        a=[r for r in rows if r['area']==area and r['arm']=='TwoAgent'];b=[r for r in rows if r['area']==area and r['arm']=='SingleReflect']
        source_sr.append(np.mean([r['success'] for r in a])-np.mean([r['success'] for r in b]))
        source_sg.append(np.mean([r['sg'] for r in a])-np.mean([r['sg'] for r in b]))
    rng=np.random.default_rng(4121);idx=rng.integers(0,20,(4000,20))
    for value,name in [(source_sr,'source_SR_interval95'),(source_sg,'source_SG_interval95')]:
        interval=np.quantile(np.array(value)[idx].mean(1),[.025,.975])
        check(np.allclose(interval,summary[name],rtol=0,atol=1e-12),'paired sourcebootstrap')
    checks=dict(SR_gain_2pp=statistics.mean(sr)>=.02-1e-12,two_positive_API_repeats=sum(x>1e-12 for x in sr)>=2,
        SG_no_worse=statistics.mean(sg)<=1e-12,complete=all(r['status']=='completed' for r in rows),
        valid_cloud_responses=all(c['status']=='ok' for c in responses),preflight=preflight['passed'])
    check(checks==summary['checks'] and all(checks.values())==summary['candidate_numeric_passed'],'numeric releaseconditions')
    for path in (OUT/'请求意图.jsonl',OUT/'API响应.jsonl',OUT/'导航轨迹.jsonl'):
        text=path.read_text(encoding='utf-8')
        check('data:image/' not in text and '"reasoning_content"' not in text,'no image bytes or hiddenreasoning inlogs')
    diagnostics={a:dict(corrected_failures=sum(x['corrected_failure'] for x in intervention_diagnostics if x['arm']==a),
        damaged_successes=sum(x['damaged_success'] for x in intervention_diagnostics if x['arm']==a),
        changed_actions=sum(x['changed_actions'] for x in intervention_diagnostics if x['arm']==a)) for a in ('SingleOnce','SingleReflect','TwoAgent')}
    write_new(OUT/'纠正与损伤诊断.json',dict(scope='posthoc vs fixed Edge0 same20 tasks; not oracle input',by_arm=diagnostics,episodes=intervention_diagnostics))
    receipt=dict(passed=True,planned_records=200,completed_records=sum(r['status']=='completed' for r in rows),
        replayed_actions=action_count,cloud_requests=len(responses),role_calls=dict(role_counts),
        checks=['input/code/default hashes','score-free20source selection','allintent-response pairs','exact pixel request hashes',
                'no current proposal toverifier','isolated per-episode role memory','strict final JSONparse','checkpoint and environmentreplay',
                'independent coordination','planned-denominator metrics','paired source bootstrap','boundedcalls/no retry/nohiddenreasoning'],
        candidate_numeric_passed=summary['candidate_numeric_passed'],default_changed=False,training_steps=0,
        audit_authorship='separate implementation by same executing agent; no separate-person claim',
        registration_sha256=digest(OUT/'预登记.json'),summary_sha256=digest(OUT/'对照汇总.json'),
        trajectories_sha256=digest(OUT/'导航轨迹.jsonl'),responses_sha256=digest(OUT/'API响应.jsonl'),
        intents_sha256=digest(OUT/'请求意图.jsonl'))
    write_new(OUT/'独立复核.json',receipt)
    write_new(OUT/'验收结论.json',dict(audit_passed=True,candidate_passed=summary['candidate_numeric_passed'],
        allow_larger_development=summary['candidate_numeric_passed'],default_changed=False,new_source_files_used=0,
        formal_generalization_passed=False,target_mechanism_proven=False,audit_sha256=digest(OUT/'独立复核.json'),
        summary_sha256=digest(OUT/'对照汇总.json'),checks=checks))
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':audit()
