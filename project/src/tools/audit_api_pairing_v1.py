"""Additional pure-JSON check of every API input/action pair and paired harms."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'平台/实验工作台_v4/单模型API统一接入与同题对照_v1'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def audit():
    gid=read(OUT/'真实API组身份.json')['id'];base=OUT/'工作区';d=read(base/'api_comparisons'/gid/'definition.json');s=read(base/'api_comparisons'/gid/'progress.json')
    tasks={t['key']:t for t in read(ROOT/'平台/实验工作台_v4/航拍统一接入与小队列_v1/首真实动作前绑定.json')['tasks']};used=[];cases=[];models=set();local_calls=0
    for row in s['rows']:
        if row['status']!='completed':continue
        path=next((base/'scenes').glob('*/runs/'+row['saved_ref']['id']+'.json'));raw=read(path);task=tasks[row['task_key']];visited=[task['start']];pos=task['start'];local_calls+=raw['policy_calls'];outside=revisits=0
        for i,e in enumerate(raw['events']):
            decision=e['decision'];cloud=decision['cloud'];index=cloud['journal_index'];used.append(index)
            request=read(base/'api_requests'/gid/f'{index:04d}.request.json');result=read(base/'api_requests'/gid/f'{index:04d}.result.json')['record'];public=request['public_input']
            if request['run_id']!=raw['id'] or public['public_state']!=dict(position=[pos//5,pos%5],grid_size=5,remaining_budget=10-i,visited=visited,legal_actions=['up','right','down','left']):raise ValueError('Public state/input pairing mismatch')
            if public['current_image_sha256']!=decision['current_image_sha256'] or public['target_image_sha256']!=decision['target_image_sha256'] or result['selected_action']!=e['action']:raise ValueError('Image/action pairing mismatch')
            if result['fallback'] and result['selected_action']!=result['local_fallback_action']:raise ValueError('Fallback action mismatch')
            model=result.get('transport',{}).get('provider_reported_model')
            if model:models.add(model)
            outside+=e['out_of_bounds'];revisits+=e['to'] in visited;pos=e['to'];visited.append(pos)
        cases.append(dict(task_key=row['task_key'],source=task['area'],distance=task['dist'],success=row['result']['success'],SG_cells=row['result']['SG_cells'],steps=row['step'],out_of_bounds=outside,revisits=revisits))
    count=len(list((base/'api_requests'/gid).glob('*.request.json')))
    if sorted(used)!=list(range(1,count+1)):raise ValueError('Unpaired/repeated API request')
    paired={}
    for c in ['M0','frontier']:
        pairs=[(d['baseline'][r['task_key']][c]['result']['success'],r['success']) for r in cases]
        paired[c]=dict(completed=len(pairs),planned=10,recovered=sum(not a and b for a,b in pairs),damaged=sum(a and not b for a,b in pairs),net=sum(int(b)-int(a) for a,b in pairs))
    groups={field:{str(v):dict(n=len(rows:=[r for r in cases if r[field]==v]),SR=sum(r['success'] for r in rows)/len(rows),SG_cells=sum(r['SG_cells'] for r in rows)/len(rows)) for v in sorted({r[field] for r in cases})} for field in ['source','distance']}
    return dict(passed=True,request_action_pairs=len(used),local_M0_invocations=local_calls,baseline_reexecuted=False,provider_reported_models=sorted(models),paired=paired,groups=groups,cases=cases,
        total_out_of_bounds=sum(r['out_of_bounds'] for r in cases),total_revisits=sum(r['revisits'] for r in cases),true_distance_used_only_posthoc=True,new_independent_geographical_sources=0,formal_evaluation=False,independent_personnel_review=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if __name__=='__main__':
    v=audit()
    with (OUT/'输入动作与恢复损伤复核.json').open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
    print(json.dumps(v,ensure_ascii=False))
