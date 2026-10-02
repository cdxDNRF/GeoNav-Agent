"""A separate, bounded diagnostic batch; never a navigation benchmark."""
import base64
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from agents.cloud_gateway import MODELS,GatewayConfig,make_gateway_policy,registered_options
from agents.vlm import APITrialError,_parse_json_object
from eval.local_ledger_expansion import ROOT,SRC,read,write,digest

OUT=ROOT/'DATA/processed_data/Masa/评测结果/云端8790接口检查_v1'
COLORS=dict(red=(235,30,30),green=(20,200,40),blue=(25,60,230),yellow=(245,225,20))
KEYS=('upper_left','upper_right','lower_left','lower_right')
ORDERS=(('red','green','blue','yellow'),('blue','red','yellow','green'))
SYSTEM='Return only the requested JSON object. Do not include reasoning or additional keys.'


def prepare():
    if OUT.exists():raise ValueError('immutable diagnostic output exists')
    registered_options();OUT.mkdir();(OUT/'诊断图像').mkdir()
    fixtures=[]
    for i,order in enumerate(ORDERS):
        image=Image.new('RGB',(240,240));draw=ImageDraw.Draw(image)
        for j,color in enumerate(order):
            r,c=divmod(j,2);draw.rectangle((c*120,r*120,c*120+119,r*120+119),fill=COLORS[color])
        path=OUT/f'诊断图像/quadrants_{i}.png';image.save(path)
        fixtures.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),expected=dict(zip(KEYS,order))))
    code=['agents/cloud_gateway.py','eval/gateway_smoke.py','tests/test_gateway_and_protection.py']
    for n in code:
        dest=OUT/'源码快照'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((SRC/n).read_bytes())
    reg=dict(version='8790-cloud-options-smoke-v1',utc=datetime.now(timezone.utc).isoformat(),
        providers=[GatewayConfig(m).public() for m in MODELS],max_model_list_requests=1,max_generation_requests=9,
        per_model='one text, then two counterfactual quadrant images iff text final JSON succeeds',
        no_retries=True,timeout_seconds=45,max_tokens=1024,temperature=0,stream=False,
        text_expected={'ok':True},visual_prompt='For the supplied image, name each quadrant color. Allowed values: red, green, blue, yellow. Return exactly upper_left, upper_right, lower_left, lower_right.',
        system=SYSTEM,fixtures=fixtures,new_navigation_episodes=0,new_training_steps=0,
        source_sha256={n:digest(SRC/n) for n in code},registry_sha256=digest(ROOT/'project/cloud_provider_options.json'),
        default_sha256=digest(ROOT/'project/local_policy_default.json'),
        frozen_https_source_sha256={n:digest(SRC/n) for n in ['agents/vlm.py','agents/curl_transport.py']})
    write(OUT/'预登记.json',reg);print(dict(gateway_registered=True,maximum_generations=9),flush=True)


def request(policy,payload,expected,kind,index,fixture=None):
    intent=dict(index=index,model=policy.config.model,kind=kind,utc=datetime.now(timezone.utc).isoformat(),
        payload_without_image_url={k:v for k,v in payload.items() if k!='messages'},
        system=SYSTEM,prompt=payload['messages'][-1]['content'] if kind=='text' else 'quadrant colors, frozen visual prompt',
        image_sha256=fixture['sha256'] if fixture else None,expected=expected,automatic_retries=0)
    with (OUT/'请求意图.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(intent,ensure_ascii=False)+'\n')
    try:
        data,record=policy._exchange('POST','/chat/completions',payload)
        record.update(index=index,requested_model=policy.config.model,kind=kind)
        try:
            ch=data['choices'][0];msg=ch['message'];content=msg.get('content');finish=ch.get('finish_reason')
            record.update(returned_model=data.get('model'),response_id=data.get('id'),usage=data.get('usage'),
                raw_content=content,finish_reason=finish,reasoning_present=bool(msg.get('reasoning') or msg.get('reasoning_content')))
            parsed,norm=_parse_json_object(content,finish)
            if not isinstance(parsed,dict) or set(parsed)!=set(expected):raise ValueError('strict keys required')
            record.update(parsed=parsed,format_normalization='markdown_fence' if norm else 'none',
                correct=all(type(parsed[k]) is type(v) and parsed[k]==v for k,v in expected.items()),status='ok')
        except (KeyError,IndexError,ValueError,TypeError,AttributeError):record.update(status='invalid_final_schema',correct=False)
    except APITrialError as exc:
        record=dict(exc.record,index=index,requested_model=policy.config.model,kind=kind,correct=False)
    with (OUT/'请求响应.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(dict(index=index,model=policy.config.model,kind=kind,status=record['status'],correct=record['correct'],returned_model=record.get('returned_model')),flush=True)
    return record


def run():
    if (OUT/'请求意图.jsonl').exists() or (OUT/'模型列表.json').exists():raise ValueError('batch is not resumable or repeatable')
    reg=read(OUT/'预登记.json')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source drift')
    records=[];index=0
    with make_gateway_policy(MODELS[0]) as policy:
        try:
            listing=policy.probe()
        except APITrialError as exc:listing=exc.record
        write(OUT/'模型列表.json',listing)
    print(dict(model_list_status=listing['status']),flush=True)
    # A transport failure on the single list request closes the batch; no repeated connection attempts.
    if listing['status']!='transport_error':
        for model in MODELS:
            with make_gateway_policy(model) as policy:
                def payload(content):return dict(model=model,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],temperature=0,max_tokens=1024,stream=False,response_format={'type':'json_object'})
                index+=1;record=request(policy,payload('Return exactly {"ok":true}.'),{'ok':True},'text',index);records.append(record)
                if not record['correct']:continue
                for fixture in reg['fixtures']:
                    encoded=base64.b64encode((ROOT/fixture['path']).read_bytes()).decode('ascii')
                    content=[dict(type='text',text=reg['visual_prompt']),dict(type='image_url',image_url={'url':'data:image/png;base64,'+encoded})]
                    index+=1;records.append(request(policy,payload(content),fixture['expected'],'image',index,fixture))
    write(OUT/'检查汇总.json',dict(models={m:dict(text_passed=any(r['kind']=='text' and r['correct'] for r in records if r['requested_model']==m),
        image_requests=sum(r['kind']=='image' for r in records if r['requested_model']==m),
        image_correct=sum(r['kind']=='image' and r['correct'] for r in records if r['requested_model']==m),
        returned_models=sorted({str(r.get('returned_model')) for r in records if r['requested_model']==m}),
        limited_visual_check_passed=sum(r['kind']=='image' and r['correct'] for r in records if r['requested_model']==m)==2)
        for m in MODELS},generation_requests=len(records),model_list_requests=1,
        successful_http=sum(r.get('http_status')==200 for r in records),new_navigation_episodes=0,
        default_changed=False,scope='bounded text and synthetic counterfactual image diagnostics only; no navigation or collaboration gain claim'))
    # Receipt independently checks all output schemas/pixels/counts, and preservation.
    from eval.local_ledger_expansion import lines
    intents=list(lines(OUT/'请求意图.jsonl')) if records else [];responses=list(lines(OUT/'请求响应.jsonl')) if records else []
    if len(intents)!=len(responses) or len(responses)>9:raise ValueError('request accounting')
    for i,(intent,response) in enumerate(zip(intents,responses),1):
        if intent['index']!=i or response['index']!=i or intent['model']!=response['requested_model']:raise ValueError('intent correspondence')
        if response['status']=='ok':
            parsed,_=_parse_json_object(response['raw_content'],response['finish_reason'])
            correct=set(parsed)==set(intent['expected']) and all(type(parsed[k]) is type(v) and parsed[k]==v for k,v in intent['expected'].items())
            if correct!=response['correct']:raise ValueError('final-answer scoring')
    for f in reg['fixtures']:
        if digest(ROOT/f['path'])!=f['sha256']:raise ValueError('fixture hash')
        with Image.open(ROOT/f['path']) as image:
            for j,(k,v) in enumerate(f['expected'].items()):
                # Expected dict keys sorted on disk; compute position from explicit key names.
                at=KEYS.index(k);rr,cc=divmod(at,2)
                if image.getpixel((cc*120+60,rr*120+60))!=COLORS[v]:raise ValueError('pixel truth')
    for n,h in reg['frozen_https_source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('HTTPS core drift')
    if digest(ROOT/'project/local_policy_default.json')!=reg['default_sha256']:raise ValueError('default drift')
    if digest(ROOT/'project/cloud_provider_options.json')!=reg['registry_sha256']:raise ValueError('registry drift')
    write(OUT/'复核.json',dict(passed=True,generation_requests=len(records),max_generation_requests=9,
        model_list_requests=1,final_answer_and_pixel_truth_verified=True,no_hidden_reasoning_saved=True,
        original_https_sources_unchanged=True,default_changed=False,summary_sha256=digest(OUT/'检查汇总.json')))
    print(read(OUT/'检查汇总.json'),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run-frozen',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run_frozen:run()
    else:parser.error('prepare or run-frozen required')
