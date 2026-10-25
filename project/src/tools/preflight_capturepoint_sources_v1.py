"""CV-011 authorized pixel/axis preflight. No model framework or navigation."""
import argparse,hashlib,io,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
BASE=ROOT/'DATA/processed_data/MM-GAG';PLAN=BASE/'同任务来源与监督资料规划_v1'
OUT=BASE/'同任务首批图像轴向预检_v1';LEDGER=BASE/'同任务来源用途账本_v1.json'
SOURCES=['tools/preflight_capturepoint_sources_v1.py','eval/audit_capturepoint_preflight_v1.py',
         'tests/test_capturepoint_preflight_v1.py','documents/close_capturepoint_preflight_v1.py']
PROBES=[(1,2),(2,1),(2,8),(8,2),(9,1),(1,9),(5,6),(6,5)]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def entry(p):
    p=Path(p);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def verify(entries):
    for e in entries:assert sha(ROOT/e['path'])==e['sha256'],e['path']
def thumb(image):return np.array(image.resize((16,16),Image.Resampling.BOX),np.uint8)
def clean_pair(target,current):
    result={}
    for k,im in [('target_rgb',target),('current_rgb',current)]:
        a=np.array(im.convert('RGB'),np.uint8);a.setflags(write=False);result[k]=a
    return result
def mapped(name,r,c):
    return {'row':(r,c),'column':(c,r),'flip_row':(10-r,c),'flip_column':(r,10-c),
            'rotate180':(10-r,10-c),'rotate90':(c,10-r),'rotate270':(10-c,r),'antitranspose':(10-c,10-r)}[name]
def axis_scores(tiles,canvas):
    scores={}
    for name in ['row','column','flip_row','flip_column','rotate180','rotate90','rotate270','antitranspose']:
        values=[]
        for j,(r,c) in enumerate(PROBES):
            y,x=mapped(name,r,c);candidate=tiles[(y-1)*9+x-1]
            values.append(float(np.abs(candidate.astype(np.float64)-canvas[j].astype(np.float64)).mean()))
        scores[name]=float(np.mean(values))
    order=sorted(scores,key=lambda n:(scores[n],n));margin=scores[order[1]]-scores[order[0]]
    return {'scores':scores,'best':order[0],'margin':margin,'passed':scores['row']<=8 and order[0]=='row' and margin>=3}


def freeze():
    assert not OUT.exists() and not LEDGER.exists(),'Inspect existing state; no overwrite'
    pseal=read(PLAN/'阶段封存.json');verify(pseal['files']+pseal['sources'])
    protected=read(PLAN/'输入与源码绑定.json')['protected'];verify(protected)
    proposal=read(PLAN/'待审九五六分组.json');refs=read(PLAN/'候选代表与监督引用.json')
    split=read(BASE/'跨视角感知最小验证_v1/划分与任务.json')
    raw={e['path']:e for e in read(BASE/'数据核验与跨视角准入_v1/输入文件与源码绑定.json')['raw_inputs']}
    selected=[];expected=[]
    for r in refs['representatives']:
        if r['proposed_role']=='confirmation':continue
        assert len(r['stitched_reference_inventory_paths'])>=1
        tiles=[refs['references'][i] for i in r['aerial_cell_reference_indices']]
        selected.append({'source_id':r['source_id_evaluator_only'],'role':r['proposed_role'],
                         'group':r['proposal_group_id'],'ground':r['ground_reference'],
                         'tiles':[raw[t['path']] for t in tiles],
                         'stitch':raw[sorted(r['stitched_reference_inventory_paths'])[0]]})
        expected.extend([selected[-1]['ground'],*selected[-1]['tiles'],selected[-1]['stitch']])
    assert len(selected)==14 and len(expected)==1162 and len({e['path'] for e in expected})==1162
    OUT.mkdir();(OUT/'核验').mkdir()
    approval={'task':'CV-011','date':'2026-10-17','question_id':'call_3f1454395d3243ac9765da1423b6cef2',
        'user_answer':'采用草案，进行14来源工程预检（推荐）','approved_proposal':entry(PLAN/'待审九五六分组.json'),
        'scope':'9fit-preparation,5development,6confirmation withheld;14representative pixel/axis preflight only',
        'training_authorized':False,'model_forward_authorized':False,'confirmation_pixels_authorized':False}
    put('用户授权记录.json',approval)
    records=[{'source_id':sid,'group':'legacy_dev_'+str(i),'role':'known_development'} for i,g in enumerate(split['dev_groups']) for sid in g]
    for g in proposal['groups']:
        for sid in g['members']:
            records.append({'source_id':sid,'group':g['proposal_group_id'],
                'role':{'fit':'fit_preparation','development':'development','confirmation':'confirmation_reserved'}[g['proposed_role']],
                'representative':sid==g['representative_source_id'],'pixel_scope':'this_preflight_representative_only' if g['proposed_role']!='confirmation' else 'withheld',
                'consumption_log':(OUT/'输入消费.jsonl').relative_to(ROOT).as_posix()})
    ledger={'type':'current_source_roles','authorization':entry(OUT/'用户授权记录.json'),'records':records,
            'historical_roles_snapshot':entry(BASE/'跨视角感知最小验证_v1/划分与任务.json'),
            'current_preflight':OUT.relative_to(ROOT).as_posix(),'training_started':False,'confirmation_opened':False}
    text=json.dumps(ledger,ensure_ascii=False,indent=2)
    with LEDGER.open('x',encoding='utf-8') as f:f.write(text)
    put('用途激活快照.json',text)
    put('首像素冻结协议.json',{'task':'CV-011','sources':selected,'raw_expected':expected,'unique_images':1162,
        'maximum_decodes':1162,'audit_thumbnail_cache_max_bytes':2*1024**2,'whole_output_max_bytes':8*1024**2,
        'axis_probes':[list(v) for v in PROBES],'axis_gate':'row-first,8-probe mean absolute error<=8 and margin>=3; inherited CV-002 thresholds, probes restricted1..9',
        'quality_gate':'all decodes/RGB, ground informative and center informative:16px BOX channel std max>1; other uniform negatives diagnostic only',
        'source_stop_rule':'if any required source gate fails, preserve failure; no replacement representative or threshold change',
        'actual_goal':'CSV/code camera-center proxy; not independently georeferenced',
        'training_steps':0,'model_calls':0,'navigation_episodes':0,'confirmation_sources':6,'confirmation_pixels':0})
    put('首像素输入源码绑定.json',{'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],
        'inputs':[entry(PLAN/n) for n in ['阶段封存.json','待审九五六分组.json','候选代表与监督引用.json']],
        'protocol':entry(OUT/'首像素冻结协议.json'),'approval':entry(OUT/'用户授权记录.json'),
        'role_snapshot':entry(OUT/'用途激活快照.json'),'current_role_at_start':entry(LEDGER),'protected':protected})
    print(json.dumps({'frozen':True,'sources':14,'images':1162,'roles_activated_before_pixels':True,'models':0}),flush=True)


def run():
    assert not (OUT/'输入消费.jsonl').exists(),'Consumed/partial preflight: inspect log; never blindly retry'
    binding=read(OUT/'首像素输入源码绑定.json')
    verify(binding['sources']+binding['inputs']+binding['protected']+[binding['protocol'],binding['approval'],binding['role_snapshot'],binding['current_role_at_start']])
    protocol=read(OUT/'首像素冻结协议.json');sources=protocol['sources']
    ground=np.lib.format.open_memmap(OUT/'地面缩略.npy',mode='w+',dtype=np.uint8,shape=(14,16,16,3))
    tiles=np.lib.format.open_memmap(OUT/'图块缩略.npy',mode='w+',dtype=np.uint8,shape=(14,81,16,16,3))
    canvas=np.lib.format.open_memmap(OUT/'拼图探针.npy',mode='w+',dtype=np.uint8,shape=(14,8,16,16,3))
    receipts=[];results=[]
    def log(value):
        with (OUT/'输入消费.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()
    def decode(e,is_ground=False):
        assert len(receipts)<1162
        p=(ROOT/e['path']).resolve();assert p.is_relative_to((ROOT/'DATA/raw_data/MM-GAG').resolve())
        log({'event':'attempt','index':len(receipts),'path':e['path']})
        data=p.read_bytes();actual=hashlib.sha256(data).hexdigest();assert actual==e['sha256'] and len(data)==e['bytes'],e['path']
        with Image.open(io.BytesIO(data)) as src:
            orientation=int(src.getexif().get(274,1));original=list(src.size)
            im=(ImageOps.exif_transpose(src) if is_ground else src).convert('RGB').copy()
        im.info.clear()
        rec={'index':len(receipts),'path':e['path'],'sha256':actual,'bytes':len(data),'original_size':original,
             'sanitized_size':list(im.size),'orientation':orientation,'output_mode':im.mode,'metadata_keys_after':0}
        receipts.append(rec);log({'event':'decoded','index':rec['index'],'path':e['path']});return im
    try:
        for i,s in enumerate(sources):
            g=decode(s['ground'],True);ground[i]=thumb(g);ground.flush();center=None
            for j,e in enumerate(s['tiles']):
                im=decode(e);assert im.size==(256,256)
                tiles[i,j]=thumb(im);tiles.flush()
                if j==40:center=im.copy()
            stitched=decode(s['stitch']);assert stitched.size==(2816,2816)
            for j,(r,c) in enumerate(PROBES):canvas[i,j]=thumb(stitched.crop((c*256,r*256,(c+1)*256,(r+1)*256)))
            canvas.flush();axis=axis_scores(tiles[i],canvas[i]);public=clean_pair(g,center)
            assert set(public)=={'target_rgb','current_rgb'} and all(not a.flags.writeable for a in public.values())
            gstd=float(ground[i].astype(np.float64).std(axis=(0,1)).max())
            cstd=float(tiles[i,40].astype(np.float64).std(axis=(0,1)).max())
            result={'source_id':s['source_id'],'group':s['group'],'role':s['role'],'axis':axis,
                'ground_std16_max':gstd,'center_std16_max':cstd,'input_only_two_readonly_RGB':True,
                'passed':axis['passed'] and gstd>1 and cstd>1,'raw_decodes':83,'goal_geographic_truth_verified':False}
            results.append(result);put('核验/'+s['source_id']+'.json',result)
            print(s['source_id']+': '+('PASS' if result['passed'] else 'FAIL')+', axis='+axis['best'],flush=True)
        passed=all(r['passed'] for r in results)
        put('逐图解码与字节回执.json',{'receipts':receipts,'count':len(receipts),'raw_hash_reads':len(receipts),'raw_decode_calls':len(receipts)})
        put('工程汇总.json',{'task':'CV-011','sources':results,'planned_sources':14,'raw_decodes':len(receipts),'all_required_passed':passed,
            'training_steps':0,'model_calls':0,'new_predictions':0,'navigation_episodes':0,'confirmation_pixels':0,'API_calls':0,'downloads':0,
            'training_ready':False,'next':'freeze same-task training protocol only if engineering passed; no training authorized this batch'})
        assert sum(p.stat().st_size for p in [OUT/'地面缩略.npy',OUT/'图块缩略.npy',OUT/'拼图探针.npy'])<=2*1024**2
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=8*1024**2
        print(json.dumps({'completed':True,'engineering_passed':passed,'raw_decodes':len(receipts),'models':0}),flush=True)
    except BaseException as exc:
        put('中断记录.json',{'error':type(exc).__name__,'message':str(exc),'completed_decodes':len(receipts),'completed_sources':len(results),
            'recovery':'inspect JSONL and cached coverage; no automatic rerun','pid':os.getpid()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);a=p.parse_args();freeze() if a.mode=='freeze' else run()
