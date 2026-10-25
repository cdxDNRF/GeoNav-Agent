"""CV-018 only:249 extra-source image checks, then research-design freeze, no models."""
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_coverage_protocol_v1 import build_queue,sample_indices
from tools.preflight_capturepoint_sources_v1 import axis_scores,PROBES

BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'三新增来源工程与配对覆盖协议_v1'
CAPACITY=BASE/'训练正配对容量与划分追溯_v1'
NEW_SOURCES=['data/capturepoint_coverage_protocol_v1.py','tools/prepare_capturepoint_coverage_v1.py',
    'eval/audit_capturepoint_coverage_v1.py','tests/test_capturepoint_coverage_v1.py','documents/close_capturepoint_coverage_v1.py']
OLD_MATH=['tools/preflight_capturepoint_sources_v1.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def entry(p):
    p=Path(p);raw=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def put(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def verify(items):
    for e in items:
        if entry(ROOT/e['path'])!=e:raise ValueError(('Bound file changed',e['path']))


def freeze():
    if OUT.exists():raise FileExistsError('Existing/partial batch: inspect before any recovery')
    cap=read(CAPACITY/'训练容量与取样追溯汇总.json');proposal=read(CAPACITY/'九对与十二对同预算开发候选草案_未执行.json')
    ids=sorted(x['additional_source'] for x in cap['additional_fit_candidates'])
    if ids!=['mmgag_031','mmgag_050','mmgag_065']:raise ValueError('Only approved3 fit candidates')
    roles={r['source_id']:r for r in read(BASE/'同任务来源用途账本_v1.json')['records']}
    catalogue={s['id']:s for s in read(BASE/'数据核验与跨视角准入_v1/逐源准入清单.json')['sources']}
    raw={r['path']:r for r in read(BASE/'数据核验与跨视角准入_v1/输入文件与源码绑定.json')['raw_inputs']}
    selected=[];expected=[]
    for sid in ids:
        r=roles[sid];s=catalogue[sid]
        if r['role']!='fit_preparation' or r['representative']:raise ValueError('Only extra fit members, never old14/dev/confirmation')
        prefix=s['aerial_directories'][0]
        stitches=sorted(x['path'] for x in s['auxiliary_images_excluded_from_grid'] if x['valid'] and x['size']==[2816,2816])
        if not stitches:raise ValueError('Native11x11 reference missing; do not replace source')
        source={'source_id':sid,'group':r['group'],'role':r['role'],'ground':raw[s['ground_files'][0]],
            'tiles':[raw[prefix+f'/sat_{i}_{j}.jpg'] for i in range(1,10) for j in range(1,10)],'stitch':raw[stitches[0]]}
        selected.append(source);expected.extend([source['ground'],*source['tiles'],source['stitch']])
    if len(expected)!=249 or len({e['path'] for e in expected})!=249:raise ValueError('249 unique pixel inputs required')
    OUT.mkdir();(OUT/'核验').mkdir()
    authorization={'task':'CV-018','date':'2026-10-18','user_request':'开始：先核验新增3份素材，再冻结9对与12对、同8192参数和4500次样本使用的对照',
        'engineering_sources':ids,'raw_reads_and_decodes_max':249,'freeze_design_authorized':True,
        'model_forward_authorized':False,'training_authorized':False,'confirmation_access_authorized':False,'role_changes_authorized':False}
    put('新增三来源工程授权.json',authorization)
    put('首像素冻结范围.json',{'task':'CV-018','sources':selected,'raw_expected':expected,'maximum_raw_reads':249,'maximum_decodes':249,
        'axis_probes':[list(p) for p in PROBES],'axis_gate':'inherited8 probes:row best, MAE<=8, runner-up margin>=3',
        'quality_gate':'every decode/dimension; ground and center16px BOX max-channel std>1; other uniform cells diagnostic only',
        'stopping':'any required source failure closes engineering; no source or threshold replacement',
        'source_roles_unchanged':True,'model_calls':0,'training_steps':0,'confirmation_access':0,
        'thumbnail_max_bytes':512*1024,'output_max_bytes':4*1024**2})
    paths=[CAPACITY/'阶段封存.json',CAPACITY/'九对与十二对同预算开发候选草案_未执行.json',CAPACITY/'训练容量与取样追溯汇总.json',
        BASE/'同任务来源用途账本_v1.json',BASE/'数据核验与跨视角准入_v1/逐源准入清单.json',BASE/'数据核验与跨视角准入_v1/输入文件与源码绑定.json',
        BASE/'同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json',BASE/'同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json',
        BASE/'同任务拍摄点对齐执行_v1/首forward绑定.json',BASE/'同任务拍摄点对齐执行_v1/首更新前特征绑定.json',
        BASE/'同任务拍摄点对齐执行_v1/阶段封存.json',BASE/'同任务拍摄点对齐执行_v1/开发对照汇总.json']
    protected=read(BASE/'同任务拍摄点对齐执行_v1/首forward绑定.json')['protected'][:4]
    put('输入与源码绑定.json',{'task':'CV-018','inputs':[entry(p) for p in paths],
        'sources':[entry(ROOT/'project/src'/p) for p in NEW_SOURCES],
        'readonly_math_dependencies':[entry(ROOT/'project/src'/p) for p in OLD_MATH],
        'protected':[entry(ROOT/e['path']) for e in protected],
        'protocol':entry(OUT/'首像素冻结范围.json'),'authorization':entry(OUT/'新增三来源工程授权.json'),
        'scope':'new3 pixel/axis checks and research design only; no old14 pixels, model/feature/weight loads, training, predictions or navigation'})
    print(json.dumps({'frozen_sources':ids,'raw_images':249,'models':0},ensure_ascii=False))


def preflight():
    if (OUT/'输入消费.jsonl').exists():raise FileExistsError('Pixels attempted; inspect existing receipts/caches, never blindly retry')
    binding=read(OUT/'输入与源码绑定.json');verify(binding['inputs']+binding['sources']+binding['readonly_math_dependencies']+binding['protected']+[binding['protocol'],binding['authorization']])
    scope=read(OUT/'首像素冻结范围.json');allowed={e['path'] for e in scope['raw_expected']};receipts=[];results=[]
    def event(data):
        with (OUT/'输入消费.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n');f.flush()
    def decode(e,ground=False):
        path=(ROOT/e['path']).resolve()
        if e['path'] not in allowed or not path.is_relative_to((ROOT/'DATA/raw_data/MM-GAG').resolve()) or any(r['path']==e['path'] for r in receipts):raise ValueError('Unplanned/repeated/outside raw input')
        event({'event':'attempt','path':e['path'],'index':len(receipts)})
        raw=path.read_bytes()
        if len(raw)!=e['bytes'] or hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('Raw bytes differ from frozen original')
        with Image.open(io.BytesIO(raw)) as image:
            rec=dict(e,size=list(image.size),orientation=int(image.getexif().get(274,1)))
            rgb=(ImageOps.exif_transpose(image) if ground else image).convert('RGB').copy()
        rgb.info.clear();rec['sanitized_size']=list(rgb.size);receipts.append(rec);event({'event':'decoded','path':e['path'],'index':len(receipts)-1})
        return rgb
    tiny=lambda im:np.asarray(im.resize((16,16),Image.Resampling.BOX),np.uint8)
    try:
        for source in scope['sources']:
            start=len(receipts);g=decode(source['ground'],True);gt=tiny(g);tile=[]
            for e in source['tiles']:
                image=decode(e)
                if image.size!=(256,256):raise ValueError('Tile dimensions changed')
                tile.append(tiny(image))
            stitch=decode(source['stitch'])
            if stitch.size!=(2816,2816):raise ValueError('Native11x11 dimensions changed')
            probes=np.stack([tiny(stitch.crop((c*256,r*256,(c+1)*256,(r+1)*256))) for r,c in PROBES]);tiles=np.stack(tile)
            axis=axis_scores(tiles,probes);gstd=float(gt.astype(np.float64).std(axis=(0,1)).max());cstd=float(tiles[40].astype(np.float64).std(axis=(0,1)).max())
            cache=OUT/'核验'/f"{source['source_id']}_缩略.npz"
            np.savez_compressed(cache,ground=gt,tiles=tiles,probes=probes)
            result={'source_id':source['source_id'],'group':source['group'],'role':source['role'],'raw_decodes':len(receipts)-start,
                'axis':axis,'ground_std16_max':gstd,'center_std16_max':cstd,'other_uniform_cells_diagnostic_only':int((tiles.astype(float).std(axis=(1,2)).max(axis=1)<=1).sum()),
                'passed':axis['passed'] and gstd>1 and cstd>1,'thumbnail':entry(cache),'source_labels_unchanged':True,'GPS_to_pixel_verified':False}
            results.append(result);put('核验/'+source['source_id']+'_回执.json',{'result':result,'receipts':receipts[start:]})
            print(source['source_id']+': '+('PASS' if result['passed'] else 'FAIL')+', axis='+axis['best'],flush=True)
        put('像素工程汇总.json',{'task':'CV-018','sources':results,'planned_sources':3,'raw_reads':len(receipts),'raw_decodes':len(receipts),
            'all_required_passed':all(x['passed'] for x in results),'model_calls':0,'training_steps':0,'new_predictions':0,'confirmation_access':0})
        if sum(p.stat().st_size for p in (OUT/'核验').glob('*_缩略.npz'))>512*1024:raise ValueError('Thumbnail cap exceeded')
        print(json.dumps({'engineering_passed':all(x['passed'] for x in results),'raw_decodes':len(receipts),'models':0},ensure_ascii=False))
    except BaseException as exc:
        put('中断记录.json',{'error':type(exc).__name__,'message':str(exc),'successful_decodes':len(receipts),'sources_completed':len(results),
            'recovery':'inspect append-only consumption log and per-source receipts; no automatic pixel retry'});raise


def protocol():
    if not read(OUT/'像素工程汇总.json')['all_required_passed'] or not read(OUT/'核验/工程另实现复核.json')['passed']:raise ValueError('All3 engineering and alternate audit required')
    binding=read(OUT/'输入与源码绑定.json');verify(binding['inputs']+binding['sources']+binding['readonly_math_dependencies']+binding['protected'])
    scope=read(OUT/'首像素冻结范围.json');oldq=read(BASE/'同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
    queue=build_queue(oldq,scope['sources']);put('固定训练与开发队列.json',queue)
    old=read(BASE/'同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json')
    draft=copy.deepcopy(read(CAPACITY/'九对与十二对同预算开发候选草案_未执行.json'))
    draft.update({'task':'CV-018','protocol_id':'capturepoint-rank8-coverage12-v1','status':'research_design_frozen_engineering_passed_not_executed',
        'research_design_frozen':True,'source_engineering_passed':True,'training_authorized':False,'model_forward_authorized':False,
        'actual_executor_binding':None,'actual_feature_binding':None,'candidate_final_checkpoint':None,
        'engineers':entry(OUT/'像素工程汇总.json'),'engineering_audit':entry(OUT/'核验/工程另实现复核.json'),
        'fixed_queue':entry(OUT/'固定训练与开发队列.json'),'gate_function':'project/src/data/capturepoint_coverage_protocol_v1.py::coverage_gate',
        'only_new_factor':'9->12 distinct training pairs in unchanged9fit groups; identical batch9/500updates/4500query uses and frozen8192head/encoders',
        'mean_ground_indices':list(range(9)),'labels':'original CSV/code center proxy; not visible-landmark relabel or independent geographic survey',
        'encoder_assets_reference_only':old['encoder_assets'],'preprocessing':old['preprocessing'],
        'prior_feature_binding_reference_only':entry(BASE/'同任务拍摄点对齐执行_v1/首更新前特征绑定.json'),
        'baseline_seal_reference':entry(BASE/'同任务拍摄点对齐执行_v1/阶段封存.json'),
        'current_activity':{'new_encoder_inputs':0,'new_model_forward_calls':0,'new_training_steps':0,'new_rankings':0,'navigation_SR':None,'confirmation_access':0}})
    draft.pop('not_authorized_by_this_inventory',None)
    draft['execution_admission_required']=['implement/bind real sampler-aware executor and engineering tests before model forward',
        'reuse and verify old features/checkpoints/1250 saved rankings with matching tensor/raw/preprocessing/model identities',
        'bind all actual new feature tensors/fit12/mean9 before first update',
        'explicit later runtime authorization; one fixed final run, no development-driven tuning or confirmation consumption']
    draft['sampler']['mathematically_verified_uses']=[sum(i==k for t in range(500) for i in sample_indices(12,t)) for k in range(12)]
    draft['evaluation']['actual_new_rankings']=0
    put('配对覆盖同预算冻结协议.json',draft)
    put('协议源码与输入绑定.json',{'sources':binding['sources'],'readonly_math_dependencies':binding['readonly_math_dependencies'],
        'inputs':binding['inputs'],'protocol':entry(OUT/'配对覆盖同预算冻结协议.json'),'queue':entry(OUT/'固定训练与开发队列.json'),
        'engineering':entry(OUT/'像素工程汇总.json'),'engineering_audit':entry(OUT/'核验/工程另实现复核.json'),
        'actual_model_executor_bound':False,'actual_features_bound':False,'training_executed':False})
    print(json.dumps({'research_design_frozen':True,'training_queries':12,'query_uses':4500,'planned_rankings':1625,'training_executed':False},ensure_ascii=False))


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('freeze','preflight','protocol'):raise SystemExit('freeze | preflight | protocol; each once')
    {'freeze':freeze,'preflight':preflight,'protocol':protocol}[sys.argv[1]]()
