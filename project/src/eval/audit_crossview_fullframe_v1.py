"""CV-005 independent pixel equations and scalar recomputation; no model imports."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'拍摄点语义与完整目标图验证_v1';PREV=BASE/'跨视角感知最小验证_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def unit(v):
    x=list(map(float,v));norm=math.sqrt(math.fsum(t*t for t in x));assert norm>1e-12
    return [t/norm for t in x]
def audit(save=False):
    state=read(OUT/'执行状态.json');assert state['state']=='completed'
    first=read(OUT/'首forward前绑定.json');checked=0
    for key in ['files','sources','dependencies','raw_used','protected','references','previous_seals']:
        for e in first[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path'];checked+=1
    prior=0
    for entry in first['previous_seals']:
        old=read(ROOT/entry['path'])
        for e in old['files']+old['sources']:assert sha(ROOT/e['path'])==e['sha256'];prior+=1
    task=read(OUT/'冻结16对副本.json');assert task==read(PREV/'冻结执行任务.json');pairs=task['pairs']
    split=read(PREV/'划分与任务.json');dev={s for group in split['dev_groups'] for s in group};reserved={s for group in split['reserved_groups'] for s in group};assert not dev&reserved
    assert len(pairs)==16 and len(split['dev_groups'])==10 and len(split['reserved_groups'])==20 and all(p['source_id_evaluator_only'] in dev for p in pairs)
    semantics=read(OUT/'目标语义约定.json');assert semantics['localization_target']=='capture_point' and semantics['unchanged_pairs']==16 and semantics['source_center']==[5,5]
    engineering={p['pair_id']:p for p in read(OUT/'零forward工程核验.json')['pairs']};cfg=read(ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json');mean=np.array(cfg['image_mean'],np.float32);std=np.array(cfg['image_std'],np.float32)
    pixelchecks=0
    for p in pairs:
        with Image.open(ROOT/p['ground_path']) as im:rgb=ImageOps.exif_transpose(im).convert('RGB');w,h=rgb.size
        # Independent scalar geometry, no import of candidate processor.
        nw=224 if w>=h else max(1,int(224*w/h));nh=224 if h>=w else max(1,int(224*h/w));left=(224-nw)//2;top=(224-nh)//2
        body=np.asarray(rgb.resize((nw,nh),Image.Resampling.BICUBIC),np.float32)/np.float32(255);full=np.zeros((3,224,224),np.float32)
        for channel in range(3):full[channel,top:top+nh,left:left+nw]=(body[:,:,channel]-mean[channel])/std[channel]
        ref=engineering[p['pair_id']];assert ref['fullframe_input_sha256']==hashlib.sha256(full.tobytes()).hexdigest();pixelchecks+=1
        assert ref['geometry']['content_box']==[left,top,left+nw,top+nh] and ref['geometry']['source_region']==[0,0,w,h]
        cw=224 if w<h else int(224*w/h);ch=224 if h<=w else int(224*h/w)
        arr=np.asarray(rgb.resize((cw,ch),Image.Resampling.BICUBIC).crop(((cw-224)//2,(ch-224)//2,(cw-224)//2+224,(ch-224)//2+224)),np.float32)/np.float32(255)
        crop=np.ascontiguousarray(((arr-mean)/std).transpose(2,0,1));assert ref['crop_input_sha256']==hashlib.sha256(crop.tobytes()).hexdigest();pixelchecks+=1
    resource=read(OUT/'运行资源.json');protocol=read(OUT/'冻结协议.json');replay=read(OUT/'原裁剪复验.json')
    for e in resource['features']:assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['bytes']
    old=np.load(PREV/'地面特征.npy',allow_pickle=False);crop=np.load(OUT/'CropReplay地面特征.npy',allow_pickle=False)
    delta=float(np.abs(old.astype(float)-crop.astype(float)).max());assert delta<=protocol['crop_replay_tolerance'] and delta==replay['maximum_delta'] and replay['fullframe_not_started_before_pass']
    aerial=np.load(PREV/'航拍特征.npy',allow_pickle=False);assert aerial.shape==(400,512)
    report=read(OUT/'完整目标图对照汇总.json');rows=read(OUT/'逐对排序结果.json')['rows'];assert len(rows)==96
    lookup={(x['arm'],x['pair_id'],x['condition']):x for x in rows};assert len(lookup)==96
    index={p['pair_id']:i for i,p in enumerate(pairs)};maxdiff=0;outcomes={}
    for arm in ['CropReplay','FullFrame']:
        feat=np.load(OUT/(arm+'地面特征.npy'),allow_pickle=False);assert feat.shape==(16,512) and feat.dtype==np.float32 and np.isfinite(feat).all()
        g=[unit(v) for v in feat];avg=unit([math.fsum(g[i][j] for i in range(16))/16 for j in range(512)])
        counts=dict.fromkeys(['Full','Mean','Wrong'],0);top5=dict(counts);ranks={k:[] for k in counts};groups={}
        for i,p in enumerate(pairs):
            grp=groups.setdefault(p['group_id'],{'planned':0,**dict.fromkeys(counts,0)});grp['planned']+=1
            for condition,target in [('Full',g[i]),('Mean',avg),('Wrong',g[index[p['wrong_target_pair_id']]])]:
                scores=[math.fsum(a*b for a,b in zip(unit(aerial[i*25+j]),target)) for j in range(25)];row=lookup[arm,p['pair_id'],condition]
                maxdiff=max(maxdiff,max(abs(a-b) for a,b in zip(scores,row['scores'])))
                ranking=sorted(range(25),key=lambda j:(-row['scores'][j],hashlib.sha256(f"CV-002/tie/v1:{p['pair_id']}:{j}".encode()).hexdigest()));pos=ranking.index(p['goal_index_evaluator_only'])+1
                assert row['ranking']==ranking and row['true_rank']==pos and row['top1']==(pos==1) and row['top5']==(pos<=5)
                counts[condition]+=pos==1;top5[condition]+=pos<=5;grp[condition]+=pos==1;ranks[condition].append(pos);outcomes[arm,p['pair_id'],condition]=pos==1
        s=report['arms'][arm];assert counts==s['top1_counts'] and top5==s['top5_counts'] and groups==s['per_group'] and {k:statistics.median(v) for k,v in ranks.items()}==s['median_true_rank']
        positive=sum(x['Full']>x['Mean'] and x['Full']>x['Wrong'] for x in groups.values());assert positive==s['positive_groups_against_both']
        gates={'all16_complete':True,'full_top1_at_least4':counts['Full']>=4,'full_minus_mean_at_least2':counts['Full']-counts['Mean']>=2,'full_minus_wrong_at_least2':counts['Full']-counts['Wrong']>=2,'positive_groups_at_least3':positive>=3}
        assert gates==s['gates'] and all(gates.values())==s['candidate_passed']
        for ctl in ['Mean','Wrong']:
            rec=sum(outcomes[arm,p['pair_id'],'Full'] and not outcomes[arm,p['pair_id'],ctl] for p in pairs);harm=sum(outcomes[arm,p['pair_id'],ctl] and not outcomes[arm,p['pair_id'],'Full'] for p in pairs)
            assert s['target_contrasts'][ctl]=={'recovered':rec,'harmed':harm,'net':rec-harm}
    assert maxdiff<=5e-12
    for metric in ['top1','top5']:
        rec=sum(lookup['FullFrame',p['pair_id'],'Full'][metric] and not lookup['CropReplay',p['pair_id'],'Full'][metric] for p in pairs)
        harm=sum(lookup['CropReplay',p['pair_id'],'Full'][metric] and not lookup['FullFrame',p['pair_id'],'Full'][metric] for p in pairs)
        assert report['fullframe_vs_crop'][metric]=={'recovered':rec,'harmed':harm,'net':rec-harm}
    original=read(PREV/'感知对照汇总.json')
    assert report['arms']['CropReplay']['top1_counts']==original['top1_counts'] and report['arms']['CropReplay']['top5_counts']==original['top5_counts']
    assert state['ground_input_attempts']==resource['ground_input_attempts']==32 and state['successful_forward_calls']==resource['successful_forward_calls']==4
    assert state['CropReplay_done']==state['FullFrame_done']==16 and resource['input_tensor_hash_checks']==32
    assert resource['parameters_used']==87849216 and resource['model_frozen'] and resource['peak_cuda_allocated_bytes']<=2*1024**3
    for key in ['new_aerial_forward_calls','training_steps','navigation_episodes','model_api_calls','downloads','reserved_encoded']:assert resource[key]==0
    assert report['candidate_passed']==report['arms']['FullFrame']['candidate_passed']==state['candidate_passed'] and report['navigation_SR'] is None
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=protocol['output_limit_bytes']
    result={'task':'CV-005','passed':True,'binding_checks':checked,'old_sealed_files_sources_checked':prior,'planned_pairs':16,'development_groups':10,'reserved_groups_untouched':20,
      'model_input_hashes_independently_recomputed':pixelchecks,'crop_global_replay_max_delta':delta,'scalar_scores_recomputed':2400,'maximum_score_delta':maxdiff,
      'candidate_passed':report['candidate_passed'],'new_model_forward_calls_in_audit':0,'independent_implementation':True,'independent_person_review':False,'output_bytes':total,'navigation_SR':None}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
