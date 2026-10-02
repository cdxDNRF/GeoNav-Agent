"""Matched negative-only rotation study with exact original-head reproduction."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import EdgeTargetCueHead
from agents.decoupled_cue import decoupled_loss
from data.rotation_negative import extract,banks,matrix,negative_angles
from data.orientation_views import banks as view_banks
from agents.candidate_reliability import winners,threshold_metrics
from eval.candidate_calibration import ROOT,SRC,MASA,S2,OLD_DEV,DEFAULT,MEANS,SEEDS,source_rows,paths as reliability_paths,old_paths
from train.dyncur_tiny import digest,set_seed
from train.local_capacity import read,lines
from train.curiosity_controlled import write_new,append
DATA=MASA/'旋转负样本_v1/拟合特征'
OUT=MASA/'训练结果/旋转负样本视觉头对照_v1'
DOC=ROOT/'选题报告相关/旋转负样本视觉头对照方案_v1.md'
PLAN=ROOT/'选题报告相关/旋转负样本新源图计划_v1.json'
TESTS=ROOT/'选题报告相关/旋转负样本测试记录_v1.json'
ARMS=('ReplayR0','RotNeg')


def freeze():
    if OUT.exists() or DATA.exists():raise ValueError('immutable training batch exists')
    tests=read(TESTS)
    if not tests['successful'] or tests['failures'] or tests['errors']:raise ValueError('tests')
    OUT.mkdir(parents=True);split=read(OLD_DEV/'源图划分.json');sources=source_rows(split['head_fit'])
    for n,p in [('冻结方案.md',DOC),('新源图计划.json',PLAN),('测试记录.json',TESTS),('冻结默认配置.json',DEFAULT),('运行前README.md',ROOT/'README.md'),('源图划分.json',OLD_DEV/'源图划分.json'),('拟合配对.json',OLD_DEV/'fit_配对样本.json'),('诊断配对.json',OLD_DEV/'calibration_配对样本.json')]:
        (OUT/n).write_bytes(p.read_bytes())
    for n in MEANS:(OUT/n).write_bytes((S2/n).read_bytes())
    reference=[p for p in read(S2/'预登记.json')['plans'] if p['arm']=='Edge'];plans=[]
    for arm in ARMS:
        for p in reference:
            s=p['seed'];folder=OUT/f'{arm}_s{s}';folder.mkdir();initial=OLD_DEV/f'Edge_s{s}/初始化.pt'
            plan=dict(arm=arm,seed=s,folder=folder.name,initial_path=initial.relative_to(ROOT).as_posix(),initial_sha256=digest(initial),
                original_head_path=p['head_path'],original_head_sha256=p['head_sha256'],explorer_path=p['explorer_path'],explorer_sha256=p['explorer_sha256'],threshold=.5)
            (folder/'原初始化.pt').write_bytes(initial.read_bytes());(folder/'explorer.pt').write_bytes((ROOT/p['explorer_path']).read_bytes());write_new(folder/'配置.json',plan);plans.append(plan)
    files={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','data','env','eval','train','tests')}
    for n,p in files.items():dst=OUT/'源码快照'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    last=reliability_paths('development')[1];protected=read(last/'预登记.json')['historical_sha256'].copy();protected[last.relative_to(ROOT).as_posix()]={f.relative_to(last).as_posix():digest(f) for f in last.rglob('*') if f.is_file()}
    old_data={'DATA/processed_data/Masa/papr_train_sat_embeds_grid_5.npy':digest(MASA/'papr_train_sat_embeds_grid_5.npy')}
    for root in (reliability_paths('calibration')[0],old_paths('development')[0]):old_data.update({f.relative_to(ROOT).as_posix():digest(f) for f in root.rglob('*') if f.is_file()})
    reg=dict(version='rotation-negative-head-v1',utc=datetime.now(timezone.utc).isoformat(),sources=sources,plans=plans,arms=list(ARMS),seeds=list(SEEDS),
        epochs=16,batch_size=512,learning_rate=3e-4,weight_decay=.01,clip=1,parameters=136837,optimizer_steps_per_head=1632,pair_uses_per_head=835200,
        positive_pair_uses_per_head=111360,rotated_negative_uses_per_RotNeg=361920,each_angle_uses_per_RotNeg=120640,
        fit_pairs=52200,positive_pairs=6960,negative_pairs=45240,fit_sources=87,diagnostic_sources=22,held_sources=28,
        fixed_threshold=.5,threshold_calibration_steps=0,cloud_calls=0,encoder_frozen=True,explorer_frozen=True,
        model_training_started=False,model_evaluation_started=False,fit_cache_root=DATA.relative_to(ROOT).as_posix(),
        default_sha256=digest(DEFAULT),encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
        source_sha256={n:digest(p) for n,p in files.items()},historical_sha256=protected,read_only_data_sha256=old_data,
        frozen_inputs_sha256={f.name:digest(f) for f in OUT.iterdir() if f.is_file()},
        gate=dict(training_gain=.02,positive_seeds=2,rotation_gain=.05,clean_loss=.02,clean_SG_increase=.10,target_gain=.05))
    write_new(OUT/'预登记.json',reg);write_new(OUT/'执行状态.json',dict(status='registered',trained_models=0));return reg


def feature_pool(values,bank):
    pool=[];hashes={}
    for angle in (0,90,180,270):
        x=matrix(bank,values,angle);hashes[str(angle)]=sha256(x.tobytes()).hexdigest()
        if angle==0:
            count=0
            with (OLD_DEV/'Edge_s0/fit_Full_预测.jsonl').open('r',encoding='utf-8') as stream:
                for i,line in enumerate(stream):
                    row=json.loads(line)
                    if row['features_sha256']!=sha256(x[i].tobytes()).hexdigest():raise ValueError('original fitting feature mismatch')
                    count+=1
            if count!=52200:raise ValueError('original fit prediction count')
        pool.append(torch.as_tensor(x,device='cuda'));del x
    return pool,hashes


def train_one(plan,pool,labels):
    seed=plan['seed'];arm=plan['arm'];folder=OUT/plan['folder'];set_seed(seed);model=EdgeTargetCueHead().to('cuda')
    model.load_state_dict(torch.load(folder/'原初始化.pt',map_location='cuda',weights_only=True));torch.save(model.state_dict(),folder/'初始化.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);y=torch.as_tensor(labels,device='cuda');steps=uses=positive=0;exposures=np.zeros(4,np.int64);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    original_logs=lines(OLD_DEV/f'Edge_s{seed}/训练日志.jsonl')
    for epoch in range(16):
        order=torch.randperm(len(labels),device='cuda');order_hash=sha256(order.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
        if order_hash!=original_logs[epoch]['sample_order_sha256']:raise ValueError('original sample order mismatch')
        angles=np.zeros(len(labels),np.int64) if arm=='ReplayR0' else negative_angles(labels,epoch);angle_ids=torch.as_tensor(angles,device='cuda')
        loss_sum=binary_sum=direction_sum=0.;epoch_pos=0;epoch_counts=np.bincount(angles,minlength=4)
        for ids in order.split(512):
            if arm=='ReplayR0':x=pool[0][ids]
            else:
                # A row is assembled from exactly one given-target view; labels stay physical.
                choices=angle_ids[ids];x=pool[0][ids].clone()
                for angle_index in (1,2,3):
                    take=choices==angle_index;x[take]=pool[angle_index][ids[take]]
            loss,binary,directional=decoupled_loss(model.raw_logits(x),y[ids]);count=int((y[ids]<4).sum())
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids);binary_sum+=float(binary.detach())*len(ids);direction_sum+=float(directional.detach())*count;epoch_pos+=count
            steps+=1;uses+=len(ids);positive+=count
        exposures+=epoch_counts
        record=dict(epoch=epoch+1,loss=loss_sum/len(labels),adjacency_bce=binary_sum/len(labels),direction_ce=direction_sum/epoch_pos,optimizer_steps=steps,pair_uses=uses,
            adjacent_pair_uses=positive,sample_order_sha256=order_hash,target_view_counts=epoch_counts.tolist(),target_view_counts_total=exposures.tolist(),angle_plan_sha256=sha256(angles.tobytes()).hexdigest())
        append(folder/'训练日志.jsonl',record)
        if (epoch+1)%4==0:print(json.dumps(dict(model=plan['folder'],epoch=epoch+1,loss=record['loss'],steps=steps)),flush=True)
    torch.save(model.state_dict(),folder/'head.pt');write_new(folder/'训练资源.json',dict(parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,pair_uses=uses,positive_pair_uses=positive,
        view_counts=exposures.tolist(),seconds=time.monotonic()-began,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),includes_feature_pool=True,excludes_driver_and_other_processes=True))
    if arm=='ReplayR0':
        old=torch.load(ROOT/plan['original_head_path'],map_location='cpu',weights_only=True);new={k:v.cpu() for k,v in model.state_dict().items()}
        equal=all(torch.equal(new[k],old[k]) for k in old);diff=max(float((new[k]-old[k]).abs().max()) for k in old)
        write_new(folder/'原头复现.json',dict(all_parameters_exact=equal,maximum_abs_difference=diff,original_head_sha256=digest(ROOT/plan['original_head_path']),new_head_sha256=digest(folder/'head.pt')))
    del model;return steps


def diagnostic(reg):
    data=reliability_paths('calibration')[0]/'方向视图';values,payloads=view_banks(data);bank=read(OUT/'诊断配对.json');models={};probabilities={p['folder']:np.empty((13200,4,5),np.float32) for p in reg['plans']}
    for p in reg['plans']:
        h=EdgeTargetCueHead().eval();h.load_state_dict(torch.load(OUT/p['folder']/'head.pt',map_location='cpu',weights_only=True));models[p['folder']]=h
    with torch.inference_mode():
        for j,angle in enumerate((0,90,180,270)):
            x=matrix(bank,values,angle)
            for p in reg['plans']:
                key=p['folder'];h=models[key]
                for i in range(13200):probabilities[key][i,j]=h(torch.as_tensor(x[i])[None]).softmax(-1)[0].numpy()
            print(json.dumps(dict(diagnostic_angle=angle,predictions=13200*6)),flush=True)
    results={}
    for p in reg['plans']:
        key=p['folder'];folder=OUT/key;np.save(folder/'诊断四视图概率.npy',probabilities[key]);ids,raw=winners(probabilities[key],payloads,bank);np.save(folder/'诊断所选视图.npy',ids)
        m=threshold_metrics(raw,bank,.5);write_new(folder/'诊断结果.json',m);results[key]=m
    write_new(OUT/'诊断汇总.json',results)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);began=time.monotonic();reg=freeze()
    try:
        extract(DATA,MASA,OLD_DEV,reg['sources'],ROOT/'models/Sat2Cap');values=banks(DATA,MASA,OLD_DEV,reg['sources']);bank=read(OUT/'拟合配对.json');labels=np.array([r['label'] for r in bank],np.int64)
        write_new(OUT/'特征冻结结束.json',dict(model_training_started=False,files_sha256={f.relative_to(DATA).as_posix():digest(f) for f in DATA.rglob('*') if f.is_file()}))
        pool,hashes=feature_pool(values,bank);write_new(OUT/'拟合矩阵哈希.json',hashes);steps=0
        for i,p in enumerate(reg['plans']):
            steps+=train_one(p,pool,labels);(OUT/'执行状态.json').write_text(json.dumps(dict(status='training',trained_models=i+1,optimizer_steps=steps)),encoding='utf-8')
        del pool;torch.cuda.empty_cache();write_new(OUT/'全部训练结束.json',dict(models=6,optimizer_steps=steps,model_evaluation_started=False,heads_sha256={p['folder']:digest(OUT/p['folder']/'head.pt') for p in reg['plans']}))
        if not all(read(OUT/f'ReplayR0_s{s}/原头复现.json')['all_parameters_exact'] for s in SEEDS):raise ValueError('original heads not exactly reproduced; no evaluation allowed')
        diagnostic(reg)
        (OUT/'执行状态.json').write_text(json.dumps(dict(status='completed',trained_models=6,optimizer_steps=steps,diagnostic_predictions=316800,seconds=time.monotonic()-began)),encoding='utf-8')
    except Exception as ex:write_new(OUT/'执行异常.json',dict(type=type(ex).__name__,message=str(ex)));raise


if __name__=='__main__':main()
