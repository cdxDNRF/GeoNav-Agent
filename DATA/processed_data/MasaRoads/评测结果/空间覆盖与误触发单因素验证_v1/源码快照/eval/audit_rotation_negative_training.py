"""Independent pixel reconstruction, labels, matched training replay and frozen diagnostics."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train.rotation_negative import ROOT,SRC,MASA,S2,OLD_DEV,DATA,OUT,DOC,PLAN,TESTS,DEFAULT,MEANS,SEEDS,ARMS,reliability_paths
from agents.edge_cue import EdgeTargetCueHead
from env.episode import ACTIONS
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from eval.audit_candidate_calibration import manual_metrics
from eval.audit_orientation_confirmation import images as view_images
from eval import audit_trusted_cue as checks
from train.dyncur_tiny import digest
from train.local_capacity import read,lines
from train.curiosity_controlled import write_new
need=checks.need;same=checks.same;close=checks.close


def registration():
    r=read(OUT/'预登记.json');same(r['version'],'rotation-negative-head-v1','version');same(r['arms'],['ReplayR0','RotNeg'],'matched arms');same(r['seeds'],[0,1,2],'seeds')
    expected=dict(epochs=16,batch_size=512,learning_rate=3e-4,weight_decay=.01,clip=1,parameters=136837,optimizer_steps_per_head=1632,pair_uses_per_head=835200,positive_pair_uses_per_head=111360,rotated_negative_uses_per_RotNeg=361920,each_angle_uses_per_RotNeg=120640,fit_pairs=52200,positive_pairs=6960,negative_pairs=45240,fit_sources=87,diagnostic_sources=22,held_sources=28,fixed_threshold=.5,threshold_calibration_steps=0,cloud_calls=0)
    for k,v in expected.items():same(r[k],v,'fixed '+k)
    need(r['encoder_frozen'] and r['explorer_frozen'] and not r['model_training_started'] and not r['model_evaluation_started'],'registration before training/evaluation')
    need(read(OUT/'执行状态.json')['status']=='completed' and not (OUT/'执行异常.json').exists(),'training completed')
    for n,h in r['source_sha256'].items():need(digest(SRC/n)==digest(OUT/'源码快照'/n)==h,'source '+n)
    for folder,files in r['historical_sha256'].items():
        for n,h in files.items():need(digest(ROOT/folder/n)==h,'historical '+folder+'/'+n)
    for n,h in r['read_only_data_sha256'].items():need(digest(ROOT/n)==h,'read-only data '+n)
    for n,h in r['frozen_inputs_sha256'].items():need(digest(OUT/n)==h,'frozen input '+n)
    for original,n in [(DOC,'冻结方案.md'),(PLAN,'新源图计划.json'),(TESTS,'测试记录.json'),(DEFAULT,'冻结默认配置.json')]:need(digest(original)==digest(OUT/n),'live input '+n)
    need(digest(DEFAULT)==r['default_sha256'],'default unchanged');need(digest(ROOT/'models/Sat2Cap/model.safetensors')==r['encoder_sha256'],'encoder');need(digest(ROOT/'models/Sat2Cap/config.json')==r['encoder_config_sha256'],'encoder config')
    split=read(OUT/'源图划分.json');fit=set(split['head_fit']);cal=set(split['head_calibration']);held=set(split['held']);need(len(fit)==87 and len(cal)==22 and len(held)==28 and not fit&cal and not fit&held and not cal&held,'roles')
    need({s['area'] for s in r['sources']}==fit,'fit sources only');need(read(TESTS)['successful'] and read(TESTS)['errors']==read(TESTS)['failures']==0,'tests')
    for p in r['plans']:
        f=OUT/p['folder'];same(read(f/'配置.json'),p,'plan');need(digest(f/'原初始化.pt')==digest(ROOT/p['initial_path'])==p['initial_sha256'],'paired initial');need(digest(f/'explorer.pt')==digest(ROOT/p['explorer_path'])==p['explorer_sha256'],'unchanged explorer')
    for n in MEANS:need(digest(OUT/n)==digest(S2/n),'mean '+n)
    finish=read(OUT/'全部训练结束.json');need(finish['models']==6 and finish['optimizer_steps']==9792 and not finish['model_evaluation_started'],'all training before diagnostic')
    for p in r['plans']:
        need(digest(OUT/p['folder']/'head.pt')==finish['heads_sha256'][p['folder']],'frozen final head');need((OUT/p['folder']/'诊断四视图概率.npy').stat().st_mtime_ns>=(OUT/'全部训练结束.json').stat().st_mtime_ns,'no early selection')
    return r


def pairs(areas):
    result=[]
    for a in sorted(areas):
        for c in range(25):
            for t in range(25):
                if c==t:continue
                label=next((i for i,(dr,dc) in enumerate(ACTIONS.values()) if (t//5-c//5,t%5-c%5)==(dr,dc)),4);wrong=(t+1)%25
                while wrong in (c,t):wrong=(wrong+1)%25
                result.append(dict(area=a,current=c,target=t,label=label,wrong=wrong))
    return result


def fit_images(reg):
    frozen=read(OUT/'特征冻结结束.json');need(not frozen['model_training_started'],'feature freeze')
    for n,h in frozen['files_sha256'].items():need(digest(DATA/n)==h,'fit feature '+n)
    need(not list(DATA.rglob('*.png')),'no redundant fit images written');manifest=read(DATA/'数据清单.json');need(manifest['source_count']==87 and manifest['rotated_views']==6525 and manifest['PNG_files_written']==0,'manifest')
    origin=read(DATA/'旋转像素来源.json');need(len(origin)==6525,'all fit pixel provenance');values={};maxg=maxl=0.
    from train.dyncur_tiny import EmbeddingStore
    base_g=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy').data
    with np.load(OLD_DEV/'局部区域特征.npz') as f:base_l={a:f[a] for a in base_g}
    with np.load(OLD_DEV/'图块边缘profile.npz') as f:base_p={a:f[a] for a in base_g}
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to('cuda').eval();encoder.requires_grad_(False)
    with torch.inference_mode():
        for angle in (0,90,180,270):
            columns=( {s['area']:base_g[s['area']] for s in reg['sources']}, {s['area']:base_l[s['area']] for s in reg['sources']}, {s['area']:base_p[s['area']] for s in reg['sources']} ) if angle==0 else []
            if angle:
                for name in ('全局特征','局部特征','边缘profile'):
                    with np.load(DATA/f'R{angle}_{name}.npz') as f:columns.append({a:f[a] for a in f.files})
            for row in reg['sources']:
                area=row['area'];inputs=[];need(digest(ROOT/row['raw_path'])==row['raw_sha256'],'raw fit source')
                with Image.open(ROOT/row['raw_path']) as raw:
                    raw.load()
                    for cell in range(25):
                        rr,cc=divmod(cell,5);buf=BytesIO();raw.crop((cc*300,rr*300,cc*300+300,rr*300+300)).save(buf,format='JPEG',quality=75)
                        path=MASA/'patches/train'/area/f'patch_{cell}.jpg';need(buf.getvalue()==path.read_bytes(),'original fit JPEG from raw')
                        with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                        buf=BytesIO();Image.fromarray(rgb).save(buf,format='PNG');original=buf.getvalue();rot=np.rot90(rgb,k=-(angle//90)).copy();im=Image.fromarray(rot);buf=BytesIO();im.save(buf,format='PNG')
                        if angle:same(origin[f'R{angle}/{area}/{cell}'],dict(source_jpeg_sha256=digest(path),source_payload_sha256=sha256(original).hexdigest(),rotated_payload_sha256=sha256(buf.getvalue()).hexdigest()),'independent transformed pixels')
                        np.testing.assert_array_equal(manual_profile(rot),columns[2][area][cell]);pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                        x=((pixels-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)).transpose(2,0,1);inputs.append(torch.from_numpy(x))
                output=encoder(torch.stack(inputs).to('cuda'));g=output.image_embeds.cpu().numpy();grid=encoder.vision_model.post_layernorm(output.last_hidden_state[:,1:]).reshape(25,7,7,768)
                l=torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),1).cpu().numpy()
                maxg=max(maxg,float(np.abs(g-columns[0][area]).max()));maxl=max(maxl,float(np.abs(l-columns[1][area]).max()));np.testing.assert_array_equal(g,columns[0][area]);np.testing.assert_array_equal(l,columns[1][area])
            values[angle]=tuple(columns);print(json.dumps(dict(reconstructed_fit_angle=angle,images=2175)),flush=True)
    del encoder;torch.cuda.empty_cache();return values,dict(original_cells=2175,rotated_pixels=6525,global_max_abs_error=maxg,local_max_abs_error=maxl)


def manual_matrix(bank,values,angle):
    g,l,p=values[0];tg,tl,tp=values[angle];matrix=np.empty((len(bank),1061),np.float32)
    for start in range(0,len(bank),1024):
        part=bank[start:start+1024];current=np.stack([g[r['area']][r['current']] for r in part]);target=np.stack([tg[r['area']][r['target']] for r in part]);cl=np.stack([l[r['area']][r['current']] for r in part]);tl_batch=np.stack([tl[r['area']][r['target']] for r in part]);cp=np.stack([p[r['area']][r['current']] for r in part]);target_p=np.stack([tp[r['area']][r['target']] for r in part])
        matrix[start:start+len(part),:1041]=checks.manual_features(target,current,tl_batch,cl);matrix[start:start+len(part),1041:]=manual_seam(target_p,cp)
    return matrix


def train_replay(reg,values):
    bank=pairs(s['area'] for s in reg['sources']);same(read(OUT/'拟合配对.json'),bank,'independent52200 pair bank');labels=np.array([r['label'] for r in bank],np.int64);need(int((labels<4).sum())==6960,'positive count')
    hashes={};pool=[]
    for a in (0,90,180,270):
        x=manual_matrix(bank,values,a);hashes[str(a)]=sha256(x.tobytes()).hexdigest();pool.append(torch.as_tensor(x,device='cuda'));del x
    same(read(OUT/'拟合矩阵哈希.json'),hashes,'all fitting matrix bytes')
    with torch.inference_mode(False):
        for plan in reg['plans']:
            seed=plan['seed'];folder=OUT/plan['folder'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=EdgeTargetCueHead().to('cuda');model.load_state_dict(torch.load(folder/'原初始化.pt',map_location='cuda',weights_only=True))
            init=torch.load(folder/'初始化.pt',map_location='cuda',weights_only=True)
            for k,v in model.state_dict().items():need(torch.equal(v,init[k]),'same initial '+k)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);ys=torch.as_tensor(labels,device='cuda');logs=lines(folder/'训练日志.jsonl');steps=uses=pos=0;total_counts=np.zeros(4,np.int64)
            for epoch in range(16):
                order=torch.randperm(len(bank),device='cuda');angles=np.zeros(len(bank),np.int64)
                if plan['arm']=='RotNeg':
                    rank=0
                    for i,label in enumerate(labels):
                        if label==4:
                            category=(rank+epoch)%6;angles[i]=0 if category<3 else category-2;rank+=1
                need((angles[labels<4]==0).all(),'positives always native');view_counts=np.bincount(angles,minlength=4);total_counts+=view_counts;angles_gpu=torch.as_tensor(angles,device='cuda')
                order_hash=sha256(order.cpu().numpy().astype(np.int64).tobytes()).hexdigest();same(logs[epoch]['sample_order_sha256'],order_hash,'paired order');same(logs[epoch]['angle_plan_sha256'],sha256(angles.tobytes()).hexdigest(),'negative-only plan');same(logs[epoch]['target_view_counts'],view_counts.tolist(),'per epoch exposure')
                totals=[0.,0.,0.];epoch_pos=0
                for ids in order.split(512):
                    x=pool[0][ids].clone() if plan['arm']=='RotNeg' else pool[0][ids]
                    if plan['arm']=='RotNeg':
                        selected=angles_gpu[ids]
                        for view in (1,2,3):take=selected==view;x[take]=pool[view][ids[take]]
                    # Separate functional forward/loss, retaining parameters for gradients.
                    h=F.linear(x[:,:1041].contiguous(),model.network[0].weight,model.network[0].bias)+F.linear(x[:,1041:],model.edge_projection.weight)
                    h=F.layer_norm(h,(128,),model.network[1].weight,model.network[1].bias,model.network[1].eps);raw=F.linear(torch.tanh(h),model.network[3].weight,model.network[3].bias)
                    adjacent=ys[ids]<4;count=int(adjacent.sum());bce=F.binary_cross_entropy_with_logits(raw[:,4],adjacent.to(raw.dtype));ce=F.cross_entropy(raw[adjacent,:4],ys[ids][adjacent]) if count else raw[:,:4].sum()*0;loss=bce+ce
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();steps+=1;uses+=len(ids);pos+=count;epoch_pos+=count
                    totals[0]+=float(loss.detach())*len(ids);totals[1]+=float(bce.detach())*len(ids);totals[2]+=float(ce.detach())*count
                for k,v in [('optimizer_steps',steps),('pair_uses',uses),('adjacent_pair_uses',pos),('target_view_counts_total',total_counts.tolist())]:same(logs[epoch][k],v,'budget '+k)
                for k,v in [('loss',totals[0]/len(bank)),('adjacency_bce',totals[1]/len(bank)),('direction_ce',totals[2]/epoch_pos)]:close(logs[epoch][k],v,'loss replay '+k,atol=1e-7)
            saved=torch.load(folder/'head.pt',map_location='cuda',weights_only=True)
            for k,v in model.state_dict().items():need(torch.equal(v,saved[k]),'exact independently trained final '+plan['folder']+'/'+k)
            need(steps==1632 and uses==835200 and pos==111360,'same training budget')
            if plan['arm']=='ReplayR0':
                original=torch.load(ROOT/plan['original_head_path'],map_location='cuda',weights_only=True)
                for k,v in model.state_dict().items():need(torch.equal(v,original[k]),'original head exactly reproduced')
            else:same(total_counts.tolist(),[473280,120640,120640,120640],'treatment exposure')
            print(json.dumps(dict(replayed_training=plan['folder'],optimizer_steps=steps,all_final_parameters_exact=True)),flush=True);del model
    del pool;torch.cuda.empty_cache()


def diagnostic(reg):
    previous=reliability_paths('calibration')[1];prior_reg=read(previous/'预登记.json');values,payloads,image_info=view_images(previous,prior_reg);bank=pairs(r['area'] for r in prior_reg['sources']);same(read(OUT/'诊断配对.json'),bank,'22 diagnostic pairs');results={}
    for plan in reg['plans']:
        folder=OUT/plan['folder'];head=EdgeTargetCueHead().eval();head.load_state_dict(torch.load(folder/'head.pt',map_location='cpu',weights_only=True));saved=np.load(folder/'诊断四视图概率.npy');pred=np.empty_like(saved)
        with torch.inference_mode():
            for j,angle in enumerate((0,90,180,270)):
                x=manual_matrix(bank,values,angle)
                for i in range(13200):
                    p=head(torch.as_tensor(x[i])[None]).softmax(-1)[0].numpy();close(p,manual_joint(head,torch.as_tensor(x[i])[None])[0],'independent head',atol=2e-6);pred[i,j]=p
        np.testing.assert_array_equal(saved,pred);chosen=[];raw=[]
        for i,r in enumerate(bank):
            keys=[sha256(payloads[a][r['area'],r['target']]).hexdigest() for a in (0,90,180,270)];scores=[float(max(p[:4])) for p in pred[i]];j=min(range(4),key=lambda j:(-scores[j],keys[j]));chosen.append(j);raw.append(pred[i,j])
        np.testing.assert_array_equal(chosen,np.load(folder/'诊断所选视图.npy'));m=manual_metrics(np.array(raw),bank,.5);same(read(folder/'诊断结果.json'),m,'precision/recall and source interval');results[plan['folder']]=m
        if plan['arm']=='ReplayR0':np.testing.assert_array_equal(saved,np.load(previous/f'Edge_s{plan["seed"]}/四候选原始概率.npy'))
        print(json.dumps(dict(replayed_diagnostic=plan['folder'],pairs=13200)),flush=True)
    same(read(OUT/'诊断汇总.json'),results,'all diagnostics');return image_info


def main():
    if (OUT/'独立复核.json').exists():raise ValueError('immutable audit exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=registration();values,fit_info=fit_images(reg);train_replay(reg,values);diag_info=diagnostic(reg)
    audit=dict(status='passed',same_execution_agent_separate_algorithm=True,all_six_training_replays_exact=True,original_heads_reproduced_exact=True,
        models=6,optimizer_steps=9792,pair_uses=5011200,positive_pair_uses=668160,rotated_negative_uses=1085760,diagnostic_predictions=316800,diagnostic_precision_intervals=6,
        fit_images=fit_info,diagnostic_images=diag_info,default_unchanged=True,encoder_and_explorer_unchanged=True,
        artifacts_sha256={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()})
    write_new(OUT/'独立复核.json',audit);receipt=dict(status='completed_and_audited',quality_and_replay_passed=True,allow_development=True,training_factor_effectiveness_not_yet_established=True,
        default_changed=False,default_sha256=reg['default_sha256'],audit_sha256=digest(OUT/'独立复核.json'),heads_sha256=read(OUT/'全部训练结束.json')['heads_sha256'],cloud_calls=0,new_SwissView_evaluations=0)
    write_new(OUT/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
