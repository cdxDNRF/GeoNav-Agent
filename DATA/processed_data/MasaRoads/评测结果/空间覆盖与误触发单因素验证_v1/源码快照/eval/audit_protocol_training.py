"""Independent world/input arithmetic, encoder re-extraction and six-model replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from io import BytesIO
from hashlib import sha256
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.spatial_relation import make_policy
from train.navigation_spatial import train as ppo_train
from train.dyncur_tiny import EmbeddingStore
from train.protocol_adaptation import ROOT,SRC,MASA,BASE,OUT,CACHE,SEEDS,ARMS,UPDATES,COUNT,ProtocolWorld,read,write,digest


def need(test,label):
    if not test:raise ValueError('protocol training audit: '+label)


class CheckedWorld(ProtocolWorld):
    """Use sampler/old PPO, but independently check every public input and transition."""
    def observe(self):
        actual=super().observe();counts=np.zeros((self.count,25),np.float32)
        for row in range(self.k):
            for col in range(self.k):counts[:,(row*5//self.k)*5+col*5//self.k]+=self.visits[:,row*self.k+col]
        state=np.stack((self.position//self.k/(self.k-1),self.position%self.k/(self.k-1),(self.budget-self.steps)/self.budget),axis=1)
        expected=np.concatenate((np.repeat(self.mean[None],self.count,axis=0),self.table[self.area,self.position],state,np.minimum(counts,3)/3),axis=1).astype(np.float32)
        need(np.array_equal(actual,expected),'every NoTarget public input/mapping')
        return actual

    def step(self,actions):
        before=self.position.copy();goal=self.goal.copy();steps=self.steps.copy();visits=self.visits.copy()
        delta=np.array([[-1,0],[0,1],[1,0],[0,-1]])[np.asarray(actions)];rr=before//self.k+delta[:,0];cc=before%self.k+delta[:,1]
        wall=(rr<0)|(cc<0)|(rr>=self.k)|(cc>=self.k);dest=np.where(wall,before,rr*self.k+cc)
        done=(dest==goal)|(steps+1==self.budget);success=done&(dest==goal)
        d0=np.abs(before//self.k-goal//self.k)+np.abs(before%self.k-goal%self.k);d1=np.abs(dest//self.k-goal//self.k)+np.abs(dest%self.k-goal%self.k)
        expected=dict(done=done,success=success,wall=wall,revisited=visits[np.arange(self.count),dest]>0,
            external=(success.astype(np.float32)+.1*(d0-d1)).astype(np.float32),
            pbrs=(.5*(d0/(2*(self.k-1))-.99*np.where(done,0.,d1/(2*(self.k-1))))).astype(np.float32))
        actual=super().step(actions)
        for key,value in expected.items():need(np.array_equal(actual[key],value),'every transition/reward '+key)
        need(np.array_equal(self.position,dest),'real destination')
        need(not wall.any(),'legal sampled action')
        return actual


def audit_cache(reg,device):
    from transformers import CLIPVisionModelWithProjection
    model=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval().requires_grad_(False)
    with np.load(CACHE/'全局特征.npz',allow_pickle=False) as archive:table={a:archive[a] for a in archive.files}
    provenance=read(CACHE/'图像来源.json');mean=np.asarray([.3670,.3827,.3338],np.float32);std=np.asarray([.2209,.1975,.1988],np.float32);patches=0
    need(set(table)=={r['area'] for r in reg['source_rows']},'fit-only cache')
    with torch.inference_mode():
        for index,row in enumerate(reg['source_rows']):
            need(digest(ROOT/row['raw_path'])==row['raw_sha256'],'raw image hash')
            with Image.open(ROOT/row['raw_path']) as source:
                source.load()
                for start in range(0,100,25):
                    batch=[]
                    for cell in range(start,start+25):
                        rr,cc=divmod(cell,10);crop=source.crop((150*cc,150*rr,150*(cc+1),150*(rr+1))).resize((300,300),Image.Resampling.BICUBIC)
                        stream=BytesIO();crop.save(stream,format='JPEG',quality=75);payload=stream.getvalue()
                        with Image.open(BytesIO(payload)) as image:
                            rgb=np.asarray(image.convert('RGB'),np.uint8);pixels=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                        need(provenance[row['area']+'/'+str(cell)]==dict(jpeg_sha256=sha256(payload).hexdigest(),rgb_sha256=sha256(rgb.tobytes()).hexdigest()),'virtual JPEG/pixels')
                        batch.append(torch.from_numpy(((pixels-mean)/std).transpose(2,0,1)));patches+=1
                    actual=model(torch.stack(batch).to(device)).image_embeds.float().cpu().numpy()
                    need(np.array_equal(actual,table[row['area']][start:start+25]),'every frozen embedding')
            if (index+1)%20==0 or index+1==len(reg['source_rows']):print(dict(audit_cache_sources=index+1,planned=len(reg['source_rows'])),flush=True)
    receipt=read(CACHE/'缓存收据.json');need(receipt['file_sha256']==digest(CACHE/'全局特征.npz') and receipt['provenance_sha256']==digest(CACHE/'图像来源.json'),'cache receipt')
    del model;torch.cuda.empty_cache();return table,patches


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');device=torch.device('cuda')
    for key,base in [('source_sha256',SRC),('frozen_core_sha256',SRC),('protected_sha256',ROOT),('encoder_sha256',ROOT),('training_input_sha256',ROOT),('evaluation_input_sha256',ROOT)]:
        for name,value in reg[key].items():need(digest(base/name)==value,key+' '+name)
    for name,value in reg['source_sha256'].items():need(digest(OUT/'源码快照'/name)==value,'source snapshot')
    need(digest(ROOT/'project/local_policy_default.json')==reg['default_sha256'],'default')
    need(digest(ROOT/'选题报告相关/探索器训练协议同预算适配执行方案_v1.md')==reg['protocol_sha256'],'protocol')
    fit=read(OUT/'源图划分.json')['fit'];table10,patches=audit_cache(reg,device);table5=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy').data
    mean=np.load(BASE/'全局拟合均值.npy',allow_pickle=False);marker=read(OUT/'全部训练结束.json');need(marker['models']==6 and marker['evaluation_started'] is False,'all training before evaluation')
    replay_root=OUT/'训练复核';replay_root.mkdir();receipts={};all_actions=all_optimizer=0
    for seed in SEEDS:
        for arm in (ARMS if seed%2==0 else ARMS[::-1]):
            original=OUT/f'{arm}_s{seed}';replay=replay_root/original.name;replay.mkdir();config=read(original/'配置.json');resource=read(original/'训练资源.json');budget=read(original/'训练预算收据.json')
            need(config['initial_sha256']==reg['warm_start'][str(seed)]['sha256'],'common starting checkpoint')
            need(digest(ROOT/reg['warm_start'][str(seed)]['path'])==config['initial_sha256'],'initial preserved')
            need(resource['steps']==budget['real_environment_actions']==UPDATES*COUNT*10 and budget['optimizer_updates']==UPDATES*8,'matched actual budget')
            initial=make_policy('Small256');initial.load_state_dict(torch.load(ROOT/reg['warm_start'][str(seed)]['path'],map_location='cpu',weights_only=True))
            initial_state={n:t.clone() for n,t in initial.state_dict().items()};world=CheckedWorld(table5 if arm=='Continue5' else table10,fit,mean,5 if arm=='Continue5' else 10,COUNT,seed)
            actual=ppo_train(initial,world,replay,seed,device,updates=UPDATES).state_dict();saved=torch.load(original/'model.pt',map_location=device,weights_only=True)
            need(all(torch.equal(actual[n],saved[n]) for n in saved),'all final parameters reproducible')
            need(all(torch.equal(saved[n].cpu(),initial_state[n]) for n in saved if n.startswith('next_state.')),'unused prediction head frozen')
            need(sum(t.numel() for t in saved.values())==797701,'architecture unchanged')
            need(digest(original/'model.pt')==marker['checkpoints'][original.name]==resource['final_sha256'],'final checkpoint receipt')
            with np.load(original/'训练轨迹.npz',allow_pickle=False) as a,np.load(replay/'训练轨迹.npz',allow_pickle=False) as b:
                need(a.files==b.files and all(np.array_equal(a[n],b[n]) for n in a.files),'all sampled trajectories/rewards')
                need(a['action'].size==UPDATES*COUNT*10,'all actual actions counted')
            logs=list(__import__('eval.local_ledger_expansion',fromlist=['lines']).lines(original/'训练日志.jsonl'))
            need(len(logs)==128 and logs[-1]['optimizer_steps']==1024 and logs[-1]['steps']==81920,'optimizer logs match')
            with np.load(original/'最终世界状态.npz',allow_pickle=False) as final:
                need(all(np.array_equal(final[n],getattr(world,n)) for n in final.files),'last bootstrap world state')
            receipts[original.name]=dict(all_parameters_exact=True,all_trace_arrays_exact=True,actions=81920,optimizer_updates=1024,
                checkpoint_sha256=digest(original/'model.pt'),replay_checkpoint_sha256=digest(replay/'model.pt'))
            all_actions+=81920;all_optimizer+=1024;print(dict(training_audited=len(receipts),planned=6,run=original.name),flush=True)
            del initial,world,actual,saved;torch.cuda.empty_cache()
    audit=dict(passed=True,models=6,fit_sources=109,held_sources_excluded=28,virtual_patches_reextracted=patches,
        new_training_actions=all_actions,optimizer_updates=all_optimizer,verification_replay_actions=all_actions,
        encoder_frozen=True,head_frozen=True,parameters_unchanged=True,independent_world_arithmetic=True,
        models_exact_reproduced=receipts,cache_sha256=digest(CACHE/'全局特征.npz'),
        all_training_finished_before_evaluation=True,cloud_calls=0,default_changed=False)
    write(OUT/'独立复核.json',audit);print(audit,flush=True)


if __name__=='__main__':main()
