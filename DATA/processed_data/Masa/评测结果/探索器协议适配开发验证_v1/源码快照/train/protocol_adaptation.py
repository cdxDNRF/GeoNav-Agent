"""Equal-action/update PPO continuation; architecture and visual head stay frozen."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
from pathlib import Path
import csv,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.spatial_relation import make_policy
from agents.scaled_edge_navigator import scaled_policy_features
from train.navigation_spatial import train as ppo_train
from train.dyncur_tiny import EmbeddingStore,set_seed
from data.protocol_fit_cache import build_cache
from eval.local_ledger_expansion import ROOT,SRC,OLD,DATA,SEEDS,OUT as LEDGER,read,write,digest

MASA=ROOT/'DATA/processed_data/Masa'
BASE=MASA/'训练结果/PBRS完整导航与局部匹配对照_v1'
OUT=MASA/'训练结果/探索器协议适配同预算对照_v1'
CACHE=MASA/'探索器协议适配_v1/训练特征'
EVAL=MASA/'评测结果/探索器协议适配开发验证_v1'
DOC=ROOT/'选题报告相关/探索器训练协议同预算适配执行方案_v1.md'
TESTS=ROOT/'选题报告相关/探索器协议适配测试_v1.json'
ARMS=('Continue5','Adapt10');UPDATES=128;COUNT=64
MOVES=np.asarray([(-1,0),(0,1),(1,0),(0,-1)],np.int64)


class ProtocolWorld:
    def __init__(self,table,fit,mean,k,count,seed):
        if k not in (5,10) or not fit or not set(fit)<=set(table):raise ValueError('invalid fitting table/protocol')
        self.k=k;self.budget=10 if k==5 else 20;self.areas=sorted(fit)
        self.table=np.stack([table[a] for a in self.areas]);self.mean=np.asarray(mean,np.float32)
        if self.table.shape!=(len(fit),k*k,512) or self.mean.shape!=(512,):raise ValueError('invalid input features')
        self.count=count;self.rng=np.random.default_rng(seed)
        self.distances=tuple(range(4,9)) if k==5 else tuple(range(12,17))
        self.pairs={d:[(a,b) for a in range(k*k) for b in range(k*k) if abs(a//k-b//k)+abs(a%k-b%k)==d] for d in self.distances}
        self.pool=(np.arange(k*k)//k*5//k)*5+(np.arange(k*k)%k*5//k)
        self.pool_matrix=np.eye(25,dtype=np.float32)[self.pool]
        self.area=np.zeros(count,np.int64);self.position=np.zeros(count,np.int64);self.goal=np.zeros(count,np.int64)
        self.steps=np.zeros(count,np.int64);self.initial_distance=np.zeros(count,np.int64);self.visits=np.zeros((count,k*k),np.float32)
        self.reset(np.ones(count,bool))

    def reset(self,mask):
        for i in np.flatnonzero(mask):
            self.area[i]=self.rng.integers(len(self.areas));d=int(self.rng.integers(self.distances[0],self.distances[-1]+1))
            self.position[i],self.goal[i]=self.pairs[d][self.rng.integers(len(self.pairs[d]))]
            self.initial_distance[i]=d;self.steps[i]=0;self.visits[i]=0;self.visits[i,self.position[i]]=1

    def observe(self):
        pooled=self.visits@self.pool_matrix
        return np.concatenate((np.broadcast_to(self.mean,(self.count,512)),self.table[self.area,self.position],
            self.position[:,None]//self.k/(self.k-1),self.position[:,None]%self.k/(self.k-1),
            (self.budget-self.steps[:,None])/self.budget,np.minimum(pooled,3)/3),axis=1).astype(np.float32)

    def step(self,actions):
        actions=np.asarray(actions,np.int64)
        if actions.shape!=(self.count,) or (actions<0).any() or (actions>3).any():raise ValueError('invalid action indices')
        before=self.position.copy();delta=MOVES[actions];r=before//self.k+delta[:,0];c=before%self.k+delta[:,1]
        wall=(r<0)|(r>=self.k)|(c<0)|(c>=self.k);self.position=np.where(wall,before,r*self.k+c)
        revisited=self.visits[np.arange(self.count),self.position]>0;self.visits[np.arange(self.count),self.position]+=1;self.steps+=1
        done=(self.position==self.goal)|(self.steps==self.budget);success=done&(self.position==self.goal)
        d0=np.abs(before//self.k-self.goal//self.k)+np.abs(before%self.k-self.goal%self.k)
        d1=np.abs(self.position//self.k-self.goal//self.k)+np.abs(self.position%self.k-self.goal%self.k)
        external=success.astype(np.float32)+.1*(d0-d1);potential_next=np.where(done,0.,-d1/(2*(self.k-1)))
        pbrs=.5*(.99*potential_next+d0/(2*(self.k-1)))
        return dict(done=done,success=success,wall=wall,revisited=revisited,external=external.astype(np.float32),pbrs=pbrs.astype(np.float32))


def prepare():
    if OUT.exists() or EVAL.exists() or CACHE.exists():raise ValueError('immutable experiment/cache exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('tests required')
    split=read(BASE/'源图划分.json');fit,held=set(split['fit']),set(split['held'])
    if len(fit)!=109 or len(held)!=28 or fit&held:raise ValueError('source split')
    for field in ('source_by_area','source_feature_sha256','source_patch_sha256'):
        if {split[field][a] for a in fit}&{split[field][a] for a in held}:raise ValueError('source leakage')
    metadata=list(csv.DictReader((MASA/'metadata.csv').open(encoding='utf-8-sig')))
    original={r['img_id']:r for r in metadata if r['split']=='train'};evaluation=read(DATA/'数据清单.json')['sources']
    rows=[]
    for area in sorted(fit):
        item=original[area];p=ROOT/'DATA/raw_data/Masa/png/train'/item['source_tile']
        with Image.open(p) as image:
            if image.mode!='RGB' or image.size!=(1500,1500):raise ValueError('raw crop dimensions')
        rows.append(dict(area=area,source=item['source_tile'],raw_path=p.relative_to(ROOT).as_posix(),raw_sha256=digest(p)))
    if {r['source'] for r in rows}&{r['source_tile'] for r in evaluation} or {r['raw_sha256'] for r in rows}&{r['raw_sha256'] for r in evaluation}:raise ValueError('fit/evaluation source overlap')
    prior=read(LEDGER/'预登记.json')
    if digest(ROOT/'project/local_policy_default.json')!=prior['default_sha256']:raise ValueError('default drift')
    if read(BASE/'独立复核.json').get('status')!='passed':raise ValueError('audited initial training required')
    OUT.mkdir();EVAL.mkdir()
    for target in (OUT,EVAL):
        (target/'执行协议.md').write_bytes(DOC.read_bytes());(target/'测试记录.json').write_bytes(TESTS.read_bytes())
        (target/'源图划分.json').write_bytes((BASE/'源图划分.json').read_bytes())
    names=['train/protocol_adaptation.py','train/navigation_spatial.py','train/curiosity_controlled.py','train/dyncur_tiny.py',
        'data/protocol_fit_cache.py','data/scaled_masa.py','data/process_masa.py','agents/scaled_edge_navigator.py',
        'agents/precomputed_scaled_edge.py','agents/spatial_relation.py','agents/boundary_policy.py',
        'eval/audit_protocol_training.py','eval/protocol_adaptation_navigation.py','eval/audit_protocol_navigation.py',
        'tests/test_protocol_adaptation.py']
    for target in (OUT,EVAL):
        for n in names:
            dest=target/'源码快照'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((SRC/n).read_bytes())
    protected=dict(prior['protected_sha256'])
    for folder in (BASE,LEDGER,MASA/'评测结果/目标邻域覆盖规划验证_v1'):
        protected.update({p.relative_to(ROOT).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()})
    warm={str(s):dict(path=(BASE/f'Small256_NoTarget_s{s}/model.pt').relative_to(ROOT).as_posix(),sha256=digest(BASE/f'Small256_NoTarget_s{s}/model.pt')) for s in SEEDS}
    encoder_files={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'models/Sat2Cap').glob('*') if p.is_file()}
    reg=dict(version='protocol-adaptation-v1',utc=datetime.now(timezone.utc).isoformat(),arms=list(ARMS),seeds=list(SEEDS),
        updates=UPDATES,vector_envs=COUNT,rollout_horizon=10,actions_per_model=UPDATES*COUNT*10,optimizer_steps_per_model=UPDATES*8,
        models=6,total_new_training_actions=6*UPDATES*COUNT*10,architecture='Small256',parameters=797701,trainable_parameters=666117,
        no_target_inputs=True,source_rows=rows,warm_start=warm,new_primary_records=3000,reused_M0_records=1500,conditional_target_records=4500,
        candidate_gate=dict(sr_gain_vs_Continue5=.02,positive_seeds_vs_Continue5=2,SG_no_worse=True,SR_vs_M0_positive=True,positive_seeds_vs_M0=2),
        target_gate=dict(sr_gain=.05,positive_seeds=2,SG_no_worse=True),default_replacement_allowed=False,
        source_sha256={n:digest(SRC/n) for n in names},frozen_core_sha256=prior['frozen_core_sha256'],protected_sha256=protected,
        evaluation_input_sha256=prior['input_sha256'],encoder_sha256=encoder_files,
        training_input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [MASA/'papr_train_sat_embeds_grid_5.npy',BASE/'全局拟合均值.npy',MASA/'metadata.csv']},
        task_sha256=digest(LEDGER/'导航任务.json'),wrong_plan_sha256=digest(LEDGER/'错误目标计划.json'),
        default_sha256=prior['default_sha256'],protocol_sha256=digest(DOC),cloud_calls=0,new_unseen_evaluation_sources=0,
        cloud_preferences_sha256=digest(ROOT/'project/cloud_provider_preferences.json'),
        runtime=dict(device='cuda',torch=torch.__version__,cuda=torch.version.cuda,deterministic=True))
    write(OUT/'预登记.json',reg);(EVAL/'预登记.json').write_bytes((OUT/'预登记.json').read_bytes())
    for n in ('导航任务.json','错误目标计划.json'):(EVAL/n).write_bytes((LEDGER/n).read_bytes())
    print(dict(registered=True,models=6,actions_per_model=reg['actions_per_model'],optimizer_steps_per_model=reg['optimizer_steps_per_model']),flush=True)


def train_all():
    if list(OUT.glob('*_s*')) or (OUT/'全部训练结束.json').exists():raise ValueError('no training rerun')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');device=torch.device('cuda')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source drift')
    rows=[dict(r,absolute_raw_path=str(ROOT/r['raw_path'])) for r in reg['source_rows']]
    fit=read(OUT/'源图划分.json')['fit'];mean=np.load(BASE/'全局拟合均值.npy',allow_pickle=False)
    table10=build_cache(CACHE,rows,ROOT/'models/Sat2Cap',device);table5=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy').data
    if not np.array_equal(mean,np.concatenate([table5[a] for a in sorted(fit)]).mean(0)):raise ValueError('fit-only frozen mean')
    completed={}
    for seed in SEEDS:
        for arm in (ARMS if seed%2==0 else ARMS[::-1]):
            folder=OUT/f'{arm}_s{seed}';folder.mkdir();set_seed(seed);initial=make_policy('Small256')
            initial.load_state_dict(torch.load(ROOT/reg['warm_start'][str(seed)]['path'],map_location='cpu',weights_only=True))
            write(folder/'配置.json',dict(arm=arm,seed=seed,initial_sha256=reg['warm_start'][str(seed)]['sha256'],
                grid_size=5 if arm=='Continue5' else 10,budget=10 if arm=='Continue5' else 20,registration_sha256=digest(OUT/'预登记.json')))
            world=ProtocolWorld(table5 if arm=='Continue5' else table10,fit,mean,5 if arm=='Continue5' else 10,COUNT,seed)
            model=ppo_train(initial,world,folder,seed,device,updates=UPDATES)
            np.savez(folder/'最终世界状态.npz',**{n:getattr(world,n) for n in ('area','position','goal','steps','visits','initial_distance')})
            write(folder/'训练预算收据.json',dict(real_environment_actions=UPDATES*COUNT*10,optimizer_updates=UPDATES*8,
                final_update=UPDATES,last_checkpoint_used=True,boundary_filter=True,next_state_frozen=True,
                source_areas=world.areas,grid_size=world.k,budget=world.budget,future_state_sha256=digest(folder/'最终世界状态.npz')))
            completed[folder.name]=digest(folder/'model.pt');print(dict(trained=len(completed),planned=6,run=folder.name),flush=True)
            del initial,model,world;torch.cuda.empty_cache()
    write(OUT/'全部训练结束.json',dict(utc=datetime.now(timezone.utc).isoformat(),models=6,checkpoints=completed,evaluation_started=False))
    write(OUT/'执行状态.json',dict(status='training_completed_pending_audit',models=6,training_actions=6*UPDATES*COUNT*10,cloud_calls=0))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--train-frozen',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.train_frozen:train_all()
    else:p.error('prepare or train-frozen required')
