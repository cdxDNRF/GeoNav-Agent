"""Preregistered test250 confirmation; all policy artifacts remain frozen from S2."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch

if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import image_profiles
from agents.spatial_relation import quadrant_features
from eval.edge_s2_confirmation import (ROOT, MASA, SRC, PREVIOUS, DEFAULT, MEANS, SEEDS,
    EDGE_CONDITIONS, navigation_result)
from eval.evaluate import verify_task_file
from eval.local_s2_confirmation import aggregate
from train.curiosity_controlled import write_new
from train.dyncur_tiny import EmbeddingStore, digest
from train.local_capacity import read, lines, rules
from train.target_controlled import wrong_cue_plan

S2 = MASA / '评测结果/边缘线索S2正式复验_v1'
OUTPUT = MASA / '评测结果/边缘线索S3独立源图确认_v1'
DOC = ROOT / '选题报告相关/边缘线索S3独立源图确认方案_v1.md'
STANDARD = ROOT / '选题报告相关/分阶段验收标准_v1.md'
TEST_RECORD = ROOT / '选题报告相关/边缘线索S3测试记录_v1.json'
GATE = dict(SR=.55, SG=2., rule_gain=.05, positive_sources=6, revisit=.10,
    Q=.99, complete_normal_terminals=250, target_gain=.05, positive_seeds=2)


def protocol_info(episodes):
    if len(episodes) != 250 or len({e.episode_id for e in episodes}) != 250:
        raise ValueError('test250 task count/identity')
    if any(e.split != 'test' or e.grid_size != 5 or e.budget != 10 for e in episodes):
        raise ValueError('test split/grid/budget')
    sources = sorted({e.area for e in episodes})
    if len(sources) != 10 or Counter(e.dist for e in episodes) != {d:50 for d in range(4,9)}:
        raise ValueError('source/distance counts')
    if any(Counter(e.dist for e in episodes if e.area == a) != {d:5 for d in range(4,9)} for a in sources):
        raise ValueError('within-source balance')
    unique = len({(e.area,e.start,e.goal) for e in episodes})
    return dict(source_count=10, unique_routes=unique, duplicate_draws=250-unique,
        by_distance={str(d):50 for d in range(4,9)}, episodes=250)


def isolated_splits(episodes, manifests):
    pairs = {}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if {e.source_tile for e in episodes[a]} & {e.source_tile for e in episodes[b]}:
            raise ValueError('source tile overlap '+a+'/'+b)
        if set(manifests[a]['area_sha256'].values()) & set(manifests[b]['area_sha256'].values()):
            raise ValueError('patch group overlap '+a+'/'+b)
        pairs[a+'/'+b] = dict(source_names_disjoint=True, patch_group_file_hashes_disjoint=True)
    return pairs


def interval(left, right, metric):
    areas = sorted(left[0]['by_source'])
    values = np.array([np.mean([a['by_source'][area][metric]-b['by_source'][area][metric]
        for a,b in zip(left,right)]) for area in areas])
    idx = np.random.default_rng(3031).integers(len(areas),size=(2000,len(areas)))
    return dict(mean=float(values.mean()), interval95=np.quantile(values[idx].mean(1),[.025,.975]).tolist(),
        source_count=len(areas),resamples=2000,scope='test source-file groups; not verified geographic groups')


def effect(left, right):
    gains = [a['metrics']['sr']-b['metrics']['sr'] for a,b in zip(left,right)]
    maps = {area:float(np.mean([a['by_source'][area]['sr']-b['by_source'][area]['sr']
        for a,b in zip(left,right)])) for area in sorted(left[0]['by_source'])}
    return dict(gain=float(np.mean(gains)), gains_by_seed=gains, positive_seeds=sum(v>1e-12 for v in gains),
        sg_change=float(np.mean([a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'] for a,b in zip(left,right)])),
        gains_by_source=maps,positive_sources=sum(v>1e-12 for v in maps.values()),
        source_interval=interval(left,right,'sr'),sg_source_interval=interval(left,right,'mean_sg_all_episodes'))


def engineering_checks(main, rule):
    m = aggregate(main)
    sources = sum(v['sr'] > rule['by_source'][a]['sr']+1e-12 for a,v in m['by_source'].items())
    return dict(all_250_normal_terminals=all(r['metrics']['episodes']==250 for r in main),
        mean_SR_55pct=m['sr_mean']>=.55-1e-12,mean_SG_2p0=m['sg_mean']<=2.+1e-12,
        rule_SR_gain_5pp=m['sr_mean']>=rule['metrics']['sr']+.05-1e-12,
        SG_no_worse_than_rule=m['sg_mean']<=rule['metrics']['mean_sg_all_episodes']+1e-12,
        six_of_ten_sources_positive=len(m['by_source'])==10 and sources>=6,
        each_seed_revisit_10pct=all(v<=.10+1e-12 for v in m['repeat_by_seed']))


def summarize(results, rule_runs):
    get = lambda a,c:[results[a,s,c] for s in SEEDS]
    main = get('Edge','CueFull')
    averages = dict(Edge={c:aggregate(get('Edge',c)) for c in EDGE_CONDITIONS},
        ZeroEdge=dict(CueFull=aggregate(get('ZeroEdge','CueFull'))))
    effects = {n:effect(main,get(a,c)) for n,a,c in (
        ('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),
        ('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong'))}
    # These repetitions align deterministic rule rows with the three policy seeds;
    # bootstrap still has precisely ten source groups, not thirty rule samples.
    for n,r in rule_runs.items(): effects[n] = effect(main,[r]*3)
    strongest = max(rule_runs,key=lambda n:(rule_runs[n]['metrics']['sr'],-rule_runs[n]['metrics']['mean_sg_all_episodes']))
    engineering = engineering_checks(main,rule_runs[strongest])
    target = {n:v['gain']>=.05-1e-12 and v['positive_seeds']>=2 and v['sg_change']<=1e-12
        for n,v in effects.items() if n not in rule_runs}
    supported = {n:v['source_interval']['interval95'][0]>0 for n,v in effects.items()}
    return dict(averages=averages,effects=effects,rules=rule_runs,strongest_rule=strongest,
        engineering_checks=engineering,target_transfer_checks=target,SR_gain_CI_positive=supported,
        formal_S3_numeric_checks_passed=all(engineering.values()),formal_S3_passed=False,audit_pending=True,
        target_transfer_numeric_checks_passed=all(target.values()),planned_neural_episodes=3750,
        planned_rule_episodes=500,training_steps=0,cloud_calls=0,
        diagnostics=dict(SR_seed_range=max(averages['Edge']['CueFull']['sr_by_seed'])-min(averages['Edge']['CueFull']['sr_by_seed'])),
        data_scope='Fixed test250, 10 source-file groups outside train/val. Geographic nonoverlap not established.')


def extract_test_features(store, out, device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    model = CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval()
    model.requires_grad_(False);local={};profiles={};provenance={};maximum=0.
    with torch.inference_mode():
        for area in sorted(store.data):
            paths = [MASA/f'patches/test/{area}/patch_{j}.jpg' for j in range(25)]
            result = model(torch.stack([preprocess_patch(p) for p in paths]).to(device))
            values = result.image_embeds.float().cpu().numpy()
            maximum=max(maximum,float(np.abs(values-store.data[area]).max()))
            np.testing.assert_allclose(values,store.data[area],atol=2e-3,rtol=2e-3)
            local[area]=quadrant_features(model.vision_model.post_layernorm(result.last_hidden_state[:,1:])).float().cpu().numpy()
            if not np.isfinite(local[area]).all():raise ValueError('nonfinite local test features')
            ps=[]
            for j,p in enumerate(paths):
                with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                ps.append(image_profiles(rgb))
                provenance[f'{area}/{j}']=dict(path=p.relative_to(ROOT).as_posix(),file_sha256=digest(p),
                    rgb_sha256=sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape))
            profiles[area]=np.stack(ps)
    np.savez(out/'测试局部区域特征.npz',**local);np.savez(out/'测试边缘profile.npz',**profiles)
    write_new(out/'测试图块来源.json',provenance)
    write_new(out/'特征核验.json',dict(sources=len(local),patches=25*len(local),frozen=True,
        global_cache_max_abs_error=maximum,local_cache_sha256=digest(out/'测试局部区域特征.npz'),
        encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),
        quadrant_recipe='post_layernorm non-CLS 7x7 tokens; nonoverlapping 3/4 row and column quadrants',
        torch=torch.__version__,transformers=__import__('transformers').__version__))
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return local


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    if out.exists():raise ValueError('immutable output already exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tests=read(TEST_RECORD)
    if not tests['successful'] or tests['failures'] or tests['errors']:raise ValueError('tests not passed')
    receipt=read(S2/'验收结论.json');s2reg=read(S2/'预登记.json')
    if not receipt['formal_S2_passed'] or read(S2/'独立复核.json')['status']!='passed':raise ValueError('S2 not audited/passed')
    for name,field in [('独立复核.json','audit_sha256'),('对照汇总.json','summary_sha256'),('默认加载核验.json','default_loading_audit_sha256')]:
        if digest(S2/name)!=receipt[field]:raise ValueError('S2 evidence mismatch '+name)
    if digest(DEFAULT)!=receipt['default_config_sha256'] or read(DEFAULT)['selected_arm']!='EdgeTargetCue':raise ValueError('default changed')
    plans=s2reg['plans']
    for p in plans:
        if digest(ROOT/p['head_path'])!=p['head_sha256'] or digest(ROOT/p['explorer_path'])!=p['explorer_sha256']:
            raise ValueError('weight changed')
        if p['threshold']!=(.5 if p['arm']=='Edge' else None):raise ValueError('threshold changed')
    episodes={};manifests={}
    for split in ('train','val','test'):
        episodes[split],manifests[split]=verify_task_file(MASA,MASA/f'任务清单_v2/episodes_{split}.jsonl',split)
    scope=protocol_info(episodes['test']);isolation=isolated_splits(episodes,manifests)
    out.mkdir(parents=True);began=time.monotonic()
    try:
        for n in MEANS:
            if digest(PREVIOUS/n)!=digest(S2/n):raise ValueError('frozen mean mismatch')
            (out/n).write_bytes((S2/n).read_bytes())
        for n in ('预登记.json','独立复核.json','验收结论.json','默认加载核验.json'):
            (out/('S2来源_'+n)).write_bytes((S2/n).read_bytes())
        for n,p in [('冻结默认配置.json',DEFAULT),('冻结方案.md',DOC),('冻结标准.md',STANDARD),
            ('测试记录.json',TEST_RECORD),('复验前README.md',ROOT/'README.md'),('研究资料快照.md',ROOT/'research.md')]:
            (out/n).write_bytes(p.read_bytes())
        selected=episodes['test'];wrong=wrong_cue_plan(selected)
        write_new(out/'导航任务.json',[asdict(e) for e in selected]);write_new(out/'错误目标计划.json',wrong)
        for p in plans:
            f=out/p['folder'];f.mkdir()
            (f/'head.pt').write_bytes((ROOT/p['head_path']).read_bytes())
            (f/'explorer.pt').write_bytes((ROOT/p['explorer_path']).read_bytes());write_new(f/'配置.json',p)
        sources={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py')
            if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for n,p in sources.items():
            target=out/'源码快照'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        inputs=['metadata.csv','papr_test_sat_embeds_grid_5.npy']+[f'任务清单_v2/{kind}_{s}.{ext}'
            for s in ('train','val','test') for kind,ext in [('episodes','jsonl'),('manifest','json')]]
        reg=dict(version='edge-s3-confirmation-v1',utc=datetime.now(timezone.utc).isoformat(),plans=plans,seeds=list(SEEDS),
            planned_neural_episodes=3750,planned_rule_episodes=500,training_steps=0,cloud_calls=0,test_used=True,
            model_evaluation_started=False,protocol=scope,source_isolation=isolation,gate=GATE,
            wrong_cue_distance_exceptions=sum(not x['matched_distance'] for x in wrong.values()),
            source_sha256={n:digest(p) for n,p in sources.items()},data_sha256={n:digest(MASA/n) for n in inputs},
            split_area_sha256={s:m['area_sha256'] for s,m in manifests.items()},
            frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
            S2_artifacts_sha256={p.relative_to(S2).as_posix():digest(p) for p in S2.rglob('*') if p.is_file()},
            default_sha256=digest(DEFAULT),encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),
            encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
            runtime=dict(device=str(device),torch=torch.__version__,python=sys.version),
            bootstrap=dict(seed=3031,resamples=2000,unit='source file; mean paired seeds first'),
            inference='argmax; reset per episode; no tuning; no new research factors')
        write_new(out/'预登记.json',reg);write_new(out/'执行状态.json',dict(status='registered',completed_jobs=0,neural_episodes=0))
        print(json.dumps(dict(registered=str(out),protocol=scope,wrong_exceptions=reg['wrong_cue_distance_exceptions'])),flush=True)
        store=EmbeddingStore(MASA/'papr_test_sat_embeds_grid_5.npy');local=extract_test_features(store,out,device)
        write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,cache_sha256={n:digest(out/n) for n in
            ('测试局部区域特征.npz','测试边缘profile.npz','测试图块来源.json','特征核验.json')}))
        means={n:np.load(out/n,allow_pickle=False) for n in MEANS};results={};jobs=0
        rule_results=rules(out,selected);rule_runs={}
        from eval.evaluate import metrics
        for n,v in rule_results.items():
            rows=lines(out/f'{n}_轨迹.jsonl')
            rule_runs[n]=dict(metrics=v,by_source={a:metrics([r for r in rows if r['area']==a]) for a in sorted(store.data)})
        for p in plans:
            for c in p['conditions']:
                v=navigation_result(out/p['folder'],p,c,selected,store,local,means,wrong,device)
                results[p['arm'],p['seed'],c]=v;jobs+=1
                print(json.dumps(dict(job=jobs,arm=p['arm'],seed=p['seed'],condition=c,sr=v['metrics']['sr'],sg=v['metrics']['mean_sg_all_episodes'])),flush=True)
                (out/'执行状态.json').write_text(json.dumps(dict(status='evaluating',completed_jobs=jobs,neural_episodes=jobs*250,rule_episodes=500),indent=2)+'\n','utf-8')
        summary=summarize(results,rule_runs);write_new(out/'对照汇总.json',summary)
        if digest(DEFAULT)!=reg['default_sha256']:raise ValueError('default changed during S3')
        (out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=jobs,neural_episodes=3750,
            rule_episodes=500,seconds=time.monotonic()-began,audit_pending=True),indent=2)+'\n','utf-8')
        print(json.dumps(dict(numeric_passed=summary['formal_S3_numeric_checks_passed'],target_transfer=summary['target_transfer_numeric_checks_passed'],audit_pending=True)),flush=True)
    except BaseException as exc:
        write_new(out/'执行异常.json',dict(error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
