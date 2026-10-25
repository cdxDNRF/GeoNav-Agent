"""Independent compact tensor, probability and streamed PPO-count/reward checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_bindings(protocol):
    for name,h in protocol['bindings'].items():
        if sha(ROOT/name)!=h:raise ValueError('binding drift: '+name)


def engineering(protocol_path, output):
    import torch
    p=read(protocol_path);verify_bindings(p);result=read(Path(output)/'result.json')
    assert result['passed'] and result['protocol_sha256']==sha(protocol_path)
    assert result['history_forward_calls']==12 and result['actor_optimizer_steps']==5
    assert result['elapsed_seconds']<=300 and result['peak_reserved_mib']<=2560
    assert result['legacy_original_update_equal'] and result['partial_rollout_exact_recovery'] and result['history_frozen']
    metadata=read(ROOT/p['tensor_evidence']);checked=0
    for seed in (0,1,2):
        entry=p['histories'][str(seed)];raw=torch.load(ROOT/entry['path'],map_location='cpu',mmap=True,weights_only=True)
        record=next(v for v in metadata['records'] if v['compact']['path']==entry['path'])
        assert set(raw['model'])==set(record['tensors'])
        for name,t in raw['model'].items():
            v=t.contiguous().view(torch.uint8).numpy();h=hashlib.sha256(memoryview(v).cast('B')).hexdigest()
            assert h==record['tensors'][name]['sha256'];checked+=1
        del raw
    for case in result['cases']:
        r,c=divmod(case['observed_cell'],5);mask=[r>0,c<4,r<4,c>0]
        probabilities=case['actor_probabilities'];weights=[a*int(m)+1e-6 for a,m in zip(probabilities,mask)]
        expected=[x/sum(weights) for x in weights]
        assert max(abs(a-b) for a,b in zip(expected,case['acting_probabilities']))<1e-6
        assert abs(case['candidate_initial_ratio']-1)<1e-6
        assert len(case['history_state'])==512 and max(case['original_output_differences'])<=1e-6
    return dict(passed=True,compact_tensor_hashes_checked=checked,independent_probability_cases=6,
        history_forward_calls=12,engineering_head_updates=5,independent_person_review=False,
        no_new_sr=True,protocol_sha256=sha(protocol_path),result_sha256=sha(Path(output)/'result.json'))


def training(protocol_path, output):
    p=read(protocol_path);verify_bindings(p);folder=Path(output);records=[];total_steps=total_updates=0
    first_rollouts={}
    for seed,arm in p['run_order']:
        run=folder/f'seed_{seed}'/arm;result=read(run/'result.json')
        assert result['status']=='completed' and result['steps']==40960 and result['updates']==64 and result['optimizer_steps']==256
        assert result['history_unchanged'] and result['protocol_sha256']==sha(protocol_path)
        assert sha(run/'final_head.pt')==result['final_sha256']
        steps=updates=0;last=None;best=None;first=[];max_ratio=0
        for line in (run/'events.jsonl').open(encoding='utf-8'):
            row=json.loads(line)
            if row['event']=='step':
                steps+=1;assert row['step']==steps and row['seed']==seed and row['arm']==arm
                cells=row['cells'];actions=row['actions'];goal=row['goal'];before,after=cells[-2:]
                if len(actions)==1:best=abs(cells[0]//5-goal//5)+abs(cells[0]%5-goal%5)
                else:assert last['cells']==cells[:-1] and last['actions']==actions[:-1] and last['goal']==goal and last['source']==row['source'] and not last['terminal_goal']
                r,c=divmod(before,5);dr,dc=((-1,0),(0,1),(1,0),(0,-1))[actions[-1]];rr,cc=r+dr,c+dc
                expected=rr*5+cc if 0<=rr<5 and 0<=cc<5 else before
                assert after==expected and row['terminal_goal']==(after==goal)
                ext=-1 if before==after else (2 if after==goal else (-1 if after in cells[:-2] else (1 if (after//5-goal//5)**2+(after%5-goal%5)**2<best else -1)))
                expected_reward=ext+.25*(2*(row['prediction_mse']-.8)/.1-1)
                assert abs(expected_reward-row['reward'])<1e-9
                if expected_reward>=.6:best=abs(after//5-goal//5)+abs(after%5-goal%5)
                assert row['source'] in p['fit_areas']
                if steps<=640:first.append({k:row[k] for k in ('source','goal','cells','actions','reward','prediction_mse','terminal_goal')})
                last=row
            elif row['event']=='update':
                updates+=1;assert row['steps']==updates*640 and row['updates']==updates and len(row['losses'])==4
                assert all(math.isfinite(x) for x in row['losses'])
                max_ratio=max(max_ratio,row['initial_ratio']['max_abs_from_one'])
        assert steps==40960 and updates==64
        if arm=='ConsistentMaskedEvaluate':assert max_ratio<=1e-6
        first_rollouts[(seed,arm)]=(first,result['initial_parameter_sha256'])
        total_steps+=steps;total_updates+=updates
        assert len(list(run.glob('recovery_step_*.pt')))<=2
        records.append(dict(seed=seed,arm=arm,steps=steps,updates=updates,max_initial_ratio_deviation=max_ratio,final_sha256=result['final_sha256']))
    for seed in (0,1,2):
        a=first_rollouts[(seed,'LegacyUnmaskedEvaluate')];b=first_rollouts[(seed,'ConsistentMaskedEvaluate')]
        assert a==b,'same new initialization/first rollout not matched'
    assert total_steps==245760 and total_updates==384
    return dict(passed=True,environment_steps=total_steps,ppo_updates=total_updates,optimizer_steps=total_updates*4,
        first_paired_rollouts_exact=True,per_arm=records,independent_person_review=False,new_sr=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['engineering','training']);parser.add_argument('protocol');parser.add_argument('output');parser.add_argument('--save')
    args=parser.parse_args();result=engineering(args.protocol,args.output) if args.mode=='engineering' else training(args.protocol,args.output)
    if args.save:
        with Path(args.save).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
