"""离线复算每条轨迹、API输入边界、规则对照及摘要；不调用模型。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.exploration import FrontierPolicy
from agents.vlm import APIConfig
from data.make_episodes import MASA, json_bytes
from env.environment import GridWorldEnv
from env.episode import Episode, write_immutable_files
from eval.diagnostics import analyze_positions
from eval.validation_suite import summarize


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(root: Path, folder: Path):
    configuration = json.loads((folder/"运行配置.json").read_text(encoding="utf-8"))
    summary = json.loads((folder/"验证汇总.json").read_text(encoding="utf-8"))
    episodes = {ep["episode_id"]:Episode.from_dict(ep) for ep in configuration["selected_episodes"]}
    calls = load_jsonl(folder/"API调用.jsonl")
    records = load_jsonl(folder/"任务结果.jsonl")
    events = load_jsonl(folder/"动作事件.jsonl")
    assert len(records) == len(episodes)*3
    assert len({(r['policy'],r['episode_id']) for r in records}) == len(records)
    assert len(episodes) == 20 and {ep.split for ep in episodes.values()} == {'val'}
    assert len({(ep.area,ep.dist) for ep in episodes.values()}) == 20
    assert configuration['task_sha256'] == sha256((root/'任务清单_v2/episodes_val.jsonl').read_bytes()).hexdigest()
    for name, expected in summary['log_sha256'].items():
        assert sha256((folder/name).read_bytes()).hexdigest() == expected, name
    src = Path(__file__).resolve().parents[1]
    for name, expected in configuration['source_sha256'].items():
        assert sha256((src/name).read_bytes()).hexdigest() == expected, name
    env = GridWorldEnv(root)
    state_checks = 0
    for record in records:
        ep = episodes[record['episode_id']]
        obs = env.reset(ep)
        positions = [ep.start]
        assert record['positions'][0] == ep.start
        episode_calls = [c for c in calls if c['episode_id']==ep.episode_id] if record['policy']=='vlm' else []
        for index, action_record in enumerate(record['actions'],1):
            if record['policy']=='vlm':
                matching = [c for c in episode_calls if c['action_step']==index]
                assert matching and matching[-1]['status']=='ok'
                assert matching[-1]['action']==action_record['action']
                for call in matching:
                    expected_state = {'position_row_col':list(obs.position), 'grid_size':5,
                                      'remaining_budget':obs.remaining_budget,
                                      'visited_cell_ids':list(obs.visited), 'legal_actions':list(obs.legal_actions)}
                    assert call['public_state']==expected_state
                    assert call['image_count']==2
                    assert call['current_image_sha256']==sha256(obs.current_image).hexdigest()
                    assert call['target_image_sha256']==sha256(obs.target_image).hexdigest()
                    assert call['system_prompt_sha256']==sha256(configuration['system_prompt'].encode()).hexdigest()
                    state_checks+=1
            if record['policy']=='frontier':
                assert FrontierPolicy().act(obs)==action_record['action']
            obs, done, info = env.step(action_record['action'])
            positions.append(obs.position[0]*5+obs.position[1])
            assert action_record['patch_id']==positions[-1]
            assert action_record['out_of_bounds']==info.out_of_bounds
            assert action_record['revisited']==info.revisited
        assert positions==record['positions']
        assert analyze_positions(positions,ep.goal)==record['diagnostics']
        if record['completion']=='completed':
            assert env.done and env.evaluator_result()==record['evaluation']
        else:
            assert not env.done and record['evaluation'] is None
        assert record['actions']==[e for e in events if e['episode_id']==ep.episode_id and e['policy']==record['policy']]
    recomputed = summarize(records,calls,configuration)
    for key,value in recomputed.items():
        assert summary[key]==value,key
    # 只做本机检查，不将凭据写入任何输出。
    secret = APIConfig.load().api_key
    for p in folder.iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl','.md','.txt'):
            content=p.read_text(encoding='utf-8')
            assert secret not in content,p.name
            assert 'data:image/' not in content,p.name
    result={'status':'passed','task_count':len(episodes),'policy_records':len(records),
            'api_attempts':len(calls),'api_input_state_checks':state_checks,
            'offline_only':True,'executed_model_calls':0,
            'checks':['source_hashes','frozen_task_hash','log_hashes','all_trajectories_replayed',
                      'rule_policy_replayed','api_public_states_and_image_hashes','metrics_recomputed',
                      'no_key_or_image_base64_in_logs']}
    write_immutable_files({folder/'离线审计.json':json_bytes(result)})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=MASA)
    parser.add_argument('--run-dir',type=Path,default=MASA/'评测结果/三项验证_v1连接修复')
    args=parser.parse_args()
    print(json.dumps(audit(args.dataset_root,args.run_dir),ensure_ascii=False,indent=2))
