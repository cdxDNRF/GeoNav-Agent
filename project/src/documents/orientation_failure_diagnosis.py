"""Posthoc paired failures; evaluator truth never enters policy or calibration."""
import json
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.orientation_confirmation import paths,ROOT
from train.dyncur_tiny import digest
from train.local_capacity import read,lines
from train.curiosity_controlled import write_new


def analyze(out):
    tasks={e['episode_id']:e for e in read(out/'导航任务.json')};losses=[];gains=[];sources={}
    for seed in range(3):
        files=[out/f'Edge_s{seed}/导航_Clean_{a}_轨迹.jsonl' for a in ('Original','Candidate4')]
        sources.update({p.relative_to(ROOT).as_posix():digest(p) for p in files})
        for original,candidate in zip(*[lines(p) for p in files]):
            if original['success']==candidate['success']:continue
            if candidate['success']:gains.append(dict(seed=seed,episode_id=candidate['episode_id']));continue
            i=next(i for i,(a,b) in enumerate(zip(original['decisions'],candidate['decisions'])) if a['action']!=b['action'])
            a,b=original['decisions'][i],candidate['decisions'][i];goal=tasks[candidate['episode_id']]['goal'];dest=candidate['trajectory'][i+1]['patch_id']
            losses.append(dict(seed=seed,episode_id=candidate['episode_id'],area=candidate['area'],distance=candidate['distance'],
                first_action_divergence_step=i+1,public_position=b['public_position'],original_action=a['action'],candidate_action=b['action'],
                candidate_cue=b['cue_action'],selected_relative_clockwise=b['selected_relative_clockwise'],selection_score=b['selection_score'],
                first_divergence_reaches_goal=dest==goal,goal_used_only_for_posthoc_audit=True,original_steps=original['steps'],candidate_steps=candidate['steps'],candidate_SG=candidate['sg']))
    return dict(scope='posthoc known SwissView pilot; no calibration/threshold change; divergence is not a single-action causal proof',source_sha256=sources,
        extra_clean_failures=len(losses),new_clean_successes=len(gains),losses=losses,gains=gains,
        unused_confirmation_source_ids=list(range(40,60)),formal_confirmation_started=False)


def main():
    out=paths('pilot')[1];receipt=read(out/'验收结论.json');assert receipt['quality_and_replay_passed'] and not receipt['allow_next_stage']
    if (out/'试跑失败配对诊断.md').exists():raise ValueError('immutable diagnosis exists')
    result=analyze(out);write_new(out/'试跑失败配对诊断.json',result)
    rows=['# 新源图试跑：干净题损失来自哪里','',
        f"候选比原干净策略额外失败{result['extra_clean_failures']}题、新增成功{result['new_clean_successes']}题（20题×3权重，共60配对执行）。本批未放行，不执行40…59正式留出；试跑4图已经成为已见证据，后续不能再冒充新独立图。",'',
        '| 题/种子 | 首次动作分歧 | 原动作→候选 | 选择的相对角度 | 最大原方向分数 | 候选SG |','|---|---:|---|---:|---:|---:|']
    for x in result['losses']:
        rows.append(f"| {x['episode_id']}/s{x['seed']} | {x['first_action_divergence_step']} | {x['original_action']}→{x['candidate_action']} | {x['selected_relative_clockwise']}° | {x['selection_score']:.4f} | {x['candidate_SG']} |")
    rows+=['','三题均在seed1；首次分歧选择了非0度候选并接受该视觉方向，分数约0.88…0.92，但动作均没有立即到达目标。干净情况下原接受线索54次/52正确（96.30%），新候选71次/49正确（69.01%）；错误接受从2增至22。本轮发现搜索候选增加了错误接受，是下一项可靠性校准的依据。', '',
        '这不是每个目标都应该选0度的监督规则，实际运行不能拿真朝向作判定；完全对称图也不存在唯一朝向。首次分歧记录只能解释配对轨迹如何不同，不能证明单一动作是全部损失的唯一原因。四源文件区间下界0不能当稳定旋转收益证明，三权重也不增加源文件样本数。','',
        '下一因素宜在原Masa开发数据单独校准多候选接受可靠性，保持探索器/权重/预算不变，再使用新的SwissView源文件确认。不在本批4图调到过线，不叠加裁剪修复，不默认简单提高阈值就能解决：错误接受也出现高原始分数。', '',
        '[逐题数据与原轨迹哈希](试跑失败配对诊断.json) / [完整报告](方向候选验证报告.md) / [最终验收](验收结论.json)。']
    (out/'试跑失败配对诊断.md').write_text('\n'.join(rows)+'\n','utf-8');print(json.dumps(dict(extra_clean_failures=result['extra_clean_failures'],formal_confirmation_started=False)))


if __name__=='__main__':main()
