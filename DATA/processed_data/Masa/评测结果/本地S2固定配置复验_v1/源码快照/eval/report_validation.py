"""将冻结验证结果汇总为中文报告，不调用API、不改原始日志。"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.make_episodes import MASA
from env.episode import write_immutable_files

LABELS = {'random':'Random', 'frontier':'规则探索', 'vlm':'最小VLM'}


def report(folder):
    folder=Path(folder)
    s=json.loads((folder/'验证汇总.json').read_text(encoding='utf-8'))
    configuration=json.loads((folder/'运行配置.json').read_text(encoding='utf-8'))
    records=[json.loads(line) for line in (folder/'任务结果.jsonl').read_text(encoding='utf-8').splitlines()]
    api=s['api']
    lines=['# 三项验证：固定20题结果与诊断','',
           '## 1. 实验边界','',
           '- 使用Masa验证集全部4个区域，每个区域距离4–8各1题，共20题；每题最多10步。题目在调用模型前锁定，选择与输出表现无关。',
           '- 比较同一任务上的Random、规则探索和gemma4:31b最小VLM。模型、提示词、双图和公开状态与上一轮修好解析器的策略一致。',
           '- 模型仅获得目标图、当前图、位置、预算和visited，不获得真实目标位置、距离、未访问图块或对话历史。',
           '- 规则策略不看任何图像及目标真值，只按已知网格拓扑找最近未访问格，必要时回退；同距离按上/右/下/左固定打破平局。',
           '- 原首轮Python/httpx遇到TLS握手EOF，VLM40次网络尝试均无有效回答。首轮原始日志保留在三项验证_v1，未作为模型失败率。',
           '- 本轮改用系统curl/Schannel传输，启用证书验证，不使用insecure或自动重定向。Key与请求体通过stdin，不进命令行参数/临时文件。网络错误允许一次重试，所有尝试分别记录。',
           '- 仅验证集开发诊断，不用test、不与原论文不同split指标作直接数值比较。','',
           '## 2. 配对总体结果','',
           '| 策略 | 完整题数 | 成功题数 | SR | 平均SG | 平均不同格数（含起点） | 重复访问率micro | 两格循环题数 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy in LABELS:
        result=s['policies'][policy]['all']; metric=result['metrics_completed_only']; d=result['completed_diagnostics']
        if metric:
            lines.append(f"| {LABELS[policy]} | {result['completed']}/{result['scheduled']} | {metric['successes']} | {metric['sr']:.1%} | {metric['mean_sg_all_episodes']:.2f} | {d['mean_unique_cells']:.2f} | {metric['repeat_visit_rate_micro']:.1%} | {d['episodes_with_two_cell_cycles']} |")
        else:
            lines.append(f"| {LABELS[policy]} | 0/{result['scheduled']} | — | — | — | — | — | — |")
    lines += ['', 'SR只对正常完成episode统计，任何API错误另列；本表必须同时看完整题数。SG为所有正常完成任务终止时到目标的曼哈顿距离，成功题SG=0。重复访问包括撞墙原地，micro=总重复动作/总执行动作。两格循环指ABAB窗口且A≠B，单次ABA回退不算持续循环。','',
              '## 3. 按初始距离比较','',
              '| 距离 | Random成功/完整 | 规则探索成功/完整 | VLM成功/完整 | VLM平均SG |','|---|---:|---:|---:|---:|']
    for distance in range(4,9):
        values=[]
        for policy in LABELS:
            row=s['policies'][policy]['by_distance'][str(distance)]
            m=row['metrics_completed_only']
            values.append(f"{m['successes'] if m else 0}/{row['completed']}")
        m=s['policies']['vlm']['by_distance'][str(distance)]['metrics_completed_only']
        lines.append(f"| {distance} | {' | '.join(values)} | {m['mean_sg_all_episodes'] if m else '—'} |")
    lines += ['', '## 4. 循环与目标邻域诊断','',
              '以下真实距离只由评测器在事后计算，从未作为动作输入。','',
              '| 策略 | 两格循环比例 | 最小目标距离均值 | 曾相邻但未到达题数 | 从相邻格离开的动作次数 |',
              '|---|---:|---:|---:|---:|']
    for policy in LABELS:
        d=s['policies'][policy]['all']['completed_diagnostics']
        rate=f"{d['cycle_episode_rate']:.1%}" if d['cycle_episode_rate'] is not None else '—'
        minimum=f"{d['mean_minimum_distance']:.2f}" if d['mean_minimum_distance'] is not None else '—'
        lines.append(f"| {LABELS[policy]} | {rate} | {minimum} | {d['adjacent_but_not_reached_episodes']} | {d['departures_from_adjacent']} |")
    lines += ['', '### VLM逐题路径（事后诊断）','',
              '| 题目 | 结果 | 路径 | 最小距离 | 两格循环 |','|---|---|---|---:|---|']
    for r in records:
        if r['policy']!='vlm': continue
        result='成功' if r['evaluation'] and r['evaluation']['success'] else ('未到达' if r['completion']=='completed' else 'API错误')
        lines.append(f"| {r['episode_id']} | {result} | {' → '.join(map(str,r['positions']))} | {r['diagnostics']['minimum_distance']} | {'是' if r['diagnostics']['two_cell_cycle_present'] else '否'} |")
    lines += ['', '## 5. API稳定性与用量','',
              f"- 实际请求尝试：{api['requests']}；合法动作回答：{api['valid_actions']}；错误尝试：{s['api_error_attempts']}；围栏格式归一化：{s['format_normalized_responses']}。",
              f"- 完整请求平均耗时：{api['latency_mean_seconds']:.3f}秒，P50={api['latency_p50_seconds']:.3f}秒，P95={api['latency_p95_seconds']:.3f}秒（包含失败尝试时延）。" if api['requests'] else '- 无推理请求。',
              f"- 供应商报告token汇总：`{json.dumps(api['reported_token_totals'],ensure_ascii=False)}`。",
              '- 非流式，未测首token延迟/解码token每秒；未核验价格，不估算金额。使用量只记供应商提供值，不把缺失算零。','',
              '## 6. 解释限制','',
              '- 这20题来自4个区域且每个格子目标是航拍图；不能将20题当20个独立地理样本，不给显著性或泛化结论。每种策略仅一次运行。',
              '- 规则探索成功说明按访问记录组织覆盖可以找到目标，不说明其理解图片。固定方向优先级可能与小样本起终点布局偶然契合；规则SR不是对任意地图的保证。',
              '- 最小VLM策略若有循环，说明当前提示/上下文/探索管理组合存在问题，不等同于模型视觉能力全面不足；如果成功，也不能单凭本轮证明视觉有贡献，尚无无图/错图消融。',
              '- 曾到目标邻格不代表已识别目标。所有诊断真值严格留在评测端。',
              '- 不因观察结果修改题目或提示。本轮没有加三Agent、研究记忆或层级搜索，也没有把规则模块偷偷融合进VLM。','',
              '## 7. 文件与复现','',
              '- 运行配置.json：固定选题、模型、提示词、传输、重试与源码hash。',
              '- API调用.jsonl：每次请求尝试、状态、耗时、usage、最终响应与动作；不含Key/图片base64/隐藏推理。',
              '- 动作事件.jsonl / 任务结果.jsonl：所有策略逐步事件、逐题结果及事后诊断。',
              '- 验证汇总.json：总体/分距离/分区域统计及配对结果。',
              '- 离线审计.json：完整轨迹与规则动作离线重放、API输入公开状态/图像hash、指标重新计算、文件hash检查。',
              '- 自动测试结果.txt：回归测试证据。','',
              '执行脚本：project/src/eval/validation_suite.py（会调用云端API）；审计与本报告分别由audit_validation.py和report_validation.py生成，不调用API。已有目录禁止直接重跑覆盖，新的实验目录须先登记根README。']
    payload=('\n'.join(lines)+'\n').encode('utf-8')
    write_immutable_files({folder/'验证分析.md':payload})
    return folder/'验证分析.md'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=MASA/'评测结果/三项验证_v1连接修复')
    args=parser.parse_args()
    print(report(args.run_dir))
