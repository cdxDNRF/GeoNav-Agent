"""Create a new small Markdown/HTML evidence reading artifact, never overwrite old reports."""
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/单表征目标对照开发_v1'


def main():
    summary=json.loads((OUT/'对照汇总.json').read_text(encoding='utf-8'))
    run=json.loads((OUT/'执行结果.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'另实现复核.json').read_text(encoding='utf-8'))
    detail=json.loads((OUT/'评分明细.json').read_text(encoding='utf-8'))
    passed=summary['screen']['all_effect_gates_passed']
    conclusion='通过开发信号筛选，仅可另立局部导航协议' if passed else '未通过完整开发筛选，收束此冻结表征，不训练追分'
    grid=summary['screen']['grid'];scene=summary['screen']['scene']
    lines=['# CV-023：单表征真实／禁用／均值／错误目标开发对照','',
        f'2026-10-20。结论：**{conclusion}。** 五个来源已在之前开发中消费，不是新源确认；以下均为检索诊断，不能称导航SR。','',
        '唯一新因素是冻结的Sample4Geo/CVUSA表征管线（指定权重与公开resize一起改变），不将相对旧Identity的差异归因于单独骨干。拍摄点CSV/原代码中心代理格保持，不按可见地标重标。',
        '', '| 目标条件 | 精确格Top1 | Top5 | 平均rank | MRR | 场景Top1 |','|---|---:|---:|---:|---:|---:|']
    for c in ('Full','Disabled','Mean','Wrong'):
        x=grid[c];lines.append(f'| {c} | {x["top1_hits"]}/5 | {x["top5_hits"]}/5 | {x["mean_rank"]:.2f} | {x["MRR"]:.4f} | {scene[c]["top1_hits"]}/5 |')
    lines+=['','## 逐来源精确格证据','',
        '| 已知开发来源 | Full rank | Disabled rank | Mean rank | Wrong rank | 旧Identity Full rank | Full选格曼哈顿偏差（非SG） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    cases={(x['source'],x['condition']):x for x in detail['grid_cases']}
    old={(x['source'],x['condition']):x for x in detail['old_identity_reference']}
    for sid in ['mmgag_062','mmgag_073','mmgag_079','mmgag_039','mmgag_081']:
        r=[cases[sid,c]['rank'] for c in ('Full','Disabled','Mean','Wrong')]
        lines.append(f'| {sid} | '+ ' | '.join(map(str,r))+f' | {old[sid,"Full"]["rank"]} | {cases[sid,"Full"]["selected_cell_manhattan_error_not_SG"]} |')
    lines+=['','## 目标有效性与损伤','',
        '| Full相对条件 | rank胜/平/负（来源） | Top1恢复/损伤 | 平均MRR差 | 来源bootstrap 95%区间 |',
        '|---|---:|---:|---:|---|']
    for p in summary['paired_grid']:
        ci=p['source_bootstrap']['MRR_delta'];bounds=ci['CI95_percentile']
        lines.append(f'| {p["control"]} | {p["rank_wins"]}/{p["rank_ties"]}/{p["rank_losses"]} | {p["recovered_top1"]}/{p["damaged_top1"]} | {ci["mean"]:.4f} | [{bounds[0]:.4f}, {bounds[1]:.4f}] |')
    lines+=['','区间按5来源全部5^5=3125次经验bootstrap重采样，未把81格、四条件或重复图当成独立地理样本。极小且已消费开发样本，区间只描述当前数据，不证实地理泛化；区间是否正并非本轮通过门槛。','',
        '## 首分数前冻结的门槛','', '| 门槛 | 通过 | 实际值 | 要求 |','|---|---|---|---|']
    for g in summary['screen']['gates']:
        lines.append(f'| {g["id"]} | {"是" if g["passed"] else "否"} | {json.dumps(g["actual"],ensure_ascii=False)} | {g["required"]} |')
    identity=summary['old_identity']['Full'];delta=summary['Full_vs_old_identity']
    lines+=['','## 旧Identity参照与统计口径','',
        f'旧Identity Full为Top1 {identity["top1_hits"]}/5、Top5 {identity["top5_hits"]}/5、MRR {identity["MRR"]:.4f}。新Full相对旧Full的来源rank胜/平/负为{delta["rank_wins"]}/{delta["rank_ties"]}/{delta["rank_losses"]}，Top1恢复/损伤{delta["recovered_top1"]}/{delta["damaged_top1"]}。',
        '只重读CV021已存分数，并核对CV012原图SHA/候选顺序/标签。此报告把并列统一为分数降序、物理索引升序；旧报告与旧哈希不改，Disabled统一为零分参照。原Identity编码器与预处理不同，所以这是共同队列参考，不是骨干因果消融或论文公平复现结果。',
        '', '全81格与5场景库只在评测器检索中使用，未交给Agent。场景认对不等于精确拍摄格认对；最佳格曼哈顿偏差只是检索诊断，不是终局SG。本轮没有环境动作、预算轨迹、导航SR、移动距离或默认升级。',
        '', '## 资源、复核与恢复','',
        f'419逻辑输入去重后{run["unique_inputs"]}个：新增编码{run["new_encoded_inputs"]}、引用CV023A成功fit向量1；9fit代表只用于均值，5已消费dev各81格，6确认组未读取。单模型87566465参数冻结，CPU4线程/batch1，运行{run["elapsed_seconds"]:.1f}秒，进程25ms采样峰值RSS {run["sampled_peak_CPU_RSS_bytes"]/1024**3:.2f}GiB。CUDA未测，不声称通过CUDA门槛。0更新/新下载/模型API/导航。',
        '逐输入before/read/forward、SQLite事务提交与after回执，成功输入不重复编码；提交后缺回执可从已存特征补记，forward前标记存在但没有持久特征则视为不确定并停止。一个特征库和小型日志保留，未复制权重/原图/压缩包。',
        f'8项合成边界/恢复/均值/错配/来源权限测试通过。另实现不用torch、不重读原图或模型：重算1620格分数与100场景分数、秩/恢复损伤/18组区间，最大余弦差{audit["max_cosine_error"]:.3g}。同Agent另一实现，不是独立人员审查。历史保护与232平台资产/62用户保持。',
        '', '恢复先读本批冻结/绑定/事件/特征，不重跑旧prepare/collect/train或本次已完成编码。只读复核：`D:\\PYTHON\\python.exe -B project/src/eval/audit_sample4geo_controls_v1.py`。',
        '', '## 后续边界','',
        ('仅表明有理由另立一个只看当前和合法历史的局部匹配/探索协议，尚不启动真实导航或确认。' if passed else
         '本候选按冻结停止条件收束。不能凭场景命中、单个来源好转或Top5替代未过的必要门槛；不追加训练、改resize/FoV、换权重、调阈值或读取6确认追分。先整理跨视角能力边界，再选择有明确证据的新方向。'),
        '平台4.7与PLATFORM-017真实地面导航拒绝保持；本批HTML只是本地评测证据阅读页，不是新生产平台/自由照片定位功能。',
        '', '[冻结协议](冻结协议.json) / [逐格及场景结果](评分明细.json) / [汇总与来源区间](对照汇总.json) / [另实现复核](另实现复核.json) / [执行](执行结果.json)']
    text='\n'.join(lines)+'\n'
    with (OUT/'目标对照开发报告.md').open('x',encoding='utf-8') as f:f.write(text)
    # A small offline text reading page; no remote assets, models or hidden automatic actions.
    def inline(line):
        value=html.escape(line)
        value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
    blocks=[];table=False
    for line in lines:
        if line.startswith('|---'):continue
        if line.startswith('|'):
            first=not table
            if first:blocks.append('<div class="scroll"><table>');table=True
            tag='th' if first else 'td'
            blocks.append('<tr>'+''.join(f'<{tag}>'+inline(x.strip())+f'</{tag}>' for x in line.strip('|').split('|'))+'</tr>')
            continue
        if table:blocks.append('</table></div>');table=False
        if not line:continue
        if line.startswith('## '):blocks.append('<h2>'+inline(line[3:])+'</h2>')
        elif line.startswith('# '):blocks.append('<h1>'+inline(line[2:])+'</h1>')
        else:blocks.append('<p>'+inline(line)+'</p>')
    if table:blocks.append('</table></div>')
    body=''.join(blocks)
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV-023开发证据</title><style>body{margin:0;background:#f1f5f8;color:#183445;font:16px/1.7 system-ui,sans-serif}main{max-width:1100px;margin:30px auto;background:white;padding:clamp(16px,4vw,48px);border-radius:20px;box-shadow:0 12px 40px #16314314}header{color:#087d82;font-weight:700}h1{font-size:clamp(23px,3vw,34px)}h2{margin-top:36px;font-size:22px}p,td{overflow-wrap:anywhere}a{color:#087d82}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px 12px;border-bottom:1px solid #dbe6eb;text-align:left}th{background:#eaf4f3}code{background:#eef2f6;padding:2px 4px;overflow-wrap:anywhere}footer{color:#587080;border-top:1px solid #dbe6eb;padding-top:16px}</style><main><header>GEONAV · CV-023 · 开发诊断，不是导航验收</header><p><a href="目标对照开发报告.md">Markdown报告</a> · <a href="对照汇总.json">完整结构化证据</a></p>'+body+'<footer>0训练 · 0导航 · 0确认组消费 · 当前平台默认保持</footer></main></html>'
    with (OUT/'对照阅读页.html').open('x',encoding='utf-8') as f:f.write(document)
    print(conclusion)


if __name__=='__main__':main()
