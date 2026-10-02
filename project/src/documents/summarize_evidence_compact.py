"""Traceable engineering/message/navigation results, with no retrospective gate changes."""
from collections import Counter
from pathlib import Path
import json
import statistics
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.evidence_compact_g import ROOT,OUT,G,read,lines,write,digest
from agents.evidence_compact_adapter import parse_compact_evidence,parse_compact_plan
from documents import stage_figures as graphics
from matplotlib import pyplot as plt

FIG=ROOT/'绘图';NAME='29_证据账本同预算闭环成功率与距离'
LABELS={'M0':'冻结策略','M1':'确定性账本规划','M2':'强单Agent','M3':'单Agent反思','M4':'双角色协作'}


def text_new(path,content):
    with Path(path).open('x',encoding='utf-8') as f:f.write(content)


def resources(rs,episodes):
    result={}
    for arm in ('M2','M3','M4'):
        rows=[r for r in rs if r['tag']['arm']==arm];lat=[r['latency_seconds'] for r in rows if r.get('latency_seconds') is not None]
        import numpy as np
        elapsed=[r['elapsed_seconds'] for r in episodes if r['arm']==arm]
        result[arm]=dict(requests=len(rows),image_inputs=sum(r['image_count'] for r in rows),
            valid=sum(r['status']=='ok' for r in rows),HTTP_successes=sum(r.get('http_status')==200 for r in rows),
            provider_usage_reported=sum(isinstance(r.get('usage'),dict) for r in rows),
            prompt_tokens=sum((r.get('usage') or {}).get('prompt_tokens',0) for r in rows),
            completion_tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in rows),
            total_tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in rows),
            request_P50_seconds=float(np.median(lat)) if lat else None,request_P95_seconds=float(np.quantile(lat,.95)) if lat else None,
            episode_P50_seconds=float(np.median(elapsed)),episode_P95_seconds=float(np.quantile(elapsed,.95)),
            dollar_cost=None,hidden_reasoning_content_saved=False)
    return result


def main():
    verdict=read(G/'验收结论.json');summary=read(G/'对照汇总_220条计数勘误.json');audit=read(G/'独立复核.json')
    if not verdict['G_audit_passed'] or verdict['audit_sha256']!=digest(G/'独立复核.json') or verdict['summary_sha256']!=digest(G/'对照汇总_220条计数勘误.json'):
        raise ValueError('audited final results required')
    d=read(OUT/'阶段D/机制汇总.json');da=read(OUT/'独立复核.json');legacy=read(OUT/'离线重放/汇总.json')
    rs=list(lines(G/'API响应.jsonl'));rows=list(lines(G/'导航轨迹.jsonl'));cost=resources(rs,rows)
    write(G/'资源汇总.json',cost)
    failures=[]
    for r in rs:
        if r['status']=='ok':continue
        item=dict(call_id=r['call_id'],tag=r['tag'],status=r['status'],error_type=r.get('error_type'),http_status=r.get('http_status'))
        if r.get('http_status')==200:
            state={k:r['public_input'][k] for k in ('grid_size','position','remaining_budget','ledger','candidates','local_proposals')}
            try:
                if r['workflow_stage']=='evidence':parse_compact_evidence(r['raw_content'],r['finish_reason'],state,r['image_labels'])
                else:parse_compact_plan(r['raw_content'],r['finish_reason'],state,r['provided_message'])
            except (ValueError,KeyError,TypeError) as e:item.update(rejection=str(e),raw_content=r['raw_content'])
        failures.append(item)
    write(G/'接口失败诊断.json',failures)
    decision='达到后续独立确认候选条件' if verdict['collaboration_candidate_passed'] else '未达到协作扩大的冻结条件，保留原默认'
    report=['# 云端简化接口、消息作用与同预算闭环验证','',
        f'**工程和消息D均通过；闭环G完成并复核，{decision}。** 旧14请求与新D46及G{len(rs)}累计{60+len(rs)}/420；本批未训练、未消耗预留源图。','',
        '## 接口问题与修复范围','',
        '程序负责公开账本、合法1–2步路径、精确覆盖/重访/边界量及几何引用。E仅输出候选排序、可见粗关系和来源标签；P仅输出最终候选和引用。模型不再抄写派生数量或生成嵌套自由文本事实。保留Gemma4:31b、512输出token、原观察权限、本地权重/阈值/触发规则，不以提高模型数量修复格式。','',
        f"离线重放旧{legacy['responses']}条：原有效{legacy['original_valid']}，只规范化已知类型事实别名和null空反证后接受{legacy['accepted_after_equivalent_representation']}；第{legacy['repaired_call_ids']}条保存收据，第{legacy['rejected_call_ids']}条因来源仍不明而拒绝。未猜数字索引或补缺失来源，未重写原日志/验收，也没有把旧输出转成新接口结果。",'',
        f"新D工程6、消息诊断40，共{da['new_raw_responses']}请求，HTTP成功{da['HTTP_successes']}、schema有效{da['schema_valid']}，现场规范化{da['normalized_responses']}。该有限样本说明本版接口可用，不保证所有未来请求稳定，也不证明视觉粗关系正确。",'',
        '闭环准备期间发现新接入层将环境tuple直接交给要求list的账本，已在新G输入边界转换；原账本/环境保持。失败测试记录保留，修复后4项检查通过，20条原策略路线的逐动作、GRU输入/输出、线索与结果完全复现旧S4。此前303项全套测试通过。','',
        '## D：收到消息与有效使用分别核查','',
        '| 固定状态 | 完整/消融选择（两轮） | 几何质量严格改善且重复 |','|---|---|---|']
    for effect in d['paired_message_effects']:
        ids='；'.join(f"r{p['repeat']}: {p['full_candidate']}/{p['ablation_candidate']}" for p in effect['pairs'])
        report.append(f"| {effect['state']} | {ids} | {'是' if effect['repeated_useful_message_effect'] else '否'} |")
    report+=['',
        '两状态通过功能门槛；另两状态虽然也换候选，质量未严格更好，不能算成功消息作用。隐藏与打乱变换事前固定，共4状态/2API重复，只支持初筛。全部候选几何事实由程序验证；similar/different仅是模型对已给图的粗关系描述，既非邻接核验也非目标坐标证据。','',
        '## G：冻结20题同预算导航','',
        '20个已经用于S4的源文件，每文件1题，初始距离C12…16各4题，10×10/B20，固定本地checkpoint0。总体按源/任务哈希选择，不按成败选题。三轮API重复不增加独立地图数，不是三训练seed。M0/M1各20条，三个云端臂各60条，共220条，所有计划题保留。','',
        '原G预登记/汇总把总数写为200，执行循环实际按20+20+3×3×20执行220，属于漏加M1的计数算术错误。运行中已另登记[计数勘误](阶段G/计数勘误预登记.json)，保留原预登记/源码/汇总，用[勘误汇总](阶段G/对照汇总_220条计数勘误.json)和独立勘误审计验收。每臂分母、SR/SG、配对区间、任务/方法/重复/请求预算完全不变，不添加题目或调用，候选判定仍按原收益条件。','',
        '| 方法 | 正常完成 | 成功数 | SR | SG（格） | 改变原动作 | G请求 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('M0','M1','M2','M3','M4'):
        m=summary['arms'][arm];v=m['metrics'];calls=cost[arm]['requests'] if arm in cost else 0
        report.append(f"| {arm} {LABELS[arm]} | {v['episodes']} | {v['successes']} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} | {m['changed_actions']} | {calls} |")
    report+=['',
        'M2和M4均是同模型、同账本/工具、同图像权限、同两次调用。M2保留自己E的上下文后规划，M4由独立规划上下文读取当前E；这是上下文工作流对照，不能仅凭角色数量归因。M3同Agent计划→反思也限两调用。M1同一公开账本确定性覆盖评分，0云端调用。每题至多1事件，每步更新原GRU；新接受cue优先打断option。','',
        'SR按计划轨迹数计算；SG为所有轨迹的终点曼哈顿距离，成功记0。模型调用失败仍按预定回退走完并保留，另计解析质量；不能把模型回复有效率当导航SR。','',
        '| 配对比较 | SR差（百分点） | 95%源文件区间 | SG差（格） | 正收益API轮数 |','|---|---:|---|---:|---:|']
    for name,e in summary['effects'].items():
        lo,hi=e['source95_SR'];report.append(f"| {name} | {100*e['sr_gain']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {e['sg_change']:+.3f} | {e['positive_API_repeats']}/3 |")
    report+=['',
        '区间为20源文件配对成组bootstrap（4000重采样，seed4121），先在源文件内平均三API重复；是逐项描述区间，不是多比较同时保证或独立地理泛化确认。单轮每题=5个百分点，不能用60轨迹夸大精度。','',
        '### 冻结判定','']
    for key,value in summary['candidate_checks'].items():report.append(f"- {key}: {'通过' if value else '未通过'}")
    report+=['',f'最终：{decision}。门槛包括M4对强M2至少+2个百分点、至少2/3正收益、SG不差、保护M0和所有有效回复；同预算单Agent或零调用M1追平，不能宣布多角色更优。即使点估计通过，也仅获得后续独立确认资格。','',
        '恢复/损伤均按同题原M0配对，三轮重复累计：','',
        '| 云端臂 | 原失败恢复 | 原成功损伤 |','|---|---:|---:|']
    for arm,v in summary['recovery'].items():report.append(f"| {arm} | {v['recovered_failures']} | {v['harmed_successes']} |")
    baseline={r['episode_id']:r for r in rows if r['arm']=='M0'}
    deterministic=[r for r in rows if r['arm']=='M1']
    recovered=sum(r['success'] and not baseline[r['episode_id']]['success'] for r in deterministic)
    harmed=sum(not r['success'] and baseline[r['episode_id']]['success'] for r in deterministic)
    report+=['',f'零调用M1单轮恢复{recovered}条原失败、损伤{harmed}条原成功；这是20题开发点估计，不能凭一题差异替换完整验收过的默认。D的局部覆盖score改善也不保证整个20步导航的SR收益。']
    report+=['','## 资源与复核','',
        '| 臂 | 请求 | 图片输入 | input/output tokens | 请求P50/P95（秒） |','|---|---:|---:|---|---|']
    for arm,c in cost.items():
        med=c['request_P50_seconds'];q=c['request_P95_seconds']
        report.append(f"| {arm} | {c['requests']} | {c['image_inputs']} | {c['prompt_tokens']}/{c['completion_tokens']} | {med:.2f}/{q:.2f} |")
    report+=['',
        'tokens只汇总provider实际返回的usage；失败请求若没有usage，其成本未知，不补零并声称已完全计费。调用上限相同不意味着实耗tokens/时延相同。未核验单价，不编造金额；没有保存API key、图片base64或隐藏思考。','',
        f"G中{len(failures)}次失败保留在[接口失败诊断](阶段G/接口失败诊断.json)：传输失败{sum(r['status']=='transport_error' for r in failures)}，已返回但解析拒绝{sum(r.get('http_status')==200 for r in failures)}。因此D46/46有效不能推为G所有请求必然有效。",'',
        '反思若用evidence_ids替代evidence_refs，原严格schema仍拒绝。检查显示反思请求里的上一计划采用内部标准化形状，包含evidence_ids；这可能诱发旧字段沿用，属于本地输入边界仍可改善的因素，不应全归云端能力。当前批不回写改判或再改提示；未来应把反思历史序列化回仅candidate_id/evidence_refs，并先离线检查。', '',
        f"独立脚本重建{audit['G_requests']}个请求的实际访问前缀、像素/消息哈希，重放{audit['navigation_records']}条完整轨迹/{audit['actions']}动作及全部原GRU/线索，复算指标和{audit['bootstrap_contrasts']}配对区间。{audit['inputs_verified']}项输入和{audit['protected_files_verified']}项历史/权重文件哈希一致；默认保持。这里的独立是另行实现的复核，不是不同人员盲审。",'',
        '当前仍是一个导航实体；E/P是云端决策角色，不是多架无人机。原S2/S3及正式S4/SwissView结论保持；本批是已见地图开发验证，10×10只是切图密度扩展，未验证更大实际地理面积/真实跨时相/任意角度。','',
        '[执行协议](执行协议.md) / [离线重放](离线重放/汇总.json) / [D机制](阶段D/机制汇总.json) / [D复核](独立复核.json) / [G汇总](阶段G/对照汇总_220条计数勘误.json) / [G验收](阶段G/验收结论.json) / [G复核](阶段G/独立复核.json) / [G资源](阶段G/资源汇总.json) / [G轨迹](阶段G/导航轨迹.jsonl) / [G请求](阶段G/请求意图.jsonl) / [G响应](阶段G/API响应.jsonl)']
    text_new(OUT/'简化接口与闭环验证报告.md','\n'.join(report)+'\n')
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(7.8,4.35))
    arms=('M0','M1','M2','M3','M4');colors=[graphics.GRAY,graphics.TEAL,graphics.BLUE,graphics.ORANGE,'#CC79A7']
    labels=['M0\n冻结策略','M1\n确定性规划','M2\n强单Agent','M3\n单Agent反思','M4\n双角色']
    for panel,(field,scale,unit) in enumerate([('sr',100,'SR：成功率（%）'),('mean_sg_all_episodes',1,'SG：平均终点距离（格）')]):
        ax=axes[panel];values=[summary['arms'][a]['metrics'][field]*scale for a in arms]
        ax.bar(range(5),values,color=colors,width=.60,edgecolor='#34424F',linewidth=.7,zorder=2)
        high=[]
        for i,a in enumerate(arms):
            repeats=[m[field]*scale for m in summary['arms'][a]['by_repeat'].values()]
            high.append(max([values[i]]+repeats))
            for rep,v in enumerate(repeats):ax.scatter(i+(rep-1)*.11,v,marker=['o','^','s'][rep],s=24,facecolor='white',edgecolor='#26333C',lw=.7,zorder=4)
            ax.text(i,high[-1]+(2 if panel==0 else .06),f'{values[i]:.1f}' if panel==0 else f'{values[i]:.3f}',ha='center',fontsize=10)
        ax.set_xticks(range(5),labels);ax.set_ylabel(unit);ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,108 if panel==0 else max(high)*1.25+.1)
        if panel==0:ax.set_yticks([0,20,40,60,80,100])
        ax.set_title('(a) 计划题成功率' if panel==0 else '(b) 全部轨迹终点距离',loc='left')
    fig.subplots_adjust(left=.08,right=.99,bottom=.23,top=.89,wspace=.30)
    fig.text(.5,.075,'20个已见源文件/20题；10×10，B=20；本地权重0。',ha='center',fontsize=9)
    fig.text(.5,.025,'白色圆/三角/方形：三轮API重复，非训练seed或置信区间；M0/M1各执行一次。',ha='center',fontsize=9)
    e=summary['effects']['M4_vs_M2']
    caption=f"双角色相对同预算强单Agent的SR差为{100*e['sr_gain']:+.2f}个百分点，{decision}。M0冻结策略、M1零调用确定性账本规划、M2同Agent整理后规划、M3同Agent计划后反思、M4双角色整理后规划；后三者同模型/观察权限/每事件两调用。20已见源文件/20哈希选题，10×10/B20、固定本地权重0，云端三API重复共60条/臂，不增加独立地图数；SG为所有轨迹终点曼哈顿距离，成功记0。白点仅为API重复，配对源文件区间见报告；本图不替代完整S4或独立泛化确认。"
    for ext in ('png','pdf','svg'):
        if (FIG/f'数据结果图/{NAME}.{ext}').exists():raise ValueError('immutable figure exists')
    graphics.save(fig,FIG/'数据结果图',NAME,caption,[(G/'对照汇总_220条计数勘误.json').relative_to(ROOT).as_posix(),(G/'独立复核.json').relative_to(ROOT).as_posix()],summary['scope'])
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/summarize_evidence_compact.py'
    write(FIG/'绘图数据/证据账本简化接口闭环_v2.json',dict(figures=[item],summary_sha256=digest(G/'对照汇总_220条计数勘误.json'),audit_sha256=digest(G/'独立复核.json'),minimum_font_pt=9,recommended_width_cm=18))
    text_new(FIG/'证据账本简化接口闭环图注.md','# 简化接口闭环图注\n\n'+caption+'\n')
    print(dict(report=(OUT/'简化接口与闭环验证报告.md').as_posix(),figure=(FIG/f'数据结果图/{NAME}.png').as_posix()))


if __name__=='__main__':main()
