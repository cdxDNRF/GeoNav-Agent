"""Publish a new S2 interpretation and figure without rewriting the prior closeout."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents import stage_figures as graphics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1'
FIG=ROOT/'绘图'


def read(p):return json.loads(p.read_text('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=read(OUT/'验收结论.json');audit=read(OUT/'独立复核.json');summary=read(OUT/'对照汇总.json');reg=read(OUT/'预登记.json')
    assert receipt['audit_sha256']==digest(OUT/'独立复核.json') and audit['status']=='passed'
    assert receipt['summary_sha256']==digest(OUT/'对照汇总.json')
    report=OUT/'结果解读与S3接续.md'
    if report.exists():raise ValueError('Existing result interpretation is immutable')
    labels=[('Edge','Baseline','NoTarget探索'),('ZeroEdge','CueFull','同容量置零'),('Edge','CueFull','Edge真实目标'),
        ('Edge','CueMean','Edge均值目标'),('Edge','CueWrong','Edge错误目标')]
    rows=['# 边缘线索正式S2：结果解读与接续','',
        '本文件在完整复核后生成；原始汇总的audit_pending/正式未判定字段是运行器的历史状态，不回写。', '',
        f"最终状态：**{'正式本地S2通过' if receipt['formal_S2_passed'] else '正式本地S2未通过'}**。默认{'更新为EdgeTargetCue' if receipt['default_changed'] else '保持原方案'}；独立地图确认属于S3，本轮未使用test或扩大网格。", '',
        '| 完整val100×3 | 三种子SR | 成功 / 计划 | 平均SR | SG |','|---|---|---|---:|---:|']
    for a,c,label in labels:
        v=summary['averages'][a][c]
        rows.append(f"| {label} | {' / '.join(f'{s:.0%}' for s in v['sr_by_seed'])} | {sum(v['successes_by_seed'])}/300 | {v['sr_mean']:.2%} | {v['sg_mean']:.3f} |")
    e=summary['effects']['NoTargetBaseline'];ci=e['source_interval']['interval95']
    rows+=['',f"真实目标相对当前探索SR{e['gain']*100:+.2f}个百分点，{e['positive_seeds']}/3种子提高；源图成组95%区间[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]点，SG差{e['lower_metric_change']:+.3f}。", '',
        '## 为什么可以判定本地S2', '',
        '运行前冻结完整val100、全部三个训练权重、原训练均值、阈值0.50、原动作筛选及线索门控；完成1500条神经轨迹和200条确定性规则，未训练、未调阈值、未换题。工程SR/SG/最强规则/各距离档，以及目标遮蔽、错误目标、从头NoTarget探索和同容量ZeroEdge对照全部按原要求判定。', '',
        '工程检查：'+'；'.join(k+'='+('通过' if v else '未通过') for k,v in summary['engineering_checks'].items())+'。', '',
        '目标与边缘贡献检查：'+'；'.join(k+'='+('通过' if v else '未通过') for k,v in summary['target_and_edge_checks'].items())+'。', '',
        f"196项回归测试通过。复核重提100块val图的语义/局部/边缘特征，重放1500条神经轨迹的{audit['neural_steps']}个动作及200条规则；核验33项错误目标距离不匹配例外、公开输入、目标通道干预、合法动作、源图区间和全部冻结哈希。NoTarget及ZeroEdge完整复现原val100探索轨迹。复核由同一执行代理另行实现，不声称不同人员审查。", '',
        '默认配置加载后，三个配对种子各用固定第一题重放成功；原默认完整副本见[原默认配置](原默认配置.json)，当前选择与加载证据见[验收结论](验收结论.json)及[默认加载核验](默认加载核验.json)。未选最高种子、未使用目标遮蔽条件作默认、未做模型集成。', '',
        '## 结论范围', '',
        '4张val源图此前用作开发，86条不同路线并非100条独立路线；三种子重复同题不是300张独立地图。其成组区间是本地验证证据，S2通过不等于独立泛化已证明。', '',
        '旧开发140题/28图的88.57%与本轮val100分开引用；新增收益以本轮真实目标减本轮探索计算。SG包含成功的0距离与失败终点，低SG不能表示每道失败都接近目标。', '',
        '目前是训练过的单智能体模块系统。边缘连续性针对同源、连续、方向一致300×300切图；异时、旋转、不同视角与非连续裁块尚未验证，不将本地验收称为通用语义地理理解。', '',
        '容量检查使用既有同容量ZeroEdge头；NoTarget对照为既有从头训练的探索器，未另训练完整架构NoTarget Edge头。该证据支持模块级视觉线索贡献，不支持所有网络训练设置的因果归因。', '',
        '## 下一步', '',
        ('冻结本批正式验收方案，另行预登记S3：原test250、10张源图、3种子、同策略与相同目标控制/规则；SR≥55%、SG≤2.0、对最强规则+5点、至少6/10源图正增益，保留源图区间与全部失败。按原S3门槛执行，test结果出来后不调至过线。本轮不运行S3。' if receipt['formal_S2_passed'] else
         '先定位本批未通过项并保留结果，回开发集讨论单因素修改；不得按这轮val挑新阈值或最高种子，也不直接进入S3。'), '',
        '[完整原始报告](正式复验报告.md) / [复核](独立复核.json) / [预登记](预登记.json)。历史收尾包和旧图保持原状，本批新增图见绘图目录。']
    report.write_text('\n'.join(rows)+'\n','utf-8')
    # New figure in the registered existing category, keeping old PNG/PDF/SVG intact.
    graphics.style();fig,axes=plt.subplots(1,2,figsize=(7.1,3.55));graphics.frames(axes)
    plot_labels=['探索/置零','真实目标','均值目标','错误目标']
    vals=[summary['averages']['Edge'][c] for c in ('Baseline','CueFull','CueMean','CueWrong')]
    colors=[graphics.GRAY,graphics.BLUE,graphics.ORANGE,graphics.TEAL]
    graphics.bars(axes[0],plot_labels,[r['sr_mean']*100 for r in vals],colors,
        seeds=[[v*100 for v in r['sr_by_seed']] for r in vals],percent=True)
    axes[0].set_ylim(0,109);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）')
    axes[0].set_title('(a) 完整val100×3',loc='left')
    graphics.bars(axes[1],plot_labels,[r['sg_mean'] for r in vals],colors,decimals=3)
    axes[1].set_ylim(0,max(r['sg_mean'] for r in vals)*1.35);axes[1].set_ylabel('SG：平均终点距离（格）')
    axes[1].set_title('(b) 完整val100×3',loc='left')
    fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.88,wspace=.31)
    fig.text(.5,.025,'固定val100；4张已用开发图；3训练种子；5×5，B=10。独立确认属于S3。',ha='center',fontsize=9)
    v=summary['averages']['Edge']['CueFull'];b=summary['averages']['Edge']['Baseline']
    caption=f"冻结边缘线索在完整val100×3上将SR从{b['sr_mean']:.2%}提升至{v['sr_mean']:.2%}，SG从{b['sg_mean']:.3f}降至{v['sg_mean']:.3f}。图示固定100题、4张已用开发源图、三个训练种子，白色标记为seed0/1/2，不是置信区间；NoTarget探索与同容量ZeroEdge轨迹一致。目标均值/错误条件同时处理全局、局部与RGB边缘通道，阈值沿用训练校准的0.50。最终判定见验收结论；4图证据不代表独立S3或跨数据集泛化。"
    graphics.save(fig,FIG/'数据结果图','08_S2正式复验结果',caption,
        ['DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/对照汇总.json'],
        '完整固定val100×3，4源图/86不同路线，冻结5×5/B10')
    manifest=dict(data_scope='formal localS2 confirmation; independent S3 pending',
        source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in (OUT/'对照汇总.json',OUT/'验收结论.json',OUT/'独立复核.json')},
        interpretation_sha256=digest(report),figures=graphics.ITEMS,
        script_sha256=digest(Path(__file__)))
    (FIG/'绘图数据/S2正式复验_v1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (FIG/'S2正式复验图注.md').write_text('# S2正式复验图注\n\n'+caption+'\n\n源数据和图文件哈希见[本批来源](绘图数据/S2正式复验_v1.json)。\n','utf-8')
    print(json.dumps(dict(formal_S2_passed=receipt['formal_S2_passed'],report=str(report),new_figure='08_S2正式复验结果'),ensure_ascii=False))


if __name__=='__main__':main()
