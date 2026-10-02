"""Add current reviewed orientation results to project/figure indexes."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_orientation import ROOT,FIG,paths,read


def insert(path,heading,entry):
    text=path.read_text('utf-8');assert heading not in text
    first=text.find('\n## ');assert first>=0
    path.write_text(text[:first]+'\n'+entry+'\n'+text[first:],'utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1]
    summary=read(out/'对照汇总.json');receipt=read(out/'验收结论.json');reg=read(out/'预登记.json');a=summary['averages'];stem={'development':'方向候选开发','pilot':'方向候选试跑','confirmation':'方向候选独立确认'}[mode]
    first={'development':17,'pilot':19,'confirmation':21}[mode];names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护'];folder=out.relative_to(ROOT).as_posix()
    heading=f'## 已完成：{stem}验证（2026-10-01）';passed=receipt['candidate_numeric_passed'];m=a['Clean']['Candidate4'];gain=summary['effects']['rotation_recovery'];lo,hi=gain['source_interval']['interval95']
    entry=f'''{heading}

固定{reg['source_count']}源文件/{reg['episodes']}题×3、两目标条件/三策略，共{18*reg['episodes']}在线轨迹。四方向候选Clean/Rot90CW均SR **{m['sr_mean']:.2%}**、SG **{m['sg_mean']:.3f}**；原策略分别为{a['Clean']['Original']['sr_mean']:.2%}/{a['Rot90CW']['Original']['sr_mean']:.2%}，NoTarget{a['Clean']['NoTarget']['sr_mean']:.2%}。旋转恢复+{gain['gain']*100:.2f}点，源文件95%区间[{lo*100:+.2f},{hi*100:+.2f}]点。预定放行检查{'全部通过' if passed else '未全部通过'}，全部像素/特征/动作/统计复核通过。

[报告]({folder}/方向候选验证报告.md) / [最终验收]({folder}/验收结论.json) / [完整复核]({folder}/独立复核.json) / [交付核验]({folder}/交付核验.json) / [结果图{first}](绘图/数据结果图/{names[0]}.pdf) / [配对图{first+1}](绘图/数据结果图/{names[1]}.pdf)。

只加给定目标0/90/180/270度候选，原模型、四份均值、阈值0.50、5×5/B10原样；不提供真朝向。默认未替换；整90度集合不变性不证明任意角度、裁剪、异时或跨视角鲁棒。Masa28是已见开发；SwissView04…07/40…59提前选定，未用旧正式20图调本项，源文件隔离不等于严格地理无重叠。运行前248测试＋审计修复2回归通过。修复仅针对完全对称PNG的并列角度标签，原审计器、实验和旧验收保持冻结。
'''
    if passed and mode!='confirmation':entry+='\n当前批允许继续预登记下一批，以每批验收/交付为准；开发或试跑不是正式独立确认。\n'
    elif passed:entry+='\n本轮四方向候选已通过新20源文件正式独立确认。SwissView累计48源文件已跑策略，剩余52未跑策略；本轮正式500题的独立单位只有20源文件。后续裁剪/尺度等另起因素，保留干净原默认。\n'
    else:entry+='\n本候选停止在当前批，不再消耗后续SwissView留出，不改门槛或默认；负结果已保存。\n'
    insert(ROOT/'README.md',heading,entry)
    fheading=f'## 新增：{stem}';links=[]
    for n in names:links.append(f'| {n} | [PNG](数据结果图/{n}.png) / [PDF](数据结果图/{n}.pdf) / [SVG](数据结果图/{n}.svg) |')
    figure_entry=f'''{fheading}

{reg['source_count']}源文件/{reg['episodes']}题×3；四方向候选干净/旋转均SR{m['sr_mean']:.2%}、SG{m['sg_mean']:.3f}。{'全部预定门槛通过' if passed else '未全部通过预定门槛'}；原权重/阈值/预算不变，候选原始最大分数未作角度校准。

| 图 | 可直接取用 |
|---|---|
'''+ '\n'.join(links)+f'''

[报告](../{folder}/方向候选验证报告.md) / [完整图注]({stem}图注.md) / [源数据与哈希](绘图数据/{stem}_v1.json) / [设计与审阅](审阅/{stem}设计与核验_v1.md)。PNG300dpi，PDF嵌入字体与SVG可编辑文字；推荐插入宽度≥18cm以保留8pt以上字号。三权重标记不是CI，配对区间按源文件成组，不增加独立样本数。旧图不覆盖。
'''
    insert(FIG/'README.md',fheading,figure_entry)
    design=f'''# {stem}科研图设计与核验

类型：实验结果图。图{first}用分组柱对照干净/旋转两条件，SR/SG分左右面板；图{first+1}用四项全部配对效应森林图，带源文件95%区间和预定门槛短线。柱从0开始，SG包含所有终点；不用颜色代替标签，柱纹样/权重形状支持灰度阅读。

画布7.6×4.25 / 7.6×3.85英寸；最小9pt、微软雅黑与统一色盲友好蓝/灰/青。Matplotlib程序重现，300dpiPNG及无位图PDF/SVG。推荐宽度18cm，缩放后最小约8.39pt。图注定义SR/SG、样本、控制、seed、区间与适用边界；不声称真朝向预测或任意几何鲁棒。

已人工查看两张实际PNG，检查图例、数值、区间、轴和图注无裁切/重叠；结构、字体、DPI、哈希及文档链接由[交付核验](../../{folder}/交付核验.json)复查。未发现CRITICAL/MAJOR/MINOR图形问题。重复权重不是地图样本，开发/试跑/正式的结果分开取用。
'''
    (FIG/f'审阅/{stem}设计与核验_v1.md').write_text(design,'utf-8')


if __name__=='__main__':main()
