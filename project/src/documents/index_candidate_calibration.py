"""Current-stage navigation and figure entry points, preserving previous reports."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_candidate_calibration import ROOT,FIG,paths,read,LABELS


def prepend(path,heading,entry):
    t=path.read_text('utf-8');assert heading not in t;offset=t.find('\n## ');assert offset>=0;path.write_text(t[:offset]+'\n'+entry+'\n'+t[offset:],'utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1]
    receipt=read(out/'验收结论.json');reg=read(out/'预登记.json');summary=read(out/'对照汇总.json');m=summary['averages']['Clean'];rot=summary['averages']['Rot90CW'];stem={'development':'可靠性校准开发','pilot':'可靠性校准试跑','confirmation':'可靠性校准正式'}[mode];first={'development':21,'pilot':23,'confirmation':25}[mode]
    names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护'];folder=out.relative_to(ROOT).as_posix();cal=paths('calibration')[1].relative_to(ROOT).as_posix();passed=receipt['candidate_numeric_passed']
    heading=f'## 已完成：{stem}（2026-10-01）';entry=f'''{heading}

只校准每权重接受阈值，原22头部校准源图/13200自然配对独立选择；三阈值为{summary['thresholds']}。模型/候选选择/探索器/四均值/5×5/B10不变。当前{reg['source_count']}源文件/{reg['episodes']}题×3、两条件/四臂，共{24*reg['episodes']}在线轨迹完整复核。

| 本批策略 | 干净SR / SG | 旋转90°SR / SG |
|---|---:|---:|
'''
    for a in ('Original','Naive4','Calibrated4','NoTarget'):entry+=f"| {LABELS[a]} | {m[a]['sr_mean']:.2%} / {m[a]['sg_mean']:.3f} | {rot[a]['sr_mean']:.2%} / {rot[a]['sg_mean']:.3f} |\n"
    entry+=f'''
预定放行条件{'全部通过' if passed else '未全部通过'}，原默认保持。离线精度和召回必须一起看，95%校准接受精度不能当导航95%成功或未知域保证；Masa28仍是已见开发，4图试跑不当20图稳定确认。253测试、158400离线四候选评分/33筛选区间和当前批全量动作/10个SR/SG区间独立复核。

[结果报告]({folder}/接受可靠性校准报告.md) / [最终验收]({folder}/验收结论.json) / [完整复核]({folder}/独立复核.json) / [交付核验]({folder}/交付核验.json) / [接受与失败诊断]({folder}/接受与失败诊断.json) / [校准报告]({cal}/校准报告.md) / [结果图{first}](绘图/数据结果图/{names[0]}.pdf) / [配对图{first+1}](绘图/数据结果图/{names[1]}.pdf)。
'''
    if not passed:
        entry+='\n本批停止后续新源文件评测，保留原S2/S3/S4与干净SwissView验收；新20正式留出不消耗，不回写旧候选失败。下一因素由本轮精度/覆盖/导航配对证据决定，不在此批再补策略调到过线。\n'
    elif mode=='confirmation':entry+='\n新20源文件正式确认已完成；方向鲁棒候选范围仅整90度变换/连续航拍切图，原默认未自动替换。\n'
    else:entry+='\n允许推进已预登记下一批，阈值完全冻结。新的08…11/40…59与28历史Swiss文件隔离；04…07不参与选门槛。\n'
    prepend(ROOT/'README.md',heading,entry)
    heading=f'## 新增：{stem}';links='\n'.join(f'| {n} | [PNG](数据结果图/{n}.png) / [PDF](数据结果图/{n}.pdf) / [SVG](数据结果图/{n}.svg) |' for n in names)
    entry=f'''{heading}

当前{reg['source_count']}源文件/{reg['episodes']}题×3，校准四方向SR{m['Calibrated4']['sr_mean']:.2%}/SG{m['Calibrated4']['sg_mean']:.3f}，两目标条件动作不变。{'通过' if passed else '未通过'}预定放行门槛；同时展示原方向策略、0.50四方向、校准与无目标探索，避免以离线精度替代导航性能。

| 图 | 直接取用 |
|---|---|
{links}

[报告](../{folder}/接受可靠性校准报告.md) / [图注]({stem}图注.md) / [数值与哈希](绘图数据/{stem}_v1.json) / [设计与审阅](审阅/{stem}设计与核验_v1.md)。PNG300dpi，PDF嵌入字体与SVG可编辑文本，无包裹位图；推荐宽度≥18cm。三权重标记不是CI，五项区间按源文件成组、逐项解释；旧图不覆盖。
'''
    prepend(FIG/'README.md',heading,entry)
    design=f'''# {stem}科研图设计与核验

类型：实验结果。图{first}分组柱展示四个冻结策略与两个目标条件，SR/SG分面板；图{first+1}展示全部五项配对效应，其中校准−0.50直接隔离新因素。柱从0开始、三seed全部展示，SG含失败终点；颜色/纹样/标签和seed形状双重编码支持灰度。

画布7.8×4.4 / 7.8×4.15英寸，最小9pt。统一微软雅黑与色盲友好蓝/灰/橙/青，推荐插入宽度≥18cm（缩放后约8.18pt）。Matplotlib重现PNG300dpi、矢量PDF/SVG；图注定义SR/SG、样本、控制、区间与边界。不声称校准分数是角度概率，不从95%接受精度推断95%SR。

已查看两张实际PNG，检查图例/数字/轴/区间无裁切重叠；结构/嵌入字体/DPI/哈希和文档链接由[交付核验](../../{folder}/交付核验.json)复查。未发现CRITICAL/MAJOR/MINOR图形问题。
'''
    (FIG/f'审阅/{stem}设计与核验_v1.md').write_text(design,'utf-8')


if __name__=='__main__':main()
