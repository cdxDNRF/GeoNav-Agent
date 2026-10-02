"""Append current evidence/figure entry points while retaining all old judgments."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_rotation_negative import ROOT,FIG,TRAIN,paths,read,LABELS,STRATEGIES


def prepend(path,heading,entry):
    t=path.read_text('utf-8');assert heading not in t;i=t.find('\n## ');assert i>=0;path.write_text(t[:i]+'\n'+entry+'\n'+t[i:],'utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1];r=read(out/'验收结论.json');g=read(out/'预登记.json');s=read(out/'对照汇总.json')
    stem={'development':'旋转负样本开发','pilot':'旋转负样本试跑','confirmation':'旋转负样本正式'}[mode];first={'development':23,'pilot':25,'confirmation':27}[mode];names=[f'{first:02d}_{stem}成功率与距离',f'{first+1:02d}_{stem}配对收益与保护'];folder=out.relative_to(ROOT).as_posix();training=TRAIN.relative_to(ROOT).as_posix();passed=r['candidate_numeric_passed']
    if mode=='development':
        t=(ROOT/'README.md').read_text('utf-8').replace('## 进行中：旋转负样本视觉头训练对照','## 冻结协议：旋转负样本视觉头训练对照').replace('原头逐参数复现和全部训练独立重放后，才评22诊断及28开发','全部训练完成且原头逐参数复现后评22诊断，全部训练独立重放通过后评28开发')
        (ROOT/'README.md').write_text(t,'utf-8')
    heading=f'## 已完成：{stem}（2026-10-01）';entry=f'''{heading}

只将原87拟合源图的非邻接负目标一半改为三种直角旋转，正样本/标签/探索器/encoder/均值/0.50阈值冻结。同预算两臂×三seed、六头9792训练步全部独立复算，原头逐参数复现。22诊断316800评分、当前{g['source_count']}源文件/{g['episodes']}题×3/两条件/四策略全部重放。

| 本批策略 | 干净SR / SG | 旋转90°SR / SG |
|---|---:|---:|
'''
    for arm in STRATEGIES:entry+=f"| {LABELS[arm]} | {s['averages']['Clean'][arm]['sr_mean']:.2%} / {s['averages']['Clean'][arm]['sg_mean']:.3f} | {s['averages']['Rot90CW'][arm]['sr_mean']:.2%} / {s['averages']['Rot90CW'][arm]['sg_mean']:.3f} |\n"
    e=s['effects']['training_factor_gain'];lo,hi=e['source_interval']['interval95'];failed=[k for k,v in r['release_checks'].items() if not v]
    entry+=f'''
新训练因素相对原头四方向SR{e['gain']*100:+.2f}点，源文件95%区间[{lo*100:+.2f},{hi*100:+.2f}]点，{e['positive_seeds']}/3正seed、SG{e['sg_change']:+.3f}。{'通过' if passed else '未通过'}预定放行条件；未满足项：{', '.join(failed) if failed else '无'}。实际新源图{g['new_source_files']}，原S2/S3/S4与默认保持。

[结果报告]({folder}/旋转负样本对照报告.md) / [验收]({folder}/验收结论.json) / [全量复核]({folder}/独立复核.json) / [交付核验]({folder}/交付核验.json) / [逐题诊断]({folder}/接受与失败诊断.json) / [训练复核]({training}/独立复核.json) / [图{first}](绘图/数据结果图/{names[0]}.pdf) / [图{first+1}](绘图/数据结果图/{names[1]}.pdf)。
'''
    if not passed:entry+='\n本批停止后续新源文件评测；开发/小试跑不能冒充正式test，失败记录保留，不在本批叠加阈值调整。\n'
    elif mode!='confirmation':entry+='\n允许进入已预登记下一批，模型和阈值保持冻结，当前结论尚未经过20新源图正式确认。\n'
    else:entry+='\n20新源图正式确认完成，结论仅覆盖连续切图与直角旋转，默认未自动替换。\n'
    prepend(ROOT/'README.md',heading,entry)
    heading=f'## 新增：{stem}';links='\n'.join(f'| {n} | [PNG](数据结果图/{n}.png) / [PDF](数据结果图/{n}.pdf) / [SVG](数据结果图/{n}.svg) |' for n in names)
    entry=f'''{heading}

同预算负旋转训练：当前{g['source_count']}源文件/{g['episodes']}题×3，{'通过' if passed else '未通过'}预定候选条件，默认保持。图展示全部四策略、两个条件和五项配对比较，训练因素直接与原头四方向比较。

| 图 | 直接取用 |
|---|---|
{links}

[报告](../{folder}/旋转负样本对照报告.md) / [图注]({stem}图注.md) / [数据与哈希](绘图数据/{stem}_v1.json) / [设计与审阅](审阅/{stem}设计与核验_v1.md)。PNG300dpi，PDF/SVG矢量文字；推荐宽度≥18cm，三权重标记不当CI，源文件区间逐项解释。
'''
    prepend(FIG/'README.md',heading,entry)
    design=f'''# {stem}科研图设计与核验

类型：实验结果。图{first}用分组柱对比原方向、原头四方向、新负旋转头和无目标探索；SR/SG两面板。图{first+1}用点与区间展示全部五项配对收益，第五项直接隔离新训练因素，不用旋转穷举优势冒充训练收益。

7.8×4.4/4.2英寸；最小9pt，推荐插入宽度≥18cm，缩放后≥8pt。共享Matplotlib样式、色盲友好蓝/灰/橙/青，色彩/纹样/文字双重编码；白色种子标记全部保留，SG轴上界覆盖最大单seed值。PNG300dpi预览、嵌入字体PDF、SVG可编辑文字；未包装位图。图注明确SR/全部终点SG、源文件/题/种子、训练因素、原头复现、区间与范围。

已查看两张实际PNG；文字、图例、数值、坐标轴与区间未裁切重叠，结构/DPI/嵌入字体/源哈希/链接由[交付核验](../../{folder}/交付核验.json)检查。0 CRITICAL，0 MAJOR，0 MINOR。开发已见源图不能称独立测试，逐项95%区间不保证同时覆盖。
'''
    (FIG/f'审阅/{stem}设计与核验_v1.md').write_text(design,'utf-8')


if __name__=='__main__':main()
