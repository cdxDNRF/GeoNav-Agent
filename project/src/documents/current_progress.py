"""A dated, evidence-linked project status snapshot after the current experiment closes."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.hard_rotation_navigation import ROOT,TRAIN,paths
from train.local_capacity import read
from train.dyncur_tiny import digest
STAGES=[('S2本地稳定验证','Masa/评测结果/边缘线索S2正式复验_v1','formal_S2_passed','5×5/B10',100),
        ('S3本地独立源图','Masa/评测结果/边缘线索S3独立源图确认_v1','formal_S3_passed','5×5/B10',250),
        ('S4网格密度扩展','Masa/评测结果/S4十乘十正式扩展_v1','formal_S4_passed','10×10/B20',500),
        ('SwissView同模态迁移','SwissView/评测结果/五乘五正式迁移_v1','formal_cross_dataset_passed','5×5/B10',500)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1];r=read(out/'验收结论.json');g=read(out/'预登记.json');s=read(out/'对照汇总.json')
    document=ROOT/'选题报告相关/当前阶段与进展_2026-10-01_v1.md'
    if document.exists():raise ValueError('immutable progress snapshot exists')
    sources={};text=['# 当前阶段与进展：2026-10-01','',
        '**项目处于S4扩展之后的目标图鲁棒性优化阶段。** 基础工程及已冻结主方法的正式实验已经完成，正在验证新的旋转鲁棒候选。当前候选的放行状态与原主方法的S2/S3/S4状态分别记录。','',
        '## 已完成主线','',
        '环境/奖励/评测基础修复、统一任务与日志、最小云端VLM接入、单因素对照、阶段收尾与科研图均已建立。当前主方法是本地训练过的Small256 NoTarget探索器+冻结EdgeTargetCue视觉线索头，阈值0.50；探索器利用当前图和公开状态，视觉头利用目标与当前图的细边缘/语义匹配，仅对可信合法未访问线索接管。不是已经交付的零训练多智能体系统。','',
        '| 验收 | 源文件 | 评测题数（3权重） | 协议 | SR | SG | 状态/证据 |','|---|---:|---:|---|---:|---:|---|']
    for label,sub,field,protocol,N in STAGES:
        folder=ROOT/'DATA/processed_data'/sub;receipt=read(folder/'验收结论.json');summary=read(folder/'对照汇总.json');metric=summary['averages']['Edge']['CueFull']
        assert receipt[field] and digest(folder/'独立复核.json')==receipt['audit_sha256'] and digest(folder/'对照汇总.json')==receipt['summary_sha256']
        for name in ('验收结论.json','对照汇总.json','独立复核.json'):sources[(folder/name).relative_to(ROOT).as_posix()]=digest(folder/name)
        link=(folder/'验收结论.json').relative_to(ROOT).as_posix();text.append(f"| {label} | {metric['source_count']} | {N}×3 | {protocol} | {metric['sr_mean']:.2%} | {metric['sg_mean']:.3f} | [通过](../{link}) |")
    text+=['','三权重是三份训练参数，重复运行或采样不会增加独立源图。不同协议分数不合并为总成功率，也不与不同论文预算直接相减。10×10是在同一源图内提高切图密度，未验证更大真实地理范围；SwissView是另一数据集的同模态连续航拍迁移，未证明严格地理非重叠或编码器预训练无覆盖。','',
        '## 最近三项补充实验','',
        '提高接受门槛至0.90：Masa开发SR86.19%，SG0.412；接受精度提高但丢失真实线索，未通过干净保护。','',
        '均匀旋转负样本：Masa开发SR89.05%，SG0.324，相对原头四方向+0.24点，只有1/3seed正收益，未通过新因素放行。','',
        f"本轮高分旋转负样本：87拟合源图挖掘，六头同预算训练；当前{g['source_count']}源文件/{g['episodes']}题×3、五策略、两条件。",'',
        '| 本轮策略 | 干净SR / SG | 90°目标SR / SG |','|---|---:|---:|']
    names={'Original':'原方向策略','Raw4':'原头四方向','Replay4':'同预算均匀负旋转','Hard4':'高分负旋转','NoTarget':'无目标探索'}
    for arm,name in names.items():
        clean=s['averages']['Clean'][arm];rot=s['averages']['Rot90CW'][arm];text.append(f"| {name} | {clean['sr_mean']:.2%} / {clean['sg_mean']:.3f} | {rot['sr_mean']:.2%} / {rot['sg_mean']:.3f} |")
    e=s['effects']['training_factor_gain'];lo,hi=e['source_interval']['interval95'];failed=[n for n,v in r['release_checks'].items() if not v]
    text+=['',f"挖掘相对均匀采样SR{e['gain']*100:+.2f}点，源文件95%区间[{lo*100:+.2f},{hi*100:+.2f}]点，{e['positive_seeds']}/3正seed，SG{e['sg_change']:+.3f}。本轮{'通过' if r['candidate_numeric_passed'] else '未通过'}预定放行；未满足项：{', '.join(failed) if failed else '无'}。工程质量与完整重放通过，本批新源图{g['new_source_files']}；原默认和既有正式验收保持。",'',
        f"[本轮报告](../{out.relative_to(ROOT).as_posix()}/高分旋转负样本对照报告.md) / [验收](../{out.relative_to(ROOT).as_posix()}/验收结论.json) / [训练复核](../{TRAIN.relative_to(ROOT).as_posix()}/独立复核.json) / [科研图入口](../绘图/README.md)。",'',
        '## 尚未完成与后续边界','',
        '新方向鲁棒候选仍须经过未见SwissView源文件确认才能称泛化成立。真实异时/季节变化、任意角度旋转、跨视角、MM-GAG和25×25尚未验证；GUI不是评测前置，当前没有交互轨迹回放界面。Gemma/DeepSeek云端仅有4图20题同策略比较，原云端G的完整S2未完成，不能把本地验收移用于云端。','',
        '当前科研材料足以支持课程项目的工程与受控实验结果；论文主张应限于已验证范围。若本轮仍无明确SR收益，先收束旋转训练因素并整理失败边界，再为下一假设设置有限预算与新确认源图。不要用同一开发集持续调到过线。','',
        '## 来源SHA','', '```json']
    for folder in (out,TRAIN):
        for name in ('验收结论.json','独立复核.json','对照汇总.json'):
            f=folder/name
            if f.exists():sources[f.relative_to(ROOT).as_posix()]=digest(f)
    text+=[json.dumps(sources,ensure_ascii=False,indent=2),'```',''];document.write_text('\n'.join(text),'utf-8');print(str(document))


if __name__=='__main__':main()
