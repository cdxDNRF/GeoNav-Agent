"""Summarize independently verified, fixed scalar threshold screening."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.candidate_calibration import paths
from train.local_capacity import read
from train.dyncur_tiny import digest


def main():
    out=paths('calibration')[1];receipt=read(out/'验收结论.json');assert receipt['quality_and_replay_passed'] and digest(out/'独立复核.json')==receipt['audit_sha256']
    report=out/'校准报告.md'
    if report.exists():raise ValueError('immutable report exists')
    text=['# 四方向候选接受阈值：离线校准','',
        '只调整每份原权重一个标量接受门槛，不更新模型/探索器/候选选择。阈值完全由原22头部校准源图确定，先冻结并独立复核，再进行140题开发导航；本报告没有新导航SR。', '',
        '| 权重 | 选定阈值 | 0.50精度/邻接召回 | 选定精度/邻接召回 | 接受数 | 95%源文件筛选区间 |', '|---|---:|---:|---:|---:|---|']
    for s in range(3):
        x=read(out/f'Edge_s{s}/校准阈值.json');b=x['grid'][0]['metrics'];t=x['threshold']
        if t is None:text.append(f'| {s} | null/全弃权 | {b["accepted_precision"]:.2%}/{b["adjacent_recall"]:.2%} | 无接受/0 | 0 | [0,0] |');continue
        m=next(r['metrics'] for r in x['grid'] if r['threshold']==t);lo,hi=m['precision_interval']['interval95']
        text.append(f'| {s} | {t} | {b["accepted_precision"]:.2%}/{b["adjacent_recall"]:.2%} | {m["accepted_precision"]:.2%}/{m["adjacent_recall"]:.2%} | {m["accepted"]} | [{lo:.2%},{hi:.2%}] |')
    text+=['','支持门槛≥100接受/≥5源图，精度≥95%、普通源文件筛选区间下界≥90%；11个固定阈值完整保留，选最小满足值保留覆盖。四候选取最大原始方向概率，五类向量/合法最高动作筛选不变；此分数没有变成角度后验概率。', '',
        '22源图的13200自然全配对含1760邻接与11440非邻接；不同于在线策略遇见的状态分布。校准源图参加过探索器109拟合、未参加视觉头87梯度训练；与28开发源文件及文件哈希隔离。source4119/2000重采样保留零接受源图，零接受重采样精度记0。筛选区间有选择偏差，不能宣称未知域保证或导航成功率95%。', '',
        '253测试、2200旋转候选/550当前格及全部global/local/profile、158400原始四候选评分、33精度区间与三个门槛选择全量独立复核通过。提高精度可能降低目标召回，必须以冻结140题/三权重导航评SR与SG，不能只看离线精度。旧模型、默认和全部历史验收不改变，无训练/云调用/下载。', '',
        '[运行前登记](预登记.json) / [阈值冻结](阈值冻结结束.json) / [独立复核](独立复核.json) / [本批验收](验收结论.json)。']
    report.write_text('\n'.join(text)+'\n','utf-8');print(str(report))


if __name__=='__main__':main()
