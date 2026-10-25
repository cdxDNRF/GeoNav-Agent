"""CV-011 freeze preflight results without authorizing or running training."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.preflight_capturepoint_sources_v1 import OUT,read,put,entry,verify
def main():
    s=read(OUT/'工程汇总.json');audit=read(OUT/'核验/另实现复核.json');assert audit['passed']
    b=read(OUT/'首像素输入源码绑定.json');verify(b['sources']+b['inputs']+b['protected'])
    table='\n'.join(f"| {r['source_id']} | {r['role']} | {r['axis']['best']} | {r['axis']['scores']['row']:.4f} | {r['axis']['margin']:.4f} | {r['ground_std16_max']:.2f}/{r['center_std16_max']:.2f} | {'过' if r['passed'] else '未过'} |" for r in s['sources'])
    put('首批图像与轴向工程预检报告.md',f'''# 同拍摄点首14来源工程预检

CV-011 / 2026-10-17。**{'14来源工程门槛通过' if s['all_required_passed'] else '存在必要工程门槛未过，停止推进训练'}**；这不是目标匹配或导航验收，训练协议和效果门槛仍须另冻，本轮没有训练/模型forward/新预测。

用户明确批准CV-010具体9fit/5开发/6确认，已在首张新图前新记[授权](用户授权记录.json)、[用途快照](用途激活快照.json)和当前来源用途账本。旧20保留划分/待批JSON保留历史原意，不回写。9组12素材转训练准备、5组6素材转开发；每组只看1代表，6确认组24素材及所有同组非代表图未读。旧16/10开发组保持已知开发。

每个代表读取1地面、81个1…9数字格、1完整拼图，共14×83=1162原图。每个文件一次读取原字节/核SHA与字节数、一次解码，地面按EXIF矫正，转RGB后去元数据；JPEG256格和2816拼图尺寸通过断言。16×16 BOX缩略仅用于质量/轴向统计，不是编码器特征或Agent候选库。中心格/地面缩略通道标准差最大值>1；负格均匀只作诊断，不拿它们追过线。

| 来源 | 用途 | 最优轴 | row MAE | 次优间隔 | 地面/中心std | 门槛 |
|---|---|---|---|---|---|---|
{table}

8个非对称探针全部位于1…9网格，比较行/列、翻转与旋转共8映射；row均值绝对差≤8且次优差≥3为冻结门槛。阈值沿用CV-002，探针位置本批在首像素前冻结，不根据结果修改。双轴中心代理标签仍是sat_5_5、9×9中40号，没有认证像素到GPS、北向或米制尺度；不能将拍摄点代理当可见物体所在格。

4个内存PIL/像素测试验证非对称轴区分、全均匀歧义拒绝、全部变换不超出81缓存和只读双RGB接口。另实现从保存的无损16px缩略重算112个轴分数、质量和全部消费/角色计数，不再打开原图；不是对原始解码/EXIF变换的第二套独立实现，也不是独立人员审查。1162原图回执和2324行尝试/完成事件绑定；6确认像素0。

输出≤8MiB、单份缩略≤2MiB，无完整原图/特征/模型副本。生产M0/GUI/Word与旧验收保持；0训练/模型/导航/API/下载，无新Top-k或SR。当前用途已按用户授权调整，不能把这14代表再称未消费确认来源；6确认继续封存。

下一项{'可准备一次同任务对齐的完整训练协议：沿用8192参数/冻结骨干/固定500更新作资源参考，先固定基线、目标控制、来源和效果门槛，再绑定真实执行器/特征' if s['all_required_passed'] else '只保留工程失败，不换来源/代表或阈值；先处理已记录的工程阻塞'}。本轮的授权仅工程，不自动启动模型或训练，也不把工程过当目标有效性过。

仓库根只读核验：`D:/PYTHON/python.exe project/src/eval/audit_capturepoint_preflight_v1.py`（不加--save）。freeze/run/报告已消费，禁止盲重跑；中断查输入消费JSONL和缩略覆盖后另案恢复。
''')
    put('README.md','''# 同任务首批图像与轴向预检

CV-011。已按用户授权进行首14代表工程，6确认组仍封存。工程结果以[报告](首批图像与轴向工程预检报告.md)和[汇总](工程汇总.json)为准；不等于训练/目标效果通过。

- [授权](用户授权记录.json) / [当前用途激活快照](用途激活快照.json)
- [首像素协议](首像素冻结协议.json) / [源码输入绑定](首像素输入源码绑定.json)
- [解码/字节回执](逐图解码与字节回执.json) / [消费日志](输入消费.jsonl)
- [另实现缩略统计核验](核验/另实现复核.json) / [封存](阶段封存.json)

0模型forward/训练/新预测/导航/API/下载，6确认像素0。旧默认/GUI/Word保持；新14代表已有工程消费，不再称未见确认。只读恢复audit_capturepoint_preflight_v1.py（不加--save）；原freeze/run/报告不能重跑覆盖。
''')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['README.md','阶段封存.json']]
    put('阶段封存.json',{'task':'CV-011','files':files,'sources':b['sources'],'sealed_bytes':sum(e['bytes'] for e in files),
        'mutable_entries_excluded':['README.md'],'engineering_passed':s['all_required_passed'],'model_calls':0,'confirmation_pixels':0})
    print(json.dumps({'sealed_files':len(files),'engineering_passed':s['all_required_passed'],'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}),flush=True)
if __name__=='__main__':main()
