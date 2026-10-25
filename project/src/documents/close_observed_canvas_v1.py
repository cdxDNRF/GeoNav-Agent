"""Publish CV032 synthetic-interface evidence only. Does not run a model or change GUI."""
import hashlib
import json
from project.src.eval.observed_canvas_engineering_v1 import ROOT,OUT,SOURCES,read,put,entry,verify


def publish():
    frozen=verify();done=read(OUT/'合成执行完成.json');audit=read(OUT/'另实现复核.json');data=read(OUT/'合成流结果.json')
    assert done['passed'] and audit['passed']
    report=f'''# CV-032：累计已访问画布合成接口

**合成接口工程通过。尚未验证目标感知收益或自主导航。** 已实现可供后续适配的独立核心`project/src/platform_core/observed_canvas_v1.py`，本轮没有接入平台真实运行或GUI，正式4.12/service11/pack23保持。

## 接口与输入权限

会话begin只接显式给定目标RGB、5×5固定网格与预算；observe_public只接当前位置、网格、连续步数、剩余预算、合法四动作和当前125×125 RGB图块。外部历史、父图、路径、源身份、GPS/heading、目标位置和未访图块字段均拒绝。输入来源真实性仍由宿主环境负责，Python对象接口不是恶意代码沙箱。

对第t步，访问掩码M_t为本题真实接受的观测格并集：已访部分保留第一次观测的静态RGB，未访部分固定为ImageNet均值。重访不新增覆盖，边界原地仍耗一步，位置必须与上一步动作一致；静态同格像素变化会拒绝，扰动/异时需要另案。超过像素预算拒绝且不部分提交，也不悄悄丢弃旧观测。跨题begin（包括非法begin）或close都会清除会话目标与历史。

输出为原生625浮点RGB及已访掩码，以及缩放512后归一化NCHW数组；不返回方向、置信度、目标坐标或检索库。原生未知像素精确等于固定均值；双线性缩放边界存在来自合法已访格的混合值，visible_support记录其支持范围，不声称缩放边界完全无混合。掩码只作合法可见范围/审计，不自动增加为神经网络通道。输出是与会话分离的只读数组副本，即使调用者强制修改快照，也不改变内部历史。调用方须串行提交本题观测。

单会话最多保存25幅125图块，RGB字节上限{done['max_stored_RGB_bytes']}（约1.12MiB），目标另限3MiB；渲染时还会产生原生浮点画布/归一化数组，这不是整个Python进程RSS上限。快照由调用方管理，不无界保留。网格与尺寸本轮固定，不外推其它数据协议。

## 实际检查

- {frozen['tests']['count']}项纯内存检查通过：真实未知均值/掩码、行列角点、新格/重访、越界耗步、跳步/重放/预算不符、额外字段/整父图拒绝、预算末步、内存无静默淘汰、新题/错误begin清空、快照隔离、静态重访变化拒绝、关闭清空、全25格压力序列。
- 旧CV030 prepare_local_rgb函数在25个公开格位置上的归一化数组逐项一致；用NumPy张量薄替身代替torch.from_numpy/permute等转换，不导入真实torch，不构造网络。这只核图像预处理的数值兼容，不证明真实PyTorch执行或模型输出等价。
- 合成工程2条流、{done['public_observations']}个公开观测包，保存{done['saved_snapshots']}个压缩数组快照。包含边界原地、新格、回访、跨题和25格压力；它们是人工构造的协议样例，不是32条真实导航或新增SR。
- 另实现按像素坐标反推格号重建{audit['native_pixels_reconstructed']}个原生像素、独立计数访问集合/预算/边界/回访，并对保存数组/哈希核验通过；同Agent另一实现，不是独立人员审查。缩放512的归一化兼容由上述25位置测试覆盖，另实现不声称独立重写PIL重采样内核。

[冻结与测试绑定](合成执行冻结.json) · [合成流结果](合成流结果.json) · [另实现复核](另实现复核.json) · [执行计数](合成执行完成.json)。原生数组与掩码快照仅为合成颜色块，没有读取或复制真实图像。

## 下一步与停止边界

具备准备一个新因素对照的接口基础：**相同合法观测序列下，单格画布与累计已访问画布的感知差异**。真实目标与禁用/均值/错误目标使用完全相同的前缀与行动/信息预算；观察顺序应由预先固定的目标盲规则产生，不能用真实拍摄点规划。原CV031单格失败原件保持。

真实冻结前仍需落实：固定来源/角色与任务清单、目标盲起点/路径、可见ROI与累计掩码的评测定义、校准队列与各条件共享规则、原权重/Mean/预处理/全部执行代码、调用/内存/保存上限、门槛与停止条件。输入分布变化时，应在预登记规则下重新用校准源校准一次，不能改旧阈值或用开发成绩选阈值。当前没有新的真实任务队列或执行授权，不直接启动模型、更不接自主方向。

若累计观测仍不能形成可靠定位与拒绝信号，应按新协议停止该适配，不加大骨干或循环改阈值追分；原预训练条件/FoV/监督差异仍是未分离因素。6确认组继续封存。平台可继续使用当前人工连续观测与只读证据功能，本次核心尚未挂到这些真实流程上。

本批0真实图像读取/解码、torch或模型加载/forward、训练、真实导航、下载、模型API、确认组消费、平台重启、Git提交或推送，无网络/代理或C盘任务输出。新接口与工程/审计/测试/交付5源已首执行前绑定；恢复只读seal/结果，不重跑freeze/execute/写盘audit/publish或旧封存实验。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(report)
    put('下一批准备条件.json',dict(candidate='same lawful observation prefix:current tile vs cumulative visited canvas',
        synthetic_interface_passed=True,real_perception_validated=False,real_queue_frozen=False,real_inference_authorized=False,navigation_admitted=False,
        required=['target-blind start/path frozen before evaluation','source roles and planned denominators','identical prefix across goal controls',
                  'explicit visible ROI and cumulative mask rules','calibration-only tau selection and fixed gates','weight/Mean/source hashes','resource/storage caps and stop rule'],
        unvisited_pixels_allowed=False,confirmation_consumed=0))
    p=ROOT/'项目导航/实验索引.json';old=p.read_bytes();idx=read(p);before=idx['batches'];assert len(before)==158
    assert hashlib.sha256(json.dumps(before,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==frozen['old_index_sha']
    rel=OUT.relative_to(ROOT).as_posix();idx['date']='2026-10-24';idx['batches']=before+[dict(category='CV-032',dataset='5×5合成RGB观测，0真实来源',kind='合法历史接口工程',name='累计已访问画布合成接口_v1',
        status='15合成检查/32观测接口与另实现通过；无模型或导航验收',path=rel,reports=[rel+'/README.md'],evidence=[rel+'/合成执行冻结.json',rel+'/另实现复核.json'])]
    assert p.read_bytes()==old;p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'docs/维护/版本生命周期.json';old=p.read_bytes();life=read(p);assert life['current']==frozen['runtime_current'];life['updated']='2026-10-24'
    life['cv032_observed_canvas']=dict(status='synthetic_interface_passed_not_integrated',report=rel+'/README.md',
        module='project/src/platform_core/observed_canvas_v1.py',replaces='CV032 planned interface only;no model/default/platform replaced',
        dependencies=['RGBFrame contract','fixed public5x5 transitions'],recovery='read sealed engineering outputs;no rerun',navigation_admitted=False)
    assert p.read_bytes()==old;p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;old=p.read_bytes();s=p.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n');prefix='' if name=='README.md' else '../'
        link=f'[CV-032 · 累计已访问画布合成接口]({prefix}{rel}/README.md)：15合成检查/未知均值/重访/跨题与预算通过；0真实图/模型/导航。'
        assert p.read_bytes()==old;p.write_text(h+'\n\n'+link+'\n\n'+rest,encoding='utf-8')
    p=ROOT/'docs/PROJECT_STATE.md';old=p.read_bytes();s=p.read_text(encoding='utf-8-sig');h,_,rest=s.partition('\n')
    text=f'''\n\n### CV-032：累计已访问画布合成接口通过（2026-10-24）

[结果](../{rel}/README.md)。独立observed_canvas_v1核心只收合法连续当前RGB125/公开位置与预算，累计已访纹理、未知均值、静态重访无新增、边界耗步、跨题/关闭清空与像素上限通过。15合成检查、25位置旧预处理数值兼容（NumPy张量替身、无真实torch）、32合成观测4快照和1562500原生像素另实现复核过；0真实来源/模型/预测/导航或新SR。同Agent另一实现非独立人员。

接口尚未接入平台真实流程。4.12/server19/service11/pack23与M0及6确认保持，current入口未变；159索引、旧158值保持。下一先固定目标盲观测序列/角色/队列/ROI与资源门槛，再决定真实当前单格与累计画布对照，不自动启动模型。\n'''
    assert p.read_bytes()==old;p.write_text(h+text+'\n'+rest,encoding='utf-8')
    verify()
    files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()]
    put('阶段封存.json',dict(files=files,sources=frozen['sources'],current_runtime_preserved=True,index=159,seal_itself_excluded=True))
    size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert size<=3*1024**2
    print(json.dumps(dict(published=True,index=159,evidence_bytes=size,real_calls=0),ensure_ascii=False))


if __name__=='__main__':publish()
