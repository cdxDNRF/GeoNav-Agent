"""CV030 one-shot publication. Never performs image/model work."""
import hashlib
import json
from pathlib import Path
from project.src.tools.ccvpe_engineering_v1 import ROOT,OUT,read,entry,put,verify_binding


def publish():
    binding=verify_binding();resources=read(OUT/'资源护栏回执.json')
    done=read(OUT/'工程执行完成.json') if (OUT/'工程执行完成.json').exists() else {}
    audit=read(OUT/'另实现复核.json') if (OUT/'另实现复核.json').exists() else {}
    passed=bool(resources['passed'] and done.get('passed') and audit.get('passed'))
    status='工程准入通过；感知筛查/导航未执行' if passed else '工程准入未通过；停止不扩大预算'
    counts={'parameters':done.get('parameters'),'calls':done.get('pipeline_calls'),'rss':resources['peak_sampled_rss'],'wall':resources['wall_seconds']}
    text=f'''# CV-030：CCVPE真实工程准入

**{status}。本批不是感知效果、目标定位能力或导航SR验收。**

唯一官方工件为KITTI/no_orientation_prior/model.pt，242293725字节；本地SHA256：`{binding['checkpoint']['sha256']}`。[官方仓库](https://github.com/tudelft-iv/CCVPE/tree/3028aa5c1585d3c274c3a531bdce19846c9c3bd2)、[官方无朝向先验权重目录](https://drive.google.com/drive/folders/1ZqQPpJLNFrMV3lc12VD3RX1cbYRjE8Nu)。此SHA为本地取得计算，非作者签名或公布校验。仅一份权重，无第二骨干下载、优化器或整仓库副本；5份必要官方文件共150910字节，原字节保存。

## 实际结果

- 总参数：{counts['parameters']}；冻结参数、全部模块eval，严格核对键/shape/dtype和逐张量SHA；未训练。
- 完整管线前向：{counts['calls']}（4对原实现/Full包装器共8次，另4次合成控制），两个骨干各12次。原8图各解码一次，未新增图像来源。
- CPU工作进程采样峰值RSS：{counts['rss']/1024**3:.3f}GiB；受护栏执行总时长{counts['wall']:.2f}秒（含导入、载入、解码、摘要与逐张量校验，非纯推理速度基准）。4GiB/1800秒内；0GPU/云模型。
- Full与原前向在全部9输出、六层地面描述子逐项核对；最大绝对差：{done.get('equivalence_maxabs')}。
- Disabled两不同合成目标的全部输出相同，六层匹配精确为零；Mean仅由原校准0/1目标生成且全部六层替换；Wrong实际输入及六层描述子等于指定替代目标，六层匹配保持开启。
- 另实现复核状态：{audit.get('passed',False)}；同一Agent另一实现，不称独立人员审查。复核只读绑定和已保存数值，不再载入模型、解码或前向。

失败时以上完整性以原始运行事件/护栏回执为准，不把缺失调用写成成功。[实际执行](工程执行完成.json)、[控制证据](全部控制检查.json)、[资源护栏](资源护栏回执.json)、[另实现复核](另实现复核.json)、[首前向绑定](首前向冻结绑定.json)。

## 实现与输入权限

固定官方CVM_KITTI类、EfficientNet源及提交。仅构造函数把两个from_pretrained改为from_name，随后严格恢复唯一checkpoint全部张量，避免暗中下载ImageNet权重。原forward保留；AST控制版仅在六处matching_score加入Disabled分支，Full的原表达式/顺序完全保留。执行进程在导入第三方模块前禁止socket连接，无缺包自动安装。

原上游IPython/绘图/TensorBoard/CUDA探测等顶层代码不执行；只选审阅过的三个辅助定义与模型类。控制版和参考版共享同一严格恢复的eval模型，逐次调用无跨样本策略状态。当前工程完整父图属于特权参考验证，**不得把它作为Agent未知区域观测**；没有使用GPS、heading或真实拍摄点旋转/位移图像。

地面照片固定保AR letterbox256×1024、RGB bilinear、浮点ImageNet均值填边及归一化；完整父航片固定512×512。后续局部接口只接125块+公开格号，用均值填未知区域；该接口仅有内存合成测试，本次没有运行局部感知筛查。普通照片FoV与KITTI不同，预处理不同不称原论文完整复现；工程等价不证明跨域泛化。

## 保存与下一步

权重只保存`models/ccvpe_kitti_public_v1/model.pt`，官方必要运行源码只保存`project/.runtime/ccvpe_engineering_v1/`，均排除Git；没有复制旧8图、多朝向热图或整库特征。Mean为本批校准0/1张量均值，旧模型/均值/阈值/标签不改。

{'下一项可另行认领CV-031，按CV029原方案执行20校准Full与80开发控制输出，先明确这100次真实感知的授权与资源，不重复本批12次调用。' if passed else '工程未过，保留失败并停止；不得继续20+80感知、改预算或盲目重跑。'}拒绝阈值尚未生成、未判断真实局部信号；即使之后感知通过也不自动获准自主导航。MM-GAG6封存组、M0/原S2S3S4及平台4.11服务不变，0训练/新导航/SR/默认升级。

公开取得经既有7897代理，前序网页工具无法读取官方目录后使用代理直接访问官方资源，无镜像或持久配置改动。0C盘任务输出、包安装、作者联系、Git提交或推送。恢复先读seal/已落盘回执，不重跑acquire/bind/execute/worker/写盘audit/publish。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(text)
    put('下一步边界.json',dict(engineering_passed=passed,perception_screen_run=False,navigation_admitted=False,
        next_task='CV-031' if passed else 'stop',remaining_future_main_pipeline_budget=100,completed_engineering_not_to_repeat=12,
        original_protocol='调研/20_轻量空间关系感知与拒绝协议_v1.json',confirm_groups_consumed=0))
    p=ROOT/'项目导航/实验索引.json';before=p.read_bytes();idx=read(p);old=idx['batches'];assert len(old)==154
    assert hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==binding['index_canonical']
    report=(OUT/'README.md').relative_to(ROOT).as_posix();path=OUT.relative_to(ROOT).as_posix()
    idx['date']='2026-10-23';idx['batches']=old+[dict(dataset='OpenCVL既有4工程组/CCVPE公开KITTI权重',kind='实际模型工程',name='CCVPE真实工程准入_v1',category='CV-030',status=status,
        scope='8图/8配对+4合成CPU调用；无感知筛查导航',path=path,reports=[report],evidence=[path+'/首前向冻结绑定.json',path+'/另实现复核.json',path+'/资源护栏回执.json'])]
    assert p.read_bytes()==before;p.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'docs/维护/版本生命周期.json';before=p.read_bytes();life=read(p);assert life['current']==binding['runtime_current']
    life['updated']='2026-10-23';life['cv030_ccvpe_engineering']=dict(status=status,report=report,replaces='CV029 artifact/executor pending only;no runtime/model default replaced',
        dependencies=[binding['checkpoint']['path'],'project/.runtime/ccvpe_engineering_v1/', '调研/20_轻量空间关系感知与拒绝协议_v1.json'],recovery='read sealed receipts;do not repeat engineering calls',
        actual_artifact_bound=True,actual_executor_bound=True,engineering_passed=passed,screen_run=False,navigation_run=False)
    assert p.read_bytes()==before;p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;before=p.read_bytes();t=p.read_text(encoding='utf-8-sig');head,_,rest=t.partition('\n');prefix='' if name=='README.md' else '../'
        line=f'[CV-030 · CCVPE真实工程]({prefix}{report})：{status}，固定唯一权重/12次CPU工程；0感知筛查/导航，旧默认保持。'
        assert p.read_bytes()==before;p.write_text(head+'\n\n'+line+'\n\n'+rest,encoding='utf-8')
    p=ROOT/'docs/PROJECT_STATE.md';before=p.read_bytes();t=p.read_text(encoding='utf-8-sig');h,_,rest=t.partition('\n')
    paragraph=f'''\n\n### CV-030：{status}（2026-10-23）

[实际证据](../{report})。用户批准唯一242293725字节权重、必要源码≤256KiB/证据≤16MiB与CPU4GiB1800秒工程后，实际严格载入{counts['parameters']}参数；12完整管线/24骨干调用、8原图各一次解码。4对原/Full全部输出差{done.get('equivalence_maxabs')}，Disabled/Mean/Wrong六层控制与冻结张量核验{done.get('controls_passed',False)}，另实现复核{audit.get('passed',False)}。采样RSS{counts['rss']/1024**3:.3f}GiB/护栏总时长{counts['wall']:.2f}秒。全部CPU，不安装依赖，不停其它进程。

仅验证特权完整父图参考前向/普通照片接口与控制工程；未执行20校准+80局部感知筛查，无拒绝阈值、新感知准确率或导航SR，工程过线不代表空间关系可靠。CV029原设计不回写；后续需另授权100调用筛查，不能重用12次工程预算或直接导航。原4组仍工程开发，MM-GAG6封存与M0/原S2S3S4、平台4.11服务保持。当前155索引，旧154值保持；单权重/源码排除Git，0C盘任务输出/训练/云模型/Git提交推送。
'''
    assert p.read_bytes()==before;p.write_text(h+paragraph+'\n'+rest,encoding='utf-8')
    verify_binding()
    files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()]
    put('阶段封存.json',dict(task='CV-030',files=files,sources=binding['sources'],checkpoint=binding['checkpoint'],vendor=binding['vendor'],passed=passed,
        current_runtime_kept=True,old154_index_values_kept=True,seal_itself_excluded=True))
    size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert size<=16*1024**2
    print(json.dumps(dict(published=True,index=155,passed=passed,evidence_bytes=size),ensure_ascii=False))


if __name__=='__main__':publish()
