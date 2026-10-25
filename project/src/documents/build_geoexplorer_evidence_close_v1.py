"""Build a closure report from sealed results, never regenerate earlier reports."""
from pathlib import Path
from eval.compile_geoexplorer_evidence_close_v1 import ROOT,BATCH,INPUTS,read,write_new,verify_binding


def link(role,label=None):
    return f'[{label or role}](../../{INPUTS[role]})'


def main():
    protocol=read(BATCH/'元数据/冻结收束协议_v1.json');verify_binding(protocol)
    data=read(BATCH/'核验/证据与静态几何汇总_v1.json')
    if read(BATCH/'核验/另实现复核_v1.json')['status']!='passed':raise ValueError('audit required')
    train=data['training'];nav=data['navigation'];diag=data['diagnostics'];old,new=data['coordinate_math']
    lines=['# GeoExplorer有限预算复现：差异、失败证据与本轮收束', '',
           'BENCH-012 / 2026-10-08。当前结论：工程复现与审计已完成，目标关系的开发泛化及强学习基线充分性尚未证明。三个旧因素收束；最多保留一个非周期位置编码变体，尚未启动训练或性能验证。', '',
           '本轮只读既有结果与代码，新增位置编码解析/NumPy FP32算术；0模型导入/forward/预测/训练/导航/API/下载/备用消费、没有新SR。原M0默认、GeoExplorer成绩、历史验收及失败结论不变。', '',
           '## 1. 哪些结果成立', '',
           '| 同题本地Masa 5×5/B10 | 成功/计划 | SR | 全部终局SG（格） | 实际移动（格） |',
           '|---|---:|---:|---:|---:|']
    for row in nav['metrics']:
        lines.append(f"| {row['method']} / {row['condition']} | {row['successes']}/{row['planned']} | {100*row['sr']:.2f}% | {row['sg_cells_all_planned']:.3f} | {row['movement_cells_all_planned']:.3f} |")
    contrast=nav['contrasts'][0]
    lines += ['',f"M0相对本预算GeoExplorer的SR差{contrast['sr_difference']*100:+.2f}点，10源文件组95%区间[{contrast['sr_difference_ci95'][0]*100:+.2f},{contrast['sr_difference_ci95'][1]*100:+.2f}]点，恢复{contrast['recovered']}/损伤{contrast['harmed']}。这是本地实现对照，不是优于原论文的证据。M0三目标门槛通过，GeoExplorer三个均未过。",'',
              '6000正常终局、50695动作和目标输入权限复核来自 '+link('comparison','BENCH-007原另实现审计')+'；本轮仅复核封存、汇总分母及恢复/损伤算术，不重新计算轨迹。', '',
              '该队列是已消费test250、10源文件、221不同路线/29重复，三权重和四控制不增加地理样本。10源文件地理足迹不重叠尚未确认；137/4/10与论文1188张的差异未核实。SG为所有终局曼哈顿格数，移动距离另列；不是米或失败集合均值。', '',
              '## 2. 复现差异与公平性边界', '',
              f"固定公开代码commit `{train['official_commit']}`。原`get_model`自身设置12层/1152hidden/12heads/512特征，随机构造网络；不是加载Falcon7B语言权重，也不是本项目自行将原公开网络缩至这个尺寸。实际history网络238629248参数、ActorCritic164997参数，结构来源与现场配置绑定。", '',
              '| 项目 | 原公开代码/约定 | 本地执行与影响 |', '|---|---|---|',
              '| 预训练 | 300epoch、BCE+MSE、累积256、验证loss选checkpoint | 每种子32700微批，109路径/epoch，每epoch末flush，合计300优化更新；最终epoch300，不按开发挑最好 |',
              '| PPO | 配置上限1e8环境步 | 每种子163840步、256rollout更新×4=1024头更新；远小于配置上限，不能证明论文实际用了该上限或训练已充分 |',
              '| 数据 | 上游作者数据切分/特征 | 本地109fit、28已知开发；不等于论文832训练/179测试来源，来源规模差异尚未确证原因 |',
              '| 框架/采样 | Lightning、16workers、原依赖 | 串行确定性采样、自有PyTorch执行器、Windows隔离兼容依赖；局部方程与恢复审计通过，不称整条Lightning流水线等价 |',
              '| 推理训练语义 | 原masked acting/unmasked evaluate、horizon/奖励/截断 | 保留原语义，关闭无用history梯度有输出等价检查；并非偷偷优化原方法 |',
              '| 两方法计算/监督 | 原方法不同机制 | 两者共同fit来源/环境交互量；M0有额外边缘头训练、局部特征，Geo有300epoch预训练；总监督/总计算未对齐 |',
              '| 两方法输入 | 当前图/目标图的合法观测 | M0除512全局还用局部/边缘信息，Geo用512全局；信息权限同等不代表表征或参数量相同，51.2点不是纯架构因果差 |',
              '| 测试动作与预算 | 与上游具体协议需逐项核对 | 本地四动作、越界原地耗步、首次/最后一步到达成功、逐题清空；不因存在STOP token声称五动作 |', '',
              '本地已声明八项执行差异：', '']
    lines += ['- '+item for item in train['differences']]
    lines += ['',f"成功三种子各{train['runs'][0]['seconds']/3600:.2f}/{train['runs'][1]['seconds']/3600:.2f}/{train['runs'][2]['seconds']/3600:.2f}小时，总{train['total_seconds']/3600:.2f}小时，训练reserved最高{max(r['reserved_mib'] for r in train['runs']):.0f}MiB；57checkpoint seal、2785原训练输入绑定由原审计核查。旧失败尝试22762微批/208更新成本另列，不能计入成功预算或忽略。耗时受共享GPU负载影响，不是速度基准。", '',
              '完整预算/资源/依赖见 '+link('training_protocol','原执行协议')+'、'+link('training_audit','原完整训练审计')+'；复跑入口和输入绑定见 '+link('reproduction_note','原复现说明')+'。本轮不重新读取全部权重和训练日志，不将当前核验称为新的全训练审计。', '',
              '## 3. 目标泛化失败与三个已关闭方向', '',
              '| 检查 | 拟合来源证据 | 已知开发/必要门槛 | 结论 |', '|---|---|---|---|']
    for row in diag['goal']:
        lines.append(f"| 目标探针 {row['partition']} / {row['head']} | Full DA {100*row['full_accuracy']:.2f}%，Mean {100*row['mean_accuracy']:.2f}% | Full−Mean {100*row['full_minus_mean']:+.2f}点；目标门槛{'过' if row['diagnostic_target_gate_all_passed'] else '未过'} | {'fit内利用目标' if row['partition']=='fit' else '未证明目标泛化'} |")
    ratio=diag['gradient_epoch300_backbone']['median_state_action_norm']
    unseen=diag['context']['unseen_share_of_active']
    lines += [f'| 损失权重λ=0.1 | epoch300骨干MSE/BCE梯度范数比{ratio:.5f} | 0/327预定BCE一阶上升，5机制门槛均未过 | 收束，不复活扫描 |',
              '| 仅增加预训练 | fit150→300目标收益+7.38点，CI为正 | 开发约0、CI跨0、1/3种子正；4门槛未过 | 不支持直接加预算 |',
              f'| 未见目标情境不足 | 有效方向622887，其中未见目标595282，占{100*unseen:.2f}% | 三种子>95%，8粗层109fit均无缺层，3门槛未过 | 收束，不以精确前缀稀疏换主门槛 |', '',
              'DA是方向标签正确率，不是闭环SR。梯度检查只是一阶普通梯度；曲线为已有检查点，不证明全部延长训练必然无效；BCE权重曝光不是梯度贡献。28开发源已反复用于开发且地理足迹独立性未知，不称新地理确认。目标在fit有效、在开发未过，不等于整网完全忽略目标。', '',
              '原证据：'+link('goal','BENCH-008目标')+' / '+link('gradient','BENCH-009梯度')+' / '+link('curve','BENCH-010曲线')+' / '+link('context','BENCH-011情境')+'。原文件/失败结论和回退入口保留，不用本总结回写旧判定。', '',
              '## 4. 新发现的静态机制及其限度', '',
              '原位置特征为 `(sin(2πc/(k−1)), sin(2πr/(k−1)), cos(2πc/(k−1)), cos(2πr/(k−1)))`。第0与k−1行/列的完整圆周编码在解析数学上相同，但矩形网格没有环绕连通。', '',
              f"5×5的25位置在解析数学上只有{old['class_count']}类；NumPy FP32、L2容差1e−6得到相同{old['class_count']}类，{old['near_pair_count']}个不同格近重合对、{old['non_singleton_classes']}个非单例类、涉及{old['cells_in_non_singletons']}格。最大近重合距离{old['maximum_near_pair_l2']:.3g}；实际FP32不是字节完全相同，本轮没有Torch/CUDA数值等价验证。", '',
              f"共有{len(old['coordinate_only_disjoint_direction_labels'])}个（近重合当前格对，固定离线目标）组合要求互不相交的前进方向。例如当前(0,0)/(0,4)、目标(0,2)分别应右/左。这里目标坐标只用于离线标签，绝不进入策略。", '',
              f"只将圆周系数2π改为π、保留四维正余弦输入，解析及FP32均得{new['class_count']}类、0近重合对，最小格间L2距离{new['minimum_distinct_cell_l2']:.6f}。候选无需增加输入维度/参数，但必须重新训练比较，不能直接替换旧权重的编码。", '',
              '**限定推论：** 这证实原坐标通道存在边界歧义，没有证实完整视觉输入碰撞，更没有证实它是开发目标失效的原因。各格图像、历史和合法动作可能消除歧义。两个编码同时有其它几何距离变化，不能把后续任何收益全部归于端点分离。本因素也不是EVAL-004的大网格坐标/访问桶漂移修复。', '',
              '## 5. 下一项决策与未完成论文证据', '',
              '保留且只保留“半圆位置编码”自拟训练变体，下一批先完成资源/方程/输入权限检查，再做同初始化、同路径、同更新预算的训练/开发对照。详见[未启动冻结草案](下一项位置编码单因素草案_v1.md)。本轮没有开始训练，也没有选择新的默认或证明SR提升。', '',
              '仍缺：可靠学习基线目标有效性；两方法在未参与选型、足迹核验的新区域上的共同确认；目标控制/消融与训练资源差异充分交代。即使候选开发成功，它也是我们的GeoExplorer变体，不能改名为原论文忠实复现或用来删除原35.60%失败记录。', '',
              '新区域来源/资格目前未冻结，不能直接借用三个MassGIS备用区；若没有合适未见来源，先停在开发结论。完整网络复跑需重新授权/登记执行版本，不能因本草案存在就启动。', '',
              '## 6. 代码锚点与复核', '', '| 核对项 | 冻结源码位置 |', '|---|---|']
    for name,a in data['source_contract'].items():
        lines.append(f"| {name} | [{Path(a['file']).name}:{a['start']}–{a['end']}](../../{a['file']}#L{a['start']}) |")
    lines += ['', '本轮[绑定协议](元数据/冻结收束协议_v1.json)、[JSON汇总](核验/证据与静态几何汇总_v1.json)、[另实现复核](核验/另实现复核_v1.json)与5份源码快照可追溯。解析同余类/邻格距离标签与主实现的向量聚类/相对方向标签分别实现；同一Codex完成，不称独立人员审查。原轨迹/梯度/曝光审计仅复用并核验封存与算术。', '']
    write_new(BATCH/'复现差异与诊断收束报告_v1.md','\n'.join(lines))
    write_new(BATCH/'README.md', '# BENCH-012：有限预算学习基线诊断收束\n\n[当前报告](复现差异与诊断收束报告_v1.md) / [绑定](元数据/冻结收束协议_v1.json) / [另实现复核](核验/另实现复核_v1.json) / [下一项未启动草案](下一项位置编码单因素草案_v1.md)。\n\n本交付替代BENCH-011的主线进度入口，保留BENCH-007至011原址证据与各自验收；不替代M0默认或原GeoExplorer基线。不训练/forward/导航，无新SR。\n\n在仓库根设定PYTHONPATH为project/src后，使用D:/PYTHON/python.exe执行`-m tools.prepare_geoexplorer_evidence_close_v1 verify`进行只读SHA核验；`-m eval.audit_geoexplorer_evidence_close_v1`另实现复核不加--save。禁止在本批重跑prepare、--run、报告生成或--save；复现算术需另登记新批次并核对输入绑定。\n')
    print('new report and README written; original evidence untouched')


if __name__=='__main__':main()
