"""CV-024 single delivery closure. No model, new navigation or old generator."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/目标语义与局部闭环工程_v1'
REL=OUT.relative_to(ROOT).as_posix()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def entry(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}


def put(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
        f.write('\n')


def main():
    if (OUT/'README.md').exists():raise FileExistsError('Closure already consumed; inspect rather than rerun')
    result=read(OUT/'结果汇总.json');audit=read(OUT/'另实现复核.json');done=read(OUT/'执行完成.json')
    assert audit['passed'] and audit['episodes']==100 and done['actions']==914
    queue=read(ROOT/'DATA/processed_data/MM-GAG/同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
    roles=read(ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json')['records']
    tasks=read(OUT/'任务清单.json')['tasks']; original={t['pair_id']:t for t in queue['development_tasks']}
    source_roles={r['source_id']:r for r in roles}
    for t in tasks:
        old=original[t['parent_window']]
        for key in ('source_id_evaluator_only','group_evaluator_only','ground_feature_index','wrong_ground_feature_index','aerial_feature_indices','goal_index_evaluator_only'):
            assert t[key]==old[key]
        assert source_roles[t['source_id_evaluator_only']]['role']=='development'
        # Alternate mapping: identify the numeric(5,5) filename, then recover
        # crop origin from the first tile and locate its row-major index.
        coords=[tuple(map(int,re.search(r'/sat_(\d+)_(\d+)\.jpg$',queue['aerial_encoder_inputs'][i]['path']).groups())) for i in t['aerial_feature_indices']]
        origin=coords[0]
        goal=(5-origin[0])*5+(5-origin[1])
        assert coords.index((5,5))==goal==t['goal_index_evaluator_only']
        assert coords==[(origin[0]+r,origin[1]+c) for r in range(5) for c in range(5)]
        assert abs(t['start']//5-goal//5)+abs(t['start']%5-goal%5)==t['distance_evaluator_only']
    assert len({r['group'] for r in roles if r['role']=='confirmation_reserved'})==6
    put('交付边界核验.json',{'passed':True,'alternate_numeric_filename_to_goal_mapping':20,
        'all_tasks_match_prior_windows_and_development_roles':True,'reserved_groups_still_registered':6,
        'reserved_pixels_or_features_read':0,'same_agent_not_independent_person':True})
    lines=['# CV-024：地面目标局部闭环工程与开发结果','',
        '**最小闭环工程通过；局部语义趋势因素未达效果门槛，按协议收束。** 这是5个已消费开发来源、20道固定题/100条条件轨迹的有界验证；不是新区域确认、论文复现或平台默认升级。','',
        '| 条件 | 成功/计划 | SR | SG（格） |','|---|---:|---:|---:|']
    labels={'Full':'真实地面目标','Disabled':'禁用目标','Mean':'9个fit均值目标','Wrong':'固定错误目标','Frontier':'同规则Frontier'}
    for arm,v in result['arms'].items():lines.append(f"| {labels[arm]} | {v['success']}/20 | {v['SR']:.0%} | {v['SG_cells']:.2f} |")
    lines+=['','真实目标对Frontier恢复4题、损伤6题、净−2题（−10个百分点）；分来源配对SR差的95%区间[−40,+20]个百分点。对均值净−4题，对错误目标净−5题；SG低于Frontier但高于错误目标，所以不能用较低SG替代必要的SR与目标有效性门槛。',
        'Full相对Frontier有19/20题动作序列变化，但仅2/5来源SR正收益，3项效果门槛均未过。这表明目标能影响计划，却没有证明影响方向可靠；错误目标较高的成功率不能解释为错误目标更懂地理。小样本、已知开发来源和规则路径都限制外推。','',
        '## 本轮实现','',
        '新增无GUI的GroundLocalLoop：给定目标向量→当前合法航拍向量→已访问历史的余弦变化→等路径成本frontier排序→执行动作→接收下一观测。Agent每次只接收一个当前512维向量和公开位置/网格/步数/剩余预算；完整缓存、任务源、裁剪原点、真实目标和距离仅在评测器中。没有把静态全图检索当导航，也没有用目标相似度作到达真值；成功由评测器首次到达标签格判定。',
        '本次用既有编码缓存运行新的逐步动作闭环，不是重放旧路线；未进行新的照片编码，不能声称未知上传照片端到端导航已可用。Web4.8/现有M0/地面导航拒绝保持，未重启生产服务。此模块可作为后续真实编码适配的调用接口，但当前因素不推荐升级默认。','',
        '## 语义与资产','',
        '公开GOMAA-Geo/models/ppo.py的validate_ground_level_custom使用行[5−goal//5:10−goal//5]、列[5−goal%5:10−goal%5]，把配对数组中心[5,5]置于目标格。本批全部20窗和原队列/轴向另算一致。这里定位的是数据约定的配对中心代理格；坐标CSV存在并不能证明实测拍摄点的精确像素映射，也不能自动改成可见地标格。历史capture_point约定、旧标签和失败结论未修改。',
        '复用已有公开CLIP ViT-B/32地面特征和本地Sat2Cap航拍特征，不使用旧8192小头或CVUSA权重。缓存/模型/预处理来源绑定与SHA通过，512维工程兼容不等于已证明语义对齐有效。本地Sat2Cap卡仍是模板，原论文run/编码器映射未齐，不能称公平复现原论文。','',
        '## 分层证据','',
        '| 开发来源 | 真实目标成功/4 | Frontier成功/4 | 真实目标SG | Frontier SG |','|---|---:|---:|---:|---:|']
    for s,v in result['arms']['Full']['by_source'].items():
        c=result['arms']['Frontier']['by_source'][s]
        lines.append(f"| {s} | {v['success']}/4 | {c['success']}/4 | {v['SG']:.2f} | {c['SG']:.2f} |")
    lines+=['','| 初始距离 | 真实目标成功/5 | Frontier成功/5 |','|---|---:|---:|']
    for d,v in result['arms']['Full']['by_distance'].items():lines.append(f"| C{d} | {v['success']}/5 | {result['arms']['Frontier']['by_distance'][d]['success']}/5 |")
    lines+=['','5个来源仍是5个开发来源；20题、5条件、914动作不增加独立地理样本。C7未纳入这次20题工程队列，不能据此宣称复现完整C4—8论文评测。米数尺度未核实，只报告SG格数和移动格数，不编造米数。','',
        '## 验证、资源和使用','',
        '11项合成权限/状态/预算/清空/目标作用测试通过；另实现用标量余弦、协方差特征分解、另一BFS/几何、多项式计数bootstrap复核100条/914动作及四组配对差，通过，余弦最大差1.67e−16。另实现不是独立人员审查。140历史保护项/模型/缓存/旧平台资产SHA保持；角色账本仍6确认组，未读取其图像或特征。',
        f"缓存导航计算约{done['seconds']:.2f}秒，仅为这次CPU缓存计算，不代表含图像编码的端到端吞吐。0训练、0编码器forward、0下载/API、新C盘输出0；没有使用外网或代理，没有提交/推送Git。",
        '最小调用：导入project.src.agents.ground_local_loop_v1.GroundLocalLoop，begin(grid_size=5,budget=10,goal_feature=单个目标512向量)，随后每步observe({position,current_feature,grid_size,step_index,remaining_steps})得到action及证据；将action交给环境，再用下一真实观测回传。终止时finish，下一题重新begin。不要传整张候选特征表、源ID或答案。','',
        '## 恢复入口与历史保护','',
        '- [冻结协议](冻结协议.json)、[20题清单](任务清单.json)、[首动作绑定](首动作绑定.json)、[执行入口修复绑定](执行入口修复绑定.json)。',
        '- [逐步轨迹](轨迹.jsonl)、[完整分层/恢复损伤/区间](结果汇总.json)、[另实现复核](另实现复核.json)、[目标/资产边界](目标语义与资产核验.json)。',
        '- 初次freeze在目录创建前遇旧条目无bytes，修复后才绑定。随后v1执行在任何动作前因同名tests包遮蔽导入失败；保留4冻结源，另加v2限定导入并绑定。未改策略、队列或门槛，只运行一次100条轨迹。',
        '- 不运行freeze/bind/execute或旧封存脚本。只读复核命令（仓库根）：D:/PYTHON/python.exe -B -m project.src.tools.execute_ground_local_loop_v2 audit（不加--save）。',
        '- [下一步与停止条件](下一步与停止条件.md)。本批6新源码和全部小文件SHA见阶段封存；原模型/缓存不复制。']
    put('README.md','\n'.join(lines))
    put('下一步与停止条件.md', '''# 本因素收束，下一步如何推进

本批只回答：冻结CLIP/Sat2Cap缓存上的局部线性语义趋势，能否使同成本探索更可靠。答案是本队列未达门槛，因此不扫描阈值/历史长度/损失、不扩参、不换错目标挑成绩、不消费6确认。旧检索失败判断继续有效；新闭环不要求推翻旧判断来取得运行资格。

工程成果保留为局部观测—决策—反馈的实验接口与可复核失败案例；当前Web4.8和航拍M0保持。若平台继续迭代，应先在既有地面任务页面呈现本批“闭环已运行/效果未过”及同题控制，明确缓存闭环与未知照片实时编码的区别，不发布为稳定地面导航。

研究侧不直接开始第N个相似度变体。下一完整路线需要目标条件的顺序决策学习或可信跨视角空间监督的新证据；先明确其标签、同预算无目标强对照、训练来源和资源上限，再单独冻结/授权，不能从本轮自动推出训练许可。仅有场景相关性不能保证相似度在地图上单调，局部趋势失败也不能证明所有跨视角策略失败。

本轮没有新模型候选获得确认资格。若新监督/任务适配依据仍不足，研究支线暂时保留失败边界，以已有航拍能力及平台可用性继续交付，而不是消耗封存组追过线。
''')
    index_path=ROOT/'项目导航/实验索引.json';idx=read(index_path)
    old=list(idx['batches']);assert len(old)==142
    old_sig=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    idx['date']='2026-10-22'
    idx['batches'].append({'dataset':'MM-GAG','kind':'局部缓存闭环工程与目标对照','name':'目标语义与局部闭环工程_v1',
        'category':'CV-024','status':'工程通过，局部语义趋势效果筛选未过并收束；非确认/默认升级',
        'scope':'5已消费开发来源20题100轨迹/914动作；0编码训练API下载6确认',
        'path':REL,'reports':[REL+'/README.md',REL+'/下一步与停止条件.md'],
        'evidence':[REL+'/'+x for x in ('冻结协议.json','任务清单.json','首动作绑定.json','执行入口修复绑定.json','结果汇总.json','另实现复核.json','交付边界核验.json','阶段封存.json')]})
    index_path.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert read(index_path)['batches'][:-1]==old
    for name,title,link in [
        ('项目导航/实验索引.md','第143批CV-024局部闭环工程','../'+REL+'/README.md'),
        ('项目导航/README.md','CV-024目标语义与局部闭环','../'+REL+'/README.md')]:
        p=ROOT/name;text=p.read_text(encoding='utf-8-sig');head,sep,rest=text.partition('\n')
        p.write_text(head+'\n\n'+f'[{title}]({link})：20题100条件轨迹/914动作完成，Full4/20、Frontier6/20；11检查/另算过，工程通过但效果未过收束，0训练/编码/6确认，不改4.8。\n'+rest,encoding='utf-8')
    p=ROOT/'docs/维护/版本生命周期.json';life=read(p);life['updated']='2026-10-22'
    life['current']['crossview_local_loop_engineering']=REL+'/README.md'
    life['current']['crossview_local_loop_status']='engineering_passed_semantic_trend_screen_failed_closed_no_default_upgrade'
    life['current_research_direction']='CV-024局部缓存闭环工程完成、局部语义趋势因素筛选失败收束；旧CV021/CV023判断保持。5已消费源/20题、6确认封存，尚无可靠地面目标导航；不继续同类调参，下一完整监督路线需新证据和另冻资源。平台4.8/航拍默认保持。'
    life['crossview_development_deliveries'].append({'task':'CV-024','path':REL,'status':'engineering passed; local semantic trend closed',
        'replaces':'none: adds engineering closed-loop evidence; does not supersede aerial default or historical failed candidates',
        'dependencies':['existing CLIP/Sat2Cap caches','v1 policy/runner/audit/tests frozen before first attempt','v2 pre-action import repair'],
        'rollback':'no deployment or default change; continue platform4.8',
        'recovery':'python -B -m project.src.tools.execute_ground_local_loop_v2 audit (no --save)',
        'training_or_confirmation_authorized':False})
    p.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    src=['agents/ground_local_loop_v1.py','tools/run_ground_local_loop_v1.py','tools/execute_ground_local_loop_v2.py',
         'eval/audit_ground_local_loop_v1.py','tests/test_ground_local_loop_v1.py','documents/close_ground_local_loop_v1.py']
    put('导航交付核验.json',{'old142_index_values_preserved':True,'old142_canonical_sha256':old_sig,
        'current_index_batches':143,'production_entry_or_default_modified':False,'new_sources':[entry(ROOT/'project/src'/s) for s in src],
        'no_git_commit_or_push':True,'proxy_used':False,'C_drive_task_outputs':[]})
    files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()]
    put('阶段封存.json',{'task':'CV-024','date':'2026-10-22','status':'engineering passed; mechanism screen failed and closed',
        'files':files,'sources':[entry(ROOT/'project/src'/s) for s in src],
        'bytes_excluding_seal':sum(e['bytes'] for e in files),'cap_bytes':8*1024**2,
        'coordination_and_navigation_mutable':True,'not_independent_person_review':True,
        'recovery':'read-only v2 audit; do not refreeze/reexecute/close again'})
    size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file());assert size<=8*1024**2
    print(json.dumps({'sealed_files':len(files),'sources':len(src),'bytes_including_seal':size,'index':143,'old142_preserved':True}))


if __name__=='__main__':main()
