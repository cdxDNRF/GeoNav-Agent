"""Document protocol adoption, compatibility evidence and dataset limitations."""
from pathlib import Path
import json
import re
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
from eval import area_compatibility as p
from agents.area_policy import resolve_area_config
from env.area_protocol import GRID10, GRID15


def replace_write(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def main():
    reg = p.check()
    audit = p.read(p.OUT / '独立工程复核.json')
    tests = p.read(p.OUT / '测试记录.json')
    catalog = p.read(p.OUT / '十五乘十五数据可行性_仅目录.json')
    packing = p.read(p.OUT / '目录候选空间打包复核.json')
    revision = p.read(p.OUT / '目录格式适配登记_v2.json')
    assert audit['passed'] and audit['old_episodes_exact'] == 900
    assert tests['successful'] and tests['tests_run'] == 18
    assert packing['input_sha256'] == p.digest(p.OUT / '十五乘十五数据可行性_仅目录.json')
    for mapping in (packing['source_sha256'], revision['source_sha256'], revision['protected_input_sha256']):
        for name, expected in mapping.items():
            assert p.digest(ROOT / name) == expected, name
    assert p.digest(p.OUT / '目录格式适配登记_v2.json') == p.read(p.OUT / '目录格式适配封存_v2.json')['sha256']
    cfg, _ = resolve_area_config(ROOT, GRID10)
    experimental, _ = resolve_area_config(ROOT, GRID15, allow_experimental=True)
    assert cfg['status'] == 'confirmed_default' and experimental['status'] == 'engineering_candidate_only'
    assert p.digest(ROOT / 'project/local_policy_default.json') == reg['old_default_sha256']
    text = ('已将Coverage3Radial登记为300米/格、10×10/B20物理区域协议默认；旧5×5配置与跨数据集入口保留。'
        f"18项接口测试、900旧轨迹/{audit['old_actions_exact']}动作精确回归、6合成轨迹/{audit['synthetic_actions']}动作及"
        f"{audit['planner_states_checked']}公开规划状态的独立复核通过。15×15/B20参数化兼容通过，未训练、未下载、没有新真实SR。")
    report = p.OUT / '默认版本化与十五乘十五兼容报告.md'
    draft = p.OUT / '下一项真实十五乘十五数据准备草案.md'
    rows = ['# 默认版本化与十五乘十五兼容准备', '', text, '',
        '本轮是工程阶段，不是15×15性能验收。10×10独立确认的三权重平均SR仍为68.04%，'
        '参考seed0的SR为64.80%；新版默认固定使用事先声明的seed0，不从测试结果选最高分seed1。', '',
        '| 作用域 | 当前策略 | 验收状态 |', '|---|---|---|',
        '| MassachusettsRoads / 300米 / 10×10 / B20 | Coverage3Radial新版默认 | 已独立确认；本轮入口回归通过 |',
        '| 原5×5及原跨数据集脚本 | 原M0配置与入口 | 不回写历史验收 |',
        '| MassachusettsRoads / 300米 / 15×15 / B20 | 显式工程候选 | 仅合成接口通过；无真实SR |', '',
        '新配置引用原三探索器、三视觉头、均值、0.50与原覆盖/径向参数；未复制权重，也未增加模型参数。'
        '新默认加载入口会检查配置、模型和独立确认依据哈希；未确认的15协议默认拒绝自动加载，需要显式工程开关。', '',
        '参数化模块在新文件中实现。探索器仍接收1052维输入，公开访问历史仍汇总到5×5；15×15扩大到4500×4500像素、20.25km²，'
        '图块仍为300×300像素/300米。合成场景的面积只是协议值，并不证明已经获得真实地理影像。', '',
        '回归从10已消费区域每层固定取前2题，共60路线×3权重×5条件=900记录；决策、logits、概率、特征哈希、动作、观察及终局逐项精确相等。'
        '它们属于旧数据重放，不能当作新的性能实验或独立确认。', '',
        '合成图块为225张确定性色彩图，全局/局部数组采用固定随机数，6条轨迹只验证接口与转移。独立检查使用手写GRU/头部方程、集合规划、输入哈希和终局米数，'
        '没有把合成结果折算为遥感SR。18项接口测试、4项清单格式检查、4项打包图检查均通过。', '',
        f"目录可行性：官方缓存{catalog['official_catalog_sources']}项，完整3×3文件名组{catalog['complete_filename_3x3_groups']}组，"
        f"排除151原足迹及24已消费连续区域后，扣除已知缓存质量拒绝，剩{catalog['provisional_candidates']}候选。"
        f"贪心选择{packing['greedy_count']}区；精确打包选择{packing['selected_count']}区，最大数量证明={packing['exact_upper_bound_proved']}。", '',
        '这个上限只针对当前缓存train目录、文件名推断坐标与已知质量筛选，不能推断整个作者数据集最多只有这么多可用区。'
        '尚未验证候选的真实GeoTIFF标签和完整质量，因此可用数量还可能减少。', '',
        '当前目录不足以直接满足“2个全新区工程+10个全新区确认”设计。下一项优先补全官方其它分区目录，检查能否补齐同模态连续影像；'
        '工程接口可使用已消费区域扩展，正式确认池单独保留。在实际坐标/质量检查后、第一次真实导航前，再冻结队列与数量。'
        '本轮不降低3公里间隔、不把已消费区域改名当独立区域、不将不够数量的样本宣称为10区确认。', '',
        '格式适配v2只将已封存“已消费14区域”的对象中的regions列表读出，未改变排除集合或顺序，冻结v1文件原样保留。', '',
        '[工程复核](独立工程复核.json) / [旧轨迹回归](回归/完整核验.json) / [目录可行性](十五乘十五数据可行性_仅目录.json) / '
        '[空间打包复核](目录候选空间打包复核.json) / [下一项草案](下一项真实十五乘十五数据准备草案.md)', '']
    p.write(report, '\n'.join(rows))
    p.write(p.OUT / 'README.md', '# 协议默认与15×15工程入口\n\n'+text+'\n\n[报告](默认版本化与十五乘十五兼容报告.md) / [下一项](下一项真实十五乘十五数据准备草案.md)。\n')
    p.write(draft, '# 下一项：真实15×15数据准备草案（待另行冻结）\n\n'
        '先补足数据池，再采集影像；本轮目录最多9个相互隔离候选且未完成真实质量检查，暂不启动正式导航。\n\n'
        '1. 补全作者官方valid/test等目录，合并同模态源索引时按实际坐标去重，保留上游分区身份。官方train不等于本项目训练集。\n'
        '2. 只选择9源组成的原生3×3、1米发布像素、4500×4500图；核验EPSG26986、方向、PixelIsArea、无缝坐标、图块像素和空白比例。'
        '保留300米/格，不用放大或更密切图替代实际面积扩展。\n'
        '3. 正式区域与原151足迹、已消费24连续区、工程区域及其它确认区保持至少3km边界间隔；工程可复用消费区，但不能混入正式确认。'
        '预训练编码器地理范围仍未知。\n'
        '4. 首轮B20、原C12—16/C8—12三个队列、原模型/0.50及覆盖/径向规则冻结，只考察区域规模；访问分区按公开位置投影5×5。'
        '目标坐标、初始距离、源身份和分层均为评测器信息，控制器只使用公开输入。\n'
        '5. 正式确认目标为10新区，每区75不同路线、三权重；在数据足够前不消费用于确认的模型输出。若无法满足，先公开提出新的数据/队列协议，'
        '不得在看到导航成绩后改数量或降低门槛。\n'
        '6. 完成数据工程后先做接口，再按新协议冻结M0/候选同题对照及条件性目标控制。建议沿用SR+2点、2/3正、混合/分层SG保护、区域CI正及目标+5点证据，'
        '不能要求更大区域与旧68.04%作跨协议直接比较。\n\n'
        '资源上限、下载并发/重试/字节记账和正式任务/错误目标种子须在下一项数据获取前单独登记。新训练、更远距离、更长预算和多Agent不纳入该单因素试验。\n')
    p.write(p.DEFAULTS / 'README.md', '# 按协议加载已确认默认\n\n'
        '物理10×10/B20/300米采用Coverage3Radial，固定参考seed0；原5×5及跨数据集脚本仍使用原入口。'
        '15×15仅为工程候选，不继承10×10验收。\n\n'
        '```python\nfrom agents.area_policy import load_area_policy\nfrom env.area_protocol import GRID10, GRID15\n'
        'agent = load_area_policy(ROOT, GRID10)  # 本协议默认，seed0\n'
        'reference = load_area_policy(ROOT, GRID10, reference_M0=True)\n'
        'engineering = load_area_policy(ROOT, GRID15, allow_experimental=True)\n```\n\n'
        '策略只接收Observation和当前/给定目标特征；环境与评测器负责图块查找，不将完整特征库、真值位置或源身份交给Agent。'
        '配置、模型、均值和独立依据均由加载器校验哈希。新15×15确认前不将工程候选写入defaults。\n')
    verdict = dict(completed=True, protocol_default_adopted=GRID10, reference_seed=0,
        legacy_M0_configuration_preserved=True, grid15_compatibility_passed=True, grid15_real_data_passed=False,
        grid15_real_SR_produced=False, tests_passed=26, interface_tests=18, metadata_graph_checks=8,
        regression_old_episodes=900, regression_actions=audit['old_actions_exact'], synthetic_episodes=6,
        synthetic_actions=audit['synthetic_actions'], independent_planner_states=audit['planner_states_checked'],
        current_catalog_provisional_maximum=packing['selected_count'], maximum_proved=packing['exact_upper_bound_proved'],
        default_registry_sha256=p.digest(p.REGISTRY), confirmed_config_sha256=p.digest(p.CONFIRMED_CONFIG),
        new_training_steps=0, cloud_calls=0, downloaded_sources=0, original_default_sha256=reg['old_default_sha256'],
        audit_sha256=p.digest(p.OUT / '独立工程复核.json'))
    p.write(p.OUT / '验收结论.json', verdict)
    p.write(p.OUT / '最终状态.json', dict(completed=True, verdict_sha256=p.digest(p.OUT / '验收结论.json'),
        next='supplement official catalogs and verify real3x3 imagery before freezing confirmation cohort'))
    idx_path = ROOT / '项目导航/实验索引.json'
    idx = p.read(idx_path)
    assert len(idx['batches']) == 70 and not any(b['path'] == p.rel(p.OUT) for b in idx['batches'])
    idx['date'] = '2026-10-03'
    idx['batches'].append(dict(dataset='MasaRoads', kind='工程准备', name=p.OUT.name,
        category='协议默认版本化与15×15兼容', status='默认登记/兼容通过；未进行真实15×15评测',
        scope='900old regressions;6synthetic cases;no new realSR;physical15x15 data pending', path=p.rel(p.OUT),
        reports=[p.rel(report), p.rel(draft)], evidence=[p.rel(p.OUT / n) for n in ('验收结论.json', '独立工程复核.json', '回归/完整核验.json')]))
    replace_write(idx_path, idx)
    root = ROOT / 'README.md'
    content = root.read_text('utf-8')
    start, end = content.index('更新至'), content.index('下一阶段先登记')
    content = content[:start]+'更新至2026年10月03日。'+text+'\n\n'+content[end:]
    latest = f'## 最新阶段：协议默认登记与15×15兼容完成\n\n{text}\n\n[报告]({p.rel(report)}) / [默认注册表]({p.rel(p.REGISTRY)}) / '
    latest += f'[加载说明]({p.rel(p.DEFAULTS / "README.md")}) / [下一项真实数据草案]({p.rel(draft)})。\n\n'
    content = content.replace('## 最新阶段：径向保护独立新区域导航确认完成', latest+'## 前一阶段：径向保护独立新区域导航确认完成', 1)
    content = content.replace('70批', '71批')
    replace_write(root, content)
    progress = ROOT / '项目导航/当前进度.md'
    content = progress.read_text('utf-8')
    start, end = content.index('截至'), content.index('## 最新：')
    content = content[:start]+'截至2026-10-03，'+text+'\n\n'+content[end:]
    content = content.replace('## 最新：径向保护独立新区域确认完成',
        f'## 最新：协议默认与15×15兼容完成\n\n{text}\n\n[报告](../{p.rel(report)}) / [加载说明](../{p.rel(p.DEFAULTS / "README.md")}) / '
        f'[下一项](../{p.rel(draft)})。\n\n## 前一阶段：径向保护独立新区域确认完成', 1)
    start = content.index('## 下一项方向')
    content = content[:start]+'## 下一项方向\n\n先补足真实3×3连续区域数据池，补全官方其它分区目录并验证实际GeoTIFF坐标/质量；当前train缓存最多9个空间隔离候选且尚未通过质量。'
    content += '正式确认10新区须在首导航前备齐并冻结，已消费区只作工程/回归。15×15工程通过不能当真实SR或默认验收。\n'
    replace_write(progress, content)
    for name in ('项目导航/README.md', '项目导航/目录结构与维护.md', '项目导航/实验索引.md'):
        path = ROOT / name
        content = path.read_text('utf-8').replace('70批', '71批').replace('70个', '71个')
        if name.endswith('实验索引.md'):
            content += f'\n\n## {p.OUT.name}\n\n{text}\n\n[报告](../{p.rel(report)}) / [复核](../{p.rel(p.OUT / "独立工程复核.json")})。\n'
        if name.endswith('目录结构与维护.md'):
            content = content.replace('3. 默认仍为', '3. 物理10×10/B20/300米新入口为`project/local_policy_defaults_v1.json`，选择Coverage3Radial；15×15只有显式工程候选。原5×5默认仍为', 1)
        replace_write(path, content)
    p.check()
    paths = [q for q in p.OUT.rglob('*') if q.is_file()]
    paths += [p.REGISTRY, p.CONFIRMED_CONFIG, p.EXPERIMENT_CONFIG, p.DEFAULTS / 'README.md', Path(__file__)]
    paths += [ROOT / n for n in ('README.md', '项目导航/当前进度.md', '项目导航/实验索引.json', '项目导航/实验索引.md',
                                '项目导航/README.md', '项目导航/目录结构与维护.md')]
    checked_links = 0
    for path in (report, draft, p.OUT / 'README.md', ROOT / 'README.md', progress):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text('utf-8')):
            target = target.strip('<>').split('#')[0]
            if target and not re.match(r'[a-z]+://', target):
                assert (path.parent / target).resolve().exists(), (path, target)
                checked_links += 1
    p.write(p.OUT / '阶段交付核验.json', dict(completed=True, protocol_default_adopted=GRID10,
        grid15_engineering_only=True, new_real_SR=False, index_entries=71, tests_passed=26,
        protected_old_files=len(reg['protected_sha256']), local_links_verified=checked_links,
        original_default_unchanged=True, downloaded_sources=0, new_training_steps=0, cloud_calls=0,
        files_sha256={p.rel(q):p.digest(q) for q in paths}))
    print(dict(delivery_completed=True, batches=71, default_registered=GRID10, grid15_real_SR=False), flush=True)


if __name__ == '__main__':
    main()
