"""Read-only method descriptions and saved-run provenance; no model imports."""
import hashlib
import json
import re
from pathlib import Path

CONFIG = 'project/local_policy_default.json'
RADIAL = 'project/defaults/masa_roads_grid10_radial_v1.json'
BENCH = 'DATA/processed_data/Masa/训练结果/GeoExplorer分布一致性成对开发_v1'
EVIDENCE = (
    ('Masa S2', 'DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/验收结论.json', 'formal_S2_passed', '5×5 / B10 · 4验证源'),
    ('Masa S3', 'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/验收结论.json', 'formal_S3_passed', '5×5 / B10 · 10测试源；地理足迹隔离未确认'),
    ('SwissView同模态', 'DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1/验收结论.json', 'formal_cross_dataset_passed', '5×5 / B10 · 20源；航拍目标'),
)
EVIDENCE_FILES = [entry[1] for entry in EVIDENCE] + [BENCH + '/' + name for name in (
    '开发评测/开发汇总.json', '核验/开发另实现复核.json', '核验/训练另实现复核.json',
    '核验/阶段封存_v1.json', '元数据/冻结执行协议.json', '训练完成与开发对照报告_v1.md')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def checked_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or p.name.startswith('.env'):
        raise ValueError('非法模型资产路径')
    p = root / p
    if not p.resolve().is_relative_to(root) or any(q.is_symlink() or (hasattr(q, 'is_junction') and q.is_junction()) for q in [p, *p.parents] if q != root and root in q.parents):
        raise ValueError('资产不允许越界或链接')
    return p


def assets(config):
    return ([dict(item, role='explorer') for item in config['checkpoints']] +
            [dict(item, role='cue') for item in config['cue_heads']] +
            [dict(item, role='mean', label=name) for name, item in config['means'].items()])


def catalogue(root):
    root = Path(root).resolve()
    path = checked_path(root, CONFIG)
    config = read(path)
    if config.get('version') != 'local-policy-default-v2' or config.get('selected_arm') != 'EdgeTargetCue':
        raise ValueError('当前说明尚未适配此模型配置')
    records = assets(config)
    for record in records:
        record['available'] = checked_path(root, record['path']).is_file()
        # No weight tensors are loaded; record the declared SHA, not a claim of model inference.
    h = digest(path)
    evidence = []
    for title, relative, field, scope in EVIDENCE:
        p = checked_path(root, relative)
        entry = dict(title=title, path=relative, scope=scope, available=p.is_file())
        if p.is_file():
            verdict = read(p)
            expected = verdict.get('default_config_sha256', verdict.get('default_sha256'))
            entry.update(passed=verdict[field] is True, matches_current=expected == h, sha256=digest(p))
        evidence.append(entry)
    radial_path = checked_path(root, RADIAL)
    radial = read(radial_path) if radial_path.is_file() else None
    methods = [dict(id='M0', title='M0 · 视觉线索导航', kind='stable', executable=True,
        description='Small256负责探索，Edge目标线索头判断邻接方向；通过门控后采用线索，否则沿探索器建议行动。',
        usage='同模态航拍、已验证方向与协议；自由上传结果仅作工程测试。',
        scope='Masa 5×5/B10；其它网格由各自适配与验收限定。'),
        dict(id='Coverage3Radial', title='覆盖与径向保护', kind='protocol_specific', executable=True,
        description='共享M0权重，加入覆盖收益规划与径向保护；接受的目标线索优先。',
        usage='仅在数据目录开放的10格协议选用；15格研究结果未通过。',
        scope='MassachusettsRoads 9km²确认通过；MassGIS新产品确认未过。'),
        dict(id='Frontier', title='Frontier · 规则探索', kind='rule', executable=True,
        description='无需学习权重，依据已访问区域和边界选择探索动作，用于同题规则对照。',
        usage='是否可运行以所选数据的方案列表为准。', scope='不使用目标视觉匹配。'),
        dict(id='Cloud', title='单模型API', kind='external', executable=True,
        description='使用用户配置的兼容模型接口；请求、耗时及本地回退单独记录。',
        usage='需先配置接口。浏览本页不会发起请求。', scope='外部模型版本和视觉能力须按实际服务验证。')]
    components = [
        dict(title='视觉编码器', name='冻结Sat2Cap', role='将当前图与给定目标图编码为512维全局特征及4组局部特征。', learned='预训练后冻结；可复用缓存', parameters=None),
        dict(title='探索器', name='Small256 / GRU', role='利用当前视觉、公开位置、剩余预算和访问历史提出动作；目标通道使用固定均值。', learned='已训练，推理冻结', parameters=797701),
        dict(title='目标线索头', name='EdgeTargetCueHead', role='比较两图语义、局部关系和20维边缘特征，输出四方向或非邻接。', learned='已训练，推理冻结', parameters=136837),
        dict(title='动作门控', name='合法方向 + 置信阈值', role='可信且合法、未访问的邻接线索可接管；其余情况采用探索器建议。', learned='确定性规则，无可训练参数', parameters=0),
    ]
    research = dict(id='GeoExplorer', title='GeoExplorer · 学习基线', available=False, executable=False,
        status='研究证据未随当前交付携带', summary='独立研究对照，不随训练自动替换M0。')
    summary_path = checked_path(root, BENCH + '/开发评测/开发汇总.json')
    seal_path = checked_path(root, BENCH + '/核验/阶段封存_v1.json')
    if summary_path.is_file() and seal_path.is_file():
        seal = read(seal_path)['files']
        for suffix in ('开发评测/开发汇总.json', '核验/开发另实现复核.json', '核验/训练另实现复核.json', '元数据/冻结执行协议.json'):
            name = BENCH + '/' + suffix
            if digest(checked_path(root, name)) != seal[name]:
                raise ValueError('研究结论与封存绑定不一致')
        summary = read(summary_path)
        if not read(checked_path(root, BENCH + '/核验/开发另实现复核.json'))['passed']:
            raise ValueError('研究开发复核未通过')
        research.update(available=True, status='已完成 · 候选未通过' if not summary['passed'] else '开发通过 · 尚非默认',
            passed=summary['passed'], full=summary['full'], frontier=summary['frontier'],
            target_contrasts=summary['target_contrasts'], gates=summary['gates'],
            source_path=BENCH + '/开发评测/开发汇总.json', source_sha256=digest(summary_path),
            scope='28个已消费开发源，140题×3种子，5×5/B10；不与Masa S3等不同队列直接相减。',
            summary='训练冻结历史骨干上的164997参数PPO头。修正更新分布有收益，但目标有效性与同题Frontier门槛未过；本因素收束。')
    return dict(schema='geonav-method-catalog-v1', config_path=CONFIG, config_sha256=h,
        model_version=config['version'], seeds=[0, 1, 2], ensemble=False, automatic_upgrade=False,
        assets=records, thresholds=config['thresholds'], components=components, methods=methods,
        evidence=evidence, research=research, controller=radial['controller'] if radial else None,
        radial_config_sha256=digest(radial_path) if radial else None,
        verification='路径存在与配置声明；执行器在加载时另核权重SHA。目录浏览不加载模型。',
        boundary='训练过的单导航Agent；地面照片、文本、异时影像与任意方向目标尚未验证。')


def snapshot(state, current):
    """Never substitute today's model identity for an old run with missing bindings."""
    definition = state['definition']
    bindings = definition.get('input_sha256', {})
    config_hash = bindings.get(CONFIG)
    # Masa uses the frozen copy of the same configuration, rather than CONFIG.
    if not config_hash:
        candidates = [h for name, h in bindings.items() if name.endswith('/默认配置.json')]
        config_hash = candidates[0] if len(set(candidates)) == 1 and candidates else None
    evidence = []
    for rec in current['assets']:
        saved = bindings.get(rec['path'])
        if saved is not None:
            evidence.append(dict(path=rec['path'], role=rec['role'], seed=rec.get('seed'), sha256=saved,
                matches_current=saved == rec['sha256']))
    arms = definition.get('arms', [])
    relevant = [r for r in current['assets'] if r['role'] == 'mean' or r.get('seed') in definition.get('seeds', [])]
    complete = bool(config_hash) and bool(relevant) and all(r['path'] in bindings for r in relevant)
    matches = complete and config_hash == current['config_sha256'] and all(r['matches_current'] for r in evidence)
    status = 'recorded_match' if matches else 'recorded_different' if complete else 'incomplete'
    if arms and all(arm == 'Frontier' for arm in arms):status = 'rule_only'
    # Whitelist public settings; never echo arbitrary provider credential fields.
    cloud = definition.get('cloud') or {}
    public_cloud = {key: cloud[key] for key in ('model', 'timeout', 'max_tokens') if key in cloud}
    return dict(schema='geonav-method-snapshot-v1', experiment_id=definition.get('id'),
        name=definition.get('name'), dataset_id=definition.get('dataset_id'), arms=arms,
        seeds=definition.get('seeds', []), grid=definition.get('grid'), budget=definition.get('budget'),
        task_ids=[task.get('key', task.get('episode_id')) for task in definition.get('tasks', [])],
        created_at=definition.get('created_at'), storage_mode=definition.get('storage_mode'),
        historical_readonly=bool(definition.get('historical_readonly')), binding_status=status,
        recorded_config_sha256=config_hash, current_config_sha256=current['config_sha256'],
        model_assets=evidence, source_sha256={key: value for key, value in definition.get('source_sha256', {}).items()
            if isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value)},
        cloud=public_cloud, record_provenance=state.get('provenance'),
        model_calls=0, research_acceptance=False,
        note='实验原定义中的绑定优先；缺项明确标记，不使用当前模型身份补写旧记录。完整任务/图像绑定见原实验JSON。')
