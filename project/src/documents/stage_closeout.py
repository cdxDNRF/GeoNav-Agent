"""Build a read-only stage index and small-file snapshots without moving experiments."""
from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
MASA = ROOT / 'DATA/processed_data/Masa'
OUT = ROOT / '选题报告相关/阶段收尾_v1'
SNAP = OUT / '报告与结果'
FIG = ROOT / '绘图'
MAX_COPY = 512 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def link(path, label, base=OUT):
    target = os.path.relpath(path, base).replace('\\', '/')
    return f'[{label}](<{target}>)'


# Deliberate classification; elapsed timestamps do not imply independent evidence.
DESCRIPTIONS = {
    '基础校验_v2': ('工程基础', '环境与评测基础记录', '基础校验；不作为智能策略成绩'),
    'Ollama试跑_v1': ('云端接口', '首次连接失败', '未发送图像推理，保留故障证据'),
    'Ollama试跑_v1重试': ('云端接口', '有界重试与最小 VLM 接入', '接口试跑，不是完整 S2'),
    '三项验证_v1': ('云端策略', 'TLS 故障记录', 'VLM 未成功决策，不能将完成文件视作完成评测'),
    '三项验证_v1连接修复': ('云端策略', '连接修复后的同题验证', '改用系统传输；保留失败批次'),
    '四组实验_v1': ('云端策略', '排序、动作治理与规则对照', '最终修正报告为 *_v2；固定 val20 试验'),
    '空间记忆实验_v1': ('云端策略', '单因素：空间记忆', '固定小验证集；结果不外推到 val100'),
    '邻域确认实验_v1': ('云端策略', '单因素：邻域确认', '保留负结果；不将所有因素叠加解释'),
    '层级搜索实验_v1': ('云端策略', '单因素：层级搜索', '云端后续对照的 G 策略来源'),
    'S2稳定验证_v1': ('云端验收', '批次已关闭，完整评测未完成', 'G/E 首轮各正常完成 20/100；后续有 not_run，S2 未通过'),
    'DeepSeek接口检查_v1': ('云端接口', '接口检查与截断记录', '合成双图工程探测，不计导航 SR'),
    'DeepSeek接口检查_v2': ('云端接口', '提高输出额度后的接口检查', '合成双图工程探测，不计导航 SR'),
    'Gemma与DeepSeek同策略对照_v1': ('云端能力', '默认思考截断，按协议停止', 'DeepSeek 3 次空正文；旧结果保留，修复见 v2'),
    'Gemma与DeepSeek同策略对照_v2': ('云端能力', '已完成并复核的服务配置对照', 'val20；Gemma 45%，DeepSeek 非思考 20%，Frontier 50%；DeepSeek 完成 18/20'),
    '好奇心配对验证_v3': ('本地策略', 'PBRS 与好奇心配对验证', 'val100×3；好奇心未形成稳定提升'),
    '合法动作协作验证_v1': ('本地策略', '动作筛选候选通过；视觉 S2 未通过', 'val100×3；PBRS 44.33% → 合法动作筛选 66.00%'),
    '目标图利用验证_v1': ('本地诊断', '目标监督与目标干预；候选未通过', 'val100×3；Boundary 66%，TargetPairs 58%，未改善目标利用'),
    '训练源图留出诊断_v1': ('本地诊断', '原 train 源图留出诊断', '109/28 已知开发隔离；方向预测不是导航 SR'),
    '本地模型容量与局部匹配对照_v1': ('本地能力', '18 模型受控对照；新候选未通过', '方向监督直接迁移，140 题；不可与 PBRS 完整导航混为一类'),
    'PBRS完整导航与局部匹配对照_v1': ('本地策略', '完整导航训练，局部匹配观察候选', '开发140×3；Small Full 65.95%，Spatial Full 68.81%，强探索 72.62%'),
    '本地S2固定配置复验_v1': ('本地复验', '工程检查通过；目标贡献未通过', 'val100×3；当前默认 Small NoTarget 73.00%，SG 0.800；正式视觉 S2 未通过'),
    '可信邻接目标线索与探索协作_v1': ('目标线索', '负结果：可信阈值全部弃权', '开发140×3；SR 72.62%，未增加成功；五分类邻接头'),
    '邻接与方向解耦对照_v1': ('目标线索', '负结果：可信阈值全部弃权', '开发140×3；SR 72.62%，条件方向信号不足以介入'),
    '边缘连续性可信线索对照_v1': ('目标线索', '开发候选与完整复核通过；正式 S2 待复验', '开发140×3；SR 88.57%，SG 0.348，3/3种子提升；当前默认未替换'),
}


def protect_existing():
    """Metadata covers large inputs; hashes cover source and small experimental artifacts."""
    records = []
    roots = [ROOT/'DATA', ROOT/'models', ROOT/'project/src', ROOT/'选题报告相关',
             ROOT/'中期报告相关', ROOT/'绘图', ROOT/'调研']
    for base in roots:
        if not base.exists():
            continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or OUT in p.parents:
                continue
            # Current delivery scripts are allowed to evolve while packaging.
            if p.name in ['stage_closeout.py', 'stage_figures.py', 'build_stage_summary.cjs', 'verify_stage_closeout.py']:
                continue
            st = p.stat()
            rec = dict(path=relative(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
            if (p.suffix in ['.json', '.md', '.py', '.docx', '.pdf', '.js', '.html']
                    and st.st_size <= 2 * 1024 * 1024):
                rec['sha256'] = digest(p)
            records.append(rec)
    p = ROOT/'project/local_policy_default.json'
    st = p.stat()
    records.append(dict(path=relative(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest(p)))
    return records


def has_sensitive_content(payload):
    return bool(re.search(r'\bsk-[A-Za-z0-9_-]{16,}|data:image/[^;]+;base64,|Bearer\s+[A-Za-z0-9._-]{15,}', payload))


def selected_small(path):
    if path.suffix == '.md':
        return True
    if path.suffix != '.json':
        return False
    return any(s in path.stem for s in ['汇总','结论','复核','预登记','冻结','规则结果',
                                       '执行状态','资源汇总','测试记录','来源','诊断','候选配置',
                                       '默认方案配置','供应商选项','原val清单哈希','训练结果','特征核验'])


def snapshots():
    copied, skipped, batches = [], [], []
    for kind in ['评测结果', '训练结果']:
        for batch in sorted((MASA/kind).iterdir()):
            if not batch.is_dir():
                continue
            category, status, note = DESCRIPTIONS.get(batch.name, (
                '历史调试/早期试跑', '保留历史记录，不作为当前正式比较',
                '未纳入受控配对的早期训练/故障排查；以原件为准'))
            source_files = []
            for p in sorted(batch.iterdir()):
                if not p.is_file() or not selected_small(p):
                    continue
                if p.stat().st_size > MAX_COPY:
                    skipped.append(dict(path=relative(p), reason='大于512KiB，原地保留并索引'))
                    source_files.append(relative(p))
                    continue
                text = p.read_text('utf-8')
                if has_sensitive_content(text):
                    raise ValueError(f'Sensitive pattern in selected source: {relative(p)}')
                dest = SNAP/kind/batch.name/p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
                assert digest(p) == digest(dest)
                copied.append(dict(original=relative(p), snapshot=relative(dest), bytes=p.stat().st_size, sha256=digest(p)))
                source_files.append(relative(p))
            # A copy may contain relative paths into an original model directory.
            dest = SNAP/kind/batch.name
            dest.mkdir(parents=True, exist_ok=True)
            lines = [f'# {batch.name}：阅读快照', '', status+'。'+note+'。', '',
                     '原目录是权威来源；相对模型/轨迹路径应从原目录解析，不能在此运行快照配置。', '',
                     link(batch, '打开原实验目录', dest), '', '| 小文件 | 原件 |', '|---|---|']
            for rel in source_files:
                p = ROOT/rel
                local = dest/p.name
                label = link(local, p.name, dest) if local.exists() else p.name+'（仅索引）'
                lines.append(f'| {label} | {link(p,"原件",dest)} |')
            (dest/'README.md').write_text('\n'.join(lines)+'\n', 'utf-8')
            batches.append(dict(name=batch.name, kind=kind, category=category, status=status,
                                note=note, original=relative(batch), snapshot=relative(dest), files=source_files))
    for source_dir, dest_name in [(ROOT/'选题报告相关','方案与标准'), (ROOT/'调研','调研记录')]:
        for p in sorted(source_dir.glob('*.md')):
            if p.stat().st_size > MAX_COPY:
                continue
            if has_sensitive_content(p.read_text('utf-8')):
                raise ValueError('Sensitive text in document')
            dest = SNAP/dest_name/p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            copied.append(dict(original=relative(p), snapshot=relative(dest), bytes=p.stat().st_size, sha256=digest(p)))
    write_json(OUT/'整理清单.json', dict(version='stage-closeout-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        operation='copy small selected files only; no moves/deletes', max_copy_bytes=MAX_COPY,
        copied=copied, skipped=skipped, experiments=batches))
    return batches, copied


def figure_data():
    references = {}
    def read(rel):
        p = MASA/rel
        references[relative(p)] = digest(p)
        return load(p)
    edge_rel='训练结果/边缘连续性可信线索对照_v1'
    edge=read(edge_rel+'/对照汇总.json')
    receipt=read(edge_rel+'/验收结论.json')
    audit_path=MASA/edge_rel/'独立复核.json'
    references[relative(audit_path)]=digest(audit_path)
    assert receipt['candidate_passed'] and load(audit_path)['status']=='passed'
    assert receipt['audit_sha256']==digest(audit_path)
    values=[]
    for cond, label in [('Baseline','冻结探索/置零对照'),('CueFull','边缘线索·真实目标'),
                        ('CueMean','边缘线索·均值目标'),('CueWrong','边缘线索·错误目标')]:
        by_seed=[read(edge_rel+f'/Edge_s{s}/导航_{cond}_结果.json') for s in range(3)]
        avg=edge['averages']['Edge'][cond]
        assert abs(sum(x['metrics']['sr'] for x in by_seed)/3-avg['sr'])<1e-12
        row=dict(condition=cond,label=label, sr=avg['sr'], sg=avg['sg'],
                 short_sr=avg['short_sr'],sr_by_seed=avg['sr_by_seed'],
                 sg_by_seed=[x['metrics']['mean_sg_all_episodes'] for x in by_seed],
                 successes=sum(x['metrics']['successes'] for x in by_seed), planned=420,
                 by_distance={c:dict(sr_by_seed=[x['by_distance'][c]['sr'] for x in by_seed],
                     sr=sum(x['by_distance'][c]['sr'] for x in by_seed)/3,
                     sg=sum(x['by_distance'][c]['mean_sg_all_episodes'] for x in by_seed)/3,
                     episodes_per_seed=28) for c in ['4','5','6','7','8']},
                 source_file=relative(MASA/edge_rel/'对照汇总.json'),source_field=f'averages.Edge.{cond}',
                 per_seed_source_pattern=relative(MASA/edge_rel)+f'/Edge_s{{seed}}/导航_{cond}_结果.json')
        values.append(row)
    val=read('评测结果/本地S2固定配置复验_v1/对照汇总.json')
    val_rows=[]
    for key,label,cond in [('HistoricalBoundary','历史动作筛选','Full'),('Small256_Full','Small·真实目标','Full'),
        ('Spatial256_Full','Spatial·真实目标','Full'),('Small256_NoTarget','Small·无目标（默认）','NoTarget'),
        ('Spatial256_NoTarget','Spatial·无目标','NoTarget')]:
        x=val['averages'][key][cond]
        val_rows.append(dict(label=label, sr=x['sr_mean'],sg=x['sg_mean'],sr_by_seed=x['sr_by_seed'],
            source_file=relative(MASA/'评测结果/本地S2固定配置复验_v1/对照汇总.json'),source_field=f'averages.{key}.{cond}'))
    cloud=read('评测结果/Gemma与DeepSeek同策略对照_v2/对照汇总.json')
    cloud_rows=[]
    for key,label in [('Gemma','Gemma'),('DeepSeek','DeepSeek\n非思考'),('Frontier','Frontier')]:
        x=cloud['jobs'][key]['metrics']
        cloud_rows.append(dict(label=label,model=key,sr=x['sr_gate'],sg=x['sg_gate'],
            successes=x['successes'],completed=x['completed'],planned=20,
            source_file=relative(MASA/'评测结果/Gemma与DeepSeek同策略对照_v2/对照汇总.json'),source_field=f'jobs.{key}.metrics'))
    history=[]
    # Track all primary development milestones, including negative results, with separate protocols.
    for rel,arms in [('训练结果/好奇心配对验证_v3/配对汇总.json',['PBRS','Curiosity','GatedCuriosity']),
                     ('训练结果/合法动作协作验证_v1/配对汇总.json',['Boundary'])]:
        d=read(rel)
        for arm in arms:
            v=d['arms'][arm]
            history.append(dict(label=arm,protocol='val100×3（历史批次）',sr=v['sr_mean'],sg=v['sg_mean'],
                sr_by_seed=v['sr_by_seed'],source_file=relative(MASA/rel),source_field=f'arms.{arm}'))
    pbrs=read('训练结果/PBRS完整导航与局部匹配对照_v1/对照汇总.json')
    for arm,cond,label in [('Small256','Full','PBRS完整导航'),('Spatial256','Full','PBRS＋局部匹配'),
                           ('Small256','NoTarget','冻结强探索')]:
        v=pbrs['averages'][arm][cond]
        history.append(dict(label=label,protocol='已知开发140×3',sr=v['sr'],sg=v['mean_sg_all_episodes'],
            sr_by_seed=v['sr_by_seed'],source_file=relative(MASA/'训练结果/PBRS完整导航与局部匹配对照_v1/对照汇总.json'),
            source_field=f'averages.{arm}.{cond}'))
    for rel,label in [('可信邻接目标线索与探索协作_v1','五分类可信线索'),('邻接与方向解耦对照_v1','邻接/方向解耦')]:
        d=read('训练结果/'+rel+'/对照汇总.json');v=d['averages']['CueFull']
        history.append(dict(label=label,protocol='已知开发140×3',sr=v['sr'],sg=v['sg'],sr_by_seed=v['sr_by_seed'],
            source_file=relative(MASA/'训练结果'/rel/'对照汇总.json'),source_field='averages.CueFull'))
    history.append(dict(label='边缘连续性候选',protocol='已知开发140×3',sr=values[1]['sr'],sg=values[1]['sg'],
        sr_by_seed=values[1]['sr_by_seed'],source_file=values[1]['source_file'],source_field=values[1]['source_field']))
    data=dict(version='stage-figure-data-v1',scope='snapshot 2026-09-30; no new experiments',
        metric_definitions=dict(SR='预算内首次到达的成功比例；百分数显示时乘100',
            SG='所有正常完成轨迹的终点曼哈顿距离平均，成功轨迹为0，单位：格',
            cloud_gate='计划20题分母；DeepSeek两题中断的SG按8补入，不能与SG_nav混用',
            seeds='三份独立训练权重；跨种子重复任务不算独立地图',
            ci='28张源图先平均种子，再成组bootstrap，2000次；来自原实验结果，非本次重算'),
        edge=dict(rows=values,effects=edge['effects'],short_effects=edge['short_effects'],
            offline=edge['held_full_probe']['Edge'],thresholds=edge['thresholds'],rules=edge['rules'],
            protocol='已知开发140题 / 28张源图 / 3训练种子 / 5×5 / B=10 / C4–C8',
            status='开发候选与复核通过；正式S2待固定val100×3复验；默认未替换'),
        formal_val=dict(rows=val_rows,rules=val['rules'],protocol='固定val100 / 4张已用开发源图 / 3训练种子 / 5×5 / B=10'),
        cloud=dict(rows=cloud_rows,protocol='固定val20 / 同G策略 / 单轮 / 5×5 / B=10'),
        historical_milestones=history,source_sha256=references)
    write_json(FIG/'绘图数据/阶段结果_v1.json',data)
    write_json(OUT/'关键指标.json',data)
    return data


def write_guides(batches, copied, data):
    latest=MASA/'训练结果/边缘连续性可信线索对照_v1'
    defaults=MASA/'评测结果/本地S2固定配置复验_v1'
    quick=['# 阶段成果总览', '', '快照日期：2026-09-30。本次仅整理、绘图与文档总结，没有新增训练或正式验收。', '',
        '## 五分钟了解项目', '', '任务：输入目标航拍图与当前局部航拍图，智能体在5×5网格中执行上/右/下/左动作，在10步预算内找到目标。环境提供当前位置、已访问记录等公开状态；答案仅由评测端读取。', '',
        '当前实现为单智能体加探索、视觉线索和行动治理模块；本地分支有训练，云端分支接入VLM。零训练多智能体仍是原始研究目标，尚未实现。', '',
        '| 阶段 | 状态 |', '|---|---|', '| 环境基础、统一评测日志、最小云端VLM | 已实现 |',
        '| 逐因素验证和本地目标利用 | 当前阶段，边缘连续性开发候选已通过 |',
        '| 正式S2 | 待冻结边缘候选进行val100×3复验 |',
        '| S3独立源图、S4大网格/跨数据集 | 尚未开展 |', '', '## 先找这几个文件', '',
        '| 文件 | 用途 |', '|---|---|',
        f'| {link(latest/"结果解读与下一项.md","最新结果解读")} | 最新开发SR88.57%、SG0.348及适用边界 |',
        f'| {link(latest/"验收结论.json","最新验收结论")} / {link(latest/"独立复核.json","完整复核")} | 最终状态与证据哈希；优先于审计前pending字段 |',
        f'| {link(latest/"候选配置.json","候选配置")} | 三头、三探索器、均值和阈值来源；尚未自动加载 |',
        f'| {link(ROOT/"project/local_policy_default.json","当前默认配置")} / {link(defaults/"结果解读与默认方案.md","选择依据")} | Small256 NoTarget；val100平均SR73%，SG0.800 |',
        f'| {link(ROOT/"选题报告相关/本地S2探索与证据标准_v2.md","现行S2证据标准")} | 探索放宽与正式验收的区分 |',
        f'| {link(OUT/"实验索引.md","完整实验索引")} / {link(SNAP/"README.md","小文件快照")} | 正、负、故障、未完成记录均可查 |',
        f'| {link(FIG/"README.md","科研绘图")} | 图表预览、格式、图注和源数据 |',
        f'| {link(ROOT/"中期报告相关/阶段提升总结_v1.docx","提升总结Word")} | 简短过程总结；原中期报告未修改 |', '',
        '## 读结果时保留的口径', '',
        '1. 开发140题、固定val100、云端val20分别报告，不以88.57%减73%推断提升；最新同题提升为88.57%−72.62%=15.95个百分点。',
        '2. SR衡量预算内到达；SG是所有正常完成轨迹的终点距离均值，成功记0。低SG与高SR相关，失败轨迹的剩余距离需单独解释。',
        '3. 三种子是三份训练权重，140×3=420条轨迹，不是420张独立源图。28张已知开发图支持开发结论；独立源图确认是S3。',
        '4. receipt/验收结论引用的复核为最终依据。原汇总中的audit_pending和旧报告的未通过判断保留历史原文，不回写。',
        '5. 连续同源、方向一致的300×300裁块是边缘线索的当前适用范围，不声称通用语义定位或跨数据集优势。', '',
        '## 后续接续与复现', '', '本轮无需继续加因素。下一项固定3份头、3份探索器、4份均值及0.50阈值做val100×3，保留同题基线/规则/真实/均值/错误目标、分距离和源图区间，复核后判定S2并决定默认。', '',
        '**不能直接把当前训练集专用运行器用于val**：`train/edge_cue.py` 的错误目标图片分支使用train路径，正式复验必须正确解析split并重提val图的相同profile；不得混用同名源图或重新按val调阈值。', '',
        '本次成果脚本（在项目根目录运行，使用已安装的Python/Node依赖）：', '',
        '```powershell', "& 'D:\\PYTHON\\python.exe' -X utf8 project/src/documents/stage_closeout.py",
        "& 'D:\\PYTHON\\python.exe' -X utf8 project/src/documents/stage_figures.py",
        'node project/src/documents/build_stage_summary.cjs',
        "& 'D:\\PYTHON\\python.exe' -X utf8 project/src/documents/verify_stage_closeout.py", '```', '',
        '上述命令只生成收尾成果，第一条拒绝覆盖既有快照；图表/Word脚本仅重新生成本版交付文件。原模型运行仍使用原实验路径。本地外部数据/权重不在Git中，克隆仓库不能仅靠此快照复现训练。', '',
        f'集中保存 {len(copied)} 份小文件，全部逐字节复制并登记SHA-256。详细见[整理清单](整理清单.json)与[整理核验](整理核验.json)。']
    (OUT/'README.md').write_text('\n'.join(quick)+'\n','utf-8')
    lines=['# 完整实验索引', '', '按工程基础、云端、本地和目标线索查询；原件权威，快照便于阅读。历史失败、负结果和未完成评测不删除。', '',
           '| 类别 | 批次 | 状态/范围 | 原件 / 阅读快照 |', '|---|---|---|---|']
    for b in sorted(batches,key=lambda x:(x['category'],x['name'])):
        lines.append(f'| {b["category"]} | {b["name"]} | {b["status"]}；{b["note"]} | {link(ROOT/b["original"],"原目录")} / {link(ROOT/b["snapshot"]/"README.md","快照")} |')
    lines += ['', '## 图表和数值追溯', '', '每个绘图数值的原文件和字段见[关键指标](关键指标.json)；每份快照的原路径、大小与哈希见[整理清单](整理清单.json)。图号、图注和输出见[绘图入口](../../绘图/README.md)。', '',
              '## 已有课程与调研文档', '', link(ROOT/'选题报告相关/最新版选题报告','最新版选题报告目录'), '',
              link(ROOT/'中期报告相关','中期报告与本次提升总结'), '',
              link(ROOT/'调研','已有调研原目录')+'；'+link(SNAP/'调研记录','调研快照')+'。本轮未重新检索文献，不把已有记录称为本轮独立核验。', '',
              link(SNAP/'方案与标准','方案和验收标准快照')+'。若副本包含相对链接，使用整理清单找到原件，再从原件位置解析。']
    (OUT/'实验索引.md').write_text('\n'.join(lines)+'\n','utf-8')
    snap_lines=['# 报告与结果阅读快照', '', '此目录集中小型Markdown/JSON，保留原文件名与实验批次。模型、特征、大数据和逐步轨迹原地保留。每份副本与原件逐字节一致，模型路径应从原实验目录解析。', '',
        link(OUT/'整理清单.json','原路径与SHA-256清单',SNAP), '',
        '| 分类 | 批次快照 |', '|---|---|']
    for b in batches:
        snap_lines.append(f'| {b["category"]} | {link(ROOT/b["snapshot"]/"README.md",b["name"],SNAP)} |')
    (SNAP/'README.md').write_text('\n'.join(snap_lines)+'\n','utf-8')


def main():
    if OUT.exists():
        raise SystemExit('Existing stage snapshot; create a new version instead of overwriting.')
    original=protect_existing()
    OUT.mkdir(parents=True)
    write_json(OUT/'原始文件保护清单.json',dict(policy='metadata for all existing protected files; SHA256 for small originals', files=original))
    batches,copied=snapshots()
    data=figure_data()
    write_guides(batches,copied,data)
    print(json.dumps(dict(experiments=len(batches),copied_files=len(copied),copied_bytes=sum(x['bytes'] for x in copied),protected_originals=len(original)),ensure_ascii=False))


if __name__=='__main__':
    main()
