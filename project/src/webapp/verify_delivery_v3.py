"""New delivery checks only: no model/API/nav/generator; exclusive new outputs."""
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
from .live_v2 import DEV, read, sha, write_new
from .binding_v3 import execution_bindings, source_closure, validate_bindings
from project.src.tools.repository_release_v1 import secret_locations

ROOT=Path(__file__).resolve().parents[3]
QA=ROOT/'平台/实时导航平台_v2/核验'
BATCH='DATA/processed_data/MassGIS/评测结果/平台单模型对照_v1'

def main():
    protected=read(QA/'开工历史保护_v1.json')['protected_sha256']
    validate_bindings(ROOT,protected,{})
    inputs,sources=execution_bindings(ROOT)
    old_sources=source_closure(ROOT,['project/src/webapp/server_v2.py'])
    original=read(ROOT/DEV/'核验/预登记.json')
    checked_original=0
    for key,expected in original['source_sha256'].items():
        if key in old_sources:
            assert old_sources[key]==expected,key
            checked_original+=1
    result=read(ROOT/BATCH/'主对照/同题批量结果_v1.json')
    episodes=0
    for arm,rows in result['rows'].items():
        for row in rows:
            folder=(ROOT/row['output_path']).parent
            seal=read(folder/'seal.json')['files_sha256']
            validate_bindings(folder,seal,{})
            binding=read(folder/'首动作前绑定.json')
            validate_bindings(ROOT,binding['input_sha256'],binding['source_sha256'])
            episodes+=1
    figures=read(ROOT/'绘图/平台方法图_v1/来源/图表输入与输出绑定_v1.json')
    validate_bindings(ROOT,figures['inputs_sha256'],figures['figures_sha256'])
    write_new(ROOT/BATCH/'核验/执行依赖补充核验_v1.json',dict(
        post_execution_supplement=True,not_original_preregistration=True,
        original_batch_unchanged=True,old_webapp_sources_sha256=old_sources,
        actual_input_sha256=inputs,original_science_source_matches=checked_original,
        existing_episode_seals_checked=episodes,
        note='原v2首动作绑定不完整，补充当前实际执行依赖与DATA007原科学源码一致性；不冒充事前登记，原结果不回写。'))
    stream=io.StringIO();loader=unittest.TestLoader()
    names=['test_live_platform_v2','test_cloud_platform_v2','test_replay_platform_v1','test_platform_review_fixes_v3']
    suite=loader.loadTestsFromNames(['project.src.tests.'+n for n in names])
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        tests=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    assert tests.wasSuccessful(),stream.getvalue()
    write_new(QA/'单测_v3.json',dict(tests=tests.testsRun,passed=True,output=stream.getvalue(),
        model_forward=False,network=False,synthetic_only=True))
    new_sources=list((ROOT/'project/src/webapp').glob('*_v[23].py'))+list((ROOT/'project/src/webapp/static_v2').glob('*'))
    new_sources += [ROOT/'project/src/documents/build_platform_midterm_v1.py',ROOT/'project/src/documents/polish_platform_midterm_v2.py']
    new_sources += [ROOT/'project/src/tests'/f'{n}.py' for n in names if n!='test_replay_platform_v1']
    for p in new_sources:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
    public_text=new_sources+[ROOT/p for p in ['README.md','docs/PROJECT_STATE.md','.agents/TASKS.md','.agents/HANDOFF.md','项目导航/实验索引.md','项目导航/README.md','中期报告相关/README.md','绘图/README.md']]
    for folder in [ROOT/BATCH,ROOT/'平台/实时导航平台_v2',ROOT/'总结报告相关/平台交付_v1']:
        public_text += [p for p in folder.rglob('*') if p.suffix in ('.md','.json','.jsonl','.py','.js','.css','.html')]
    issues=[]
    for p in set(public_text):
        for issue in secret_locations(p.read_text(encoding='utf-8-sig')):
            issues.append(dict(path=p.relative_to(ROOT).as_posix(),**issue))
    assert not issues,issues
    # Only live navigation entries/new material links; historic HANDOFF is not rewritten.
    links=0
    for p in [ROOT/'README.md',ROOT/'docs/PROJECT_STATE.md',ROOT/'项目导航/README.md',ROOT/'中期报告相关/README.md',ROOT/'绘图/README.md']+list((ROOT/'总结报告相关/平台交付_v1').glob('*.md')):
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
            url=raw.strip('<>')
            if '://' in url or url.startswith('#'):continue
            target=p.parent/unquote(url.split('#')[0])
            assert target.exists(),(str(p),url)
            links+=1
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in new_sources}
    for folder in [ROOT/BATCH,ROOT/'绘图/平台方法图_v1',ROOT/'总结报告相关/平台交付_v1',ROOT/'平台/实时导航平台_v2']:
        for p in folder.rglob('*'):
            if p.is_file():artifacts[p.relative_to(ROOT).as_posix()]=sha(p)
    write_new(QA/'最终交付复核_v3.json',dict(ok=True,protected_count=len(protected),
        unchanged_historical_inputs=True,figure_outputs=len(figures['figures_sha256']),
        test_count=tests.testsRun,public_text_scan=len(set(public_text)),secret_findings=0,
        checked_links=links,actual_execution_inputs=len(inputs),execution_sources=len(sources),
        old_comparison_records=episodes,artifact_sha256=artifacts,
        current_execution_input_sha256=inputs,current_execution_source_sha256=sources,
        mutable_entries_not_frozen=['README.md','docs/PROJECT_STATE.md','.agents/TASKS.md','.agents/HANDOFF.md','项目导航/实验索引.md','项目导航/实验索引.json'],
        independent_person=False,science_metrics_audit='另算批量指标_v1.json',
        no_new_training=True,no_spare_regions=True,no_default_change=True))
    print(json.dumps(dict(ok=True,tests=tests.testsRun,protected=len(protected),links=links,inputs=len(inputs),sources=len(sources)),ensure_ascii=False))

if __name__=='__main__':main()
