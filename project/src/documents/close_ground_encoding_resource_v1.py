"""Seal the two stopped 3GiB runs without rerunning any encoder."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'DATA/processed_data/MM-GAG/地面目标实时编码接口工程_v1'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest)
def put(name, obj):
    with (OUT/name).open('x', encoding='utf-8') as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2))
        f.write('\n')
def main():
    if (OUT/'阶段封存.json').exists(): raise FileExistsError('Already sealed')
    binding, repair = read(OUT/'首forward绑定.json'), read(OUT/'加载修复绑定.json')
    protected = sum((binding[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')), []) + repair['sources']
    for e in protected:
        actual = entry(ROOT/e['path'])
        assert actual['sha256'] == e['sha256'] and actual['bytes'] == e['bytes'], e['path']
    failures = [read(OUT/n) for n in ('工程中断.json','v2_工程中断.json')]
    attempts = [json.loads(s) for s in (OUT/'编码尝试.jsonl').read_text(encoding='utf-8').splitlines()]
    events = [json.loads(s) for s in (OUT/'局部事件.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [f['attempts'] for f in failures] == [0,1]
    assert all(f['completed_pairs'] == 0 and f['peak_RSS_bytes'] > 3*1024**3 for f in failures)
    assert len(attempts) == len(events) == 1 and attempts[0]['kind'] == 'ground'
    assert not (OUT/'在线工程特征.npy').exists() and not (OUT/'执行完成.json').exists()
    put('资源停止结论.json', dict(passed=False, reason='both runs exceeded original 3GiB RSS guard',
        failures=failures, actual_forward_calls=1, successful_ground_encodings=1, complete_pairs=0,
        ground_vector_not_persisted=True, cache_consistency_not_assessed=True,
        assets_sources_protected_verified=len(protected), no_navigation_SR=True,
        separate_user_authorization='4GiB new v2 batch; same8images; at most8 additional forwards; cumulative9'))
    put('README.md', '''# CV-025原3GiB资源中断（封存）

两次执行均未通过3GiB进程RSS护栏：原加载峰3.327GiB、0次forward；meta/单槽加载修复峰3.025GiB、1次地面forward成功、0完整图像对。两次均已停止，没有自动重试。

唯一成功地面向量没有落盘；begin事件证明返回，不能替代与旧缓存的数值一致性证据。没有新SR、训练、导航、API、下载或确认组消费；模型、原缓存、历史记录和绑定源码SHA不变。不能声称3GiB通过。

用户另外批准[4GiB新批次](../地面目标实时编码接口工程_v2/README.md)：同8幅图追加最多8次forward，累计实际最多9次，参数/输入/一致性门槛不变。新批次逐成功向量即时写盘；本目录、原失败回执和原绑定保持。

恢复只读核验阶段封存、首forward绑定、加载修复绑定、两份中断、编码尝试及局部事件。禁止重跑原freeze/bind/execute/close。本批不是导航验收，不升级默认。
''')
    ip=ROOT/'项目导航/实验索引.json'; idx=read(ip); old=idx['batches']; assert len(old)==145
    rel=OUT.relative_to(ROOT).as_posix()
    idx['date']='2026-10-22'
    idx['batches']=old+[dict(dataset='MM-GAG', kind='编码接口资源中断', name=OUT.name, category='CV-025资源中断',
        status='3GiB工程未过，封存；非导航验收', scope='4固定开发对，实际仅1次地面forward；0完整对/训练/导航/API/确认',
        path=rel, reports=[rel+'/README.md'], evidence=[rel+'/'+n for n in ('资源停止结论.json','首forward绑定.json','加载修复绑定.json','工程中断.json','v2_工程中断.json','阶段封存.json')])]
    ip.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert read(ip)['batches'][:-1]==old
    put('索引原值核验.json', dict(old_batches=145,new_batches=146,old_values_preserved=True,
        old_canonical_sha256=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
    files=[entry(p) for p in sorted(OUT.iterdir()) if p.is_file()]
    put('阶段封存.json', dict(task='CV-025', status='resource_failed_3GiB', files=files,
        sources=binding['sources']+repair['sources']+[entry(Path(__file__).resolve())], no_blind_rerun=True))
    print(json.dumps(dict(sealed=True,files=len(files),actual_forward_calls=1,index=146)))

if __name__=='__main__': main()
