"""PLATFORM-029 fail-closed release guard and bounded, recorded maintenance.

No model loading, inference, training, navigation or remote network requests.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '平台/实验工作台_v4/发布修复与存储收束_v1'
VERSION = 'geonav-workbench-v4.13.1'
TARGETS = (
    'project/.runtime/workbench_clean_v1/venv',
    'project/.runtime/geoexplorer_probe_v1/cache',
    '平台/实验工作台_v4/便携闭环_v1/工作副本',
)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def put(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def probe_port(port, expected=VERSION, timeout=2):
    """Require refusal or exclusive OS bind proof; HTTP failure never frees it."""
    if type(port) is not int or not 1 <= port <= 65535:
        return dict(state='unknown', port=port, reason='invalid_port')
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=timeout):
            pass
    except OSError as exc:
        if exc.errno == errno.ECONNREFUSED or getattr(exc, 'winerror', None) == 10061:
            return dict(state='free', port=port, reason='tcp_connection_refused')
        if isinstance(exc, TimeoutError):
            # Windows may silently filter a closed local port. A timeout alone
            # proves nothing; acquire exclusive ownership before calling it free.
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind(('127.0.0.1', port))
                return dict(state='free', port=port, reason='exclusive_bind_succeeded_after_connect_timeout')
            except OSError as bind_error:
                return dict(state='unknown', port=port, reason=type(bind_error).__name__)
        return dict(state='unknown', port=port, reason=type(exc).__name__)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/api/meta', timeout=timeout) as response:
            meta = json.load(response)
        if isinstance(meta, dict) and meta.get('version') == expected:
            return dict(state='current', port=port, version=expected)
        return dict(state='occupied', port=port, reason='other_service_or_version')
    except (OSError, ValueError) as exc:
        return dict(state='occupied', port=port, reason=type(exc).__name__)


def require_free(port):
    state = probe_port(port)
    if state['state'] != 'free':
        raise RuntimeError('端口非空闲或无法核实；保留现有入口与服务：' + json.dumps(state, ensure_ascii=False))
    return state


def files_under(folder):
    """Do not traverse junctions or symlinks during inventory or deletion."""
    folder = Path(folder)
    if not folder.resolve().is_relative_to(ROOT) or folder.is_symlink():
        raise ValueError('维护路径越界')
    result = []
    for dp, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            p = Path(dp) / name
            if p.is_symlink() or getattr(p.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError('维护树含重解析点，需单独核验：' + str(p))
        result.extend(Path(dp) / n for n in files)
    return sorted(result)


def assert_no_dependents():
    import psutil
    matches = []
    for p in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
        try:
            if p.pid == os.getpid() or not (p.info['name'] or '').lower().startswith(('python', 'uv')):
                continue
            args = p.info['cmdline'] or []
            roots = [str(ROOT / n).replace('\\', '/').lower() for n in TARGETS]
            if any(root in str(a).replace('\\', '/').lower() for a in [p.info['exe'] or ''] + args for root in roots):
                matches.append(p.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError('无法核实Python进程依赖，暂停清理') from exc
    if matches:
        raise RuntimeError('维护路径仍被进程使用：' + str(matches))
    return dict(passed=True, matched_pids=[])


def baseline():
    assert read(OUT / '用户授权.json')['approved']
    paths = set()
    for base in ['project/src', 'models', 'DATA/processed_data/工程冷归档_v1']:
        paths.update(p for p in (ROOT / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    new_names = {'release_maintenance_v1.py', 'test_release_maintenance_v1.py',
                 'browser_release_maintenance_v1.js', 'workbench_server_v21.py',
                 'spatial_evidence_v2.js', 'package_workbench_v25.py'}
    paths = {p for p in paths if p.name not in new_names}
    old = ROOT / '平台/实验工作台_v4/空间小头对照证据查阅_v1'
    paths.update(p for p in old.rglob('*') if p.is_file())
    batch = ROOT / 'DATA/processed_data/OpenCVL/普通RGB空间小头固定开发对照_v1'
    paths.update(p for p in batch.rglob('*') if p.is_file())
    paths.update(p for p in (ROOT / 'project').glob('*.json') if 'default' in p.name)
    paths.update(p for p in (ROOT / 'project/defaults').glob('*.json'))
    store = ROOT / 'DATA/processed_data/平台实验/工作台_v1'
    user = {p.relative_to(ROOT).as_posix(): sha(p) for p in store.rglob('*') if p.is_file() and p.suffix not in ('.log', '.lock')}
    raw = {p.relative_to(ROOT).as_posix(): p.stat().st_size for p in (ROOT / 'DATA/raw_data').rglob('*') if p.is_file()}
    entries = ['README.md', 'docs/PROJECT_STATE.md', 'project/src/README.md',
               'docs/维护/版本管理与恢复.md', 'docs/维护/版本生命周期.json',
               '平台/README.md', '平台/实验工作台_v4/README.md', '平台/启动平台.ps1',
               '项目导航/README.md', '项目导航/目录结构与维护.md']
    snaps = []
    for i, name in enumerate(entries):
        p = ROOT / name
        dest = OUT / '入口快照' / f'{i:02}.before'
        with dest.open('xb') as f:
            f.write(p.read_bytes())
        snaps.append(dict(path=name, snapshot=dest.relative_to(ROOT).as_posix(), sha256=sha(p)))
    put('基线保护.json', dict(protected={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)},
        user=user, raw_sizes=raw, entries=snaps,
        old_index=read(ROOT / '项目导航/实验索引.json')['batches'], model_calls=0))
    print('baseline protected', len(paths), 'files')


def plan_cleanup():
    assert_no_dependents()
    rows = []
    copy_root = ROOT / TARGETS[2]
    unique_bytes = 0
    for name in TARGETS:
        target = ROOT / name
        if not target.is_dir():
            raise ValueError('维护目标缺失，先查已有回执：' + name)
        for p in files_under(target):
            h = sha(p)
            row = dict(path=p.relative_to(ROOT).as_posix(), sha256=h, bytes=p.stat().st_size,
                       role='rebuildable_environment_or_cache')
            if name == TARGETS[2]:
                relative = p.relative_to(copy_root)
                original = ROOT / relative
                if original.is_file() and sha(original) == h:
                    row.update(role='byte_identical_copy', restore_source=original.relative_to(ROOT).as_posix())
                else:
                    dest = OUT / '保留记录' / relative
                    unique_bytes += row['bytes']
                    if unique_bytes > 8 * 1024**2:
                        raise ValueError('唯一记录超过8MiB，不执行清理')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dest)
                    if sha(dest) != h:
                        raise ValueError('唯一记录保全失败')
                    row.update(role='unique_preserved', retained=dest.relative_to(ROOT).as_posix())
            rows.append(row)
    packages = []
    site = ROOT / TARGETS[0] / 'Lib/site-packages'
    for info in sorted(site.glob('*.dist-info/METADATA')):
        headers = info.read_text(encoding='utf-8', errors='replace').split('\n\n', 1)[0].splitlines()
        packages.append({k: next((v[len(k)+2:] for v in headers if v.startswith(k+': ')), '') for k in ('Name', 'Version')})
    put('环境重建元数据.json', dict(pyvenv=(ROOT / TARGETS[0] / 'pyvenv.cfg').read_text(), packages=packages,
        source='existing environment metadata only;no install/network', runtime_reproducibility='requires reinstall;not a byte-identical runtime backup'))
    put('清理预登记.json', dict(targets=list(TARGETS), files=rows, bytes=sum(r['bytes'] for r in rows),
        retained_unique_bytes=unique_bytes, old_research_models_deleted=False,
        copy_restore='For each original copy path, use restore_source or retained; verify recorded SHA before restoration.'))
    print(json.dumps(dict(files=len(rows),MiB=sum(r['bytes'] for r in rows)/1024**2,unique_MiB=unique_bytes/1024**2)))


def check_cleanup():
    plan = read(OUT / '清理预登记.json')
    assert_no_dependents()
    if plan['targets'] != list(TARGETS):
        raise ValueError('目标清单变化')
    actual = {p.relative_to(ROOT).as_posix() for name in TARGETS for p in files_under(ROOT / name)}
    if actual != {r['path'] for r in plan['files']}:
        raise ValueError('清理树文件集合变化')
    for row in plan['files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('待清理文件变化')
        restore = row.get('restore_source') or row.get('retained')
        if restore and sha(ROOT / restore) != row['sha256']:
            raise ValueError('复制资产恢复来源变化')
    return dict(passed=True, files=len(actual), bytes=plan['bytes'])


def audit_protected():
    b = read(OUT / '基线保护.json')
    maintained_entries = {row['path'] for row in b['entries']}
    changed = [n for n, h in b['protected'].items() if n not in maintained_entries
               and (not (ROOT / n).is_file() or sha(ROOT / n) != h)]
    user_changed = [n for n,h in b['user'].items() if not (ROOT / n).is_file() or sha(ROOT / n)!=h]
    raw = {p.relative_to(ROOT).as_posix():p.stat().st_size for p in (ROOT/'DATA/raw_data').rglob('*') if p.is_file()}
    if changed or user_changed or raw != b['raw_sizes']:
        raise ValueError('保护原件发生变化：' + str(changed + user_changed))
    return dict(passed=True, protected_sha_files=len(b['protected'])-sum(n in maintained_entries for n in b['protected']), user_files=len(b['user']),
                raw_file_count=len(raw), raw_inventory_unchanged=True, raw_fully_rehashed=False,
                models_and_final3_sha_unchanged=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['probe','baseline','plan-cleanup','check-cleanup','audit'])
    p.add_argument('--port',type=int,default=8769)
    a=p.parse_args()
    if a.action=='probe': print(json.dumps(probe_port(a.port),ensure_ascii=False))
    elif a.action=='baseline':baseline()
    elif a.action=='plan-cleanup':plan_cleanup()
    elif a.action=='check-cleanup':print(json.dumps(check_cleanup()))
    elif a.action=='audit':print(json.dumps(audit_protected()))
