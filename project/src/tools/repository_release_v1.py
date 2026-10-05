"""Read-only Git index publication checks; never imports project models.

Run from the repository root with --output pointing to a new local JSON file.
Only newly staged additions/changes get the size and Python syntax checks;
publication boundaries and credential checks cover the complete index.
"""
import argparse
import ast
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess


def _relative(value):
    normalized = str(value).replace('\\', '/')
    path = PurePosixPath(normalized)
    if (not normalized or path.is_absolute() or PureWindowsPath(normalized).drive
            or '..' in path.parts or '.git' in path.parts):
        raise ValueError('unsafe repository-relative path')
    return path.as_posix()


def publication_issues(paths):
    issues = []
    for original in paths:
        try:
            path = _relative(original)
        except ValueError:
            issues.append({'path': original, 'reason': 'unsafe_path'})
            continue
        name = PurePosixPath(path).name
        reason = None
        if name.startswith('.env') and not (name == '.env.example' or name.endswith('.example')):
            reason = 'actual_credentials'
        elif path.startswith(('.agents/本地/', 'project/src/tools/local/', 'DATA/raw_data/', 'models/', 'GOMAA-Geo/')):
            reason = 'local_only'
        elif path.startswith('DATA/processed_data/') and path.lower().endswith(('.tif', '.tiff', '.jsonl')) and not any(fnmatch.fnmatchcase(path, pattern) for pattern in (
                'DATA/processed_data/Masa/episodes/episodes_*.jsonl',
                'DATA/processed_data/Masa/任务清单_v2/episodes_*.jsonl',
                'DATA/processed_data/Masa/评测结果/Gemma与DeepSeek同策略对照_v1/原val任务清单.jsonl',
                'DATA/processed_data/Masa/评测结果/S2稳定验证_v1/固定任务.jsonl')):
            reason = 'bulk_dataset_payload'
        elif path.lower().endswith(('.npy', '.npz', '.pt', '.pth', '.ckpt', '.safetensors')):
            reason = 'model_or_feature_payload'
        elif any(part in {'__pycache__', 'node_modules', '.venv', '.zcode', '.mimosa', '.superpowers'} for part in PurePosixPath(path).parts):
            reason = 'local_cache'
        elif path.startswith('DATA/processed_data/') and '/源码快照/' in path and name.startswith(('pi_', 'external_agent_smoke_')):
            reason = 'helper_snapshot'
        elif path.startswith('DATA/processed_data/') and '/工程数据/previews/' in path:
            reason = 'derived_preview'
        elif path.startswith(('视频/成功案例可视化/', '视频/实际用途示意/')) and (name.startswith('frame_') or name == 'start_overview.png'):
            reason = 'render_frame'
        if reason:
            issues.append({'path': original, 'reason': reason})
    return issues


def resolve_moved(root, relative, mapping):
    root = Path(root).resolve()
    original = root / _relative(relative)
    if original.exists():
        chosen = original.resolve()
    else:
        chosen = (root / _relative(mapping.get(relative, relative))).resolve()
    if not chosen.is_relative_to(root):
        raise ValueError('mapped path escapes repository')
    return chosen


def secret_locations(text):
    """Return line/type only, never return matching credential values."""
    patterns = {
        'api_token': r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}',
        'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
        'google_token': r'\bAIza[A-Za-z0-9_-]{30,}',
        'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    }
    result = []
    for kind, expression in patterns.items():
        for match in re.finditer(expression, text):
            result.append({'kind': kind, 'line': text.count('\n', 0, match.start()) + 1})
    return result


def verify_hashes(root, hashes, mapping):
    for relative, expected in hashes.items():
        path = resolve_moved(root, relative, mapping)
        actual = None
        if path.is_file():
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'hash mismatch or missing: {relative}')
    return len(hashes)


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def inspect_index(root, max_new_bytes=10 * 1024 * 1024):
    root = Path(root).resolve()
    tracked = [p.decode('utf-8') for p in _git(root, 'ls-files', '-z').split(b'\0') if p]
    changed = {p.decode('utf-8') for p in _git(root, 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').split(b'\0') if p}
    issues = publication_issues(tracked)
    # ls-files -s provides the indexed blob ID, so checks do not depend on
    # un-staged working-tree edits. One cat-file process reads all blobs.
    entries = []
    for row in _git(root, 'ls-files', '-s', '-z').split(b'\0'):
        if not row:
            continue
        header, raw_path = row.split(b'\t', 1)
        mode, oid, stage = header.decode('ascii').split()
        path = raw_path.decode('utf-8')
        if stage != '0':
            issues.append({'path': path, 'reason': 'unmerged_index'})
        else:
            entries.append((mode, oid, path))
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    manifest = []
    try:
        for mode, oid, path in entries:
            proc.stdin.write((oid + '\n').encode('ascii')); proc.stdin.flush()
            header = proc.stdout.readline().decode('ascii').split()
            if len(header) != 3 or header[1] != 'blob':
                raise ValueError(f'cannot read indexed blob: {path}')
            size = int(header[2])
            data = proc.stdout.read(size)
            if len(data) != size or proc.stdout.read(1) != b'\n':
                raise ValueError(f'incomplete indexed blob: {path}')
            manifest.append({'path': path, 'bytes': size, 'sha256': hashlib.sha256(data).hexdigest()})
            if path in changed and size > max_new_bytes:
                issues.append({'path': path, 'reason': 'new_file_too_large', 'bytes': size})
            if b'\0' not in data:
                try:
                    text = data.decode('utf-8-sig')
                except UnicodeDecodeError:
                    continue
                for location in secret_locations(text):
                    issues.append({'path': path, 'reason': 'credential_match', **location})
            if path in changed and path.endswith('.py'):
                try:
                    ast.parse(data, filename=path)
                except (SyntaxError, ValueError) as error:
                    issues.append({'path': path, 'reason': 'python_syntax', 'line': getattr(error, 'lineno', None)})
    finally:
        proc.stdin.close(); proc.stdout.close()
        if proc.wait() != 0:
            raise ValueError('git cat-file failed')
    return {'ok': not issues, 'index_file_count': len(manifest), 'index_bytes': sum(x['bytes'] for x in manifest),
            'changed_file_count': len(changed), 'changed_bytes': sum(x['bytes'] for x in manifest if x['path'] in changed),
            'python_checked': sum(p.endswith('.py') for p in changed), 'issues': issues, 'files': manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    local = (root / '.agents/本地').resolve()
    if not output.is_relative_to(local) or output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new file within the registered local verification directory')
    report = inspect_index(root)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
