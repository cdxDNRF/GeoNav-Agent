"""Bounded, single-member HTTP-range ZIP acquisition. Never fall back to a full ZIP."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'DATA/processed_data/MM-GAG/单权重取得与工程_v1'
ASSET = ROOT / 'models/sample4geo_cvusa_public_v1'
FILE_ID = '1dZ9itDjS-h5KHc23OQb8oHqJ3KoSaPyS'
URL = 'https://drive.usercontent.google.com/download?id=' + FILE_ID + '&export=download&confirm=t'
MEMBER = 'pretrained/cvusa/convnext_base.fb_in22k_ft_in1k_384/weights_e40_98.6830.pth'
TOTAL = 1626048556
CAP = 384 * 1024**2
NET_CAP = 512 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def record(path):
    p = Path(path)
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def checked_range(status, headers, start, end, total):
    if status != 206:
        raise ValueError(f'range unsupported: status={status}; full body refused')
    expected = f'bytes {start}-{end}/{total}'
    if headers.get('Content-Range') != expected:
        raise ValueError('Content-Range mismatch')
    if int(headers.get('Content-Length', end-start+1)) != end-start+1:
        raise ValueError('Content-Length mismatch')
    if headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('encoded response refused')


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(newurl)
        if u.scheme != 'https' or u.hostname not in {'drive.google.com', 'drive.usercontent.google.com'}:
            raise ValueError('unexpected redirect host')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class RangeReader:
    def __init__(self, out):
        self.out = out
        self.used = 0
        self.requests = 0
        self.started = time.monotonic()
        self.opener = urllib.request.build_opener(PublicRedirect())

    def event(self, data):
        with (self.out / '取得请求事件.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def read(self, start, end, dest=None):
        n = end-start+1
        if not (0 <= start <= end < TOTAL) or self.used+n > NET_CAP or self.requests >= 8:
            raise ValueError('range/network/request cap')
        if time.monotonic()-self.started > 1200:
            raise ValueError('acquisition wall-clock cap')
        self.requests += 1
        event = {'request': self.requests, 'start': start, 'end': end, 'expected_bytes': n}
        self.event({**event, 'phase': 'before_request'})
        req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        try:
            with self.opener.open(req, timeout=40) as r:
                event['status'] = r.status
                event['content_range'] = r.headers.get('Content-Range')
                event['content_type'] = r.headers.get('Content-Type')
                checked_range(r.status, r.headers, start, end, TOTAL)
                target = io.BytesIO() if dest is None else Path(dest).open('xb')
                got = 0
                h = hashlib.sha256()
                try:
                    while got < n:
                        if time.monotonic()-self.started > 1200:
                            raise ValueError('acquisition wall-clock cap')
                        b = r.read(min(1024**2, n-got))
                        if not b:
                            raise ValueError('short range response')
                        got += len(b)
                        self.used += len(b)
                        h.update(b)
                        target.write(b)
                    if r.read(1):
                        self.used += 1
                        raise ValueError('overlong range response')
                    result = target.getvalue() if dest is None else None
                finally:
                    if dest is not None:
                        target.close()
                self.event({**event, 'phase': 'complete', 'bytes': got, 'sha256': h.hexdigest()})
                return result
        except Exception as e:
            # HTTP errors are not read; do not persist cookies or signed URLs.
            self.event({**event, 'phase': 'failed', 'error_type': type(e).__name__,
                        'http_code': getattr(e, 'code', None), 'bytes_read_total': self.used})
            raise


def directory_location(tail, total):
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0 or pos+22 > len(tail):
        raise ValueError('EOCD missing')
    sig, disk, cd_disk, n_disk, n_all, size, offset, comment = struct.unpack_from('<4s4H2LH', tail, pos)
    if pos+22+comment != len(tail) or disk or cd_disk or n_disk != n_all:
        raise ValueError('invalid/spanned ZIP')
    if size > 2*1024**2 or n_all == 65535 or offset == 0xffffffff or n_all == 0:
        raise ValueError('ZIP64/oversize/empty directory refused')
    if offset+size != total-len(tail)+pos:
        raise ValueError('central directory geometry mismatch')
    return offset, size, tail[pos:]


def select_member(cd, eocd):
    # Feed a compact directory-only ZIP to Python's separate central directory parser.
    end = bytearray(eocd)
    struct.pack_into('<L', end, 16, 0)
    with zipfile.ZipFile(io.BytesIO(cd+end)) as z:
        entries = z.infolist()
    matches = [x for x in entries if x.filename == MEMBER]
    if len(matches) != 1:
        raise ValueError('exact checkpoint member missing/ambiguous')
    x = matches[0]
    if x.flag_bits & 1 or x.compress_type not in (0, 8) or (x.external_attr >> 16) & 0o170000 == 0o120000:
        raise ValueError('encrypted/unsupported/symlink member')
    if not (0 < x.file_size <= CAP and 0 < x.compress_size <= CAP):
        raise ValueError('checkpoint member size cap')
    return x


def data_start(header, info):
    sig, version, flags, method, mt, md, crc, csize, size, nl, el = struct.unpack('<4s5H3L2H', header[:30])
    if sig != b'PK\x03\x04' or method != info.compress_type or flags != info.flag_bits:
        raise ValueError('local header mismatch')
    return info.header_offset+30+nl+el, nl, el


def unpack_member(compressed, output, info):
    crc = 0
    count = 0
    dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
    with Path(compressed).open('rb') as f, Path(output).open('xb') as w:
        for b in iter(lambda: f.read(1024**2), b''):
            b = dec.decompress(b, min(info.file_size-count+1, CAP-count+1)) if dec else b
            count += len(b)
            if count > info.file_size or count > CAP or (dec and dec.unconsumed_tail):
                raise ValueError('decompression size cap')
            w.write(b)
            crc = zlib.crc32(b, crc)
        if dec and (not dec.eof or dec.unused_data or dec.flush()):
            raise ValueError('deflate incomplete/trailing data')
    if count != info.file_size or crc != info.CRC:
        raise ValueError('uncompressed size/CRC mismatch')
    return {'bytes': count, 'crc32': f'{crc:08x}', 'sha256': sha(output)}


def acquire():
    protocol = json.loads((OUT / '执行协议.json').read_text(encoding='utf-8'))
    if protocol['archive_bytes'] != TOTAL or protocol['member'] != MEMBER:
        raise ValueError('protocol mismatch')
    write_new(OUT / '取得开始.json', {'protocol': record(OUT/'执行协议.json'), 'source': record(__file__),
                                     'status': 'started_do_not_blindly_rerun'})
    ASSET.mkdir(exist_ok=False)
    r = RangeReader(OUT)
    try:
        tail = r.read(TOTAL-65557, TOTAL-1)
        offset, size, eocd = directory_location(tail, TOTAL)
        cd = r.read(offset, offset+size-1)
        (OUT/'中央目录.bin').write_bytes(cd)
        (OUT/'目录结束.bin').write_bytes(eocd)
        info = select_member(cd, eocd)
        h = r.read(info.header_offset, info.header_offset+29)
        begin, nl, el = data_start(h, info)
        if nl+el > 65536 or begin+info.compress_size > offset:
            raise ValueError('local member bounds')
        name_extra = r.read(info.header_offset+30, begin-1)
        if name_extra[:nl].decode('utf-8' if info.flag_bits & 2048 else 'cp437') != MEMBER:
            raise ValueError('local filename mismatch')
        metadata = {'member': MEMBER, 'header_offset': info.header_offset, 'data_start': begin,
                    'compressed_bytes': info.compress_size, 'uncompressed_bytes': info.file_size,
                    'crc32': f'{info.CRC:08x}', 'method': info.compress_type}
        write_new(OUT/'选定member.json', metadata)
        part = ASSET/'selected_member.deflate.part'
        r.read(begin, begin+info.compress_size-1, part)
        output = ASSET/'weights_e40_98.6830.pth'
        result = unpack_member(part, output, info)
        write_new(OUT/'权重取得结果.json', {'status': 'acquired', 'network_body_bytes': r.used,
                   'requests': r.requests, 'archive_total_bytes_metadata': TOTAL,
                   'archive_sha256': None, 'archive_not_downloaded': True,
                   'compressed_member_sha256': sha(part), 'model': record(output), **metadata, **result})
        # The verified selected compressed stream is this task's regenerable temporary file.
        if part.resolve().parent != ASSET.resolve():
            raise ValueError('cleanup path outside asset folder')
        part.unlink()
    except Exception as e:
        write_new(OUT/'权重取得结果.json', {'status': 'stopped', 'reason_type': type(e).__name__,
                  'reason': str(e)[:200] if not isinstance(e, urllib.error.URLError) else 'public endpoint error; see HTTP receipt',
                  'network_body_bytes': r.used, 'requests': r.requests,
                  'full_archive_downloaded': False, 'model_loaded': False,
                  'recovery': 'inspect receipts/partial paths; no automatic retry/full archive fallback'})
        print('Stopped:', type(e).__name__, 'HTTP', getattr(e, 'code', None), flush=True)
        return 2
    print('Single checkpoint acquired', result['bytes'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(acquire())
