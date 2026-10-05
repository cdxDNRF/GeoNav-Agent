"""Offline exact-origin link normalization; frozen v1 remains unchanged."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin,urlparse
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_catalog_v1 as c

OUT=c.OUT/'目录链接适配_v2'
CODE=('data/area_catalog_links_v2.py','eval/audit_area_catalog_links_v2.py','tests/test_area_catalog_links_v2.py')


def normalize(href,collection,split):
    index=c.BASE+f'{collection}/{split}/sat/index.html'
    u=urlparse(urljoin(index,href))
    prefix=f'/~vmnih/data/{collection}/{split}/sat'
    if u.scheme not in ('http','https') or u.hostname!=c.HOST or u.username or u.password or u.port or u.query or u.fragment:
        raise ValueError('origin/credentials/query changes prohibited')
    suffix=u.path[len(prefix):] if u.path.startswith(prefix) else ''
    if not re.fullmatch(r'/+\d{8}_15\.tiff',suffix):
        raise ValueError('only repeated slash immediately before TIFF basename may normalize')
    return c.BASE+f'{collection}/{split}/sat/'+suffix.lstrip('/')


class Parser(HTMLParser):
    def __init__(self,collection,split):
        super().__init__(); self.collection=collection; self.split=split; self.records={}; self.changed=0
    def handle_starttag(self,tag,attrs):
        href=dict(attrs).get('href','')
        if tag!='a' or not href.endswith('.tiff'): return
        url=normalize(href,self.collection,self.split)
        name=Path(urlparse(url).path).name
        self.changed+=urlparse(urljoin(c.BASE,href)).path!=urlparse(url).path
        self.records[name]=dict(name=name,url=url,collection=self.collection,upstream_split=self.split,
                                original_href=href,repeated_slash_format_only=True)


def run():
    if OUT.exists(): raise ValueError('immutable adapter batch exists')
    c.check()
    prior=c.read(c.OUT/'调查输出封存.json')
    for name,h in prior['files_sha256'].items():
        if c.digest(c.ROOT/name)!=h: raise ValueError('v1 output drift '+name)
    OUT.mkdir()
    sources={}
    for n in CODE:
        p=c.ROOT/'project/src'/n; target=OUT/'源码快照'/n
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(p.read_bytes())
        sources[c.rel(p)]=c.digest(p); sources[c.rel(target)]=c.digest(target)
    test=c.ROOT/'选题报告相关/十五乘十五目录链接适配单元测试_v2.json'
    if not c.read(test)['successful']: raise ValueError('adapter tests required')
    c.write(OUT/'预登记.json',dict(utc=c.stamp(),source_sha256=sources,input_sha256=prior['files_sha256'],
             test_sha256=c.digest(test),normalization='only repeated slashes before known basename; same exact host/collection/split',
             geometry_quality_order_or_exclusions_changed=False,new_requests=0,old_v1_changed=False,new_SR=False))
    records,changes,counts=[],{},{}
    for collection in c.COLLECTIONS:
        for split in c.SPLITS:
            parser=Parser(collection,split)
            parser.feed((c.OUT/'元数据'/f'官方_{collection}_{split}.html').read_text(encoding='utf-8'))
            if not parser.records: raise ValueError('complete known directory required')
            records.extend(parser.records.values())
            counts[f'{collection}/{split}']=len(parser.records); changes[f'{collection}/{split}']=parser.changed
    old=c.read(c.ROOT/c.INPUTS[1]); consumed=c.read(c.ROOT/c.INPUTS[2])['regions']+c.read(c.ROOT/c.INPUTS[4])['regions']
    cache={r['id']:r for r in c.read(c.ROOT/c.INPUTS[3])}
    heads={r['name']:r for p in (c.OUT/'元数据/GeoTIFF头').glob('*.json') if (r:=c.read(p))}
    forbidden=[r['bounds_m'] for r in old+consumed]; scopes={}
    for label,items in [('roads_only',[r for r in records if r['collection']=='mass_roads']),('publisher_rgb_union',records)]:
        tiles=c.merge(items); candidates,why=c.propose(tiles,old,consumed,cache)
        actual=[g for cand in candidates if (g:=c.actual_group(cand,heads,forbidden)) is not None]
        scopes[label]=dict(unique_proposed_tiles=len(tiles),candidate_groups=candidates,counts=why,
                          provisional_pack=c.pack(candidates),coordinate_verified_groups=actual,
                          coordinate_verified_pack=c.pack(actual,'bounds_m'),
                          all_proposed_coordinates_checked=len(actual)==len(candidates))
    name_sets={s:{r['name'][:-5] for r in records if r['collection']=='mass_buildings' and r['upstream_split']==s} for s in c.SPLITS}
    old_sets={s:{r['id'] for r in old if r['split']==('val' if s=='valid' else s)} for s in c.SPLITS}
    # Accept the recorded upstream local validation spelling without conflation.
    old_sets['valid']={r['id'] for r in old if r['split'] in ('valid','val','validation')}
    comparison={s:dict(publisher_names=len(name_sets[s]),local_names=len(old_sets[s]),
                       exact_filename_set_equal=name_sets[s]==old_sets[s],pixels_equal_proved=False) for s in c.SPLITS}
    c.write(OUT/'完整目录分析结果.json',dict(utc=c.stamp(),catalog_complete=True,records=records,source_counts=counts,
             normalized_duplicate_slash_counts=changes,scopes=scopes,Buildings_vs_local151=comparison,
             live_index_does_not_resolve_paper1188_discrepancy=True,excluded_old151=151,excluded_consumed24=24,
             geometry_or_quality_protocol_changed=False,new_requests=0,full_images_downloaded=0,new_SR=False,
             source_expansion_needs_own_protocol=True,formal_queue_frozen=False))
    c.check()
    c.write(OUT/'封存.json',dict(files_sha256={c.rel(p):c.digest(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(source_counts=counts,filename_packs={k:v['provisional_pack']['count'] for k,v in scopes.items()},
                          coordinate_packs={k:v['coordinate_verified_pack']['count'] for k,v in scopes.items()},Buildings_vs_local151=comparison),ensure_ascii=False),flush=True)


if __name__=='__main__': run()
