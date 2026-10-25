"""CV-016 alternate metadata/receipt/central-geometry audit; no raw image access."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/MM-GAG/同任务拍摄点标签与可辨识性审查_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    binding=read(OUT/'输入与源码绑定.json')
    for e in binding['inputs']+binding['sources']+binding['protected']+[binding['protocol']]:
        p=ROOT/e['path'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256'] and len(data)==e['bytes']
    p=read(OUT/'冻结审查范围.json');r=read(OUT/'像素回执与面板.json');v=read(OUT/'逐源视觉审查.json')
    expected={e['path']:e for s in p['sources'] for e in s['images'].values()}
    assert len(expected)==42 and r['raw_reads']==42 and r['raw_decode_calls']==42
    assert len(r['receipts'])==42 and {x['path'] for x in r['receipts']}==set(expected)
    for rec in r['receipts']:assert rec['sha256']==expected[rec['path']]['sha256']
    events=[json.loads(x) for x in (OUT/'输入消费.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(events)==84
    assert Counter(e['path'] for e in events if e['event']=='attempt')==Counter({k:1 for k in expected})
    assert Counter(e['path'] for e in events if e['event']=='decoded')==Counter({k:1 for k in expected})
    observations={x['source_id']:x for x in v['sources']};assert len(v['sources'])==14 and len(observations)==14
    labels={'plausible','unclear'}
    for s,panel in zip(p['sources'],r['panels']):
        assert s['source_id']==panel['source_id'];e=panel['panel'];data=(ROOT/e['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256']
        # Independent flat-channel distance from saved tiny pixels, never a second raw decode.
        left=sum(panel['center_thumbnail'],[]);right=sum(panel['mosaic_center_thumbnail'],[])
        error=sum(abs(a-b) for a,b in zip(left,right))/len(left)
        assert abs(error-panel['center_mapping_MAE16'])<1e-12 and (error<=8)==panel['center_mapping_consistent_at_old_MAE8_tolerance']
        obs=observations[s['source_id']];assert obs['scene_correspondence'] in labels
        assert obs['camera_point_identifiable']=='unknown' and obs['independent_geographic_label_verified'] is False
        assert obs['basis']=='posthoc_nonblind_observer_panel' and len(obs['observation'])>=15
        assert s['metadata']['observed_distinct_queries_in_group']==1
    assert all(not x['metadata']['pixel_to_GPS_transform_available'] for x in p['sources'])
    result={'task':'CV-016','passed':True,'sources_checked':14,'receipt_and_raw_hash_references':42,'raw_images_redecoded_by_auditor':0,
        'center_geometry_cases':14,'events':84,'image_panels_SHA_checked':14,
        'manual_labels_are_not_independent_geographic_truth':True,'independent_person_review':False,
        'new_model_forward_calls':0,'new_predictions':0,'training_steps':0,'confirmation_access':0}
    target=OUT/'核验/另实现复核.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
