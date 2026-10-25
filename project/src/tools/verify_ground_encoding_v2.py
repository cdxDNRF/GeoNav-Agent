"""CV-025 pre-forward loading repair; original v1 failure is immutable.

Meta construction and single mapped slot only. Same8images/tensors/thresholds,
max3GiB and cumulative encoding time600sec. No retry of old execute.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import unittest
import numpy as np
from project.src.tools.verify_ground_encoding_v1 import ROOT,OUT,read,sha,verify,put,entry,reference_pixels

REPAIR_SOURCES=['agents/frozen_crossview_encoder_v2.py','tests/test_ground_encoding_v2.py',
    'tools/verify_ground_encoding_v2.py','documents/close_ground_encoding_v2.py']


def bind():
    if (OUT/'加载修复绑定.json').exists():raise FileExistsError('Repair already bound')
    failure=read(OUT/'工程中断.json')
    if failure['attempts']!=0 or failure['completed_pairs']!=0 or (OUT/'编码尝试.jsonl').exists():
        raise ValueError('Only pre-forward resource failure can use this repair')
    if failure['peak_RSS_bytes']<=3*1024**3:raise ValueError('Expected memory guard failure')
    old=read(OUT/'首forward绑定.json')
    verify(sum((old[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')),[]))
    from project.src.tests.test_ground_encoding_v2 import Tests
    tests=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(tests)
    put('加载修复测试.json',{'tests':tests.testsRun,'passed':tests.wasSuccessful(),'failures':tests.failures,'errors':tests.errors,
        'actual_CLIP_Sat2Cap_forward':0,'tiny_parameter_module_only':True})
    if not tests.wasSuccessful():raise ValueError('Loader tests failed')
    put('加载修复绑定.json',{'task':'CV-025','stage':'pre-first-forward engineering repair',
        'original_binding':entry(OUT/'首forward绑定.json'),'prior_failure':entry(OUT/'工程中断.json'),
        'sources':[entry(ROOT/'project/src'/p) for p in REPAIR_SOURCES],
        'changes':['meta model construction','load_state_dict strict assign reuses CPU tensor storage','one live encoder at a time'],
        'unchanged':['model architecture/tensors','8fixed inputs','preprocessing','error/cosine thresholds','3GiB RSS cap'],
        'prior_actual_forward_calls':0,'remaining_actual_forward_calls':8,
        'prior_failed_elapsed_seconds':failure['elapsed_seconds'],'cumulative_encoding_time_limit_seconds':600,
        'v1_execute_must_not_rerun':True,'stop_if_repair_fails':True})
    print('Memory-only loader repair bound;0actual forward consumed')


def verify_repair():
    repair=read(OUT/'加载修复绑定.json')
    verify(repair['sources']+[repair['original_binding'],repair['prior_failure']])
    return repair


def execute():
    if (OUT/'v2_执行开始.json').exists():raise FileExistsError('Consumed/partial batch;inspect prefix, no automatic rerun')
    b=read(OUT/'首forward绑定.json');protocol=read(OUT/'冻结工程协议.json')
    repair=verify_repair()
    protocol=dict(protocol,max_seconds=protocol['max_seconds']-repair['prior_failed_elapsed_seconds'])
    verify(sum((b[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')),[]))
    pairs=read(OUT/'固定输入清单.json')['pairs']
    refs=[np.load(ROOT/e['path'],allow_pickle=False) for e in b['features']]
    assert refs[0].shape==(14,512) and refs[1].shape==(1134,512)
    from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
    from project.src.agents.frozen_crossview_encoder_v1 import pixels
    from project.src.platform_core.ground_encoding_v1 import GroundEncodingSession,decode_frame
    import psutil
    started=time.perf_counter();process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    put('v2_执行开始.json',{'pid':os.getpid(),'single_execution':True,'real_forward_successes_so_far':0,'device':'cpu'})
    cursor={};attempts=[];events=[];rows=[];features=[];encoder=None
    def guard():
        peak[0]=max(peak[0],process.memory_info().rss)
        if peak[0]>protocol['max_rss_bytes'] or time.perf_counter()-started>protocol['max_seconds']:raise RuntimeError('Resource limit reached;stop without retry')
    def before_forward(item):
        guard();record=dict(item,**cursor);attempts.append(record)
        with (OUT/'编码尝试.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    try:
        encoder=FrozenCrossViewEncoder(ROOT,b['models'],max_forward_calls=8,before_forward=before_forward)
        processor=encoder.processor
        for p in pairs:
            public={};preprocess=[]
            for kind in ('ground','aerial'):
                raw=(ROOT/p[kind]['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==p[kind]['sha256']
                frame=decode_frame(raw,ground=kind=='ground');reference=reference_pixels(raw,kind,processor)
                actual=pixels(frame,kind,processor)
                delta=float(np.abs(actual-reference).max())
                if delta>1e-6:raise ValueError('Preprocessor differs;no forward for this input')
                preprocess.append({'kind':kind,'max_abs':delta,'reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),
                    'actual_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'pixel_sha256':frame.fingerprint()})
                public[kind]=frame
            def sink(event):
                events.append(dict(event,pair_evaluator_only=p['source_evaluator_only']))
                with (OUT/'局部事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(events[-1],ensure_ascii=False)+'\n');f.flush()
            session=GroundEncodingSession(encoder,event_sink=sink)
            cursor.clear();cursor.update(source_evaluator_only=p['source_evaluator_only'],kind_evaluator_only='ground')
            session.begin(public['ground'],grid_size=5,budget=10)
            cursor['kind_evaluator_only']='aerial'
            packet=session.observe_public({'position':(0,0),'grid_size':5,'step_index':0,'remaining_budget':10,
                'legal_actions':('up','right','down','left'),'current_image':public['aerial']})
            checks=[]
            for kind,value,index,cache in [('ground',packet.target_feature,p['ground_index'],refs[0]),('aerial',packet.current_feature,p['aerial_index'],refs[1])]:
                current=np.asarray(value,np.float32);expected=cache[index]
                error=float(np.max(np.abs(current-expected)))
                cos=float(np.dot(current.astype(np.float64),expected.astype(np.float64))/(np.linalg.norm(current.astype(np.float64))*np.linalg.norm(expected.astype(np.float64))))
                checks.append({'kind':kind,'cache_index_evaluator_only':index,'max_abs':error,'cosine':cos,
                    'passed':error<=5e-5 and cos>=.999999})
                features.append(current)
            rows.append({'source_evaluator_only':p['source_evaluator_only'],'preprocessing':preprocess,'cache_checks':checks,
                'public_packet':asdict(packet),'event':session.events[-1]})
            session.close();guard()
        frozen=encoder.close();guard()
        put('逐对接口核验.json',{'pairs':rows,'navigation_SR':None,'evidence_not_confidence':True})
        with (OUT/'在线工程特征.npy').open('xb') as f:np.save(f,np.asarray(features,np.float32),allow_pickle=False)
        gates={'all8_complete':len(features)==8 and encoder.calls==8,'cache_consistent':all(c['passed'] for r in rows for c in r['cache_checks']),
            'preprocessing_consistent':all(c['max_abs']<=1e-6 for r in rows for c in r['preprocessing']),
            'frozen_weights':all(c['unchanged'] for c in frozen['frozen_tensor_checks']),
            'resources':peak[0]<=3*1024**3 and time.perf_counter()-started<=600}
        put('工程结果.json',{'gates':gates,'passed':all(gates.values()),'pairs':4,'known_development_sources':4,
            'max_embedding_abs_error':max(c['max_abs'] for r in rows for c in r['cache_checks']),
            'min_embedding_cosine':min(c['cosine'] for r in rows for c in r['cache_checks']),
            'real_forward_calls':8,'navigation':0,'training':0,'API':0,'confirmation':0,'frozen_encoder_report':frozen,
            'feature':entry(OUT/'在线工程特征.npy'),'no_performance_or_default_upgrade':True})
        verify(b['models']+b['features']+b['protected']+b['sources']+b['dependencies']+b['inputs'])
        put('执行完成.json',{'completed':True,'elapsed_seconds':time.perf_counter()-started,'sampled_peak_RSS_bytes':peak[0],
            'prior_failed_elapsed_seconds':repair['prior_failed_elapsed_seconds'],'cumulative_encoding_seconds':repair['prior_failed_elapsed_seconds']+time.perf_counter()-started,
            'sample_interval_seconds':.05,'device':'cpu','successful_forward_calls':encoder.calls,
            'attempted_forward_calls':len(attempts),'pairs':4,'engineering_passed':all(gates.values()),
            'raw_inputs':8,'training':0,'navigation':0,'API':0,'downloads':0,'confirmation':0})
        if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>6*1024**2:raise ValueError('Output cap exceeded')
        print(json.dumps({'engineering_passed':all(gates.values()),'calls':8,'max_abs':max(c['max_abs'] for r in rows for c in r['cache_checks']),'peak_RSS':peak[0]}))
    except Exception as error:
        put('v2_工程中断.json',{'error_type':type(error).__name__,'attempts':len(attempts),'completed_pairs':len(rows),
            'elapsed_seconds':time.perf_counter()-started,'peak_RSS_bytes':peak[0],'no_auto_retry':True})
        raise
    finally:
        stop.set();monitor.join(timeout=1)
        if encoder is not None and not encoder.closed:encoder.close()



def audit():
    verify_repair()
    from project.src.tools.verify_ground_encoding_v1 import audit as base_audit
    result=read(OUT/'执行完成.json')
    if result['cumulative_encoding_seconds']>600:raise ValueError('Cumulative time cap exceeded')
    base_audit()
    print('Repair source/failure bindings and cumulative600sec guard also verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['bind','execute','audit']);a=p.parse_args()
    {'bind':bind,'execute':execute,'audit':audit}[a.phase]()
