"""Read-only CV041 evidence kernel. No model/cache/optimizer or experiment API.

Only frozen development outputs and already-consumed images are presented.
True positions are observer-only, never an Agent input or new evaluation.
"""
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from project.src.platform_core.perception_reader_v1 import EvidenceUnavailable

BATCH='DATA/processed_data/OpenCVL/普通RGB空间小头固定开发对照_v1'
SEAL_SHA='45474565d00ebf527ca5e651e2f62b623cfd081cb86621e43400366e1a0d830d'
METHODS=('Spatial','TrainedMean','RawCos')
ARMS=('Full','Disabled','Mean','Wrong')
META=('首正式执行绑定.json','主对照汇总.json','最终判定.json','固定校准阈值.json','预测文件清单.json')


class SpatialEvidenceReader:
    def __init__(self,root):self.root=Path(root).resolve()

    def _raw(self,relative,expected):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):raise EvidenceUnavailable('证据路径超出工作区')
        try:data=path.read_bytes()
        except OSError as e:raise EvidenceUnavailable('空间对照证据尚未携带，请按搬迁清单恢复原件') from e
        if hashlib.sha256(data).hexdigest()!=expected:raise EvidenceUnavailable('空间对照证据绑定已变化，请恢复封存原件')
        return data

    def _bundle(self):
        seal=json.loads(self._raw(BATCH+'/阶段封存.json',SEAL_SHA));refs={r['path']:r['sha256'] for r in seal['files']}
        def get(name):return json.loads(self._raw(BATCH+'/'+name,refs[BATCH+'/'+name]))
        binding=get('首正式执行绑定.json');summary=get('主对照汇总.json');decision=get('最终判定.json');cal=get('固定校准阈值.json')
        protocol=json.loads(self._raw(binding['protocol']['path'],binding['protocol']['sha256']))
        cases=[(f,c) for f in protocol['folds'] for c in f['development_cases']]
        if len(cases)!=150 or summary['development_predictions']!=2850:raise EvidenceUnavailable('空间对照分母与界面规范不一致')
        return refs,binding,summary,decision,cal,protocol,cases

    @staticmethod
    def _select(cases,case,method,seed,arm):
        if type(case) is not str or not case.isascii() or not case.isdecimal() or not 0<=int(case)<len(cases):raise EvidenceUnavailable('请选择有效的固定案例')
        if method not in METHODS:raise EvidenceUnavailable('请选择有效方法')
        if method=='RawCos':
            if seed!='none':raise EvidenceUnavailable('冻结余弦只有一组确定性输出，不设随机种子')
            number=None
        else:
            if seed not in ('0','1','2'):raise EvidenceUnavailable('请选择冻结种子0/1/2')
            number=int(seed)
        if arm not in (('NoTarget',) if method=='TrainedMean' else ARMS):raise EvidenceUnavailable('该方法没有此目标条件')
        i=int(case);fold,c=cases[i]
        return i,fold,c,number

    def assets(self):
        refs,b,s,decision,cal,d,cases=self._bundle()
        assets={BATCH+'/阶段封存.json':SEAL_SHA}
        assets.update({BATCH+'/'+n:refs[BATCH+'/'+n] for n in META})
        assets[b['protocol']['path']]=b['protocol']['sha256']
        for source in d['source_table'].values():
            for k in ('ground','aerial'):assets[source[k]['path']]=source[k]['sha256']
        for fold in d['folds']:
            for method in METHODS:
                for seed in ((None,) if method=='RawCos' else (0,1,2)):
                    for arm in (('NoTarget',) if method=='TrainedMean' else ARMS):
                        path=BATCH+f"/记录/{fold['fold']}__{method}__{seed}__development__{arm}.json"
                        assets[path]=refs[path]
        return assets

    def catalog(self):
        refs,b,s,decision,cal,d,cases=self._bundle()
        errors=[e for v in s['development_means']['Spatial/Full']['by_source'].values() for e in v['positive_errors_m_all']]
        return dict(batch='CV-041',title='空间小头 · 固定开发对照',status='未通过 · 已收束',readonly=True,new_model_calls=0,navigation_admitted=False,
            methods=list(METHODS),arms=list(ARMS),seeds=['0','1','2'],geographic_sources=3,independent_confirmation_sources=0,
            base_cases=150,development_outputs=2850,all_outputs=6750,
            metrics=s['development_means'],contrasts=s['contrasts'],gates=decision['gates'],
            always_reject_macro_joint=2/3,fixed_location_upper_bound=dict(within5m=sum(e<=5 for e in errors),n=9,diagnostic_only=True),
            sources=[dict(id=k,metric_width_m=v['metric_extent_m'][0]) for k,v in d['source_table'].items()],
            cases=[dict(id=i,city=f['fold'],reference=c['reference'],cell=c['cell'],relation=c['relation'],metric_cell_m=(c['crop'][2]-c['crop'][0])*c['mpp']) for i,(f,c) in enumerate(cases)],
            seal_sha256=SEAL_SHA,limitations=['三个来源均已工程查看，当前是开发证据，不是独立地域确认。','异源负例的航片来自本折拟合来源，不能称查询/参考双端均未见。',
                '联合指标同时要求定位和接受；全部拒绝的参照值为66.67%，不是导航SR。','空间定位热图与给定照片均来自保存记录；切换不编码、不预测、不移动。',
                '当前仅覆盖三城车载前视照片，不外推任意手机照片或全景。','按数据集修正拍摄点评估，5米容差不等于独立测绘精度；raw仅敏感性分析。'])

    def detail(self,case,method='Spatial',seed='0',arm='Full'):
        refs,b,s,decision,cal,d,cases=self._bundle();i,fold,c,number=self._select(cases,case,method,seed,arm)
        path=BATCH+f"/记录/{fold['fold']}__{method}__{number}__development__{arm}.json"
        group=json.loads(self._raw(path,refs[path]))
        if (group['method'],group['fold'],group['seed'],group['phase'],group['condition'])!=(method,fold['fold'],number,'development',arm):raise EvidenceUnavailable('保存分组与所选条件不一致')
        matches=[r for r in group['outputs'] if r['case_id']==c['case_id']]
        if len(matches)!=1:raise EvidenceUnavailable('保存案例缺失或重复')
        pred=matches[0];z=pred['logits']
        if len(z)!=50 or not all(type(v) in (float,int) and math.isfinite(v) for v in z):raise EvidenceUnavailable('保存输出格式无效')
        p=[math.exp(v-max(z)) for v in z];score=sum(p[:49])/sum(p)
        if abs(score-pred['score'])>1e-12 or max(range(49),key=lambda j:z[j])!=pred['patch_index']:raise EvidenceUnavailable('保存分数与logits不相容')
        threshold=next((v for v in cal['calibrations'] if (v['method'],v['fold'],v['seed'])==(method,fold['fold'],number)),None)
        if threshold is None:raise EvidenceUnavailable('缺少固定拟合阈值')
        tau=threshold['tau'];accepted=pred['score']>=tau;truth=c['camera_xy'] if c['relation']=='inside' else None
        error=math.hypot(pred['point'][0]-truth[0],pred['point'][1]-truth[1])*c['mpp'] if truth else None
        good=accepted and error<=5 if truth else not accepted
        if truth:
            reason='接受且位置在5米容差内。' if good else '位置在容差内，但模型拒绝了真实目标。' if error<=5 else '真实目标被拒绝，位置也超出5米容差。' if not accepted else '目标被接受，但位置超出5米容差。'
        else:reason='该负例被正确拒绝；这不代表方法总体通过。' if good else '目标不在当前格内，却被模型误接受。'
        target_city=fold['wrong_target_donor'] if arm=='Wrong' else c['query']
        return dict(id=i,case_id=c['case_id'],city=fold['fold'],reference=c['reference'],cell=c['cell'],relation=c['relation'],
            method=method,seed=number,arm=arm,bounds=c['crop'],mpp=c['mpp'],metric_cell_m=(c['crop'][2]-c['crop'][0])*c['mpp'],
            prediction=pred['point'],truth_xy=truth,error_m=error,score=pred['score'],tau=tau,accepted=accepted,joint_correct=bool(good),reason=reason,
            heat=[z[r*7:r*7+7] for r in range(7)],heat_range=[min(z[:49]),max(z[:49])],logits=z,
            original_target_city=c['query'],display_target_city=target_city,
            target_is_explanatory=arm in ('Disabled','Mean','NoTarget'),readonly=True,new_model_calls=0,
            source_metric_width_m=d['source_table'][c['reference']]['metric_extent_m'][0],
            provenance=dict(seal_sha256=SEAL_SHA,prediction_file=path,prediction_sha256=refs[path],protocol_sha256=b['protocol']['sha256'],calibration_file_sha256=refs[BATCH+'/固定校准阈值.json']))

    def image(self,case,method,seed,arm,slot):
        if slot not in ('target','original','current'):raise EvidenceUnavailable('图像槽位无效')
        record=self.detail(case,method,seed,arm);refs,b,s,decision,cal,d,cases=self._bundle()
        if slot=='current':source=d['source_table'][record['reference']]['aerial']
        else:source=d['source_table'][record['original_target_city'] if slot=='original' else record['display_target_city']]['ground']
        raw=self._raw(source['path'],source['sha256'])
        if slot!='current':return raw
        from PIL import Image
        with Image.open(BytesIO(raw)) as im:
            with im.crop(tuple(record['bounds'])) as crop:
                output=BytesIO();crop.save(output,format='PNG');return output.getvalue()

    def export(self,case,method,seed,arm):
        return dict(format='geonav-spatial-evidence-v1',record=self.detail(case,method,seed,arm),
                    note='只读CV041封存静态结果；真值仅供观察者，0新模型调用/训练/导航。')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['catalog','detail','export']);p.add_argument('--root',default=str(Path(__file__).resolve().parents[3]));p.add_argument('--case',default='0');p.add_argument('--method',default='Spatial');p.add_argument('--seed',default='0');p.add_argument('--arm',default='Full');a=p.parse_args()
    reader=SpatialEvidenceReader(a.root);v=reader.catalog() if a.action=='catalog' else getattr(reader,a.action)(a.case,a.method,a.seed,a.arm)
    print(json.dumps(v,ensure_ascii=False,allow_nan=False))
