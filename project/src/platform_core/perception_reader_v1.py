"""Read-only CV031 evidence API. No torch, encoder, prediction, action or storage writes."""
from io import BytesIO
import hashlib
import json
from pathlib import Path

BATCH='DATA/processed_data/OpenCVL/CCVPE局部感知开发筛查_v1'
IMAGES='DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
SEAL_SHA='10f250ca4d72c5f631bd9a61255e3fb946c6b640542381665a05cebd622134ff'
ARMS=('Full','Disabled','Mean','Wrong')


class EvidenceUnavailable(ValueError):pass


class PerceptionReader:
    def __init__(self,root):self.root=Path(root).resolve()

    def _raw(self,relative,expected):
        p=(self.root/relative).resolve()
        if not p.is_relative_to(self.root):raise EvidenceUnavailable('证据路径超出工作区')
        try:data=p.read_bytes()
        except OSError as e:raise EvidenceUnavailable('证据或图像尚未随平台携带') from e
        if hashlib.sha256(data).hexdigest()!=expected:raise EvidenceUnavailable('证据绑定已变化，请恢复封存文件后刷新')
        return data

    def _bundle(self):
        seal=json.loads(self._raw(BATCH+'/阶段封存.json',SEAL_SHA))
        refs={e['path']:e['sha256'] for e in seal['files']}
        def read(name):return json.loads(self._raw(BATCH+'/'+name,refs[BATCH+'/'+name]))
        b=read('冻结执行绑定.json');s=read('开发筛查汇总.json');cal=read('唯一校准阈值.json')
        cases=b['cases']['development_screen']
        return refs,b,s,cal,cases

    def assets(self):
        refs,b,s,cal,cases=self._bundle()
        files={BATCH+'/'+n for n in ('阶段封存.json','冻结执行绑定.json','开发筛查汇总.json','唯一校准阈值.json')}
        for n in range(21,101):files.update((BATCH+f'/预测_{n:03d}.json',BATCH+f'/roi_{n:03d}.npz'))
        files.update(e['path'] for e in b['images'])
        return sorted(files)

    def catalog(self):
        refs,b,s,cal,cases=self._bundle()
        return dict(batch='CV-031',title='局部感知开发筛查',status='未通过 · 已收束',readonly=True,
            navigation_admitted=False,source_count=2,calibration_sources=2,total_outputs=100,development_outputs=80,
            tau=cal['tau'],metrics=s['metrics'],contrasts=s['contrasts'],gates=s['gates'],baselines=s['position_baselines'],
            always_reject_BA=s['always_reject_BA'],
            cases=[dict(id=i,source=c['query_evaluator_only'].rsplit('_',1)[-1],reference=c['reference_evaluator_only'].rsplit('_',1)[-1],
                        cell=c['public_cell'],relation=c['evaluator_only']['relation']) for i,c in enumerate(cases)],
            arms=list(ARMS),binding_sha=SEAL_SHA,
            limitations=['2个已消费开发来源，不能视作独立确认。','BA衡量接受/拒绝，定位还需单独满足5米门槛。',
                '当前是固定目标照片与局部航拍格的静态证据，不提供自主方向。',
                '完整父图仅有工程等价记录，缺少定量定位对照。'])

    @staticmethod
    def _select(cases,case,arm):
        if type(case) is not str or not case.isascii() or not case.isdecimal() or not 0<=int(case)<len(cases) or arm not in ARMS:
            raise EvidenceUnavailable('请选择有效案例与目标条件')
        return int(case),cases[int(case)]

    def detail(self,case,arm):
        refs,b,s,cal,cases=self._bundle();i,c=self._select(cases,case,arm)
        number=21+i*4+ARMS.index(arm);name=BATCH+f'/预测_{number:03d}.json'
        pred=json.loads(self._raw(name,refs[name]));assert pred['case_id']==c['case_id'] and pred['arm']==arm
        row=next(r for r in s['detailed'] if r['case_id']==c['case_id'] and r['arm']==arm)
        import numpy as np
        roi=pred['ROI'];raw=self._raw(roi['path'],roi['sha256'])
        if refs.get(roi['path'])!=roi['sha256']:raise EvidenceUnavailable('ROI双重绑定不一致')
        with np.load(BytesIO(raw),allow_pickle=False) as z:heat=z['heat'].astype(float).tolist()
        return dict(id=i,arm=arm,source=c['query_evaluator_only'].rsplit('_',1)[-1],reference=c['reference_evaluator_only'].rsplit('_',1)[-1],
            cell=c['public_cell'],relation=c['evaluator_only']['relation'],bounds=c['current_crop_bounds'],
            truth_xy=c['evaluator_only']['query_xy_native'] if c['evaluator_only']['relation']=='inside' else None,
            prediction=pred['predicted_xy_native'],score=pred['score'],tau=cal['tau'],accepted=row['accepted'],error_m=row['error_m'],joint_correct=row['joint_correct'],
            heat=heat,heat_origin=pred['heat_roi_origin_yx'],heat_range=[min(map(min,heat)),max(map(max,heat))],
            condition_metrics=s['metrics'][arm],provenance=dict(prediction_sha=refs[name],ROI_sha=roi['sha256'],seal_sha=SEAL_SHA),
            interpretation='正确拍摄点应在当前格内；需接受且定位误差≤5米。' if c['evaluator_only']['relation']=='inside' else '目标拍摄点在当前格外，应拒绝。' if c['evaluator_only']['relation']=='outside' else '地面目标来自另一来源，应拒绝。',
            original_case=c['case_id'],readonly=True)

    def image(self,case,arm,slot):
        refs,b,s,cal,cases=self._bundle();i,c=self._select(cases,case,arm)
        if slot not in ('target','original','current'):raise EvidenceUnavailable('图像类型无效')
        if slot=='current':rel=IMAGES+'/'+c['parent_file_evaluator_only']
        else:
            goal=c
            if slot=='target' and arm=='Wrong':goal=next(x for x in cases if x['query_evaluator_only']==c['wrong_query_evaluator_only'])
            rel=IMAGES+'/'+goal['target_file_evaluator_only']
        image_refs={e['path']:e['sha256'] for e in b['images']}
        raw=self._raw(rel,image_refs[rel])
        if slot!='current':return raw
        from PIL import Image
        with Image.open(BytesIO(raw)) as im:
            tile=im.crop(c['current_crop_bounds']);out=BytesIO();tile.save(out,format='PNG');tile.close();return out.getvalue()

    def export(self,case,arm):
        result=self.detail(case,arm);result.pop('heat')
        return dict(format='geonav-perception-evidence-v1',record=result,note='CV031封存预测的只读导出；未执行模型或导航。')
