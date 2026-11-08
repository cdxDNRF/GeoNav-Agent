"""Pinned CV042 camera-goal task catalog and observer image rendering.

No model/feature lookup. Full references remain private to this data adapter.
"""
from collections import OrderedDict,Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading

BATCH='DATA/processed_data/OpenCVL/拍摄点开发任务队列_v1'
SEAL_SHA='9f7e4bb8ce6e53a455e353d9d6b9b6584b00da2e69f1a9748f9b74fbd4733268'
META=('来源与配对清单_v2.json','导航任务清单_v2.json','导航真值_仅监督评测_v2.json')


def digest(data):return hashlib.sha256(data).hexdigest()


class CapturepointCatalog:
    def __init__(self,root):
        self.root=Path(root).resolve();self.cache=OrderedDict();self.lock=threading.RLock();self.aerial_decodes=0
    def _checked(self,relative,expected):
        p=(self.root/relative).resolve()
        if not p.is_relative_to(self.root) or '视频' in p.relative_to(self.root).parts:raise ValueError('拍摄点资产路径无效')
        try:raw=p.read_bytes()
        except OSError:raise ValueError('拍摄点资产缺失，请按搬迁清单恢复；不会下载或重跑') from None
        if digest(raw)!=expected:raise ValueError('拍摄点资产绑定变化，请恢复原件')
        return raw
    def _bundle(self):
        seal=json.loads(self._checked(BATCH+'/阶段封存_v2.json',SEAL_SHA));refs={e['path']:e['sha256'] for e in seal['files']}
        def get(name):return json.loads(self._checked(BATCH+'/'+name,refs[BATCH+'/'+name]))
        sources=get(META[0])['pairs'];tasks=get(META[1])['tasks'];truth=get(META[2])['labels']
        if len(sources)!=9 or len(tasks)!=192 or len(truth)!=192:raise ValueError('拍摄点固定任务分母变化')
        sm={s['pair_id']:s for s in sources};tm={t['task_id']:t for t in tasks};ym={t['task_id']:t for t in truth}
        if len(sm)!=9 or len(tm)!=192 or set(tm)!=set(ym):raise ValueError('拍摄点任务身份不唯一或真值缺失')
        return refs,sm,tm,ym
    def bindings(self):
        refs,sources,tasks,truth=self._bundle();out={BATCH+'/阶段封存_v2.json':SEAL_SHA}
        out.update({BATCH+'/'+n:refs[BATCH+'/'+n] for n in META})
        for s in sources.values():
            if s['navigation_eligible']:
                for e in s['images'].values():out[e['path']]=e['sha256']
        return out
    def task(self,task_id,validate_assets=False):
        if not isinstance(task_id,str) or len(task_id)>160:raise ValueError('请选择已登记拍摄点任务')
        refs,sources,tasks,truth=self._bundle()
        if task_id not in tasks:raise ValueError('请选择已登记拍摄点任务，窗外来源没有可达导航题')
        t=tasks[task_id];s=sources[t['pair_id']];y=truth[task_id]
        if not s['navigation_eligible'] or not s['corrected_camera_label']['inside'] or t['grid_size']!=5 or t['budget']!=10:
            raise ValueError('拍摄点任务协议无效')
        if y['goal_index']!=s['corrected_camera_label']['index'] or t['start_index']==y['goal_index']:
            raise ValueError('拍摄点目标身份无效')
        if validate_assets:
            for e in s['images'].values():self._checked(e['path'],e['sha256'])
        return dict(task=t,source=s,truth=y,input_sha256={
            BATCH+'/阶段封存_v2.json':SEAL_SHA,**{BATCH+'/'+n:refs[BATCH+'/'+n] for n in META},
            **{e['path']:e['sha256'] for e in s['images'].values()}})
    def public_task(self,task_id):
        b=self.task(task_id);t,s=b['task'],b['source']
        return self._public(task_id,t,s)
    @staticmethod
    def _public(task_id,t,s):
        return dict(task_id=task_id,city=s['city'],role=t['role'],pair_id=t['pair_id'],start=t['start_index'],grid=5,budget=10,
            cell_size_m=s['resolution']*s['native_sizes']['aerial'][0]/5,
            coordinate_status='数据集修正拍摄点；绝对测绘精度未独立确认',target_modality='ground',
            automatic_navigation=False,model_calls=0)
    def catalog(self):
        refs,sources,tasks,truth=self._bundle()
        rows=[self._public(key,t,sources[t['pair_id']]) for key,t in tasks.items()]
        return dict(title='拍摄点任务 · 开发池',task_count=192,source_count=9,navigation_sources=8,outside_reference_sources=1,
            cities=sorted({s['city'] for s in sources.values()}),roles=dict(Counter(t['role'] for t in tasks.values())),
            tasks=rows,formal_confirmation_sources=0,automatic_navigation=False,model_calls=0,
            scope='已消费开发来源；人工记录不计入Agent SR',dataset_id='opencvl-camera-development-v1')
    def image(self,task_id,cell):
        b=self.task(task_id);s=b['source']
        if cell=='target':
            e=s['images']['ground'];return self._checked(e['path'],e['sha256'])
        if type(cell) is not int or not 0<=cell<25:raise ValueError('航拍格无效')
        e=s['images']['aerial'];raw=self._checked(e['path'],e['sha256'])
        with self.lock:
            from PIL import Image
            key=e['sha256']
            if key not in self.cache:
                with Image.open(BytesIO(raw)) as image:
                    image.load()
                    if list(image.size)!=s['native_sizes']['aerial']:raise ValueError('原生图像尺寸与登记不符')
                    self.cache[key]=image.convert('RGB').copy();self.aerial_decodes+=1
                while len(self.cache)>2:
                    _,im=self.cache.popitem(last=False);im.close()
            self.cache.move_to_end(key);im=self.cache[key];r,c=divmod(cell,5)
            crop=im.crop((c*im.width//5,r*im.height//5,(c+1)*im.width//5,(r+1)*im.height//5))
            # Display-only rendering; the native half-open crop is unchanged.
            stream=BytesIO();crop.save(stream,format='PNG');return stream.getvalue()
    def close(self):
        with self.lock:
            for im in self.cache.values():im.close()
            self.cache.clear()
