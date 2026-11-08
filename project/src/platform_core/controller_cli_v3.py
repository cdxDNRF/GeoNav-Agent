"""Locked, GUI-independent unified single/queue execution on existing assets."""
import argparse
import json
from pathlib import Path
from .storage_v1 import StoreLock


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['catalog','run','queue','resume','export'])
    p.add_argument('--store',type=Path);p.add_argument('--dataset',default='masa-s2');p.add_argument('--task-key')
    p.add_argument('--controller',choices=['frontier','M0'],default='M0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--tasks',type=Path);p.add_argument('--batch-id');p.add_argument('--restart-incomplete',action='store_true')
    p.add_argument('--save',action='store_true');p.add_argument('--max-actions',type=int,default=220);p.add_argument('--format',choices=['json','csv'],default='json')
    args=p.parse_args(argv);root=Path(__file__).resolve().parents[3]
    if args.mode=='catalog':
        from project.src.webapp.masa_workbench_v1 import MasaCatalog
        from .capturepoint_catalog_v1 import CapturepointCatalog
        aerial=MasaCatalog(root);camera=CapturepointCatalog(root)
        print(json.dumps(dict(datasets=[dict(id='camera-development',tasks=camera.catalog()['task_count'],controllers=['manual','frontier']),
            *[dict(id=k,tasks=len(v['tasks']),controllers=['manual','frontier','M0']) for k,v in aerial.datasets.items() if k=='masa-s2']],model_calls=0),ensure_ascii=False));return 0
    if args.store is None or not args.store.is_dir():raise ValueError('请选择已存在且未被服务使用的独立工作目录')
    if not 1<=args.max_actions<=220:raise ValueError('单次CLI动作上限1..220')
    # Lock BEFORE importing/constructing Workbench: its legacy startup can mark
    # interrupted experiments, so a conflicting CLI must never get that far.
    lock=StoreLock(args.store);w=None;owner='f'*32
    try:
        from .service_v15 import Workbench
        from project.src.webapp.binding_v3 import source_closure
        w=Workbench(root,args.store);w.sources.update(source_closure(root,['project/src/platform_core/controller_cli_v3.py']))
        if args.mode=='run':
            if not args.task_key:raise ValueError('请选择任务编号')
            value=w.controllers.create(dict(name='CLI统一单题',dataset_id=args.dataset,task_key=args.task_key,controller=args.controller,seed=args.seed),owner)
            while value['status'] not in ['completed','cancelled','failed'] and value['step']<args.max_actions:
                value=w.controllers.step(value['id'],owner,kind=value['kind'])
            if args.save:
                if value['status'] not in ['completed','cancelled','failed']:value=w.controllers.finish(value['id'],owner,value['kind'])
                value=w.controllers.save_run(value['id'],owner,value['kind'])
            print(json.dumps(value,ensure_ascii=False));return 0
        if args.mode=='queue':
            if not args.save or args.tasks is None:raise ValueError('队列会持久保存，需显式--save及--tasks JSON编号数组')
            if args.tasks.stat().st_size>512*1024:raise ValueError('任务编号清单过大')
            tasks=json.loads(args.tasks.read_text('utf-8-sig'))
            value=w.controller_batches.create(dict(name='CLI小队列',dataset_id=args.dataset,task_keys=tasks,controllers=['M0','frontier'],seed=args.seed),owner)
            bid=value['id']
        else:
            if not args.batch_id:raise ValueError('请选择已保存队列')
            bid=args.batch_id;value=w.controller_batches.state(bid,owner)
        if args.mode=='export':
            output=w.controller_batches.export(bid,owner,args.format);print(output if isinstance(output,str) else json.dumps(output,ensure_ascii=False));return 0
        for _ in range(args.max_actions):
            if value['status'] in ['completed','cancelled','failed']:break
            value=w.controller_batches.step(bid,owner,args.restart_incomplete)
        if value['status'] not in ['completed','cancelled','failed']:value=w.controller_batches.pause(bid,owner)
        print(json.dumps(value,ensure_ascii=False));return 0
    finally:
        if w:w.shutdown()
        lock.close()


if __name__=='__main__':raise SystemExit(main())
