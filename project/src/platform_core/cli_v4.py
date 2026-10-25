"""Current ground CLI: same recoverable core, plus read-only recovery reports."""
import argparse
import json
from pathlib import Path
from .ground_tasks_v2 import GroundStore
from .storage_v1 import StoreLock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[3]));parser.add_argument('--store')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('catalog','tasks','list','recovery'):sub.add_parser(cmd)
    p=sub.add_parser('reference');p.add_argument('case');p.add_argument('--name',default='地面开发目标')
    p=sub.add_parser('upload');p.add_argument('image');p.add_argument('--name',required=True);p.add_argument('--source',required=True)
    p=sub.add_parser('create');p.add_argument('spec')
    for cmd in ('run','state','rerun','export'):
        p=sub.add_parser(cmd);p.add_argument('id')
        if cmd=='export':p.add_argument('destination')
    args=parser.parse_args();root=Path(args.root).resolve();store=Path(args.store).resolve() if args.store else root/'DATA/processed_data/平台实验/工作台_v1'
    lock=StoreLock(store);core=None
    try:
        core=GroundStore(root,store)
        if args.command=='catalog':value=core.catalog.public()
        elif args.command=='tasks':value=core.list_tasks()
        elif args.command=='list':value=core.list_runs()
        elif args.command=='recovery':value=core.recovery_report()
        elif args.command=='reference':value=core.register_reference(args.case,args.name)
        elif args.command=='upload':
            image=Path(args.image)
            if image.stat().st_size>10*1024**2:raise ValueError('图片超过10MiB')
            value=core.register_upload(image.read_bytes(),{'name':args.name,'source':args.source})
        elif args.command=='create':
            spec=json.loads(Path(args.spec).read_text(encoding='utf-8-sig'))
            if spec.get('temporary'):raise ValueError('单次CLI不能持续持有临时任务，请使用正式保存或持久Python服务')
            value=core.define(spec)
        elif args.command=='run':value=core.execute(args.id)
        elif args.command=='state':value=core.state(args.id)
        elif args.command=='rerun':value=core.rerun(args.id)
        else:
            value=core.state(args.id)
            with Path(args.destination).open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
            value={'exported':True,'id':args.id}
        print(json.dumps(value,ensure_ascii=False,allow_nan=False))
    finally:
        if core is not None:core.shutdown()
        lock.close()


if __name__=='__main__':main()
