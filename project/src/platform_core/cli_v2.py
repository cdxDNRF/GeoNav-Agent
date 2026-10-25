"""Run the same experiment engine without a browser or HTTP service.

Use an idle or separate store. The ownership lock prevents parallel writers.
"""
import argparse
import json
from pathlib import Path
import time

from .engine_v2 import STORE
from .service_v2 import Workbench
from .storage_v1 import StoreLock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--store',help='Separate/idle persisted experiment store')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('catalog');sub.add_parser('list');sub.add_parser('capabilities')
    state=sub.add_parser('state');state.add_argument('id')
    create=sub.add_parser('create');create.add_argument('spec',help='JSON file with the same specification used by the GUI')
    run=sub.add_parser('run');run.add_argument('spec');run.add_argument('--scope',choices=['episode','queue'],default='queue')
    run.add_argument('--timeout',type=int,default=600)
    args=p.parse_args();root=Path(args.root).resolve();store=Path(args.store).resolve() if args.store else root/STORE
    ownership=StoreLock(store)
    w=None
    try:
        w=Workbench(root,store)
        if args.command=='catalog':result=w.catalog()
        elif args.command=='list':result=w.listing()
        elif args.command=='capabilities':result=w.capabilities()
        elif args.command=='state':result=w.state(args.id)
        else:
            spec=json.loads(Path(args.spec).read_text('utf-8-sig'));result=w.define(spec)
            if args.command=='run':
                if not 1<=args.timeout<=3600:raise ValueError('timeout需为1到3600秒')
                w.control(result['id'],'run_'+args.scope,interval_ms=0)
                deadline=time.monotonic()+args.timeout
                while True:
                    result=w.state(result['id'])
                    if result['status'] not in ('running','pausing','cancelling'):break
                    if time.monotonic()>=deadline:
                        w.control(result['id'],'pause');raise TimeoutError('已请求暂停；保留已保存进度，不盲目重跑')
                    time.sleep(.05)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    finally:
        if w:w.shutdown()
        ownership.close()


if __name__=='__main__':main()
