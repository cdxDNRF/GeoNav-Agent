"""Single4.13 transfer builder: explicit assets, no history trees or chained builds."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile
from project.src.webapp.binding_v3 import execution_bindings,source_closure
from project.src.platform_core.method_catalog_v1 import EVIDENCE_FILES
from project.src.platform_core.ground_tasks_v1 import CATALOG
from project.src.tools.spatial_evidence_transfer_v1 import verified_spatial_assets,require_spatial_guide_links,GUIDE
from project.src.tools.perception_transfer_v1 import verified_perception_assets,verified_guide_assets,require_guide_links,verify_expected

ROOT=Path(__file__).resolve().parents[3]
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(destination,with_encoder=False,with_workspace=False,with_masa=False,manifest_only=False,with_ground_encoders=False):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('搬迁清单/包已存在，不能覆盖历史交付')
    inputs,sources=execution_bindings(ROOT)
    files=set(inputs)|set(sources)|set(source_closure(ROOT,[
        'project/src/webapp/workbench_server_v20.py','project/src/platform_core/cli_v2.py','project/src/platform_core/cli_v4.py']))
    files.update(source_closure(ROOT,['project/src/tools/ground_live_worker_v1.py','project/src/tools/ground_frame_worker_v1.py']))
    for folder in ['project/src/webapp/static_v10','project/src/webapp/static_ground_v1',CATALOG]:
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(['project/src/webapp/workflow_polish_v1.js','project/src/webapp/workflow_polish_v1.css','project/src/webapp/trial_workflow_v1.js','project/src/webapp/trial_workflow_v1.css','project/src/webapp/ground_layout_v2.css','project/src/webapp/ground_recovery_v1.js','project/src/tools/audit_ground_release_v2.py','project/src/webapp/static_v7/assets/geonav-terrain.png',
        'project/src/tools/package_workbench_v24.py','project/src/platform_core/README.md','平台/启动平台.ps1','平台/启动平台.cmd',
        '平台/实验工作台_v4/README.md','平台/实验工作台_v4/设计与验收.md','平台/实验工作台_v4/方案与模型_v1/README.md',
        '平台/实验工作台_v4/试跑保存与同题对照_v1/README.md','平台/实验工作台_v4/试跑保存与同题对照_v1/使用说明.md'])
    files.update(['project/src/webapp/workbench_server_v13.py','project/src/webapp/scene_workspace_v2.js','project/src/webapp/scene_workspace_v1.html','project/src/webapp/scene_workspace_v1.js','project/src/webapp/scene_workspace_v1.css',
        '平台/实验工作台_v4/场景配置与手动探索_v1/README.md','平台/实验工作台_v4/场景交互复查与修复_v1/README.md'])
    files.update(['project/src/webapp/ground_live_v3.js','project/src/webapp/ground_live_v1.html','project/src/webapp/ground_live_v1.js','project/src/webapp/ground_live_v1.css','project/src/platform_core/ground_live_assets_v1.json','平台/实验工作台_v4/地面证据四项修复与回归_v1/README.md','平台/实验工作台_v4/地面目标实时证据接入_v1/README.md','平台/实验工作台_v4/地面目标实时证据接入_v1/README_发布补充.md'])
    files.update(['project/src/webapp/ground_walk_v1.html','project/src/webapp/ground_walk_v1.css','project/src/webapp/ground_walk_v1.js','project/src/platform_core/ground_walk_scene_v1.json','平台/实验工作台_v4/地面目标连续观测与人工探索_v1/README.md'])
    trusted=verified_perception_assets(ROOT)
    trusted.update(verified_spatial_assets(ROOT))
    files.update(source_closure(ROOT,['project/src/tools/spatial_evidence_transfer_v1.py']))
    files.update(['project/src/webapp/spatial_evidence_v1.'+ext for ext in ('html','js','css')])
    files.add(GUIDE)
    trusted.update(verified_guide_assets(ROOT))
    files.update(trusted)
    files.update(source_closure(ROOT,['project/src/webapp/workbench_server_v18.py','project/src/tools/perception_transfer_v1.py']))
    files.add('平台/实验工作台_v4/搬迁证据修复与独立验证_v1/搬迁使用与恢复.md')
    files.update(['project/src/webapp/perception_reader_v1.html','project/src/webapp/perception_reader_v1.js','project/src/webapp/perception_reader_v1.css'])
    if with_ground_encoders:
        assets=json.loads((ROOT/'project/src/platform_core/ground_live_assets_v1.json').read_text(encoding='utf-8'))['models']
        for e in assets:
            if digest(ROOT/e['path'])!=e['sha256']:raise ValueError('真实编码资产SHA变化')
            files.add(e['path'])
    for name in EVIDENCE_FILES:
        if not with_masa and any(name.startswith(prefix) for prefix in (
            'DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/','DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/')):continue
        if (ROOT/name).is_file():files.add(name)
    if with_masa:
        from project.src.webapp.masa_workbench_v1 import MasaCatalog
        c=MasaCatalog(ROOT)
        if set(c.datasets)!={'masa-s2','masa-s3'}:raise ValueError('缺少完整Masa队列')
        for did in c.datasets:files.update(c.bindings(did))
    if with_encoder:
        for name in ('config.json','model.safetensors'):
            p=ROOT/'models/Sat2Cap'/name
            if not p.is_file():raise ValueError('缺少固定编码器工件')
            files.add(p.relative_to(ROOT).as_posix())
    if with_workspace:
        workspace=ROOT/'DATA/processed_data/平台实验/工作台_v1'
        for p in workspace.glob('experiments/*/进度.json'):
            if json.loads(p.read_text(encoding='utf-8'))['status'] in ('running','pausing','cancelling'):
                raise ValueError('请先暂停/结束实验，再打包工作台数据')
        for p in workspace.glob('ground_diagnostics/*/进度.json'):
            if json.loads(p.read_text(encoding='utf-8'))['status']=='running':
                raise ValueError('请先完成地面诊断，再打包工作台数据')
        for p in workspace.rglob('*'):
            relative=p.relative_to(workspace)
            if any(part.startswith('.pending-') for part in relative.parts):continue
            if p.is_file() and p.suffix not in ('.log','.lock','.tmp'):files.add(p.relative_to(ROOT).as_posix())
    guide_links=require_guide_links(ROOT,files)
    guide_links.extend(require_guide_links(ROOT,files,'平台/实验工作台_v4/搬迁证据修复与独立验证_v1/搬迁使用与恢复.md',2))
    guide_links.extend(require_spatial_guide_links(ROOT,files))
    for name in files:
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or p.name.startswith('.env') or name.startswith(('DATA/raw_data/','.agents/','node_modules/','GOMAA-Geo/')):
            raise ValueError('非法或缺失打包资产：'+name)
    total=sum((ROOT/name).stat().st_size for name in files)
    if total>(1600 if with_ground_encoders else 600)*1024**2:raise ValueError('搬迁资产超过所选600/1600MiB资源上限')
    launcher=(ROOT/'平台/启动平台.ps1').read_text(encoding='utf-8-sig')
    if 'workbench_server_v20' not in launcher or 'geonav-workbench-v4.13' not in launcher:raise ValueError('当前启动器未绑定本发布版本')
    dependencies={}
    for pkg in ('numpy','Pillow','torch','transformers','scipy','tqdm','psutil'):
        try:dependencies[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:pass
    manifest={'version':'geonav-transfer-v24','application_version':'4.13','server':'project.src.webapp.workbench_server_v20',
        'files':{name:digest(ROOT/name) for name in sorted(files)},'trusted_evidence_sha256':trusted,'guide_local_links':guide_links,'uncompressed_bytes':total,'dependencies':dependencies,
        'with_encoder':with_encoder,'workspace_included':with_workspace,'masa_sealed_queues_included':with_masa,
        'ground_samples_self_contained':(ROOT/CATALOG/'样例封存.json').is_file(),'ground_task_registry':True,'free_scene_registry':True,'scene_saved_snapshots_readonly':True,'scene_fault_isolation':True,'human_control_separate_metrics':True,
        'ground_saved_score_diagnostics':True,'trial_save_without_rerun':True,'same_task_fork_and_comparison':True,'ground_record_recovery':True,'historical_audit_separate_from_deployment':True,'paired_ground_manual_walk':True,'paired_scene_raw_images_included':False,'spatial_head_development_evidence':True,'spatial_head_candidate_passed':False,'local_perception_readonly_evidence':True,'local_perception_screen_passed':False,'ground_navigation_available':False,'ground_live_interface':True,'ground_live_encoder_assets_included':with_ground_encoders,'ground_live_sample_requires_original_development_data':True,
        'python_runtime_included':False,'credentials_included':False,'raw_data_included':False,
        'scope':'当前单一4.13入口及必要资产；地面目标支持实时单对编码证据与已存诊断；骨干仅显式with-ground-encoders携带，无导航能力或新确认成绩'}
    verify_expected(ROOT,trusted)
    if any(manifest['files'][name]!=expected for name,expected in trusted.items()):
        raise ValueError('计算清单期间封存资产发生变化')
    if manifest_only:
        with destination.open('x',encoding='utf-8',newline='\n') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
    else:
        with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for name in sorted(files):z.write(ROOT/name,name)
            z.writestr('搬迁清单.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            z.writestr('requirements-workbench.txt','\n'.join(p+'=='+v.split('+')[0] for p,v in dependencies.items())+'\n')
            z.writestr('开始使用.txt','GeoNav4.13\n配置已有兼容Python后，双击平台/启动平台.cmd。依赖版本见requirements-workbench.txt；不含运行时或凭据。\n地面照片支持登记/已存诊断；实时编码需已有CLIP与Sat2Cap，或显式--with-ground-encoders带入。地面导航未准入。\n')
        with zipfile.ZipFile(destination) as z:
            assert z.testzip() is None
            for name,h in manifest['files'].items():
                with z.open(name) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=h or (name in trusted and actual!=trusted[name]):raise ValueError('打包期间资产变化：'+name)
    return {'passed':True,'manifest_only':manifest_only,'files':len(files),'uncompressed_bytes':total,
        'original_evidence_verified':True,'current_guide_links_complete':True,'zip_created':not manifest_only,'spatial_head_development_evidence':True,'spatial_head_candidate_passed':False,'local_perception_readonly_evidence':True,'local_perception_screen_passed':False,'ground_navigation_available':False,'credentials_included':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');p.add_argument('--with-encoder',action='store_true');p.add_argument('--with-workspace',action='store_true');p.add_argument('--with-masa',action='store_true');p.add_argument('--manifest-only',action='store_true');p.add_argument('--with-ground-encoders',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.destination,a.with_encoder,a.with_workspace,a.with_masa,a.manifest_only,a.with_ground_encoders),ensure_ascii=False))
