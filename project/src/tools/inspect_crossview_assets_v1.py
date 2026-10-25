"""Bounded public Sample4Geo metadata/code audit, without model downloads.

Fetched source is parsed as text/AST only; never imported or executed.
Only derived facts, line references and content hashes are saved.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角公开资产准入_v1'
DEV=('mmgag_062','mmgag_073','mmgag_079','mmgag_039','mmgag_081')
MAX_REQUESTS=24;MAX_BYTES=1024**2


def allowed(url):
    parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.username or parsed.password:return False
    if parsed.hostname=='api.github.com':return parsed.path.startswith(('/repos/Skyy93/Sample4Geo','/repos/eceo-epfl/ConGeo'))
    if parsed.hostname=='raw.githubusercontent.com':return parsed.path.startswith(('/Skyy93/Sample4Geo/','/eceo-epfl/ConGeo/')) and parsed.path.endswith(('.py','.txt','.md'))
    if parsed.hostname=='drive.google.com':return parsed.path.startswith('/drive/folders/') and not parsed.query
    return False


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if not allowed(newurl):raise ValueError('只允许公开代码/文件夹元数据，拒绝认证或下载跳转')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def extract_config(text):
    tree=ast.parse(text);values={}
    for node in ast.walk(tree):
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try:values[node.target.id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):pass
    selected={k:values[k] for k in ('model','img_size','batch_size','checkpoint_start','normalize_features') if k in values}
    size=values.get('img_size')
    if isinstance(size,int) and '224 / 1232' in text:
        selected['satellite_height_width']=[size,size]
        selected['ground_height_width']=[round(224/1232*(size*2)),size*2]
    selected['windows_num_workers_branch']='os.name' in text and "'nt'" in text
    selected['pretrained_true_in_original_entry']=bool(re.search(r'pretrained\s*=\s*True',text))
    selected['non_strict_load_in_original_entry']='strict=False' in text.replace(' ','')
    selected['map_location_in_original_entry']='map_location' in text
    selected['literal_reference_lines']={k:next((i for i,line in enumerate(text.splitlines(),1) if re.search(r'\b'+re.escape(k)+r'\b',line)),None) for k in selected if k in values}
    return selected


def inspect():
    start=OUT/'公开核查开始.json';result=OUT/'公开资产核查.json'
    if start.exists() or result.exists():raise FileExistsError('Public collection started; inspect saved receipts instead of blindly rerunning')
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    start.write_text(json.dumps({'source_sha256':source_sha,'candidate':'Sample4Geo/CVUSA','max_public_requests':MAX_REQUESTS,'weights_downloads':0,'models':0},indent=2),encoding='utf-8')
    receipts=[];opener=build_opener(SafeRedirect())
    def fetch(url):
        if len(receipts)>=MAX_REQUESTS or not allowed(url):raise ValueError('Public request scope/budget exceeded')
        item={'url':url,'method':'GET','model_payload':False};receipts.append(item)
        body=None
        try:
            with opener.open(Request(url,headers={'User-Agent':'GeoNav-public-metadata-review/1.0','Accept':'application/json,text/plain,text/html'}),timeout=18) as response:
                item.update(status=response.status,final_url=response.geturl(),content_type=response.headers.get('Content-Type',''))
                if not allowed(response.geturl()):raise ValueError('Unexpected redirect')
                if not any(x in item['content_type'] for x in ('json','text/','javascript')):raise ValueError('Non-text response blocked before body read')
                raw=response.read(MAX_BYTES+1)
                if len(raw)>MAX_BYTES:raise ValueError('Text response exceeds1MiB')
                item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());body=raw.decode('utf-8')
        except (OSError,ValueError,HTTPError,URLError) as error:
            item.update(error_type=type(error).__name__,error=str(error)[:240],status=getattr(error,'code',item.get('status')))
        with (OUT/'公开请求回执.json').open('w',encoding='utf-8') as stream:json.dump(receipts,stream,ensure_ascii=False,indent=2)
        print('Public metadata request',len(receipts),item.get('status'),item.get('error_type','ok'),flush=True)
        return body

    sample={};code={}
    text=fetch('https://api.github.com/repos/Skyy93/Sample4Geo')
    if text:
        metadata=json.loads(text);sample['github_license_metadata']=metadata.get('license');sample['archived']=metadata.get('archived')
    text=fetch('https://api.github.com/repos/Skyy93/Sample4Geo/commits/main')
    revision=json.loads(text)['sha'] if text else None;sample['revision']=revision
    if revision:
        text=fetch('https://api.github.com/repos/Skyy93/Sample4Geo/git/trees/'+revision+'?recursive=1')
        tree=json.loads(text).get('tree',[]) if text else []
        sample['tree_paths']=[x['path'] for x in tree]
        sample['explicit_license_files']=[x['path'] for x in tree if Path(x['path']).name.lower().startswith(('license','copying'))]
        for rel in ('README.md','eval_cvusa.py','sample4geo/model.py','sample4geo/transforms.py','sample4geo/dataset/cvusa.py','requirements.txt'):
            code[rel]=fetch('https://raw.githubusercontent.com/Skyy93/Sample4Geo/'+revision+'/'+rel)
        if code.get('eval_cvusa.py'):sample['configuration']=extract_config(code['eval_cvusa.py'])
        model=code.get('sample4geo/model.py') or ''
        sample['model_code_facts']={'uses_timm_create_model':'timm.create_model' in model,'num_classes_zero':'num_classes=0' in model.replace(' ',''),'shared_single_image_forward':'def forward' in model,'no_import_executed':True}
        requirements=code.get('requirements.txt') or ''
        sample['declared_dependencies']=[x.strip() for x in requirements.splitlines() if x.strip() and not x.startswith('#')]
        sample['official_weight_folder_urls']=sorted(set(re.findall(r'https://drive\.google\.com/drive/folders/[A-Za-z0-9_-]+',code.get('README.md') or '')))
        if sample['official_weight_folder_urls']:
            folder=sample['official_weight_folder_urls'][0];listing=fetch(folder)
            sample['weight_folder_page_readable']=listing is not None
            if listing:
                sample['weight_folder_name_mentions']=sorted(set(re.findall(r'(?i)(?:cvusa|cvact|vigor|u1652|university)[A-Za-z0-9_. -]{0,40}',listing)))[:40]
                sample['weight_archive_name_mentions']=sorted(set(re.findall(r'[A-Za-z0-9_.-]+\.(?:zip|pth|safetensors)',listing)))[:30]
    congeo={}
    text=fetch('https://api.github.com/repos/eceo-epfl/ConGeo/commits/main')
    c_revision=json.loads(text)['sha'] if text else None;congeo['revision']=c_revision
    if c_revision:
        text=fetch('https://raw.githubusercontent.com/eceo-epfl/ConGeo/'+c_revision+'/README.md')
        congeo['README_mentions_orientation_and_FoV']=bool(text and 'FoV' in text and 'orientation' in text)
        congeo['README_weight_links']=sorted(set(re.findall(r'https://(?:drive\.google\.com|huggingface\.co)[^\s)<>]+',text or '')))
    protocol=json.loads((ROOT/'DATA/processed_data/MM-GAG/同任务首批图像轴向预检_v1/首像素冻结协议.json').read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'DATA/processed_data/MM-GAG/同任务首批图像轴向预检_v1/逐图解码与字节回执.json').read_text(encoding='utf-8'))
    old_receipts={x['path']:x for x in prior['receipts']};dimensions=[]
    for item in protocol['sources']:
        if item['source_id'] not in DEV:continue
        g=old_receipts[item['ground']['path']];dimensions.append({'source_id':item['source_id'],'ground_width_height':g['sanitized_size'],'ground_aspect_ratio':g['sanitized_size'][0]/g['sanitized_size'][1],'fov_degrees_known':False,'target_semantics':'capture_point_proxy','original_pixels_read':False})
    assert len(dimensions)==5
    result_data={'task':'CV-022','candidate':'Sample4Geo/CVUSA','sample4geo':sample,'congeo_mechanism_reference_only':congeo,
                 'known_development_dimension_metadata':dimensions,'requests':receipts,'public_fetches':len(receipts),
                 'full_third_party_sources_saved':False,'source_code_executed':False,'model_weights_downloaded':0,
                 'raw_images_read':0,'model_forward':0,'training':0,'navigation':0,'confirmation_pixels_features_or_evaluation':0,
                 'runtime_parameter_count_or_VRAM_measured':False,'public_link_is_not_a_weight_integrity_proof':True}
    with result.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result_data,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return {'completed_metadata_inspection':True,'fetches':len(receipts),'sample_revision':revision,'model_downloads':0,'configuration':sample.get('configuration'),'license_files':sample.get('explicit_license_files'),'folder_readable':sample.get('weight_folder_page_readable')}


if __name__=='__main__':print(json.dumps(inspect(),ensure_ascii=False))
