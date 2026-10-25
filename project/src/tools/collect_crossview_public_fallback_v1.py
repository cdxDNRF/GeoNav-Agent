"""One public-page fallback after unauthenticated GitHub API rate limiting.

No authenticated endpoint, repository clone, code execution or model payload.
Retains derived facts/URLs/hashes only; failed first collection is immutable.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from project.src.tools.inspect_crossview_assets_v1 import extract_config

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角公开资产准入_v1'


def run():
    output=OUT/'公开资产补充核查.json';begin=OUT/'公开补查开始.json'
    if output.exists() or begin.exists():raise FileExistsError('Fallback consumed; inspect existing receipts')
    original=json.loads((OUT/'公开资产核查.json').read_text(encoding='utf-8'))
    assert original['public_fetches']==3
    begin.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prior_requests':3,'separate_network_diagnosis_requests':3,'max_additional_requests':15,'weights_downloaded':0}),encoding='utf-8')
    records=[]
    def fetch(url):
        u=urlsplit(url)
        valid=u.scheme=='https' and not u.query and not u.username and ((u.hostname=='github.com' and u.path in ('/Skyy93/Sample4Geo','/eceo-epfl/ConGeo')) or (u.hostname=='raw.githubusercontent.com' and u.path.startswith(('/Skyy93/Sample4Geo/','/eceo-epfl/ConGeo/')) and u.path.endswith(('.py','.md','.txt'))) or (u.hostname=='drive.google.com' and u.path.startswith('/drive/folders/')))
        if not valid or len(records)>=15:raise ValueError('Public metadata fallback scope/budget')
        receipt={'url':url,'model_payload':False};records.append(receipt)
        try:
            with urlopen(Request(url,headers={'User-Agent':'GeoNav-public-review/1.0'}),timeout=18) as response:
                final=urlsplit(response.geturl())
                if final.hostname!=u.hostname or (u.hostname=='drive.google.com' and not final.path.startswith('/drive/folders/')):raise ValueError('Authentication/download redirect refused')
                mime=response.headers.get('Content-Type','')
                if not any(t in mime for t in ('text/','json','javascript')):raise ValueError('Non-text body refused')
                data=response.read(1024**2+1)
                if len(data)>1024**2:raise ValueError('Text response over1MiB')
                receipt.update(status=response.status,content_type=mime,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                return data.decode('utf-8')
        except (OSError,ValueError,HTTPError,URLError) as error:
            receipt.update(status=getattr(error,'code',None),error=type(error).__name__)
            return None
        finally:
            (OUT/'公开补查请求回执.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

    html=fetch('https://github.com/Skyy93/Sample4Geo')
    match=re.search(r'currentOid[^0-9a-f]+([0-9a-f]{40})',html or '')
    revision=match.group(1) if match else None;facts={'revision':revision,'repo_page_readable':html is not None}
    facts['root_license_link_seen']=bool(re.search(r'/Skyy93/Sample4Geo/blob/[^"<>]+/(?:LICENSE|COPYING)',html or '',re.I))
    code={}
    if revision:
        for rel in ('README.md','eval_cvusa.py','sample4geo/model.py','sample4geo/transforms.py','sample4geo/dataset/cvusa.py','requirements.txt'):
            code[rel]=fetch('https://raw.githubusercontent.com/Skyy93/Sample4Geo/'+revision+'/'+rel)
        facts['configuration']=extract_config(code['eval_cvusa.py']) if code.get('eval_cvusa.py') else None
        model=code.get('sample4geo/model.py') or '';compact=re.sub(r'\s+','',model)
        facts['feature_model']={'timm_create_model':'timm.create_model' in model,'num_classes_zero':'num_classes=0' in compact,'shared_model_attribute':'self.model(' in compact,'code_executed':False}
        facts['declared_dependencies']=[x.strip() for x in (code.get('requirements.txt') or '').splitlines() if x.strip() and not x.startswith('#')]
        facts['weights_folder_urls']=sorted(set(re.findall(r'https://drive\.google\.com/drive/folders/[A-Za-z0-9_-]+',code.get('README.md') or '')))
        if facts['weights_folder_urls']:
            listing=fetch(facts['weights_folder_urls'][0]);facts['weight_directory_metadata_readable']=listing is not None
            if listing:
                facts['weight_archive_names_seen']=sorted(set(re.findall(r'[A-Za-z0-9_.-]+\.(?:zip|pth|safetensors)',listing)))[:30]
                facts['folder_dataset_mentions']=sorted(set(re.findall(r'(?i)\b(?:cvusa|cvact|vigor|university|u1652)\b',listing)))
                facts['drive_folder_child_ids']=list(dict.fromkeys(re.findall(r'folders/([A-Za-z0-9_-]{20,})',listing)))[:10]
    html=fetch('https://github.com/eceo-epfl/ConGeo');m=re.search(r'currentOid[^0-9a-f]+([0-9a-f]{40})',html or '')
    reference={'revision':m.group(1) if m else None,'selected_as_candidate':False}
    if m:
        text=fetch('https://raw.githubusercontent.com/eceo-epfl/ConGeo/'+m.group(1)+'/README.md')
        reference['README_FoV_and_orientation']=bool(text and 'fov' in text.lower() and 'orientation' in text.lower())
        reference['README_weight_links']=sorted(set(re.findall(r'https://(?:drive\.google\.com|huggingface\.co)[^\s)<>]+',text or '')))
    installed={}
    for name in ('torch','torchvision','timm','albumentations','opencv-python','numpy'):
        try:installed[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:installed[name]=None
    report={'task':'CV-022','candidate':'Sample4Geo/CVUSA only','sample4geo':facts,
            'congeo_mechanism_reference':reference,'installed_package_metadata_only':installed,
            'local_known_development_dimensions':original['known_development_dimension_metadata'],
            'requests':records,'additional_fetches':len(records),'prior_collection_requests':3,'network_diagnosis_requests':3,
            'full_sources_retained':False,'source_execution':0,'model_weights_or_datasets_downloaded':0,
            'model_forward':0,'training':0,'navigation':0,'confirmation_pixels_features_evaluation':0,
            'checkpoint_bytes_SHA_or_parameter_count_measured':False,'runtime_VRAM_measured':False}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return {'completed':True,'fetches':len(records),'revision':revision,'config':facts.get('configuration'),
            'root_license_link_seen':facts['root_license_link_seen'],'weight_directory':facts.get('weight_directory_metadata_readable'),
            'weight_archive_names_seen':facts.get('weight_archive_names_seen'),'dependencies':installed}


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
