"""Independent literal/Drive metadata parsing, not a model or source executor."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角公开资产准入_v1'


def literal_configuration(text):
    tree=ast.parse(text);result={}
    wanted={'model','img_size','batch_size','checkpoint_start','normalize_features'}
    for node in ast.walk(tree):
        if isinstance(node,ast.AnnAssign):pairs=[(node.target,node.value)]
        elif isinstance(node,ast.Assign):pairs=[(target,node.value) for target in node.targets]
        else:continue
        for target,value in pairs:
            if isinstance(target,ast.Name) and target.id in wanted:
                try:result[target.id]={'value':ast.literal_eval(value),'line':node.lineno}
                except (ValueError,TypeError):pass
    return result


def drive_entries(text):
    match=re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'\\])*)'",text)
    if not match:return []
    decoded=ast.literal_eval("'"+match.group(1)+"'")
    tree=json.loads(decoded);entries=[]
    for row in tree[0] or []:
        if not isinstance(row,list) or len(row)<4 or not isinstance(row[2],str):continue
        entries.append({'id':row[0],'name':row[2],'mime_type':row[3],
                        'numeric_metadata_uninterpreted':{str(i):x for i,x in enumerate(row) if type(x) in (int,float) or (isinstance(x,str) and x.isdecimal())}})
    return entries


def run():
    dest=OUT/'公开事实另实现核验.json';begin=OUT/'公开事实核验开始.json'
    if dest.exists() or begin.exists():raise FileExistsError('Fact verification already started')
    prior=json.loads((OUT/'公开资产补充核查.json').read_text(encoding='utf-8'))
    revision=prior['sample4geo']['revision'];assert revision=='498c6511478053ddb4a43f24a9f4942fab238f09'
    begin.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'maximum_requests':2,'weights_downloads':0}),encoding='utf-8')
    records=[]
    def get(url):
        with urlopen(Request(url,headers={'User-Agent':'GeoNav-public-review/1.0'}),timeout=18) as response:
            if not response.headers.get('Content-Type','').startswith('text/'):raise ValueError('Only text metadata')
            raw=response.read(1024**2+1)
            if len(raw)>1024**2:raise ValueError('Metadata cap')
            records.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'status':response.status})
            return raw.decode('utf-8')
    source=get('https://raw.githubusercontent.com/Skyy93/Sample4Geo/'+revision+'/eval_cvusa.py')
    config=literal_configuration(source)
    folder=prior['sample4geo']['weights_folder_urls'][0];page=get(folder)
    entries=drive_entries(page)
    sizes=sorted(set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:GB|MB|KB|bytes)\b',page)))
    assert config['model']['value']==prior['sample4geo']['configuration']['model']
    assert config['img_size']['value']==384 and config['batch_size']['value']==128
    real_names=[x['name'] for x in entries]
    result={'task':'CV-022','passed':True,'configuration_literals_with_actual_lines':config,
            'ground_resize_derived_height_width':[round(224/1232*768),768],
            'public_drive_entries':entries,'visible_size_mentions':sizes,
            'regex_mentions_are_not_authoritative_filenames':True,
            'literal_filename_pretrained_zip_verified':'pretrained.zip' in real_names,
            'code_tree_was_parsed_not_executed':True,'requests':records,
            'checkpoint_tensor_SHA_unverified':True,'model_or_dataset_downloads':0,
            'navigation':0,'training':0,'same_agent_independent_parser_not_person_review':True}
    with dest.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


if __name__=='__main__':
    result=run();print(json.dumps({k:result[k] for k in ['passed','configuration_literals_with_actual_lines','public_drive_entries','visible_size_mentions','literal_filename_pretrained_zip_verified']},ensure_ascii=False))
