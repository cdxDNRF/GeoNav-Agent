"""CV-002 narrow engineering repair. Keep v1 inputs/queue/failure; explicitly verify PIL."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from tools.prepare_crossview_probe_v1 import OUT,MODEL,read,digest,SOURCE_PATHS
from data.mmgag_views_v1 import clean_rgb,ground_pixels,numeric_window
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def main():
    assert not (OUT/'冻结协议.json').exists(),'Already frozen; do not rerun'
    binding=read(OUT/'准备开始绑定.json')
    for e in binding['sources']+binding['models']+binding['raw_used']:assert digest(ROOT/e['path'])==e['sha256'],e['path']
    old=read(OUT/'工程核验.json');assert old['geometry']['passed'] and old['architecture_compatible']
    assert not (OUT/'执行状态.json').exists(),'Must repair before any model call'
    pairs=read(OUT/'划分与任务.json')['pairs'];pp=read(MODEL/'preprocessor_config.json')
    from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil
    processor=CLIPImageProcessorPil.from_pretrained(str(MODEL),local_files_only=True)
    deltas=[]
    for p in pairs:
        image=clean_rgb(ROOT/p['ground_path'],ground=True)
        a=ground_pixels(image,pp);b=processor(images=image.copy(),return_tensors='np')['pixel_values'][0]
        deltas.append(float(np.abs(a-b).max()))
    repair={'task':'CV-002','replaces':'processor equivalence field of failed 工程核验.json only; geometry and 16-pair split unchanged',
        'implicit_backend':'CLIPImageProcessor in transformers 5.9.0 inherits TorchvisionBackend',
        'explicit_reference_backend':'CLIPImageProcessorPil; matches original PIL image resize/crop contract',
        'original_max_delta':old['processor_max_delta'],'fixed_max_delta':max(deltas),'tolerance_unchanged':1e-5,
        'all_deltas':deltas,'passed':max(deltas)<=1e-5,'model_forward_calls':0,
        'new_sources':[{'path':p,'sha256':digest(ROOT/p)} for p in ['project/src/tools/prepare_crossview_probe_v2.py','project/src/tests/test_crossview_processors_v2.py']]}
    save('核验/预处理修复_v2.json',repair)
    assert repair['passed'],'PIL equivalence still failed; no encoding'
    for p in pairs:p['aerial_window_paths']=[x.relative_to(ROOT).as_posix() for x in numeric_window(ROOT/p['aerial_directory'],p['goal_index_evaluator_only'],old['geometry']['first_index'])]
    save('冻结执行任务.json',{'pairs':pairs,'first_numeric_index':old['geometry']['first_index']})
    protocol={'task':'CV-002','date':'2026-10-16','pairs':16,'aerial_inputs':400,'ground_inputs':16,'conditions':['Full','Mean','Wrong'],
      'dev_groups':10,'reserved_groups':20,'scoring':'L2-normalized 512D cosine; float64 diagnostic arithmetic',
      'mean':'mean of 16 normalized development ground vectors, L2 normalized','wrong':'fixed different development group target; no extra encoded images',
      'ties':'sort by decreasing score then SHA256(CV-002/tie/v1:pair_id:cell_index); never by truth',
      'ground_preprocessing':'EXIF transpose, PIL shortest-edge 224 BICUBIC, center crop 224, official CLIP mean/std; matches explicit HF PIL reference',
      'aerial_preprocessing':'PIL square224 BICUBIC; bound Sat2Cap mean/std',
      'batch_size':8,'maximum_forward_attempts_inputs':416,'cuda_allocation_limit_bytes':2*1024**3,
      'device_rule':'CUDA if free memory >=2560MiB at run start, otherwise CPU; no post-score device/model selection',
      'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'output_limit_bytes':16*1024**2,
      'candidate_gates':{'all_16_complete':True,'full_top1_minimum':4,'full_minus_mean_minimum':2,'full_minus_wrong_minimum':2,'groups_net_better_than_both_minimum':3},
      'stopping':'one candidate; no tuning or reserved encoding after results; Top-k is not SR',
      'ground_model':'models/clip_vit_b32_public_v1/pytorch_model.bin; vision_model.* + visual_projection.* only, strict load',
      'aerial_model':'models/Sat2Cap/model.safetensors; frozen',
      'runtime_input':'pixels only, metadata sanitized; candidate bank allowed only to offline evaluator, never online agent',
      'geometry':'image coordinates, numeric first index is row; code-based (5,5) label not externally surveyed; no metric SG',
      'authority':'user started CV-002 and explicitly approved one official CLIP download',
      'engineering_repair':'核验/预处理修复_v2.json; no threshold, model, source or task changes'}
    save('冻结协议.json',protocol)
    files=sorted(OUT.glob('*.json'))+[OUT/'核验/预处理修复_v2.json']
    save('首forward前绑定.json',{'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p)} for p in files],
      'sources':binding['sources']+repair['new_sources'],'models':binding['models'],'raw_used':binding['raw_used']})
    print(json.dumps({'frozen':True,'pairs':16,'dev_groups':10,'reserved_groups':20,'axis':'row','fixed_preprocessor_max_delta':max(deltas),'model_calls':0}))
if __name__=='__main__':main()
