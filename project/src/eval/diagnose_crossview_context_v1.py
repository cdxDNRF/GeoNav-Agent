"""Post-hoc cached-only coarse association. Never changes frozen CV-002 verdict."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def unit(a):
    a=np.asarray(a,np.float64);return a/np.linalg.norm(a,axis=-1,keepdims=True)
def main():
    dest=OUT/'核验/事后区域尺度诊断.json';assert not dest.exists(),'Do not overwrite exploratory output'
    inputs=['地面特征.npy','航拍特征.npy','冻结执行任务.json','感知对照汇总.json']
    initial={n:sha(OUT/n) for n in inputs}
    pairs=read('冻结执行任务.json')['pairs'];ground=unit(np.load(OUT/'地面特征.npy',allow_pickle=False));aerial=np.load(OUT/'航拍特征.npy',allow_pickle=False).reshape(16,25,512)
    targets={'center_patch_bank':unit(np.stack([aerial[i,p['goal_index_evaluator_only']] for i,p in enumerate(pairs)])),
             'full_window_mean_bank':unit(aerial.mean(axis=1,dtype=np.float64))}
    findings={}
    for name,bank in targets.items():
        scores=ground@bank.T;rows=[]
        for i,p in enumerate(pairs):
            rank=sorted(range(16),key=lambda j:(-float(scores[i,j]),hashlib.sha256(f'CV-002/context-tie/v1:{i}:{j}'.encode()).hexdigest()))
            place=rank.index(i)+1
            rows.append({'pair_id':p['pair_id'],'correct_source_rank':place,'ranking':rank,'scores':scores[i].tolist(),
                  'top1_group_match':pairs[rank[0]]['group_id']==p['group_id']})
        # Independently recompute scalar products with fsum, not matrix multiplication.
        delta=0
        for i in range(16):
            for j in range(16):delta=max(delta,abs(float(scores[i,j])-math.fsum(float(a)*float(b) for a,b in zip(ground[i],bank[j]))))
        assert delta<1e-12
        findings[name]={'planned':16,'bank_size':16,'source_top1':sum(r['correct_source_rank']==1 for r in rows),
          'source_top5':sum(r['correct_source_rank']<=5 for r in rows),'median_source_rank':float(np.median([r['correct_source_rank'] for r in rows])),
          'top1_group_matches':sum(r['top1_group_match'] for r in rows),'scalar_recompute_delta':delta,'rows':rows}
    result={'task':'CV-002','status':'exploratory_after_frozen_candidate_failure','source_code_sha256':sha(__file__),'input_sha256':initial,
      'findings':findings,'new_model_forwards':0,'new_navigation':0,'changes_original_gate':False,
      'limitations':['16 source bank differs from within-window 25-tile task; do not subtract scores between tasks',
       'several sources share provisional groups; not 16 independent geographic regions',
       'center bank uses evaluator labels; full-window mean uses all25 cells; neither provides an online policy with a legitimate full-map input',
       'coarse association does not establish direction prediction or navigation success']}
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    local=read('感知对照汇总.json');fig,axes=plt.subplots(1,2,figsize=(10.6,4.8));colors=['#386CB0','#B8C6D4'];width=.32
    x=np.arange(3)
    axes[0].bar(x-width/2,[local['top1_counts'][k] for k in ['Full','Mean','Wrong']],width,color=colors[0],label='Top-1')
    axes[0].bar(x+width/2,[local['top5_counts'][k] for k in ['Full','Mean','Wrong']],width,color=colors[1],label='Top-5')
    axes[0].set_xticks(x,['Full','Mean','Wrong']);axes[0].set_title('Frozen local task: 25 tile candidates\nCandidate gates FAILED',fontsize=10)
    keys=list(findings);x=np.arange(2)
    axes[1].bar(x-width/2,[findings[k]['source_top1'] for k in keys],width,color=colors[0],label='Source Top-1')
    axes[1].bar(x+width/2,[findings[k]['source_top5'] for k in keys],width,color=colors[1],label='Source Top-5')
    axes[1].set_xticks(x,['Single center tile','Mean of all 25 tiles']);axes[1].set_title('Exploratory region task: 16 sources\nFull-window information is offline only',fontsize=10)
    for ax in axes:
        ax.set_ylim(0,16);ax.set_yticks(range(0,17,4));ax.set_ylabel('Correct queries / 16');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
        for bars in ax.containers:ax.bar_label(bars,padding=3,fontsize=9)
    fig.suptitle('Cross-view association: frozen result and post-hoc scale diagnosis',fontsize=12)
    fig.text(.5,.02,'Different tasks and information budgets; no navigation SR, no new forward, no acceptance upgrade.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.92))
    png=OUT/'核验/冻结结果与尺度诊断.png';pdf=OUT/'核验/冻结结果与尺度诊断.pdf'
    assert not png.exists() and not pdf.exists();fig.savefig(png,dpi=170);fig.savefig(pdf);plt.close(fig)
    import fitz
    d=fitz.open(pdf);d[0].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(OUT/'核验/科研图PDF独立渲染.png')
    for n,s in initial.items():assert sha(OUT/n)==s,n
    print(json.dumps({k:{a:v for a,v in x.items() if a!='rows'} for k,x in findings.items()},ensure_ascii=False))
if __name__=='__main__':main()
