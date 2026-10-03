"""Preserve v1 and export a clearer legend in a separate figure revision."""
from pathlib import Path
from io import BytesIO
import sys
import warnings
import json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from documents import summarize_radial_confirmation_data as base
from documents import stage_figures as graphics
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Patch
import numpy as np
RUN,FIG,META,QA=base.RUN,base.FIG,base.META,base.QA
read,digest,rel,write=base.read,base.digest,base.rel,base.write
STEM=base.STEM+'_v2'
MANIFEST=FIG/'绘图数据/径向保护新区域确认准备_最终_v2.json'
CAPTION=FIG/'径向保护新区域确认准备图注_v2.md'
NOTE=RUN/'科研图排版修订_v2.md'


def main():
    original=read(base.MANIFEST)
    for mapping in (original['source_sha256'],original['other_outputs_sha256']):
        for n,h in mapping.items():assert digest(ROOT/n)==h,n
    for item in original['figures'][0]['files']:assert digest(ROOT/item['path'])==item['sha256']
    assert not MANIFEST.exists() and not CAPTION.exists() and not NOTE.exists()
    assert not any((FIG/'数据结果图'/f'{STEM}.{ext}').exists() for ext in ('png','pdf','svg'))
    audit,verdict=original['audit'],original['verdict']
    regions=original['selection']['regions']
    old=read(META/'旧151足迹.json');consumed=read(META/'已消费14连续区域.json')['regions']
    graphics.style();fig,axes=plt.subplots(1,2,figsize=(12.5,5.5),gridspec_kw={'width_ratios':[1.2,1]})
    colors=['#7C8792','#D55E00','#009E73']
    for j,group in enumerate((old,consumed,regions)):
        for r in group:
            x0,y0,x1,y1=np.asarray(r['bounds_m'])/1000
            axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=colors[j],edgecolor=colors[j],alpha=.4 if j==0 else .7,linewidth=.6))
    all_bounds=np.asarray([r['bounds_m'] for r in old+consumed+regions])/1000
    axes[0].set_xlim(all_bounds[:,0].min()-4,all_bounds[:,2].max()+4)
    axes[0].set_ylim(all_bounds[:,1].min()-4,all_bounds[:,3].max()+4)
    axes[0].set_aspect('equal',adjustable='box');axes[0].set_xlabel('东向投影坐标（公里）');axes[0].set_ylabel('北向投影坐标（公里）')
    axes[0].grid(alpha=.45);titles=[axes[0].set_title('(a) 历史足迹与预留新区',loc='left')]
    counts=[250,250,250,400,400,400];xs=np.arange(6)
    axes[1].bar(xs,counts,color=['#0072B2']*3+['#56B4E9']*3,edgecolor='#243746',linewidth=.6,zorder=3)
    for x,n in zip(xs,counts):axes[1].text(x,n+12,str(n),ha='center',fontsize=10)
    axes[1].set_xticks(xs,['长距离\nC12—16','接缝\nC8—12','内部\nC8—12','跨源\n邻接','同源\n邻接','非邻接\n控制'])
    axes[1].set_ylim(0,470);axes[1].set_ylabel('冻结条目数');axes[1].set_axisbelow(True);axes[1].grid(axis='y',alpha=.55)
    titles.append(axes[1].set_title('(b) 确认任务与几何探针',loc='left'))
    handles=[Patch(facecolor=c,label=label,alpha=.7) for c,label in zip(colors,['原151影像','已消费14区域','预留10新区'])]
    handles.append(Patch(facecolor='#0072B2',label='750导航任务'))
    handles.append(Patch(facecolor='#56B4E9',label='1200几何探针'))
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=5,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,top=.79,bottom=.23,wspace=.35)
    fig.text(.5,.055,f"最小间距：至旧151 {audit['minimum_old_region_gap_m']/1000:.2f} km；至旧14区 {audit['minimum_consumed14_region_gap_m']/1000:.2f} km；新区间 {audit['minimum_new_region_pair_gap_m']/1000:.2f} km。",ha='center',fontsize=9)
    fig.text(.5,.02,'数据与空间复核通过；真实导航尚未启动，不产生新SR。Sat2Cap预训练地理覆盖未知。',ha='center',fontsize=9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert all(not legend.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in titles)
    files=[]
    for ext in ('png','pdf','svg'):
        b=BytesIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fig.savefig(b,format=ext,dpi=300,bbox_inches='tight')
        assert not any('Glyph' in str(w.message) for w in caught)
        p=FIG/'数据结果图'/f'{STEM}.{ext}'
        with p.open('xb') as f:f.write(b.getvalue())
        files.append(dict(path=rel(p),sha256=digest(p)))
    plt.close(fig)
    caption=original['figures'][0]['caption']+' 修订版将蓝色750导航任务与浅蓝色1200几何探针分别标注，绿色仅表示预留新区；数值、坐标、数据、候选与协议全部保持。'
    write(CAPTION,'# 新区域确认准备图注 v2\n\n'+caption+'\n')
    write(NOTE,f'# 科研图49排版修订 v2\n\n保留原图、报告和来源清单；只区分右图几何探针颜色并补齐图例，不改变任何数据或结论。正式引用本修订版图注和图片。\n\n[原报告](新区域确认数据准备报告.md) / [修订图注](../../../../绘图/径向保护新区域确认准备图注_v2.md)。\n\n![图49修订版](../../../../绘图/数据结果图/{STEM}.png)\n')
    item=dict(original['figures'][0],id=STEM,files=files,caption=caption)
    final=dict(original)
    final['figures']=[item]
    final['figure_only_revision']=True
    final['source_sha256']=dict(original['source_sha256'])
    final['source_sha256'][rel(base.MANIFEST)]=digest(base.MANIFEST)
    final['source_sha256'][rel(Path(__file__))]=digest(Path(__file__))
    final['other_outputs_sha256']=dict(original['other_outputs_sha256'])
    final['other_outputs_sha256'].update({rel(p):digest(p) for p in (CAPTION,NOTE)})
    final['reproduction_script']=rel(Path(__file__))
    final['script_sha256']=digest(Path(__file__))
    write(MANIFEST,final)
    print(dict(figure_revision=STEM,original_preserved=True),flush=True)


if __name__=='__main__':main()
