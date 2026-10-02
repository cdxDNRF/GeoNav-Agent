"""Preserve first render and put the final scientific legend above the bars."""
from pathlib import Path
import sys,json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_continued_validation import ROOT,FIG,CONFIRM,read,write,digest,audit_required
from documents import stage_figures as graphics
from matplotlib import pyplot as plt


def main():
    audit_required(CONFIRM,'主对照汇总.json');s=read(CONFIRM/'主对照汇总.json');r=s['averages'];name='35_继续训练独立源图双网格确认'
    manifest=FIG/'绘图数据/继续训练独立源图确认_v1.json';old=read(manifest);before=old['figures'][0]
    for ext in ('png','pdf','svg'):
        src=FIG/f'数据结果图/{name}.{ext}';dst=FIG/f'审阅/继续训练独立源图确认_初稿.{ext}'
        with dst.open('xb') as f:f.write(src.read_bytes())
    write(FIG/'审阅/继续训练独立源图确认_初稿清单.json',old)
    graphics.style();graphics.ITEMS.clear();fig,axes=plt.subplots(1,2,figsize=(8.2,4.2));colors=[graphics.GRAY,graphics.BLUE]
    for j,(field,scale,ylabel) in enumerate([('sr',100,'SR：成功率（%）'),('sg',1,'SG：平均终点距离（格）')]):
        ax=axes[j];maximum=0
        for i,k in enumerate(('5','10')):
            for arm,color,offset in zip(('M0','Continue5'),colors,(-.17,.17)):
                z=r[k][arm];value=z[field+'_mean']*scale;weights=[x*scale for x in z[field+'_by_seed']];maximum=max(maximum,max(weights));x=i+offset
                ax.bar(x,value,width=.3,color=color,edgecolor='#34424F',lw=.7,zorder=2)
                for seed,w in enumerate(weights):ax.scatter(x+(seed-1)*.045,w,marker=['o','^','s'][seed],facecolor='white',edgecolor='#23313B',s=24,lw=.7,zorder=4)
                ax.text(x,max(weights)+(.9 if j==0 else .025),f'{value:.2f}' if j==0 else f'{value:.3f}',ha='center',fontsize=9.3)
        ax.set_xticks([0,1],['5×5 / B10','10×10 / B20']);ax.set_ylabel(ylabel);ax.set_title('(a) 新源图成功率' if j==0 else '(b) 所有终局距离',loc='left');ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,109 if j==0 else maximum*1.25+.1)
        if j==0:
            ax.set_yticks([0,20,40,60,80,100]);ax.legend([ax.containers[0],ax.containers[1]],['原M0','Continue5'],loc='upper center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.085,right=.99,bottom=.23,top=.89,wspace=.32)
    fig.text(.5,.085,'20新SwissView源文件，两协议共用；每方法/网格500任务×3最终权重。',ha='center',fontsize=9)
    fig.text(.5,.03,'白色标记为seed0/1/2，非置信区间；10×10提高同足迹密度，非更大实际面积。',ha='center',fontsize=9)
    graphics.save(fig,FIG/'数据结果图',name,before['caption'],before['sources'],before['protocol'])
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/refine_continued_source_figure.py'
    updated=dict(old,figures=[item],script_sha256=digest(Path(__file__)),first_render_manifest_sha256=digest(FIG/'审阅/继续训练独立源图确认_初稿清单.json'),
        change='legend moved into empty upper space; all numerical data/axes/labels/budget unchanged')
    manifest.write_text(json.dumps(updated,ensure_ascii=False,indent=2,sort_keys=True)+'\n','utf8')
    catalog=read(FIG/'图表来源清单.json');catalog['figures']=[item if x['id']==name else x for x in catalog['figures']]
    (FIG/'图表来源清单.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n','utf8')
    print(dict(final_figure=name,first_render_preserved=True,numerical_data_changed=False),flush=True)


if __name__=='__main__':main()
