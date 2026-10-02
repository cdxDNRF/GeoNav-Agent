"""Fix crowded draft labels; preserve draft and all numerical evidence."""
from pathlib import Path
import json
import shutil
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from documents.summarize_evidence_compact import ROOT,OUT,G,FIG,NAME,read,digest
from documents import stage_figures as graphics
from matplotlib import pyplot as plt


def main():
    manifest=FIG/'绘图数据/证据账本简化接口闭环_v2.json';prior=read(manifest)
    if 'layout_revision' in prior:raise ValueError('final figure already generated')
    for ext in ('png','pdf','svg'):
        dst=FIG/f'审阅/{NAME}_初稿.{ext}'
        if dst.exists():raise ValueError('draft exists')
        shutil.copyfile(FIG/f'数据结果图/{NAME}.{ext}',dst)
    shutil.copyfile(manifest,FIG/'审阅/证据账本简化接口闭环_初稿清单.json')
    s=read(G/'对照汇总_220条计数勘误.json');graphics.style();graphics.ITEMS.clear()
    fig,axes=plt.subplots(1,2,figsize=(7.8,4.35));arms=('M0','M1','M2','M3','M4')
    labels=['M0\n冻结策略','M1\n账本规划','M2\n单智能体','M3\n反思','M4\n双角色']
    colors=[graphics.GRAY,graphics.TEAL,graphics.BLUE,graphics.ORANGE,'#CC79A7']
    for panel,(field,scale,unit) in enumerate([('sr',100,'SR：成功率（%）'),('mean_sg_all_episodes',1,'SG：平均终点距离（格）')]):
        ax=axes[panel];v=[s['arms'][a]['metrics'][field]*scale for a in arms]
        ax.bar(range(5),v,color=colors,width=.6,edgecolor='#34424F',linewidth=.7,zorder=2);high=[]
        for i,a in enumerate(arms):
            runs=[m[field]*scale for m in s['arms'][a]['by_repeat'].values()];high.append(max([v[i]]+runs))
            for rep,y in enumerate(runs):ax.scatter(i+(rep-1)*.11,y,marker=['o','^','s'][rep],s=24,facecolor='white',edgecolor='#26333C',lw=.7,zorder=4)
            ax.text(i,high[-1]+(2 if panel==0 else .06),f'{v[i]:.1f}' if panel==0 else f'{v[i]:.3f}',ha='center',fontsize=10)
        ax.set_xticks(range(5),labels);ax.set_ylabel(unit);ax.grid(axis='y');ax.set_axisbelow(True)
        ax.set_ylim(0,108 if panel==0 else max(high)*1.25+.1)
        if panel==0:ax.set_yticks([0,20,40,60,80,100])
        ax.set_title('(a) 计划题成功率' if panel==0 else '(b) 全部轨迹终点距离',loc='left')
    fig.subplots_adjust(left=.08,right=.99,bottom=.23,top=.89,wspace=.30)
    fig.text(.5,.075,'20个已见源文件/20题；10×10，B=20；本地权重0。',ha='center',fontsize=9)
    fig.text(.5,.025,'白色圆/三角/方形：三轮API重复，非训练seed或置信区间；M0/M1各执行一次。',ha='center',fontsize=9)
    old=prior['figures'][0];graphics.save(fig,FIG/'数据结果图',NAME,old['caption'],old['sources'],old['protocol'])
    item=graphics.ITEMS[0];item['reproduction_script']='project/src/documents/finalize_evidence_compact_figure.py'
    prior['figures']=[item];prior['layout_revision']=dict(reason='abbreviated x-axis labels prevent overlap',
        numerical_data_unchanged=True,draft_manifest='绘图/审阅/证据账本简化接口闭环_初稿清单.json')
    manifest.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({'final_layout':True,'draft_preserved':True,'metrics_unchanged':True})


if __name__=='__main__':main()
