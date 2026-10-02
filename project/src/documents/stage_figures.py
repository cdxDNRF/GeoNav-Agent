"""Reproducible scientific plots and editable diagrams from the closeout snapshot."""
from pathlib import Path
import json
import hashlib
import html
import xml.etree.ElementTree as ET
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'绘图'
DATA = OUT/'绘图数据/阶段结果_v1.json'
PLOTS = OUT/'数据结果图'
ARCH = OUT/'Agent架构图'
REVIEW = OUT/'审阅'
GRAY, BLUE, ORANGE, TEAL = '#7C8792', '#0072B2', '#D55E00', '#009E73'
ITEMS = []


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10,
        'axes.labelsize':10.5, 'axes.titlesize':11.5, 'xtick.labelsize':9.5,
        'ytick.labelsize':9.5, 'legend.fontsize':9.5, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.7, 'grid.color':'#DBE1E6',
        'grid.linewidth':.6, 'axes.unicode_minus':False, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white'})


def save(fig, folder, name, caption, refs, scope, editable=None):
    folder.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    files=[]
    for fmt in ['pdf','svg','png']:
        p=folder/f'{name}.{fmt}'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.savefig(p,dpi=300)
        missing=[str(w.message) for w in caught if 'Glyph' in str(w.message)]
        if missing:
            raise ValueError('Font glyph missing: '+missing[0])
        files.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)))
    plt.close(fig)
    if editable:
        files.append(dict(path=editable.relative_to(ROOT).as_posix(),sha256=digest(editable)))
    ITEMS.append(dict(id=name,caption=caption,protocol=scope,sources=refs,
                      files=files,reproduction_script='project/src/documents/stage_figures.py'))


def frames(axes, percent=False):
    for ax in np.atleast_1d(axes).flat:
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        if percent:
            ax.set_ylim(0,105)
            ax.set_yticks([0,20,40,60,80,100])


def bars(ax, labels, means, color_list, decimals=2, seeds=None, percent=False):
    x=np.arange(len(labels))
    ax.bar(x,means,width=.60,color=color_list,edgecolor='#34424F',linewidth=.6,zorder=2)
    for i,v in enumerate(means):
        ymax=max(seeds[i]) if seeds is not None else v
        ax.text(i,max(v,ymax)+(.9 if percent else .025),f'{v:.{decimals}f}',
                ha='center',va='bottom',fontsize=10.2,fontweight='bold' if color_list[i]==BLUE else 'normal')
    if seeds is not None:
        for s,(off,marker) in enumerate([(-.13,'o'),(0,'^'),(.13,'s')]):
            ax.scatter(x+off,[v[s] for v in seeds],marker=marker,s=22,
                       facecolor='white',edgecolor='#1D2B34',linewidth=.8,zorder=4)
    ax.set_xticks(x,labels)
    ax.tick_params(axis='x',length=0)


def latest(d):
    rows=d['edge']['rows'];labels=['探索/置零','真实目标','均值目标','错误目标']
    colors=[GRAY,BLUE,ORANGE,TEAL]
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.55))
    frames(axes)
    bars(axes[0],labels,[r['sr']*100 for r in rows],colors,
         seeds=[[v*100 for v in r['sr_by_seed']] for r in rows],percent=True)
    axes[0].set_ylim(0,107);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR：成功率（%）')
    axes[0].set_title('(a) 导航成功率',loc='left')
    bars(axes[1],labels,[r['sg'] for r in rows],colors,decimals=3,seeds=[r['sg_by_seed'] for r in rows])
    axes[1].set_ylim(0,1.30);axes[1].set_ylabel('SG：平均终点距离（格）');axes[1].set_title('(b) 终点距离',loc='left')
    fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.88,wspace=.31)
    fig.text(.5,.025,'已知开发140题×3训练种子；5×5，B=10。白色标记为各训练种子，非置信区间。',ha='center',fontsize=9)
    caption='边缘连续性线索在同题开发验证中将SR从72.62%提升至88.57%，SG从0.864降至0.348。真实、均值与错误目标同时作用于全局、局部和边缘通道；探索/置零对照不改变目标信息。每条件140题、28张已知开发源图、3份训练权重；白色圆/三角/方形表示seed0/1/2，不能当420张独立源图。SG为所有正常完成轨迹的终点曼哈顿距离均值，成功记0；本图不代表正式S2验收。'
    save(fig,PLOTS,'01_边缘线索主结果',caption,[r['source_file'] for r in rows],d['edge']['protocol'])


def distances(d):
    baseline,edge=d['edge']['rows'][:2]
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.45));frames(axes)
    xs=np.arange(4,9)
    for row,color,marker,linestyle,label in [(baseline,GRAY,'s','--','冻结探索/置零'),(edge,BLUE,'o','-','边缘真实目标')]:
        sr=[row['by_distance'][str(c)]['sr']*100 for c in xs]
        sg=[row['by_distance'][str(c)]['sg'] for c in xs]
        axes[0].plot(xs,sr,color=color,marker=marker,linestyle=linestyle,lw=1.7,ms=5,label=label)
        axes[1].plot(xs,sg,color=color,marker=marker,linestyle=linestyle,lw=1.7,ms=5,label=label)
        for x,y in zip(xs,sr):
            axes[0].annotate(f'{y:.1f}',(x,y),xytext=(0,7 if color==BLUE else -13),textcoords='offset points',ha='center',fontsize=9)
    for ax in axes:
        ax.set_xticks(xs,[f'C{x}' for x in xs]);ax.set_xlabel('初始曼哈顿距离档（格）')
    axes[0].set_ylim(0,112);axes[0].set_yticks([0,20,40,60,80,100]);axes[0].set_ylabel('SR（%）')
    axes[1].set_ylim(0,max(baseline['by_distance'][str(c)]['sg'] for c in xs)*1.18);axes[1].set_ylabel('SG（格）')
    axes[0].legend(loc='lower right',frameon=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.95,wspace=.3)
    fig.text(.5,.035,'每个距离档28题×3训练种子；距离仅评测端分组，不提供给Agent。',ha='center',fontsize=9)
    caption='边缘线索的主要恢复来自C4/C5短距离档，短距离合并SR由44.64%升至74.40%。横轴为评测端使用的真实初始曼哈顿距离，Agent不读取该字段；每档28题×3训练权重，曲线为种子均值。C档限制可行起终点和路线结构，图中变化不等于连续增加任务难度；C4/C5后仍存在失败，不能由总体低SG推断所有失败都接近目标。'
    save(fig,PLOTS,'02_分距离表现',caption,[baseline['per_seed_source_pattern'],edge['per_seed_source_pattern']],d['edge']['protocol'])


def seed_plot(d):
    baseline,edge=d['edge']['rows'][:2];x=np.arange(3)
    fig,ax=plt.subplots(figsize=(5.6,3.55));frames(ax,True)
    width=.31
    for off,row,col in [(-width/2,baseline,GRAY),(width/2,edge,BLUE)]:
        vals=np.array(row['sr_by_seed'])*100
        ax.bar(x+off,vals,width,color=col,edgecolor='#34424F',linewidth=.7,label='冻结探索/置零' if row is baseline else '边缘真实目标')
        for s,v in enumerate(vals):ax.text(x[s]+off,v+1.2,f'{v:.2f}',ha='center',fontsize=10)
    ax.set_xticks(x,['seed 0','seed 1','seed 2']);ax.set_ylabel('SR（%）')
    ax.set_xlabel('独立训练权重的种子编号');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,frameon=False)
    for s in range(3):
        gain=(edge['sr_by_seed'][s]-baseline['sr_by_seed'][s])*100
        ax.text(s,12,f'增益 +{gain:.2f}点',ha='center',fontsize=10)
    fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.81)
    fig.text(.5,.035,'三个种子逐一配对；不选最高种子，不进行模型集成。',ha='center',fontsize=9)
    caption='边缘线索在三个配对训练种子上均提高SR，增益分别为17.14、14.29和16.43个百分点。每个种子执行相同140题；种子对应三份训练权重，不能视为三套独立地图。展示全部种子，不依据最高分挑模型。'
    save(fig,PLOTS,'03_三种子配对验证',caption,[baseline['source_file'],edge['source_file']],d['edge']['protocol'])


def intervals(d):
    rows=[]
    for key,label in [('Baseline','真实 − 冻结探索'),('ZeroEdgeFull','真实 − 置零对照'),
                      ('MeanCue','真实 − 均值目标'),('WrongCue','真实 − 错误目标')]:
        v=d['edge']['effects'][key]['source_interval']
        rows.append((label,v['mean']*100,np.array(v['interval95'])*100))
    v=d['edge']['short_effects']['Baseline']['source_interval']
    rows.append(('C4/C5：真实 − 探索',v['mean']*100,np.array(v['interval95'])*100))
    fig,ax=plt.subplots(figsize=(7.1,3.45))
    for i,(label,mean,ci) in enumerate(rows):
        y=len(rows)-1-i
        ax.errorbar(mean,y,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',ms=6,color=BLUE if i<4 else TEAL,capthick=1,capsize=4,lw=1.6)
        ax.text(43,y,f'+{mean:.2f}\n[{ci[0]:.2f}, {ci[1]:.2f}]',va='center',ha='left',fontsize=9)
    ax.set_yticks(range(len(rows)),[r[0] for r in reversed(rows)])
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_xlim(-1,53);ax.set_xticks([0,10,20,30,40])
    ax.set_ylim(-.65,len(rows)-.35)
    ax.set_xlabel('配对SR增益（百分点），误差线为源图成组95%区间')
    ax.grid(axis='x',alpha=.7);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.subplots_adjust(left=.29,right=.97,bottom=.24,top=.94)
    fig.text(.5,.035,'先平均训练种子，再按28张已知开发源图bootstrap；2000次重采样。',ha='center',fontsize=9)
    caption='边缘真实目标相对同题探索、置零、均值和错误目标的SR增益区间均为正。误差线直接取原实验的源图成组bootstrap：先在每张源图上平均三个训练种子，再对28张图进行2000次重采样；最后一行仅统计C4/C5。区间只支持本批已知开发数据上的优势，不代表独立地图或跨数据集保证。'
    save(fig,PLOTS,'04_目标贡献与增益区间',caption,['DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/对照汇总.json:effects,short_effects'],d['edge']['protocol'])


def historical(d):
    # A separate panel for each task list: no line bridges val100 and development140.
    fig,axes=plt.subplots(1,2,figsize=(8.0,4.2));frames(axes,True)
    vals=d['formal_val']['rows'];labels=['历史\n筛选','Small\n真实','Spatial\n真实','Small\n无目标','Spatial\n无目标']
    bars(axes[0],labels,[r['sr']*100 for r in vals],[GRAY,GRAY,GRAY,BLUE,GRAY],
         seeds=[[v*100 for v in r['sr_by_seed']] for r in vals],percent=True)
    axes[0].set_title('(a) 固定 val100×3：当前默认来源',loc='left');axes[0].set_ylabel('SR（%）')
    dev=[r for r in d['historical_milestones'] if r['protocol']=='已知开发140×3']
    labels=['PBRS\n真实','局部\n匹配','强\n探索','五分类\n线索','解耦\n线索','边缘\n线索']
    bars(axes[1],labels,[r['sr']*100 for r in dev],[GRAY]*5+[BLUE],
         seeds=[[v*100 for v in r['sr_by_seed']] for r in dev],percent=True)
    axes[1].set_title('(b) 开发140×3：方法探索过程',loc='left');axes[1].set_ylabel('SR（%）')
    for ax in axes:ax.set_ylim(0,109)
    fig.subplots_adjust(left=.07,right=.99,bottom=.2,top=.86,wspace=.26)
    fig.text(.5,.035,'两面板任务集与训练设定不同；只在各面板内阅读，不计算跨面板提升。',ha='center',fontsize=9)
    caption='本地策略的工程默认与新视觉候选分别来自不同协议，必须分开报告。(a) 固定val100的五个主条件，当前默认Small NoTarget平均SR73%；(b) 已知开发140题上的完整导航训练、强探索与线索验证，最新边缘候选88.57%。白色圆/三角/方形为三个训练种子；面板(b)各方法分批训练、假设不同，只有原配对实验支持对应因素效应，不能将此图解读为一条受控累计提升曲线。'
    refs=[r['source_file'] for r in vals+dev]
    save(fig,PLOTS,'05_同任务集历史方案',caption,sorted(set(refs)),'分面板：val100×3 与 已知开发140×3；不可跨面板相减')


def cloud(d):
    rows=d['cloud']['rows'];fig,axes=plt.subplots(1,2,figsize=(6.8,3.55));frames(axes)
    labels=[r['label'] for r in rows]
    bars(axes[0],labels,[r['sr']*100 for r in rows],[GRAY,ORANGE,TEAL],percent=True)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('SR_gate（%，分母20题）')
    bars(axes[1],labels,[r['sg'] for r in rows],[GRAY,ORANGE,TEAL],decimals=2)
    axes[1].set_ylim(0,3.5);axes[1].set_ylabel('SG_gate（格，含中断惩罚）')
    for i,r in enumerate(rows):axes[0].text(i,8,f'完成 {r["completed"]}/20',ha='center',fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.95,wspace=.32)
    fig.text(.5,.035,'同G层级搜索策略，固定val20，单轮；DeepSeek为非思考服务配置。',ha='center',fontsize=9)
    caption='同任务同G策略下，Gemma SR45%、DeepSeek非思考配置20%，均未超过Frontier规则50%。SR_gate按计划20题计，DeepSeek两题因严格解析失败而中断，SG_gate按每题8格补入；Gemma与Frontier正常完成20题，DeepSeek18题。单轮且仅4张源图，图中不画伪造误差线；服务配置、任务规模与本地分支不同，不比较其绝对分数。'
    save(fig,PLOTS,'06_云端模型同策略对照',caption,[rows[0]['source_file']],d['cloud']['protocol'])


def offline(d):
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.55));frames(axes)
    values=d['edge']['offline'];x=np.arange(3)
    for i in range(3):
        v=values[str(i)];p=v['accepted_precision']*100;lo,hi=np.array(v['precision_interval']['interval95'])*100
        axes[0].errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt=['o','^','s'][i],color=BLUE,ms=7,capsize=4)
        axes[0].text(i,hi+1.0,f'{p:.2f}',ha='center',fontsize=10)
    axes[0].set_xticks(x,['seed 0','seed 1','seed 2']);axes[0].set_ylim(0,106)
    axes[0].set_ylabel('接受线索的一步命中精度（%）');axes[0].set_yticks([0,20,40,60,80,100])
    axes[0].axhline(85,color=GRAY,ls='--',lw=.9);axes[0].text(2.35,83,'放行精度85%',ha='right',fontsize=9,color=GRAY)
    bars(axes[1],['seed 0','seed 1','seed 2'],[values[str(i)]['adjacent_recall']*100 for i in range(3)],[TEAL]*3,percent=True)
    axes[1].set_ylim(0,106);axes[1].set_ylabel('真实邻接召回率（%）');axes[1].set_yticks([0,20,40,60,80,100])
    fig.subplots_adjust(left=.1,right=.98,bottom=.21,top=.94,wspace=.32)
    fig.text(.5,.035,'固定校准阈值0.50；28张已知开发图；组件精度与召回不是导航SR。',ha='center',fontsize=9)
    caption='边缘线索在28张已知开发图上形成高精度邻接判断，为后续导航介入提供依据。左图为接受线索后一步到达目标的精度及源图成组95%区间，右图为真实邻接召回；三个种子阈值均在22张校准图上先冻结为0.50，接受数2088/2113/2052、接受覆盖率约12%。虚线表示预登记离线放行所需精度85%，该要求另有接受数、源图覆盖和区间下界；组件指标不能当导航SR或独立泛化证据。'
    save(fig,PLOTS,'07_邻接线索离线可靠性',caption,['DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/对照汇总.json:held_full_probe.Edge'],'87拟合/22校准/28已知开发源图；冻结阈值后测留出')


def diagram(name, nodes, edges, panels, caption, sources, height=960):
    """One geometry source produces native drawio plus PDF/SVG/PNG."""
    width=1500
    fig,ax=plt.subplots(figsize=(15,height/100))
    ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    root=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
    diag=ET.SubElement(root,'diagram',id=name,name=name)
    model=ET.SubElement(diag,'mxGraphModel',dx=str(width),dy=str(height),grid='1',gridSize='10',page='1',pageWidth=str(width),pageHeight=str(height))
    cells=ET.SubElement(model,'root');ET.SubElement(cells,'mxCell',id='0');ET.SubElement(cells,'mxCell',id='1',parent='0')
    for i,p in enumerate(panels):
        x,y,w,h,label,color=p
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor=color,edgecolor='#CBD5DF',lw=1))
        ax.text(x+18,y+23,label,fontsize=20,fontweight='bold',va='center',color='#243746')
        cell=ET.SubElement(cells,'mxCell',id=f'panel{i}',value=html.escape(label),vertex='1',parent='1',
            style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={color};strokeColor=#CBD5DF;fontSize=28;fontFamily=Microsoft YaHei;verticalAlign=top;align=left;spacingTop=10;spacingLeft=16;')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    for key,n in nodes.items():
        x,y,w,h,label,fill=n
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',facecolor=fill,edgecolor='#425466',lw=1.2,zorder=3))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=18,linespacing=1.4,color='#162E3F',zorder=4)
        cell=ET.SubElement(cells,'mxCell',id=key,value=html.escape(label).replace('\n','<br>'),vertex='1',parent='1',
            style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#425466;fontSize=26;fontFamily=Microsoft YaHei;spacing=8;')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    for i,e in enumerate(edges):
        source,target,points,label,dashed=e
        for j,(a,b) in enumerate(zip(points[:-1],points[1:])):
            if j==len(points)-2:
                ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,linewidth=1.3,
                    linestyle='--' if dashed else '-',color='#526674',zorder=5))
            else:
                ax.plot([a[0],b[0]],[a[1],b[1]],ls='--' if dashed else '-',color='#526674',lw=1.3,zorder=2)
        if label:
            pos=points[len(points)//2]
            ax.text(pos[0],pos[1]-14,label,ha='center',va='bottom',fontsize=16,
                    bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=6)
        cell=ET.SubElement(cells,'mxCell',id=f'edge{i}',source=source,target=target,value=html.escape(label),edge='1',parent='1',
            style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#526674;fontSize=23;fontFamily=Microsoft YaHei;'+('dashed=1;' if dashed else ''))
        geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
        arr=ET.SubElement(geo,'Array',attrib={'as':'points'})
        for x,y in points[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    ARCH.mkdir(parents=True,exist_ok=True)
    p=ARCH/f'{name}.drawio';ET.ElementTree(root).write(p,encoding='utf-8',xml_declaration=True)
    # Min intended printed width: 17cm. Labels 18pt become >=8pt at that width.
    save(fig,ARCH,name,caption,sources,'当前实际实现；单智能体模块协作；候选未替换默认',editable=p)


def architectures():
    nodes={
        'fit':(50,70,390,105,'原train：87拟合 / 22校准\n28张已知开发图只作验证','#F4F6F8'),
        'learn':(510,70,430,105,'解耦邻接/方向监督\n20维边缘投影；配对训练','#E8F3F9'),
        'freeze':(1030,70,420,105,'冻结3头、3探索器、4均值\n校准联合置信阈值0.50','#E8F4F0'),
        'obs':(50,290,290,115,'公开Observation\n当前图 + 目标图\n位置 / visited / 剩余步数','#F4F6F8'),
        'encode':(395,290,315,115,'冻结Sat2Cap\n512维全局特征\n四区域局部特征','#E8F3F9'),
        'feat':(765,290,340,115,'两图关系特征\n1041维原特征\n+20维RGB边缘特征','#E8F3F9'),
        'head':(1155,290,295,115,'Edge线索头\np(邻接) × p(方向|邻接)\n原始5类置信度','#E8F3F9'),
        'explore':(395,520,315,115,'Small256 NoTarget探索器\n当前视觉 + 公开历史\n合法动作筛选','#F4F6F8'),
        'gate':(765,520,340,115,'可信线索行动门控\n最高方向≥0.50且合法未访问\n接受线索；否则沿用探索','#E8F4F0'),
        'env':(1155,520,295,115,'环境执行一步\n更新公开Observation\n首次到达 / 预算耗尽','#F4F6F8'),
        'truth':(50,825,485,95,'评测专用：固定任务答案 / 真实距离\n仅评测端使用，不作为策略输入','#FFF4E8'),
        'eval':(735,825,715,95,'统一日志、指标与可复核证据\n动作轨迹 → SR / SG / 分档 / 目标干预 / 哈希','#FFF4E8'),
    }
    edges=[
        ('fit','learn',[(440,120),(510,120)],'',False),
        ('learn','freeze',[(940,120),(1030,120)],'',False),
        ('freeze','head',[(1300,175),(1300,290)],'',True),
        ('obs','encode',[(340,347),(395,347)],'',False),
        ('encode','feat',[(710,347),(765,347)],'',False),
        ('feat','head',[(1105,347),(1155,347)],'',False),
        ('encode','explore',[(553,405),(553,520)],'',False),
        ('obs','explore',[(195,405),(195,470),(430,470),(430,520)],'',False),
        ('head','gate',[(1300,405),(1300,465),(1030,465),(1030,520)],'',False),
        ('explore','gate',[(710,577),(765,577)],'',False),
        ('gate','env',[(1105,577),(1155,577)],'',False),
        ('env','obs',[(1300,635),(1300,718),(115,718),(115,405)],'',False),
        ('env','eval',[(1390,635),(1390,825)],'',True),
        ('truth','eval',[(535,872),(735,872)],'',False),
    ]
    panels=[(25,10,1450,205,'离线：训练与校准分离，固定配置后验证','#FAFBFC'),
            (25,245,1450,525,'在线：一个导航Agent，探索与线索为策略模块','#FAFBFC'),
            (25,785,1450,160,'评测侧：答案隔离，保留完整证据链','#FFFDF9')]
    # RGB seam descriptors bypass the semantic encoder; make that input explicit.
    for key in ['obs','encode','feat','head']:
        x,y,w,h,label,fill=nodes[key]
        nodes[key]=(x,y+30,w,h,label,fill)
    edges=[('fit','learn',[(440,120),(510,120)],'',False),
        ('learn','freeze',[(940,120),(1030,120)],'',False),
        ('freeze','head',[(1300,175),(1300,320)],'',True),
        ('obs','encode',[(340,377),(395,377)],'',False),
        ('obs','feat',[(290,320),(290,301),(935,301),(935,320)],'RGB边缘',False),
        ('encode','feat',[(710,377),(765,377)],'',False),
        ('feat','head',[(1105,377),(1155,377)],'',False),
        ('encode','explore',[(553,435),(553,520)],'',False),
        ('obs','explore',[(195,435),(195,480),(430,480),(430,520)],'',False),
        ('head','gate',[(1300,435),(1300,480),(1030,480),(1030,520)],'',False),
        ('explore','gate',[(710,577),(765,577)],'',False),
        ('gate','env',[(1105,577),(1155,577)],'',False),
        ('env','obs',[(1300,635),(1300,718),(115,718),(115,435)],'',False),
        ('env','eval',[(1390,635),(1390,825)],'',True),
        ('truth','eval',[(535,872),(735,872)],'',False)]
    diagram('01_边缘线索Agent架构',nodes,edges,panels,
        '边缘线索通过可信门控介入冻结探索策略，构成单智能体的观察—决策—行动闭环。当前图和给定目标图用于两图语义/局部/边缘特征；线索仅在原始五类最高方向置信度达到0.50、动作合法且落点未访问时介入，否则使用合法筛选后的探索建议；不重新排序线索。离线用87拟合、22校准、28已知开发源图，并冻结头、探索器和均值；实线表示观测/行动/统计流，虚线表示冻结配置或评测记录流。评测答案仅进入指标侧；图中没有未访问邻格图像输入。此为已验证开发候选，尚未替换当前默认；不是多智能体协作。',
        ['project/src/agents/edge_cue.py','project/src/agents/target_cue.py','project/src/train/edge_cue.py',
         'DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/预登记.json'],height=970)
    nodes={
        'tasks':(60,125,320,125,'固定episode任务清单\nsplit / 距离 / seed\n数据与协议哈希','#F4F6F8'),
        'env':(470,125,390,125,'部分可观测网格环境\n5×5；四方向；B=10\n公开观测与答案隔离','#F4F6F8'),
        'cloud':(1020,75,420,130,'云端分支：VLM + Governor\nGemma / DeepSeek服务对照\nG层级搜索策略','#E8F3F9'),
        'local':(1020,320,420,145,'本地分支：PyTorch策略\n默认Small NoTarget；候选Edge\n冻结视觉编码器 + 训练策略','#E8F4F0'),
        'logs':(470,380,390,125,'统一评测与结构化日志\nSR / SG / 完成率 / 重访\n逐步动作与终态','#FFF4E8'),
        'audit':(60,380,320,125,'复算与环境重放\n冻结代码/模型/输入哈希\n配对目标干预与源图区间','#FFF4E8'),
        'deliver':(390,650,680,110,'课程与科研交付\n报告 + 实验索引 + 可编辑架构图 + 来源可追溯结果图','#F4F6F8')}
    edges=[('tasks','env',[(380,187),(470,187)],'',False),
        ('env','cloud',[(860,155),(945,155),(945,140),(1020,140)],'',False),
        ('env','local',[(860,215),(945,215),(945,392),(1020,392)],'',False),
        ('cloud','env',[(1230,75),(1230,35),(650,35),(650,125)],'',False),
        ('local','env',[(1230,465),(1230,575),(890,575),(890,285),(775,285),(775,250)],'',False),
        ('env','logs',[(650,250),(650,380)],'',True),
        ('logs','audit',[(470,442),(380,442)],'',False),
        ('audit','deliver',[(220,505),(220,705),(390,705)],'',False),
        ('logs','deliver',[(650,505),(650,650)],'',False)]
    diagram('02_工程与评测边界',nodes,edges,[],
        '工程将云端VLM分支和本地训练策略分支接入相同网格环境与日志口径，两分支各自执行，尚未组成多智能体系统。实线显示任务配置、公开观测、动作回送与成果流，虚线显示评测记录；真实目标位置、初始/终点距离仅在环境及评测端使用，绝不传给Agent。当前开发默认为Small256 NoTarget，边缘候选尚待正式val100×3复验；GUI轨迹回放和大网格尚未实现。本图是实现范围概览，不表示零训练或所有模型已通过S2。',
        ['README.md','project/src/env/environment.py','project/local_policy_default.json',
         'DATA/processed_data/Masa/训练结果/边缘连续性可信线索对照_v1/候选配置.json'],height=810)


def write_index(d):
    architecture_refs={s for item in ITEMS if '架构' in item['id'] or '边界' in item['id'] for s in item['sources']}
    hashes=dict(d['source_sha256'])
    for rel in sorted(architecture_refs):
        hashes[rel]=digest(ROOT/rel)
    index=dict(version='scientific-figures-v1',data_file=DATA.relative_to(ROOT).as_posix(),
        data_sha256=digest(DATA),source_sha256=hashes,figures=ITEMS,
        script_sha256=digest(Path(__file__)))
    (OUT/'图表来源清单.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n','utf-8')
    captions=['# 科研图注与取用说明', '', 'PDF为论文/排版优先的嵌入字体矢量格式；SVG适合PPT中缩放，PNG为300dpi预览/兼容插图。架构drawio含原生可编辑节点与连线。', '',
        '结果图建议插入宽度16–18cm，历史方案图18cm；架构图17–20cm，避免过度缩成论文单栏导致文字小于8pt。SVG保留文本便于编辑，需要Microsoft YaHei字体；跨机器优先使用嵌入字体PDF或PNG。', '']
    readme=['# 绘图：科研与课程交付', '', '本目录图表全部从既有实验小型结果生成。本轮不新增实验，不改权重或默认方案。', '',
        '**核心结果：**同题开发验证SR72.62%→88.57%、SG0.864→0.348；正式S2待固定配置复验。架构按已实现的单智能体模块绘制。', '',
        '| 图号 | 内容 | 使用建议 |', '|---|---|---|']
    for item in ITEMS:
        p=ROOT/item['files'][0]['path'];base=p.relative_to(OUT).with_suffix('').as_posix()
        label=item['id']
        readme.append(f'| {label} | [PNG预览](<{base}.png>) / [矢量PDF](<{base}.pdf>) / [SVG](<{base}.svg>) | '+('架构可编辑：[drawio](<'+base+'.drawio>)' if '架构图' in base else '配套完整图注见下方')+' |')
        captions += ['## '+label, '', item['caption'], '', '**协议：**'+item['protocol'], '',
                     '**原件来源：** '+ '；'.join('`'+s+'`' for s in item['sources']), '']
    readme += ['', '## 直接取用', '', '[全图预览](审阅/全图预览.jpg)、[完整图注](图注.md)、[来源与哈希](图表来源清单.json)、[绘图数据](绘图数据/阶段结果_v1.json)、[设计与质量说明](设计与质量说明.md)。', '',
        '每个结果图都有PNG/PDF/SVG，文件同名；报告/PPT直接引用图注并保留任务集范围。当前候选图使用原train内已知开发140题，不能写成test结果、正式S2通过或通用语义定位。', '',
        '## 重绘', '', '在项目根目录运行：', '', '```powershell',
        "& 'D:\\PYTHON\\python.exe' -X utf8 project/src/documents/stage_figures.py", '```', '',
        '脚本读取本轮冻结绘图数据，并验证每个原始结果SHA-256；结果变化应另建版本。原始训练/评测路径不依赖本目录。']
    (OUT/'README.md').write_text('\n'.join(readme)+'\n','utf-8')
    (OUT/'图注.md').write_text('\n'.join(captions)+'\n','utf-8')
    design='''# 科研图设计与质量说明

结果采用同题分组柱状图、距离档折线和增益区间图；历史方案按任务集分面，云端20题另图展示。柱形以0为基线，SR轴完整包含0–100%；各训练种子用不同形状与文字标签识别，种子离散度不冒充置信区间。只有原实验明确计算的源图区间用误差线表示。

方法图采用在线/离线/评测三层架构；模块名称与源码对应，闭环观测与行动清晰，答案只留在评测侧。候选图和工程边界图分开描述局部机制与整体范围。原生drawio节点及连线可编辑，PDF/SVG均由同一几何定义生成，不是截图包成矢量。

颜色采用Okabe–Ito中的蓝、朱红、蓝绿及中性灰，配合不同标记、线型和直接文字标注。无3D、阴影或装饰渐变。正文图注逐一定义SR、SG、gate、种子及开发范围。PDF嵌入字体，SVG保留文本，PNG为300dpi。

目标印刷宽度：结果图16–18cm，历史方案18cm，架构图17–20cm；该尺寸下主要标签不小于8pt。若缩至论文单栏，建议分拆面板或重新设置画布/字号，而非直接缩小整图。

视觉检查、XML/文件来源/数值/原件保护核验见“审阅/图表核验.json”和阶段“整理核验.json”。图注由本次生成，历史原报告保持原文；本轮没有新的文献检索或新颖性断言。
'''
    (OUT/'设计与质量说明.md').write_text(design,'utf-8')


def contact_sheet():
    REVIEW.mkdir(parents=True,exist_ok=True)
    cols=2;tilew,tileh=1050,680
    sheet=Image.new('RGB',(cols*tilew,((len(ITEMS)+1)//2)*tileh),'#E8EDF1')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26)
    draw=ImageDraw.Draw(sheet)
    for i,item in enumerate(ITEMS):
        p=ROOT/next(f['path'] for f in item['files'] if f['path'].endswith('.png'))
        with Image.open(p) as im:
            preview=ImageOps.contain(im.convert('RGB'),(tilew-30,tileh-65))
        x=(i%2)*tilew;y=(i//2)*tileh
        draw.text((x+20,y+10),item['id'],font=font,fill='#203444')
        sheet.paste(preview,(x+(tilew-preview.width)//2,y+50+(tileh-65-preview.height)//2))
    sheet.save(REVIEW/'全图预览.jpg',quality=94)


def main():
    d=json.loads(DATA.read_text('utf-8'))
    for rel,h in d['source_sha256'].items():
        if digest(ROOT/rel)!=h:
            raise ValueError('Changed result source: '+rel)
    style()
    for fn in [latest,distances,seed_plot,intervals,historical,cloud,offline]:fn(d)
    architectures()
    write_index(d);contact_sheet()
    print(json.dumps(dict(figures=len(ITEMS),formats=['PDF','SVG','PNG','drawio for architectures']),ensure_ascii=False))


if __name__=='__main__':main()
