"""Seal and publish CV041 fixed development results, including negative results."""
import csv
import json
from project.src.eval.rgb_spatial_development_v1 import ROOT,OUT,ENG,entry,read,put,verify


def update(p,old,text):
    assert p.read_bytes()==old,str(p)+' changed during publication'
    p.write_bytes(text.encode('utf-8'))


def publish():
    if (OUT/'README.md').exists():raise ValueError('Already published; inspect sealed results')
    b=verify();s=read(OUT/'主对照汇总.json');a=read(OUT/'核验/另实现复核.json');r=read(OUT/'资源回执.json');done=read(OUT/'完整执行完成.json')
    assert a['passed'] and r['passed'] and done['completed']
    assert read(ROOT/'项目导航/实验索引.json')['batches']==b['index_prefix']
    assert read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current']
    passed=s['candidate_passed'];status='仅通过已知来源开发候选，尚需新来源确认' if passed else '候选未过，按停止条件收束'
    put('最终判定.json',dict(task='CV-041',complete=True,candidate_passed=passed,status=status,gates=s['gates'],
        integrity_audit_passed=True,default_upgraded=False,navigation_admitted=False,independent_confirmation_sources=0,
        frozen_protocol_not_overwritten=True,no_new_hyperparameter_search=True,
        next='freeze genuinely new source confirmation' if passed else 'close this candidate; consolidate evidence and platform status before selecting any different mechanism'))
    table='';csvrows=[]
    for method,m in s['development_means'].items():
        table+=f"|{method}|{m['positive_joint']}/{m['positive_n']}|{100*m['source_macro_joint']:.2f}%|{m['positive_error_mean_m']:.3f}|\n"
        csvrows.append(dict(method=method,positive_joint=m['positive_joint'],positive_n=m['positive_n'],source_macro_joint=m['source_macro_joint'],all_positive_error_mean_m=m['positive_error_mean_m']))
    with (OUT/'对照表.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    contrasts=''
    for name,c in s['contrasts'].items():
        p=c['by_class']['inside'];ci=c['source_paired_bootstrap95_descriptive']
        contrasts+=f"|{name}|{100*c['source_macro_joint_gain']:+.2f}|{p['recovered']}/{p['harmed']}/{p['net']:+d}|[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]|\n"
    gates='\n'.join('- '+k+'：'+('通过' if v else '未过') for k,v in s['gates'].items())
    bycity='、'.join(f"{k} {v}/3" for k,v in s['positive_by_source'].items())
    nexttext='下一步只能另冻真实新且空间隔离来源确认；当前3城不能充当那个确认池。' if passed else '本候选到此收束，不追加epoch/阈值/裁剪/种子/来源/参数搜索。下一步整理失败机制与可用平台证据，选择不同机制之前先给出新证据；不能把新增报告或训练轮数当作能力进步。'
    report=f'''# CV-041 普通RGB空间小头固定开发对照

**完整执行与复核通过；{status}。** 本轮是静态局部感知开发，表中的通过率不是主动导航SR；没有自主动作或默认升级。

## 固定队列与主要结果

3个已消费Mapillary城市各1照片/航片，三折留一已知来源开发、种子0/1/2；拟合每折2城，18头（Spatial与TrainedMean各9），每头固定200更新。匹配头仅接给定照片特征＋当前格49空间token；禁用GPS/heading/K/目标格/源身份/全图特征。共享CLIP视觉塔完全冻结，不下载新模型。

|方法/目标条件|窗内接受且≤5米|来源宏联合BA|全部窗内正例平均误差/米（含拒绝）|
|---|---:|---:|---:|
{table}
联合BA按窗内“接受且定位≤5m”、窗外拒绝、异源拒绝三类等权，再按来源等权；不是普通接受率。Spatial Full逐城市联合正例：{bycity}。RawCos无随机权重，每城只推断一次，故表中分母3；配对时将同一预测与3个种子比较，不增加独立地域或另算模型调用。

|Full相对对照|联合BA差/百分点|正例恢复/损伤/净增（9配对）|三城市配对95%区间/百分点（仅描述）|
|---|---:|---:|---|
{contrasts}
窗外/异源恢复损伤、每来源×种子误接受及所有条件、raw位姿敏感性在[主汇总](主对照汇总.json)；原预测/50logits/原生点位和每条case_id在记录目录。只看全局均值会掩盖负类与来源损伤，因此全部分层保留。

## 预登记判定

{gates}

所有必要项一起判定，不看到结果后改门槛。数据集OpenSfM修正相机点仍是主标签；raw仅敏感性，未用来调阈值/选权重。5米是对数据集估计位置的容差，不是独立测绘精度。原生100/100/41.89米范围和不同格宽逐源保留，均值误差按各自mpp折算。

{nexttext}

## 真正执行了什么

|项目|实际/计划|
|---|---:|
|新增CPU编码|72/72|
|复用工程特征|6/6（不重新编码）|
|分阶段累计编码|80（工程8＋本轮72）|
|本轮航片解码|3/3（不重读地面图）；分阶段累计9|
|训练头|18/18，每头65,537参数|
|正式优化更新/样本使用|3600/21600|
|非训练对预测|6750/6750，96个组|
|开发对预测|2850；余3900为拟合初末检查/校准|
|fit-only校准|21次：18训练头＋3 RawCos，控制复用Full阈值|
|资源|CPU采样峰值{r['peak_sampled_RSS']/1024**3:.3f}GiB，监督总时长{r['wall_seconds']:.2f}秒；上限4GiB/1800秒|

编码的实际新增输入仅原航片格1—24，不是全父图的特征；全父图只由切图器读取。旧CV040预处理/模型张量身份完全相同，首末冻结SHA与工程相等。额外解码3幅原航片的I/O补充已由用户明确授权并在首调用前绑定，未回写CV039原协议的6次记录；没有暗中扩展编码预算。

Spatial和TrainedMean同种子初始权重SHA、固定采样顺序/每批类别平衡、Adam/学习率/裁剪/最终200步一致。有限零梯度仅记录，不与工程阶段的非零梯度检查混淆；缺失/非有限梯度仍停。18个最终小头保留，只有一个活动optimizer/RNG/游标恢复点在本批目录原子更新，不复制冻结骨干，不堆优化器历史。

## 可复核性与边界

9项合成检查及{len(a['checks'])}项另实现检查通过。另实现独立核全部6750保存logits的概率/argmax/原生坐标、21次有理数校准、主/敏感性分层、全部3600采样/输入哈希和梯度记录、90个损失/标签快照、恢复/损伤、3城市聚类区间与必要门槛。复核0新编码/头forward/更新/解码；不声称重新预测了所有6750输出，也不复用已耗尽的CV04010数学预测。由同Agent另一实现完成，非独立人员审查。

来源只有3城，且全部已工程查看；每折异源参考来自拟合城市，不能声称全新查询/参考双端确认。多个种子、多个格和条件均不增加独立地理样本，bootstrap仅描述。没有任意手机照片/全景/真实无人机或跨时相泛化结论；预训练地理覆盖未知。

[执行方案与I/O补充](执行方案与I_O补充.json) · [首正式绑定](首正式执行绑定.json) · [特征绑定](完整特征绑定.json) · [18模型清单](模型执行清单.json) · [预测清单](预测文件清单.json) · [校准](固定校准阈值.json) · [主对照](主对照汇总.json) · [最终判定](最终判定.json) · [资源](资源回执.json) · [另实现](核验/另实现复核.json)。所有配置、源、输入、权重/缓存/预测哈希与实际调用计数保留，失败结果不删除或换队列。

当前批次完整封存后不要再bind/execute/worker/写盘audit/publish覆盖；恢复读清单/事件/完成/资源/判定与封存，活动恢复点只是最后一个完成模型的状态，不能盲目续训或重跑已消费编码。数据特征与最终权重为研究原件而非缓存。当前168索引且旧167值/current/旧角色/6确认保持，原M0/S2S3S4及平台4.12/server19/service11/pack23无改变，无GUI升级。

本轮0导航/云模型API/下载/确认组消费，无新导航SR；全程离线，未使用代理，没有安装/持久配置、主动C盘任务文件或Git提交推送。CPU运行不占GPU、不控制ComfyUI。结果与缺陷按原协议保留，不能用“工作完成”代替“能力通过”。
'''
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write(report)
    rel=OUT.relative_to(ROOT).as_posix();rp=rel+'/README.md'
    p=ROOT/'项目导航/实验索引.json';old=p.read_bytes();idx=read(p);idx['date']='2026-10-24'
    idx['batches'].append(dict(category='CV-041',dataset='OpenCVL三城市已消费开发',kind='普通RGB空间小头同预算/目标对照',name='普通RGB空间小头固定开发对照_v1',
        status='完整/另实现通过；'+status,path=rel,reports=[rp],evidence=[rel+'/最终判定.json',rel+'/主对照汇总.json',rel+'/核验/另实现复核.json']))
    update(p,old,json.dumps(idx,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'docs/维护/版本生命周期.json';old=p.read_bytes();life=read(p);life['updated']='2026-10-24'
    life['cv041_rgb_spatial_development']=dict(status='development_candidate_passed_needs_confirmation' if passed else 'candidate_closed_failed_development_gates',report=rp,
        replaces='CV040 pending formal development only; no default/source role or historical result replaced',
        inference_heads=rel+'/模型执行清单.json',features=rel+'/完整特征绑定.json',recovery='read sealed complete records; do not rerun or extend training',
        candidate_passed=passed,default_upgrade=False,navigation_admitted=False,
        next='new isolated source confirmation design' if passed else 'consolidate failed candidate and platform evidence; no tuning this factor')
    update(p,old,json.dumps(life,ensure_ascii=False,indent=2)+'\n')
    for name in ('README.md','项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;old=p.read_bytes();text=old.decode('utf-8-sig').replace('\r','');h,_,rest=text.partition('\n');pre='' if name=='README.md' else '../'
        link=f'[CV-041 · 普通RGB空间小头固定开发]({pre}{rp})：18头/3600更新/6750预测完成，{status}；未升级默认或导航。'
        update(p,old,h+'\n\n'+link+'\n\n'+rest)
    p=ROOT/'docs/PROJECT_STATE.md';old=p.read_bytes();text=old.decode('utf-8-sig').replace('\r','');h,_,rest=text.partition('\n');m=s['development_means']['Spatial/Full']
    note=f'''\n\n### CV-041：固定普通RGB空间小头对照完成，{status}（2026-10-24）

[完整分层/目标/基线对照](../{rp})。复用6旧特征、72新编码（累计80）/追加3父航片解码（已授权，分阶段9），18小头共3600更新21600样本，6750非训练预测/96组/21次fit-only校准完整；9合成/{len(a['checks'])}另实现过。Full窗内接受≤5米{m['positive_joint']}/9、来源宏联合BA{100*m['source_macro_joint']:.2f}%，非导航SR。必要门槛以最终判定为准，不调参重试；三城均已消费开发、独立确认0，异源参考已见限制保留。

CPU{r['peak_sampled_RSS']/1024**3:.3f}GiB/{r['wall_seconds']:.2f}秒；18最终头+一个活动恢复点，旧CV039/040原件不回写。当前168索引且旧167/current/旧角色/6确认/原M0与平台保持，无新导航SR/GUI升级；离线无代理/主动C盘输出/Git。下一{'冻结真正新来源确认' if passed else '收束本候选并整合失败边界与平台证据，不重复同因素试错'}。\n'''
    update(p,old,h+note+'\n'+rest)
    files=[p for p in OUT.rglob('*') if p.is_file()];assert sum(p.stat().st_size for p in files)<48*1024**2-100000
    put('阶段封存.json',dict(files=[entry(p) for p in files],sources=b['sources'],seal_itself_excluded=True,completed=True,candidate_passed=passed,default_upgraded=False))
    print(json.dumps(dict(published=True,index=168,candidate_passed=passed,bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__':publish()
