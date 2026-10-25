"""Close the separately authorized 4GiB engineering batch; no encoder imports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/地面目标实时编码接口工程_v2'
REL=OUT.relative_to(ROOT).as_posix()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def put(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        f.write(obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2));f.write('\n')
def prepend(path,body):
    p=ROOT/path;txt=p.read_text(encoding='utf-8-sig');heading,_,rest=txt.partition('\n')
    p.write_text(heading+'\n\n'+body+'\n'+rest,encoding='utf-8')
def main():
    if (OUT/'阶段封存.json').exists() or (OUT/'README.md').exists():raise FileExistsError('Already closed')
    result,done,audit=map(lambda n:read(OUT/n),('工程结果.json','执行完成.json','另实现复核.json'))
    assert done['completed'] and audit['passed']
    binding=read(OUT/'首forward绑定.json')
    for e in sum((binding[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')),[]):
        actual=entry(ROOT/e['path']);assert actual['sha256']==e['sha256'] and actual['bytes']==e['bytes'],e['path']
    status='工程通过' if result['passed'] else '工程未过，不自动重试'
    put('README.md',f'''# CV-025：地面目标实时编码接口工程（4GiB）

**{status}，仅编码与输入权限工程，不是地面目标导航验收。** 复用原CLIP地面编码器和Sat2Cap航拍编码器，逐幅RGB形成512维特征，接入合法局部观测会话。Web4.9.1、航拍默认和地面导航未通过结论均保持。本接口尚未接入正式Web或当前搬迁清单。

## 实际结果

- 固定4个已消费开发来源，每来源1地面照+1当前航拍图，共8次新增CPU forward；含原批次1次地面成功，累计实际9次。没有按分数重选素材，第5开发来源和6封存确认组未使用。
- 与旧缓存逐输入最大绝对误差{result['max_embedding_abs_error']:.9g}，最小余弦{result['min_embedding_cosine']:.12f}。冻结门槛仍为max_abs≤5e-5、cosine≥0.999999、另实现PIL预处理差≤1e-6。
- 本批编码耗时{done['elapsed_seconds']:.2f}秒、含两次中断累计{done['cumulative_elapsed_seconds']:.2f}秒；50ms采样进程峰RSS {done['sampled_peak_RSS_bytes']/1024**3:.3f}GiB。用户批准上限4GiB，累计编码限时600秒，CPU4线程。RSS是进程采样值，并非OS硬配额。
- 原20项接口合成检查、4项加载边界检查通过；新5项即时写盘/失败/不覆盖测试通过。另实现标量点积/范数复核8向量、4公开包、8尝试与8成功回执，历史/模型/缓存/源码SHA保持。同Agent另实现，不称独立人员审查。
- meta构造、strict assign共享CPU张量、单槽编码器；参数冻结且逐次释放前后张量SHA一致。架构、模型参数、预处理、原缓存与门槛没有改变；不扩参、不训练。原3GiB两次失败[单独封存](../地面目标实时编码接口工程_v1/README.md)，不回写为通过。

## 可用接口与权限

`project.src.platform_core.ground_encoding_v1.GroundEncodingSession`：begin显式接给定地面RGB与公开网格/预算；observe_public接当前航拍RGB、当前位置、步数、剩余预算及合法动作。observe_environment忽略旧航拍目标图与外部visited，由会话维护合法历史。输出不可变EncodedObservation；不读取目标位置、源身份、完整图块库或未访问信息。

`project.src.agents.frozen_crossview_encoder_v2.FrozenCrossViewEncoder`：只接单幅RGB与模态，strict加载已绑定本地资产，不下载。单槽复用不是减少模型参数；CPU加载的固定骨干仍是各约8785万参数。

像素指纹缓存仅题内复用，跨题清空。事件raw_similarity是未校准余弦，confidence为null、navigation_available为false。本批没有策略动作、SR或可靠拍摄点定位能力；向量可复现不能证明跨视角关联有效。编码器来源/训练地理覆盖的旧缺项不因此消失。

## 证据与恢复

[资源授权](资源授权.json)、[冻结协议](冻结工程协议.json)、[同8幅输入](固定输入清单.json)、[首forward绑定](首forward绑定.json)、[写盘测试](写盘测试.json)、[逐对接口与预处理](逐对接口核验.json)、[结果](工程结果.json)、[资源](执行完成.json)、[另实现复核](另实现复核.json)。编码尝试.jsonl在调用前写盘，编码成功.jsonl和8份单向量在成功后立即fsync；在线工程特征.npy仅8×512，用于重算。

本批freeze/execute/audit-save/close已消费，不能盲重跑。恢复只读校验阶段封存、绑定与保存向量，无需再forward。后续真实新题需单独预算、用户输入权限和实验状态；没有自动许可重复本批或消费确认组。

下一项建议把合法局部特征与证据事件接入平台的实验性地面模式及保存流程，继续区分诊断、实时编码和未验导航。产生新动作仍需有任务适配依据的决策器与同预算目标对照，不复活CV-024失败因素或小头加训。

本批0训练/导航/API/下载/确认，无代理、无C盘任务输出、无Git提交推送；总输出≤6MiB。不重启平台服务、不改变历史验收。
''')
    ip=ROOT/'项目导航/实验索引.json';idx=read(ip);old=idx['batches'];assert len(old)==146
    idx['date']='2026-10-22';idx['batches']=old+[dict(dataset='MM-GAG',kind='地面航拍真实编码接口工程',name=OUT.name,category='CV-025',
        status=status+'；非目标有效性或导航验收',scope='4已消费开发来源/8新增CPU forward/累计9；0训练/导航/API/6确认',
        path=REL,reports=[REL+'/README.md'],evidence=[REL+'/'+n for n in ('冻结工程协议.json','首forward绑定.json','逐对接口核验.json','工程结果.json','执行完成.json','另实现复核.json','阶段封存.json')])]
    ip.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');assert read(ip)['batches'][:-1]==old
    put('索引原值核验.json',dict(old_batches=146,new_batches=147,old_values_preserved=True,
        old_canonical_sha256=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
    for p in ('项目导航/README.md','项目导航/实验索引.md'):
        prepend(p,f'[第146批3GiB中断](../DATA/processed_data/MM-GAG/地面目标实时编码接口工程_v1/README.md)保留；[第147批4GiB编码接口](../{REL}/README.md)：{status}，8新增CPU编码/累计9，0导航或默认升级。\n')
    prepend('README.md',f'### CV-025真实编码工程交付（2026-10-22）\n\n[4GiB局部编码接口]({REL}/README.md)：{status}，同8图实时向量与旧缓存另算核验，20+4+5项工程检查；原3GiB失败保留。只输出合法特征和证据，不输出动作；0训练/导航/API/确认，Web4.9.1不变。\n')
    life_path=ROOT/'docs/维护/版本生命周期.json';life=read(life_path);life['updated']='2026-10-22'
    life['current']['ground_encoding_engineering']=REL+'/README.md'
    life['current']['ground_encoding_interface']='project/src/platform_core/ground_encoding_v1.py'
    life['current']['ground_encoding_loader']='project/src/agents/frozen_crossview_encoder_v2.py'
    life.setdefault('crossview_development_deliveries',[]).extend([
        dict(task='CV-025',path='DATA/processed_data/MM-GAG/地面目标实时编码接口工程_v1',status='failed_3GiB_sealed',replaces='none;failure evidence',recovery='sealed original;never rerun'),
        dict(task='CV-025',path=REL,status=status,replaces='v1 as current engineering check, not historical judgment',
            depends_on=['frozen_crossview_encoder_v1.py','frozen_crossview_encoder_v2.py','ground_encoding_v1.py','original CLIP/Sat2Cap and cache bindings'],
            navigation_available=False,production_platform_unchanged=True,recovery='saved vector/SHA audit;no blind rerun')])
    life_path.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    put('阶段封存.json',dict(task='CV-025',status=status,files=files,sources=binding['sources'],
        cumulative_actual_forwards=9,no_new_navigation_SR=True,mutable_coordination_excluded=True))
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert size<=6*1024**2
    print(json.dumps(dict(closed=True,engineering_passed=result['passed'],files=len(files),bytes=size,index=147)))

if __name__=='__main__':main()
