"""CV-025 evidence closure. Never loads encoders or starts old executors."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/MM-GAG/地面目标实时编码接口工程_v1';REL=OUT.relative_to(ROOT).as_posix()
SOURCES=['agents/frozen_crossview_encoder_v1.py','platform_core/ground_encoding_v1.py','tests/test_ground_encoding_v1.py',
    'tools/verify_ground_encoding_v1.py','documents/close_ground_encoding_v1.py']
SOURCES+=['agents/frozen_crossview_encoder_v2.py','tests/test_ground_encoding_v2.py','tools/verify_ground_encoding_v2.py','documents/close_ground_encoding_v2.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def entry(p):return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def put(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2));f.write('\n')


def main():
    if (OUT/'阶段封存.json').exists() or (OUT/'README.md').exists():raise FileExistsError('Delivery consumed')
    result,done,audit=read(OUT/'工程结果.json'),read(OUT/'执行完成.json'),read(OUT/'另实现复核.json')
    assert done['completed'] and audit['passed']
    status='工程通过' if result['passed'] else '工程未过；停止，不自动重跑'
    put('README.md',f'''# CV-025：地面目标真实编码与局部观测接口

**{status}。** 本批实现给定地面RGB→冻结CLIP目标特征、当前航拍RGB→冻结Sat2Cap观测特征的无GUI接口，并与既有缓存做一次有界工程核对；不选择动作、不导航、不训练，不把余弦分数当位置概率。

实际使用4个已消费开发来源的4对图像，共8次CPU单图forward。按原开发队列首次出现顺序取前4来源，未按模型分数选样，未使用第5来源或6个封存确认组。模型、任务、原图、缓存、预处理和源码都在首forward前绑定，原有模型只保留原份，没有下载或扩大骨干。

- 新/旧512维原始向量最大绝对误差：{result['max_embedding_abs_error']:.9g}；最小余弦：{result['min_embedding_cosine']:.12f}。
- 冻结门槛为逐输入max_abs≤5e−5、cosine≥0.999999、独立PIL预处理差≤1e−6，未看成绩后调整。
- 耗时{done['elapsed_seconds']:.2f}秒，进程50ms采样峰值RSS {done['sampled_peak_RSS_bytes']/1024**3:.3f}GiB；CPU4线程、单批≤3GiB/600秒，无GPU使用。这是编码接口工程耗时，不是导航吞吐。
- 两编码器严格加载，参数requires_grad=False、eval/inference_mode，实际张量及资产文件前后SHA一致。未更新任何参数。
- 20项接口+4项加载内存合成检查与另实现标量余弦/差分、8向量/4特征包/8尝试计数和原件SHA核验通过；同Agent核验不是独立人员审查。

## 可调用接口

`project.src.platform_core.ground_encoding_v1`提供decode_frame、GroundEncodingSession和不可变EncodedObservation。begin接收显式给定地面RGB及公开网格/预算；observe_public仅接当前图、位置、步数、剩余预算和合法动作。observe_environment桥接原Observation时刻意忽略其航拍target_image、外部visited和评测真值，历史由本会话按合法转移建立。

重复当前像素可复用本题特征，跨题缓存和目标会清空。输出目标/当前512维向量及公开几何，局部事件中的raw_similarity只是未校准诊断值，confidence始终为null、navigation_available始终为false。不会输出动作或读取完整候选图块库。

编码器实现位于`project/src/agents/frozen_crossview_encoder_v1.py`，只接受单幅RGB，不接图像文件名、源身份、目标索引或环境对象；资产路径仅在独立装配时绑定。真实推理没有使用旧小头或CVUSA候选，也没有复活CV-024局部语义趋势策略。

## 首forward前内存修复

原v1加载在首forward前因RSS约3.327GiB触发3GiB护栏，101.96秒、实际forward0；中断原件/旧源码保留，未重跑v1。另版v2只改加载：meta空权重构造、strict assign复用CPU张量、单模型驻留，四项加载测试过。架构/模型张量/预处理/8图/阈值/3GiB均未改，累计编码时间包含原失败且≤600秒。v2不是新算法试验，内存修复绑定见加载修复绑定.json。

## 证据与恢复

- [冻结协议](冻结工程协议.json) / [固定输入](固定输入清单.json) / [首forward绑定](首forward绑定.json)。
- [合成测试](合成测试.json) / [逐对预处理和缓存一致性](逐对接口核验.json) / [工程结果](工程结果.json)。
- [运行资源](执行完成.json) / [另实现复核](另实现复核.json) / 编码尝试.jsonl / 局部事件.jsonl。
- 在线工程特征.npy仅8×512 float32，不复制原特征库。输出≤6MiB，原始数据和模型只读。

v1 freeze/失败execute及v2 bind/execute/audit-save/close均已消费；不要盲重跑。恢复先核本批seal、输入与源码SHA、保存向量和回执，离线重算不需模型。模型调用中断必须保留前缀并单独登记，不能覆盖重试或换样本追通过。

本批0训练/导航/模型API/下载/确认，没有新SR或可靠地面导航结论。既有Sat2Cap卡和原论文run映射的来源缺项仍存在；工程向量兼容不能证明地面目标定位有效。当前Web4.9.1和航拍默认保持，接口尚未接入正式Web/搬迁包，不能称用户上传地面照已能自主定位。

下一项应把这一合法特征/证据接口接入地面实验模式及保存路径，继续区分已存诊断、实时编码和未验导航。若要产生行动，仍需一个有新任务适配/空间监督依据的决策器，另冻结同预算目标对照；不先扩参或重复旧失败因素。未使用代理，无C盘任务输出、Git提交或推送。
''')
    life_path=ROOT/'docs/维护/版本生命周期.json';life=read(life_path);life['updated']='2026-10-22'
    life['current']['ground_encoding_engineering']=REL+'/README.md'
    life['current']['ground_encoding_interface']='project/src/platform_core/ground_encoding_v1.py'
    life['crossview_development_deliveries'].append({'task':'CV-025','path':REL,'status':status,
        'replaces':'none;new engineering interface, production4.9.1 unchanged','depends_on':'existing public CLIP/Sat2Cap and sealed raw/preprocessing/cache references',
        'navigation_available':False,'ground_strategy_upgraded':False,'recovery':'read saved vectors/SHAs;never blindly rerun real forward'})
    life_path.write_text(json.dumps(life,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    ip=ROOT/'项目导航/实验索引.json';idx=read(ip);old=idx['batches'];assert len(old)==145
    oldsig=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    idx['date']='2026-10-22';idx['batches']=old+[{'dataset':'MM-GAG','kind':'真实地面/航拍编码接口工程','name':'地面目标实时编码接口工程_v1','category':'CV-025',
        'status':status+'，非导航或目标有效性验收','scope':'4已消费开发来源/8CPU单图forward；0训练/导航/API/6确认',
        'path':REL,'reports':[REL+'/README.md'],'evidence':[REL+'/'+n for n in ('冻结工程协议.json','首forward绑定.json','逐对接口核验.json','工程结果.json','执行完成.json','另实现复核.json','阶段封存.json')]}]
    ip.write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');assert read(ip)['batches'][:-1]==old
    put('索引原值核验.json',{'old_batches':145,'new_batches':146,'old_canonical_sha256':oldsig,'old_values_preserved':True})
    for name in ('项目导航/README.md','项目导航/实验索引.md'):
        p=ROOT/name;text=p.read_text(encoding='utf-8-sig');head,sep,rest=text.partition('\n')
        p.write_text(head+'\n\n[第146批CV-025地面目标编码接口](../'+REL+'/README.md)：'+status+'，4开发来源/8CPU编码/20内存检查，0导航/训练/API/6确认；正式Web不变。\n'+rest,encoding='utf-8')
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');head,sep,rest=text.partition('\n')
    p.write_text(head+'\n\n### CV-025局部编码接口交付（2026-10-22）\n\n[真实地面/航拍编码工程](DATA/processed_data/MM-GAG/地面目标实时编码接口工程_v1/README.md)：'+status+'，24合成/4已消费来源8CPU图像编码与旧缓存另算核验；输出合法目标/当前特征包，不输出导航动作。0训练/导航/API/6确认，Web4.9.1与原失败结论保持。\n'+rest,encoding='utf-8')
    files=[entry(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    sources=[entry(ROOT/'project/src'/n) for n in SOURCES]
    put('阶段封存.json',{'task':'CV-025','status':status,'files':files,'sources':sources,'output_cap_bytes':6*1024**2,
        'mutable_coordination_and_navigation_excluded':True,'no_new_navigation_SR':True,'recovery':'saved vector/SHA audit only;no blind rerun'})
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=6*1024**2
    print(json.dumps({'files':len(files),'sources':len(sources),'bytes':total,'index':146,'engineering_passed':result['passed']}))


if __name__=='__main__':main()
