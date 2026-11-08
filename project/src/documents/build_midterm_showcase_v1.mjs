/* DOC009 local editable showcase. Bundled Artifact Tool; no model calls. */
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const OUT=path.join(ROOT,'中期报告相关/中期成果更新_20261025');
const DEPS='C:/Users/DFWJ/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SKILL='C:/Users/DFWJ/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const TMP=path.join(OUT,'临时');
process.env.TEMP=TMP;process.env.TMP=TMP;process.env.XDG_CACHE_HOME=path.join(TMP,'cache');delete process.env.NODE_COMPILE_CACHE;
process.env.RUNTIME_NODE_MODULES=path.join(DEPS,'node/node_modules');
const require=createRequire(import.meta.url);
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(DEPS,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT='Microsoft YaHei',C={ink:'#163F4A',muted:'#557078',teal:'#087F83',pale:'#EDF5F4',amber:'#B16C1B',gray:'#8A9CA2'};
let num=0;
function text(s,str,x,y,w,h,size=26,color=C.ink,bold=false){
 const sh=s.shapes.add({name:'text-'+(++num),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=str;sh.text.style={typeface:FONT,fontSize:size,color,bold,wrap:'square',autoFit:'none',insets:0,verticalAlignment:'top'};return sh;
}
function heading(p,title,source){
 const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,64,35,1145,74,44,C.ink,true);
 text(s,`中期汇报 · 2026.10.25                                      ${String(p.slides.items.length).padStart(2,'0')}`,64,675,1150,28,17,C.muted);
 s.speakerNotes.text=source;return s;
}
function bullet(s,lines,x,y,w,size=27,gap=80){lines.forEach((t,i)=>text(s,t,x,y+i*gap,w,gap-8,size));}
function image(s,p,x,y,w,h,alt,fit='contain'){const absolute=path.isAbsolute(p)?p:path.join(ROOT,p);return s.images.add({blob:new Uint8Array(fss.readFileSync(absolute)),contentType:'image/png',alt,fit,position:{left:x,top:y,width:w,height:h}});}
function node(s,label,x,y,w,h){const sh=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'#F4F8F7',line:{fill:'#8BAEB3',width:1.5}});text(s,label,x+16,y+16,w-32,h-20,25,C.ink,true);return sh;}
function table(s,rows,widths,x,y,h){
 const t=s.tables.add({rows:rows.length,columns:widths.length,left:x,top:y,width:widths.reduce((a,b)=>a+b),height:h,columnWidths:widths,values:rows});
 t.borders.assign({fill:'#D2DFE2',style:'solid',width:1});
 for(let r=0;r<rows.length;r++)for(let c=0;c<widths.length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';cell.text.style={typeface:FONT,fontSize:24,color:C.ink,bold:r===0,alignment:c===0?'left':'center',verticalAlignment:'middle',insets:10};
 }
 return t;
}
function chart(s,values,categories,x,y,w,h,percent=true){
 const displayed=values.map(v=>Number(v.toFixed(percent?4:2)));
 const ch=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,
  series:[{name:percent?'成功率':'终局距离 米',values:displayed,fill:C.teal,points:displayed.map((v,i)=>({idx:i,fill:i===0?C.gray:C.teal}))}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:95},hasLegend:false,
  xAxis:{textStyle:{fontSize:21,fill:C.ink},line:{fill:'#C6D5D8',width:1}},
  yAxis:{min:0,...(percent?{max:1,majorUnit:.2}:{}),numberFormatCode:percent?'0%':'0',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:'#E0E9EA',width:1}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:percent?'0.0%':'0.0',textStyle:{fontSize:24,fill:C.ink,bold:true}}});
 applyPresentationChartFont(ch,{fontFamily:FONT});return ch;
}
async function blob(file,b){await fs.writeFile(file,new Uint8Array(await b.arrayBuffer()));}

async function screens(){
 const {chromium}=require('C:/Users/DFWJ/.agents/skills/playwright-skill/node_modules/playwright');
 const browser=await chromium.launch({executablePath:'C:/Users/DFWJ/AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe',headless:false,args:['--disable-gpu','--disable-background-networking','--disable-component-update','--disk-cache-size=1','--no-first-run']});
 const requests=[],blocked=[];const base='http://127.0.0.1:8769';
 try{
  const context=await browser.newContext({viewport:{width:1440,height:960}});
  await context.route('**/*',r=>{const q=r.request(),u=q.url();if(q.method()!=='GET'||(/^https?:/.test(u)&&!u.startsWith(base+'/'))){blocked.push({method:q.method(),url:u.split('?')[0]});return r.abort();}requests.push(u.split('?')[0]);return r.continue();});
  const page=await context.newPage();await page.goto(base+'/');await page.waitForSelector('#page-title');
  await page.locator('nav button[data-page="create"]').click();await page.screenshot({path:path.join(OUT,'图表/平台当前工作台.png')});
  await page.goto(base+'/ground-walk');await page.waitForSelector('#saved');
  if(await page.locator('#saved option').count()>1){await page.locator('#saved').selectOption({index:1});await page.locator('#open').click();await page.waitForFunction(()=>document.getElementById('goal').naturalWidth>0);}
  await page.screenshot({path:path.join(OUT,'图表/地面人工观察.png')});
  await page.goto(base+'/spatial-evidence?case=112&method=Spatial&seed=0&arm=Full');
  await page.waitForFunction(()=>document.getElementById('download')&&!document.getElementById('download').disabled&&document.getElementById('target-image').naturalWidth>0&&document.getElementById('current-image').naturalWidth>0);
  await page.screenshot({path:path.join(OUT,'图表/地面目标与空间证据.png')});
  await fs.writeFile(path.join(OUT,'核验/当前界面截图回执.json'),JSON.stringify({passed:true,version:'4.13.1',captured:3,source:base,allowed_GETs:requests,blocked_non_GET_or_external:blocked,model_calls:0,actions:0},null,2),{flag:'wx'});
 }finally{await browser.close();}
}

async function cardReview(){
 const {chromium}=require('C:/Users/DFWJ/.agents/skills/playwright-skill/node_modules/playwright');
 const browser=await chromium.launch({executablePath:'C:/Users/DFWJ/AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe',headless:false,args:['--disable-gpu','--disable-background-networking','--disable-component-update','--disk-cache-size=1','--no-first-run']});
 const errors=[],blocked=[];
 try{
  const context=await browser.newContext({viewport:{width:1440,height:960}});
  await context.route('**/*',r=>{if(/^https?:/.test(r.request().url())){blocked.push(r.request().url());return r.abort();}return r.continue();});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(pathToFileURL(path.join(OUT,'交付/演示操作卡.html')).href);
  await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0));
  const desktop=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,details:document.querySelectorAll('details').length}));
  await page.screenshot({path:path.join(OUT,'审阅/演示卡桌面.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(async()=>{await document.fonts.ready;await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));});
  const narrow=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight}));
  await page.screenshot({path:path.join(OUT,'审阅/演示卡窄屏_v2.png'),fullPage:true});
  if(errors.length||desktop.scroll>desktop.width||narrow.scroll>narrow.width||desktop.details!==8)throw Error(JSON.stringify({errors,desktop,narrow}));
  await fs.writeFile(path.join(OUT,'核验/演示卡可见GUI_v2.json'),JSON.stringify({passed:true,desktop,narrow,errors,blocked_external:blocked,model_calls:0,navigation_actions:0,source:'local card only'},null,2),{flag:'wx'});
 }finally{await browser.close();}
}

function build(f,coverOnly=false){
 const p=Presentation.create({slideSize:{width:1280,height:720}});
 let s=p.slides.add();s.background.fill='#FFFFFF';
 image(s,'project/src/webapp/static_v7/assets/geonav-terrain.png',690,0,590,720,'GeoNav既有装饰地形资产','cover');
 text(s,'基于主动探索的\n视觉地理定位\n方法设计与实现',64,115,620,290,58,C.ink,true);
 text(s,'综合工程设计中期汇报',68,435,580,50,30,C.teal,true);
 text(s,'陈德昕  李世显  陈龙辉\n指导教师 王笛\n2026年10月25日',68,525,590,125,25,C.muted);
 s.speakerNotes.text='研究范围截至2026-10-25。航拍已形成稳定成果，地面照片目标自主导航仍未完成。装饰图来自现有平台资产，不作为真实区域或实验结果。';
 if(coverOnly)return p;

 s=heading(p,'任务目标与阶段定位','来源：docs/PROJECT_STATE.md；当前中期内容与证据JSON。最终主线由用户调整为地面照片目标主动导航。');
 ['地面照片','空间关联证据','自主移动','对应目标格'].forEach((t,i)=>{text(s,t,64+i*300,180,230,70,32,C.teal,true);if(i<3)text(s,'→',310+i*300,180,60,70,38,C.gray);});
 text(s,'最终任务链路',64,120,1100,36,24,C.muted);
 bullet(s,['已形成成果  航拍目标导航＋可交互平台','正在攻关  地面照片与航拍位置的可靠关联','尚未完成  地面目标驱动的自主探索闭环'],64,340,1130,30,95);

 s=heading(p,'环境交互与信息权限','来源：AGENTS.md四动作/公开Observation约定；现有环境与平台内核。');
 text(s,'Agent可用',64,145,500,50,32,C.teal,true);text(s,'仅环境与观察者掌握',680,145,520,50,32,C.amber,true);
 bullet(s,['当前局部航拍、给定目标','公开位置、访问历史与剩余预算','上、右、下、左四个动作'],64,225,520,28,88);
 bullet(s,['真实目标坐标与距离','未访问图块和完整特征库','事后真值叠加、失败诊断标签'],680,225,520,28,88);
 text(s,'动作 → 新局部观测 → 更新记忆 → 下一步动作',64,535,1120,52,30,C.ink,true);
 text(s,'越界原地耗一步；首次到达成功；末步到达仍计成功。',64,601,1120,46,24,C.muted);

 s=heading(p,'航拍M0的目标线索与探索','来源：agents/boundary_policy.py、edge_cue.py、decoupled_cue.py；M0冻结配置。参数为静态封存规模，无本轮模型加载。');
 node(s,'当前航拍＋公开状态',64,165,330,105);node(s,'给定目标＋当前图',64,365,330,105);
 node(s,'冻结表征＋GRU探索\n约79.8万任务参数',465,165,315,105);node(s,'Edge邻接与方向线索\n约13.7万任务参数',465,365,315,105);
 text(s,'→',404,185,50,70,40,C.teal);text(s,'→',404,385,50,70,40,C.teal);
 text(s,'→',800,185,50,70,40,C.teal);text(s,'→',800,385,50,70,40,C.teal);
 node(s,'可信线索优先\n\nq≥0.50、合法且未访问\n否则保留探索建议',870,165,330,305);
 text(s,'q(d) = p(邻接) × p(d | 邻接)',64,532,1120,50,30,C.teal,true);
 text(s,'探索器目标通道用固定拟合均值；视觉骨干冻结，其资源仍属于整个系统。',64,599,1130,52,24,C.muted);

 s=heading(p,'航拍结果按协议分别验收',f.aerial.map(x=>x.source).join('\n')+'\n三权重重复同队列，不增加独立来源；不同数据/预算不直接排名。');
 table(s,[['批次','源/每权重题','网格 / 预算','SR','SG 格'],...f.aerial.map(x=>[x.name,`${x.sources} / ${x.tasks_per_seed}`,x.protocol,`${(x.sr*100).toFixed(2)}%`,x.sg.toFixed(3)])],[210,240,260,190,190],64,160,340);
 text(s,'S3：10个独立源文件；地理足迹无重叠尚未核实。',64,540,1130,45,25,C.muted);
 text(s,'S4提高切图密度；SwissView是同模态迁移，均不证明地面照片定位。',64,592,1130,58,25,C.muted);

 s=heading(p,'9 km²连续区域的独立确认',f.area.source+'\n完整区域分组95% SR差区间[+7.07,+11.02]个百分点。恢复295、损伤88；三权重仍是10个地理区域。');
 text(s,'10新区 · 750路线×3权重 · 300米/格 · 10×10 / B20',64,120,1140,42,25,C.muted);
 chart(s,[f.area.reference.sr,f.area.candidate.sr],['M0','径向保护'],65,190,540,335,true);
 chart(s,[f.area.reference.mean_sg_m,f.area.candidate.mean_sg_m],['M0','径向保护'],675,190,540,335,false);
 text(s,'成功率',180,540,350,44,26,C.teal,true);text(s,'全终局SG 米',810,540,350,44,26,C.teal,true);
 text(s,'SR +9.20个百分点；恢复295、损伤88，净增加207条成功。',64,601,1140,50,27,C.ink,true);

 s=heading(p,'更大新域仍有明确边界','来源：'+f.new_domain.map(()=>f.crossview.source).slice(0,0).join('')+'DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1/主对照/汇总.json。两网格共用10区域；独立确认主门槛均未过，未升级默认。');
 chart(s,f.new_domain.map(x=>x.sr),['10×10 / B20','15×15 / B20'],64,150,640,420,true);
 bullet(s,['Coverage SR 54.62% / 20.27%','两网格共用10区，不能算20地图','尺度、域与输入表征差异仍需分离'],750,200,455,26,105);
 text(s,'未过结果保留；其它航拍协议的通过结论仍按原范围有效。',64,603,1140,50,26,C.muted);

 s=heading(p,'平台已经支持真实实验流程','来源：当前4.13.1只读截图；原PLATFORM系列工程验收。截图不启动新模型或动作。');
 image(s,path.join(OUT,'图表/平台当前工作台.png'),64,135,800,490,'当前4.13.1工作台');
 bullet(s,['创建任务与场景','当前题／整个队列','人工探索与Agent接管','保存、重开、同题对照'],915,170,300,25,95);

 s=heading(p,'地面目标已接入人工观察','来源：PLATFORM-024既有四步人工工程样例截图（当时界面），不是本轮新运行。人工移动/实时编码为已实现工程能力，未通过地面目标自主方向决策。');
 image(s,'平台/实验工作台_v4/地面目标连续观测与人工探索_v1/审阅/连续探索桌面.png',64,135,810,435,'既有四步人工工程样例，含越界和重访');
 bullet(s,['目标照片固定','逐步读取当前航拍','保存已访过程和证据'],920,170,290,26,105);
 text(s,'既有四步人工工程记录 · 当时界面 · 本轮未重跑',64,580,810,26,18,C.muted);
 text(s,'当前为人工／实验模式；自主方向策略与可靠空间定位尚未完成。',64,610,1140,40,25,C.amber,true);

 s=heading(p,'跨视角小头没有通过定位门槛',f.crossview.source+'\nFull 450开发对全部拒绝；9是三城×三种子，不是9个独立区域。固定位置上界3/9仍低于6/9门槛；不追加同因素训练。');
 text(s,'0 / 9',64,165,350,120,80,C.amber,true);text(s,'接受且误差≤5米',64,300,360,60,29,C.ink,true);
 table(s,[['城市','三种子位置误差 米'],['Amsterdam','18.306 / 20.384 / 18.306'],['Krakow','6.438 / 6.438 / 6.438'],['Stockholm','1.431 / 1.431 / 1.431']],[245,520],465,160,320);
 text(s,'宏联合66.67% = 全拒绝参照，不能当作定位成功率。',64,550,1150,55,31,C.amber,true);
 text(s,'候选按停止条件收束，优先补任务匹配的监督与可辨识性证据。',64,614,1150,40,24,C.muted);

 s=heading(p,'中期成果与最终缺项','来源：当前PROJECT_STATE和报告证据表。原航拍成果继续进入最终系统；地面目标自主闭环为未完成主线。');
 text(s,'中期已经形成',64,150,530,55,33,C.teal,true);text(s,'最终仍需补齐',690,150,520,55,33,C.amber,true);
 bullet(s,['已验证航拍导航与区域拓展','可交互平台、保存与同题对照','公式、参数、科研图与失败边界'],64,245,535,28,100);
 bullet(s,['可信地面目标空间证据','同预算目标条件自主闭环','冻结新源确认与完整交付'],690,245,530,28,100);
 text(s,'多Agent、实飞和任意照片全球定位可留未来工作。',64,605,1140,48,26,C.muted);

 s=heading(p,'下一阶段计划与演示','来源：本轮中期报告四；演示只读旧结果可零模型运行。需要新实时实验时由用户显式创建临时实验，成绩不混入正式验收。');
 table(s,[['顺序','工作','检查点'],['1','固定标签与来源分工','目标定义可信，开发/确认分离'],['2','轻量空间证据与最小自主闭环','同预算规则/无目标及目标干预'],['3','冻结新来源确认','SR、SG、恢复/损伤与区域区间'],['4','整合平台与最终材料','创建→运行→保存→同题对照']],[110,430,570],64,145,335);
 text(s,'三分钟演示  工作台 → 航拍历史案例 → 地面人工观察 → 同题空间证据',64,535,1140,72,27,C.teal,true);
 text(s,'http://127.0.0.1:8769/   ·   数据与模型按绑定准备，平台不自动下载。',64,615,1140,42,23,C.muted);
 return p;
}

async function main(){
 const mode=process.argv[2]||'build';if(mode==='screens'){await screens();return;}if(mode==='card-review'){await cardReview();return;}
 if(mode==='finish'){
  const final=path.join(OUT,'交付/中期成果汇报.pptx'),pdf=path.join(OUT,'交付/中期成果汇报.pdf');
  const validation=JSON.parse(await fs.readFile(path.join(OUT,'核验/PPT结构与可编辑性.json'),'utf8'));
  const hash=createHash('sha256').update(await fs.readFile(final)).digest('hex');
  if(hash!==validation.finalSha256)throw Error('PPT changed after finalization');
  const native=JSON.parse((await fs.readFile(path.join(OUT,'核验/PPT原生PDF导出.json'),'utf8')).replace(/^\uFEFF/,''));
  if(!native.passed||(native.slides??native.pages_or_slides)!==12||!fss.existsSync(pdf))throw Error('native PDF not complete');
  await fs.writeFile(path.join(OUT,'核验/PPT生成回执.json'),JSON.stringify({passed:true,slides:12,editable_tables:[5,10,12],editable_charts:[6,7],literal_chart_rounding:'SR 4 decimal fractions;SG 2 decimal metres;unrounded values retained in facts',
   pptx_sha256:hash,pdf_backend:native.engine,artifact_pdf_failure:'missing bundled #standard-fonts/Helvetica;no dependency edits/install',source_pptx_unchanged:true,model_calls:0},null,2),{flag:'wx'});
  console.log('final PPT/native PDF and editable evidence verified');return;
 }
 const facts=JSON.parse(await fs.readFile(path.join(OUT,'核验/中期内容与证据.json'),'utf8'));const p=build(facts,mode==='probe');
 if(mode==='probe'){
  await blob(path.join(TMP,'封面试排.png'),await p.export({slide:p.slides.items[0],format:'png',scale:1}));console.log('cover rendered');return;
 }
 const stage=path.join(TMP,'ppt-finalizer-r8');await fs.mkdir(stage,{recursive:true});
 const candidate=path.join(stage,'candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidate);
 const delivery=path.join(OUT,'交付');await fs.mkdir(delivery,{recursive:true});
 const final=path.join(delivery,'中期成果汇报.pptx');
 const result=await finalizePresentation({workspaceDir:OUT,candidatePath:candidate,finalPath:final,
  pythonExecutable:path.join(DEPS,'python/python.exe'),
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[5,10,12].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[5,10,12],requiredNativeChartOwnerSlides:[6,7],
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(OUT,'核验/PPT结构与可编辑性.json')});
 const checked=await PresentationFile.importPptx(await FileBlob.load(final));
 for(let i=0;i<checked.slides.items.length;i++)await blob(path.join(OUT,'审阅',`slide-${String(i+1).padStart(2,'0')}.png`),await checked.export({slide:checked.slides.items[i],format:'png',scale:1}));
 await fs.writeFile(path.join(OUT,'核验/PPT构建回执.json'),JSON.stringify({slide_count:p.slides.items.length,editable_tables:[5,10,12],editable_charts:[6,7],literal_chart_data_from_sealed_facts:true,model_calls:0,finalizer:result,pdf_backend:'native WPS export required; bundled Helvetica resource previously unavailable'},null,2),{flag:'wx'});
 console.log('12 editable slides exported; use export_midterm_refresh_v1.ps1 -Kind slides then finish');
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
