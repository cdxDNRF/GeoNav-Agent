// Visible PLATFORM032 acceptance; owned D workspace, request-counted, no downloads.
const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/DFWJ/.agents/skills/playwright-skill/node_modules/playwright');
const ROOT=path.resolve(__dirname,'../../..'),OUT=path.join(ROOT,'平台/实验工作台_v4/航拍统一接入与小队列_v1');
process.env.TEMP=path.join(OUT,'浏览器临时');process.env.TMP=process.env.TEMP;process.env.XDG_CACHE_HOME=process.env.TEMP;
const phase=process.argv[2],label=process.argv[3]||'r2',BASE=JSON.parse(fs.readFileSync(path.join(OUT,`候选就绪_${label}.json`),'utf8')).url;
const attempt=process.argv[4]||'a1';const receipt=path.join(OUT,`GUI_${phase}_${label}_${attempt}.json`),journal=path.join(OUT,`GUI_${phase}_${label}_${attempt}.jsonl`);
if(!['preflight','accept','readonly'].includes(phase)||fs.existsSync(receipt)||fs.existsSync(journal))throw Error('Fresh phase/label required');
const log=v=>fs.appendFileSync(journal,JSON.stringify(v)+'\n');
(async()=>{
 const browser=await chromium.launch({headless:false,executablePath:'C:/Users/DFWJ/AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe'});
 const ctx=await browser.newContext({viewport:{width:1500,height:1000}}),page=await ctx.newPage(),errors=[];let sessions=0,actions=0,checks=0;
 page.on('pageerror',e=>errors.push(String(e)));
 await ctx.route('**/*',async route=>{const req=route.request(),u=new URL(req.url());if(u.origin!==BASE)return route.abort();
  if(req.method()==='POST'){
   const p=u.pathname,b=req.postDataJSON();if(phase!=='accept'&&p!=='/api/controllers/release'&&p!=='/api/controllers/heartbeat')return route.abort();
   if(phase==='accept'){
    if(!p.startsWith('/api/controllers/'))return route.abort();
    if(p.endsWith('/create')||p.endsWith('/fork')){if(++sessions>2)return route.abort()}
    if(p.endsWith('/step')){if(++actions>20)return route.abort()}
   }log({method:'POST',path:p,body:b,sessions,actions});
  }return route.continue();
 });
 async function loaded(p=page){await p.waitForFunction(()=>data&&data.datasets?.length&&!busy,{timeout:25000})}
 async function images(p=page){await p.waitForFunction(()=>['target-image','current-image'].every(id=>{const im=document.getElementById(id);return im.complete&&im.naturalWidth}),{timeout:15000})}
 async function select(key){await page.locator('#search').fill(key);await page.locator('.task-row').first().click();await page.waitForFunction(()=>!document.getElementById('create').disabled);await images()}
 try{
  await page.goto(BASE+'/controllers',{waitUntil:'networkidle'});await loaded();assert.equal(await page.locator('#dataset option').count(),2);checks++;
  assert.equal(await page.evaluate(()=>data.tasks.filter(t=>t.dataset_id==='masa-s2').length),100);checks++;
  await page.locator('#dataset').selectOption('camera-development');assert.equal(await page.locator('#initial-controller option[value=M0]').count(),0);checks++;
  await page.locator('#dataset').selectOption('masa-s2');assert.equal(await page.locator('#initial-controller option[value=M0]').count(),1);checks++;
  if(phase==='preflight'){
   assert.equal(await page.locator('#seed').inputValue(),'0');assert(await page.locator('#restart-label').isHidden());checks++;
   const first=await page.evaluate(()=>data.tasks.find(t=>t.dataset_id==='masa-s2').task_key);await select(first);
   const cells=await page.locator('.tile').evaluateAll(xs=>xs.map(e=>{const r=e.getBoundingClientRect();return [r.width,r.height]}));assert(cells.length===25&&cells.every(([w,h])=>Math.abs(w-h)<1.1));checks++;
   await page.screenshot({path:path.join(OUT,`截图/候选_${attempt}_桌面预检.png`),fullPage:true});await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));checks++;
   await page.screenshot({path:path.join(OUT,`截图/候选_${attempt}_窄屏预检.png`),fullPage:true});
   // All following model/actions are synthetic HTTP responses, never server calls.
   const mock=await ctx.newPage();mock.on('pageerror',e=>errors.push(String(e)));let job=null,q=null,syntheticSteps=0;
   const task=k=>({task_id:k,task_key:k,dataset_id:'masa-s2',city:'synthetic',role:'development',start:0,grid:5,budget:10,cell_size_m:null,target_modality:'aerial'});
   const result=()=>({success:false,SG_cells:3,SG_m:null,moved_m:null,moved_cells:job.step,steps:job.step,coverage:.12,human_actions:job.trajectory.filter(t=>t.origin==='human').length,rule_actions:0,model_actions:job.trajectory.filter(t=>t.origin==='model').length});
   function create(k='synthetic0',controller='M0'){job={id:'1'.repeat(32),kind:'aerial',name:'synthetic',dataset_id:'masa-s2',task:task(k),seed:0,definition:{input_sha256:{}},controller,locked_agent:controller==='M0'?'M0':controller==='frontier'?'frontier':null,controller_class:'unstarted',step:0,budget:10,position:0,visited:[0],trajectory:[{step:0,cell:0}],status:'ready',saved:false,readonly:false,saved_task:false,observer_revealed:false,model_calls:0,result:null};return job}
   function advance(){const old=job.position;job.step++;job.position=old+1;job.model_calls+=job.controller==='M0'?1:0;job.controller_class=job.controller==='manual'?'mixed':'model';job.trajectory.push({step:job.step,cell:job.position,action:'right',origin:job.controller==='manual'?'human':'model',reason:'synthetic'});syntheticSteps++;return job}
   const empty=()=>({batches:[],broken:0});
   await mock.route('**/api/controllers/**',async route=>{const req=route.request(),op=new URL(req.url()).pathname.split('/').at(-1),b=req.method()==='POST'?req.postDataJSON():{};let v;
    if(op==='catalog')v={datasets:[{id:'masa-s2',name:'synthetic',controllers:['manual','frontier','M0'],seed_choices:[0]}],tasks:[task('synthetic0')],saved_tasks:[],saved_runs:[],recovery:{broken:0,aerial:{count:0}}};
    else if(op==='image'||op==='preview-image')return route.fulfill({contentType:'image/png',body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=','base64')});
    else if(op==='create')v=create('synthetic0',b.controller);
    else if(op==='step')v=advance();
    else if(op==='mode'){job.controller=b.controller;v=job}
    else if(op==='finish'){job.status='cancelled';job.result=result();v=job}
    else if(op==='save-run'){job.saved=true;job.readonly=true;v=job}
    else if(op==='fork')v=create('synthetic0',b.controller);
    else v={released:true};return route.fulfill({json:v});
   });
   await mock.route('**/api/controller-batches/**',async route=>{const req=route.request(),op=new URL(req.url()).pathname.split('/').at(-1);log({mock_batch:op});let v;
    if(op==='catalog')v=empty();
    else if(op==='create'){
     q={id:'2'.repeat(32),definition:{name:'synthetic queue',dataset_id:'masa-s2',seed:0},status:'ready',controllable:true,needs_restart:false,action_attempts:0,model_calls:0,current:null,rows:['M0','frontier'].map((controller,index)=>({index,task_key:'synthetic0',controller,status:'not_run',step:0,result:null,saved_ref:null})),metrics:{arms:{M0:{successes:0,planned:1,completed:0,SR:0,SG_cells:null},frontier:{successes:0,planned:1,completed:0,SR:0,SG_cells:null}},paired:{completed:0,planned:1,baseline:'M0',comparison:'frontier',recovered:0,damaged:0,net:0}}};v=q;
    }else if(op==='step'){
     if(!q.current)create();advance();q.current=job;q.action_attempts++;q.model_calls++;q.rows[0].step=job.step;q.rows[0].status='running';q.status='paused';
     if(job.step===2){job.status='completed';job.saved=true;job.readonly=true;job.result=result();q.rows[0].status='completed';q.rows[0].saved_ref={id:job.id,kind:'aerial'};q.rows[0].result=job.result;q.metrics.arms.M0.completed=1;q.metrics.arms.M0.SG_cells=3}v=q;
    }else v=q;return route.fulfill({json:v});
   });
   await mock.goto(BASE+'/controllers',{waitUntil:'networkidle'});await loaded(mock);await mock.locator('.task-row').click();await mock.waitForFunction(()=>!document.getElementById('create').disabled);
   await mock.locator('#initial-controller').selectOption('M0');await mock.locator('#create').click();await mock.waitForFunction(()=>state?.controller==='M0'&&!busy);
   await mock.locator('#rule-step').click();await mock.waitForFunction(()=>state?.step===1&&!busy);await mock.locator('#auto-current').click();await mock.waitForFunction(()=>state?.step===2);await mock.locator('#pause').click();await mock.waitForFunction(()=>!running&&!busy);checks++;
   await mock.locator('#controller').selectOption('manual');await mock.waitForFunction(()=>state?.controller==='manual'&&!busy);await mock.locator('[data-action=left]').click();await mock.waitForFunction(()=>state?.step===3&&!busy);
   await mock.locator('#finish').click();await mock.waitForFunction(()=>state?.result&&!busy);await mock.locator('#save-run').click();await mock.waitForFunction(()=>state?.readonly&&!busy);checks++;
   assert((await mock.locator('#result').textContent()).includes('SG 3格'));checks++;
   await mock.locator('#new').click();await mock.waitForFunction(()=>state===null&&!busy);await mock.locator('.task-row').click();await mock.waitForFunction(()=>!document.getElementById('queue-add').disabled);await mock.locator('#queue-add').click();await mock.locator('#queue-create').click();await mock.waitForFunction(()=>batch&&!busy);
   await mock.locator('#batch-current').click();await mock.waitForFunction(()=>batch?.rows[0].status==='completed'&&!running&&!busy);assert.equal(await mock.evaluate(()=>batch.rows[1].status),'not_run');checks++;
   assert.equal(await mock.locator('#batch-table button').first().isDisabled(),false);checks++;await mock.close();
   assert.equal(sessions,0);assert.equal(actions,0);assert.equal(errors.length,0);fs.writeFileSync(receipt,JSON.stringify({passed:true,checks,real_sessions:0,real_actions:0,synthetic_actions:syntheticSteps,errors},null,2),{flag:'wx'});
  }else{
   const frozen=JSON.parse(fs.readFileSync(path.join(OUT,'首真实动作前绑定.json'),'utf8')),records=[];
   const bid=JSON.parse(fs.readFileSync(path.join(OUT,'真实队列身份.json'),'utf8')).id;
   await page.locator('#saved-batches').selectOption(bid);await page.locator('#queue-open').click();await page.waitForFunction(()=>batch?.status==='completed'&&!busy);assert.equal(await page.locator('#batch-table tbody tr').count(),20);checks++;
   await page.locator('#batch-table button').first().click();await page.waitForFunction(()=>state?.readonly&&!busy);await images();checks++;
   if(phase==='accept'){
    await page.locator('#new').click();await page.waitForFunction(()=>!batch&&!state&&!busy);await select(frozen.task_keys[0]);await page.locator('#initial-controller').selectOption('M0');await page.locator('#run-name').fill('验收A · M0单题');await page.locator('#create').click();await page.waitForFunction(()=>state?.step===0&&!busy);
    await page.locator('#auto-current').click();await page.waitForFunction(()=>state?.status==='completed'&&!running&&!busy,{timeout:60000});assert.equal(await page.evaluate(()=>batch),null);checks++;
    await page.locator('#save-run').click();await page.waitForFunction(()=>state?.readonly&&!busy);records.push(await page.evaluate(()=>state));
    await page.locator('#initial-controller').selectOption('M0');await page.locator('#fork').click();await page.waitForFunction(()=>!state?.readonly&&state?.step===0&&!busy);
    await page.locator('#rule-step').click();await page.waitForFunction(()=>state?.step===1&&!busy);await page.locator('#controller').selectOption('manual');await page.waitForFunction(()=>state?.controller==='manual'&&!busy);
    const a=await page.evaluate(()=>state.position%5<4?'right':'left');await page.locator('[data-action='+a+']').click();await page.waitForFunction(()=>state?.step===2&&!busy);if(!await page.evaluate(()=>['completed','cancelled','failed'].includes(state.status))){await page.locator('#finish').click();await page.waitForFunction(()=>state?.result&&!busy)}
    await page.locator('#save-run').click();await page.waitForFunction(()=>state?.readonly&&!busy);records.push(await page.evaluate(()=>state));assert.equal(records[1].controller_class,'mixed');checks++;
    await page.locator('#compare-a').selectOption('aerial:'+records[0].id);await page.locator('#compare-b').selectOption('aerial:'+records[1].id);await page.locator('#compare').click();await page.waitForFunction(()=>document.getElementById('comparison').textContent.includes('B−A')&&!busy);checks++;
    await page.reload({waitUntil:'networkidle'});await page.waitForFunction(()=>state?.readonly&&!busy);await page.locator('#replay').focus();await page.locator('#replay').press('Home');await page.waitForFunction(()=>document.getElementById('steps').textContent.startsWith('0 /'));checks++;
    assert.equal(sessions,2);assert(actions<=20);log({kind:'actual_records',records});
   }
   await images();await page.screenshot({path:path.join(OUT,`截图/统一航拍_${phase}_桌面.png`),fullPage:true});await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));checks++;
   await page.screenshot({path:path.join(OUT,`截图/统一航拍_${phase}_窄屏.png`),fullPage:true});assert.equal(errors.length,0);
   fs.writeFileSync(receipt,JSON.stringify({passed:true,checks,real_sessions:sessions,real_actions:actions,records,errors,formal_evaluation:false},null,2),{flag:'wx'});
  }
  console.log(JSON.stringify({passed:true,phase,checks,sessions,actions}));
 }catch(e){const snapshots=[];for(const p of ctx.pages()){snapshots.push(await p.evaluate(()=>({notice:document.getElementById('notice')?.textContent,step:typeof state!=='undefined'?state?.step:null,batch:typeof batch!=='undefined'?batch:null})).catch(()=>({closed:true})))}fs.writeFileSync(path.join(OUT,`GUI_failure_${phase}_${label}_${attempt}.json`),JSON.stringify({error:String(e),errors,snapshots},null,2),{flag:'wx'});throw e}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
