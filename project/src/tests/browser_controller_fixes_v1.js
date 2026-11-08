// Visible4.17 read-only/synthetic GUI acceptance; no action/model endpoints.
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/DFWJ/.agents/skills/playwright-skill/node_modules/playwright');
const ROOT=path.resolve(__dirname,'../../..'),OUT=path.join(ROOT,'平台/实验工作台_v4/队列恢复与证据修复_v1'),WORK=path.join(OUT,'工作区');
const label=process.argv[2]||'r1',BASE=JSON.parse(fs.readFileSync(path.join(OUT,`候选就绪_${label}.json`),'utf8')).url;
process.env.TEMP=path.join(OUT,'浏览器临时');process.env.TMP=process.env.TEMP;process.env.XDG_CACHE_HOME=process.env.TEMP;
const receipt=path.join(OUT,`GUI_${label}.json`);if(fs.existsSync(receipt))throw Error('Receipt already exists');
const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const bid=fs.readdirSync(path.join(WORK,'controller_batches'))[0],progress=path.join(WORK,'controller_batches',bid,'progress.json'),before=digest(progress);
const row=JSON.parse(fs.readFileSync(progress,'utf8')).rows[0],sceneDirs=fs.readdirSync(path.join(WORK,'scenes'));
const original=sceneDirs.map(s=>path.join(WORK,'scenes',s,'runs',row.saved_ref.id+'.json')).find(p=>fs.existsSync(p));
const hidden=path.join(OUT,'测试','暂存的轨迹副本.json'),badID='f'.repeat(32),badFolder=path.join(WORK,'controller_batches',badID);
if(!original||!original.startsWith(WORK+path.sep)||!badFolder.startsWith(WORK+path.sep)||fs.existsSync(hidden)||fs.existsSync(badFolder))throw Error('Unsafe/used synthetic workspace');
(async()=>{const browser=await chromium.launch({headless:false,executablePath:'C:/Users/DFWJ/AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe'}),ctx=await browser.newContext({viewport:{width:1500,height:1000}}),page=await ctx.newPage();let checks=0,blocked=0;const errors=[];
page.on('pageerror',e=>errors.push(String(e)));await ctx.route('**/*',r=>{const u=new URL(r.request().url());if(u.origin!==BASE)return r.abort();if(r.request().method()==='POST'&&!u.pathname.endsWith('/release')&&!u.pathname.endsWith('/heartbeat')){blocked++;return r.abort()}return r.continue()});
async function ready(){await page.waitForFunction(()=>data&&!busy,{timeout:20000})}
async function openBatch(){await page.locator('#saved-batches').selectOption(bid);await page.locator('#queue-open').click();await page.waitForFunction(()=>batch&&!busy)}
async function api(route){const r=await page.request.get(BASE+route);return {status:r.status(),data:await r.json()}}
try{
 await page.goto(BASE+'/controllers',{waitUntil:'networkidle'});await ready();await openBatch();assert(await page.evaluate(()=>batch.evidence_complete));assert.equal(await page.evaluate(()=>batch.metrics.arms.M0.SR),.8);assert.equal(await page.evaluate(()=>batch.metrics.arms.frontier.SG_cells),2.7);checks+=3;
 fs.renameSync(original,hidden);
 const u='/api/controller-batches/state?owner='+('a'.repeat(32))+'&id='+bid;let v=await api(u);assert.equal(v.status,200);assert.equal(v.data.evidence_complete,false);assert.equal(v.data.metrics.arms.M0.SR,null);assert.equal(v.data.cached_metrics.arms.M0.SR,.8);checks+=3;
 for(const fmt of ['json','csv']){const r=await page.request.get(BASE+'/api/controller-batches/export?owner='+('a'.repeat(32))+'&id='+bid+'&format='+fmt);assert.equal(r.status(),400)}checks++;
 await page.locator('#new').click();await page.waitForFunction(()=>!batch&&!busy);await openBatch();assert((await page.locator('#batch-summary').textContent()).includes('证据不完整'));assert(await page.locator('#batch-json').isDisabled());assert(await page.locator('#batch-csv').isDisabled());assert(await page.locator('#batch-table button').first().isDisabled());checks+=2;
 await page.screenshot({path:path.join(OUT,'截图/证据异常提示.png'),fullPage:true});assert.equal(digest(progress),before);checks++;
 fs.renameSync(hidden,original);v=await api(u);assert(v.data.evidence_complete);assert.equal(v.data.metrics.arms.M0.SR,.8);checks++;
 fs.mkdirSync(badFolder);fs.writeFileSync(path.join(badFolder,'definition.json'),'null');fs.writeFileSync(path.join(badFolder,'progress.json'),'{}');
 v=await api('/api/controller-batches/catalog?owner='+('a'.repeat(32)));assert.equal(v.status,200);assert.equal(v.data.broken,1);assert(v.data.batches.some(b=>b.id===bid));checks++;
 // Only the known regenerated malformed fixture can be deleted.
 assert(path.resolve(badFolder)===path.resolve(WORK,'controller_batches',badID));fs.rmSync(badFolder,{recursive:true});
 await page.reload({waitUntil:'networkidle'});await ready();const catalog=await page.evaluate(()=>data);
 await page.route('**/api/controllers/catalog',r=>r.fulfill({json:{...catalog,datasets:catalog.datasets.filter(d=>d.id==='camera-development'),tasks:catalog.tasks.filter(t=>t.dataset_id==='camera-development')}}));
 await page.reload({waitUntil:'networkidle'});await ready();assert.equal(await page.locator('#dataset').inputValue(),'camera-development');assert.equal(await page.locator('#seed').inputValue(),'0');assert.equal(await page.locator('.task-row').count(),192);checks+=2;
 await page.screenshot({path:path.join(OUT,'截图/仅地面目录默认选择.png'),fullPage:true});
 await page.unroute('**/api/controllers/catalog');await page.route('**/api/controllers/catalog',r=>r.fulfill({json:{...catalog,datasets:[],tasks:[]}}));
 await page.reload({waitUntil:'networkidle'});await ready();assert.equal(await page.locator('.task-row').count(),0);assert(await page.locator('#create').isDisabled());assert((await page.locator('#notice').textContent()).includes('没有可用任务'));checks++;
 await page.unroute('**/api/controllers/catalog');await page.reload({waitUntil:'networkidle'});await ready();await openBatch();await page.locator('#batch-table button').first().click();await page.waitForFunction(()=>state?.readonly&&!busy);checks++;
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(OUT,'截图/修复后窄屏.png'),fullPage:true});checks++;
 assert.equal(blocked,0);assert.equal(errors.length,0);assert.equal(digest(progress),before);fs.writeFileSync(receipt,JSON.stringify({passed:true,checks,model_calls:0,navigation_actions:0,source_records_readonly:true,progress_bytes_unchanged:true,camera_only_dataset_default:true,empty_catalog_safe:true,missing_evidence_export_rejected:true,null_definition_isolated:true,errors},null,2),{flag:'wx'});console.log(JSON.stringify({passed:true,checks,model_calls:0,navigation_actions:0}));
}catch(e){fs.writeFileSync(path.join(OUT,`GUI_failure_${label}.json`),JSON.stringify({error:String(e),errors},null,2),{flag:'wx'});throw e}
finally{if(fs.existsSync(hidden)){assert(!fs.existsSync(original));fs.renameSync(hidden,original)}if(fs.existsSync(badFolder)){assert(path.resolve(badFolder).startsWith(WORK+path.sep));fs.rmSync(badFolder,{recursive:true})}await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
