'use strict';
const $=id=>document.getElementById(id),BASE='/api/capturepoint-sessions/',PREVIEW='/api/capturepoints/',owner=crypto.randomUUID().replaceAll('-','');
let token='',data=null,selected=null,state=null,busy=false,closed=false,running=false,generation=0,previewLoaded=new Set();
const terminal=s=>['completed','cancelled'].includes(s),kinds={human:'人工',rule:'Frontier规则',mixed:'混合接管',unstarted:'未执行'},modes={manual:'人工控制',frontier:'Frontier规则'};
const headers=()=>({'Content-Type':'application/json','X-GeoNav-Session':owner,'X-GeoNav-Scene-Token':token});
const query=v=>new URLSearchParams({owner,...v}).toString(),key=v=>(v.kind||'session')+':'+v.id;
const selection=value=>{const [kind,id]=value.split(':');if(!id||!['session','legacy'].includes(kind))throw Error('请选择已保存的过程');return {kind,id}};
function notice(value=''){$('notice').textContent=value}
async function api(path,body){const response=await fetch(BASE+path,{headers:headers(),...(body===undefined?{}:{method:'POST',body:JSON.stringify(body)})});const value=await response.json();if(!response.ok)throw Error(value.error||'读取失败');return value}
function pause(){running=false;generation++;controls()}
async function act(action){if(busy)return false;busy=true;controls();try{notice();await action();return true}catch(error){notice(error.message);pause();return false}finally{busy=false;controls()}}
function controls(){const active=state&&!state.readonly&&!terminal(state.status),locked=busy||running;
 $('create').disabled=locked||!selected||!!state||previewLoaded.size!==2;$('initial-controller').disabled=locked||!!(state&&!state.saved);
 $('run-name').disabled=locked||!!(state&&!state.saved);for(const id of ['city','role','search','new','refresh','reopen-task','open-run'])$(id).disabled=locked;
 $('save-task').disabled=locked||!state||state.saved_task||state.readonly;$('save-run').disabled=locked||!state||!terminal(state.status)||state.saved;
 $('finish').disabled=locked||!active;$('export').disabled=locked||!state||!terminal(state.status);
 $('controller').disabled=locked||!active;$('controller').value=state?.controller==='frontier'?'frontier':'manual';
 document.querySelectorAll('[data-action]').forEach(b=>b.disabled=locked||!active||state.controller!=='manual');
 $('rule-step').disabled=locked||!active||state.controller!=='frontier';$('auto-current').disabled=locked||!active||state.controller!=='frontier';
 $('pause').disabled=!running;$('reveal').disabled=locked||!state||state.readonly;$('speed').disabled=busy;
 $('fork').disabled=locked||!state||!state.saved;$('compare').disabled=locked;
 document.querySelectorAll('.task-row').forEach(b=>b.disabled=locked);
}
function picture(id,url,preview=false){const image=$(id);if(image.getAttribute('src')===url){if(preview&&image.complete&&image.naturalWidth){previewLoaded.add(id);controls()}return}
 image.onload=()=>{if(preview&&image.getAttribute('src')===url){previewLoaded.add(id);controls()}};
 image.onerror=()=>{if(image.getAttribute('src')!==url)return;image.removeAttribute('src');previewLoaded.delete(id);notice('图像缺失或绑定变化，请恢复原始资产；不会补造观测。');controls()};image.src=url;
}
function board(cell,visited,imageUrl){const host=$('board');if(host.children.length!==25){host.replaceChildren();for(let c=0;c<25;c++){const tile=document.createElement('div');tile.className='tile';tile.dataset.cell=c;const number=document.createElement('span');number.textContent=c;tile.append(number);host.append(tile)}}
 for(const tile of host.children){const c=Number(tile.dataset.cell),show=visited.includes(c)||(state&&!state.readonly&&$('reveal').checked);tile.className='tile'+(visited.includes(c)?' visited':'')+(cell===c?' current':'');tile.setAttribute('aria-label','图格'+c+(cell===c?'，当前格':''));let image=tile.querySelector('img');if(show){if(!image){image=document.createElement('img');image.alt='航拍图格'+c;tile.prepend(image)}const url=imageUrl(c);if(image.getAttribute('src')!==url)image.src=url}else if(image)image.remove()}
}
function context(task){$('selected-info').textContent=`5×5 · 10步 · 起点${task.start} · ${task.cell_size_m.toFixed(2)}米/格`;$('task-context').textContent=`${task.city} · ${task.role==='development'?'开发':'训练准备'}\n${task.task_id}`}
function clearVisual(){for(const id of ['target-image','current-image'])$(id).removeAttribute('src');$('board').replaceChildren();$('task-context').textContent='选择固定开发任务';$('current-label').textContent='等待选择'}
function preview(){pause();state=null;previewLoaded.clear();$('reveal').checked=false;$('status').textContent=selected?'待创建':'待选择任务';$('steps').textContent='— / 10';$('visited').textContent='— / 25';$('model-calls').textContent='0';$('history').replaceChildren();$('result').textContent='尚未运行。人工、规则及混合过程单独标记，平台记录不作为科研验收。';$('save-state').textContent='默认临时；离页释放。显式保存只写小型JSON。';$('replay').disabled=true;$('replay').max=0;
 if(!selected){clearVisual();controls();return}context(selected);const url=cell=>PREVIEW+'preview-image?'+query({task_id:selected.task_id,cell});
 picture('target-image',url('target'),true);picture('current-image',url('start'),true);board(selected.start,[selected.start],()=>url('start'));$('current-label').textContent='起点：第'+(Math.floor(selected.start/5)+1)+'行 / 第'+(selected.start%5+1)+'列';history.replaceState(null,'',location.pathname);controls();
}
function filter(){if(!data)return;const search=$('search').value.trim().toLowerCase(),city=$('city').value,role=$('role').value;
 const rows=data.tasks.filter(t=>(!city||t.city===city)&&(!role||t.role===role)&&(!search||`${t.task_id} ${t.city} ${t.start}`.toLowerCase().includes(search)));
 $('filtered-count').textContent=rows.length+' / 192';$('task-list').replaceChildren();
 for(const task of rows){const button=document.createElement('button');button.className='task-row'+(selected?.task_id===task.task_id?' selected':'');button.dataset.task=task.task_id;const title=document.createElement('strong');title.textContent=task.city+' · 起点 '+task.start;const detail=document.createElement('small');detail.textContent=task.task_id+' · '+(task.role==='development'?'开发':'训练准备');button.append(title,detail);
 button.onclick=()=>{if(busy||running)return;if(state&&!state.saved){notice('请先保存终局，或释放当前临时过程。');return}selected=task;preview();filter()};const item=document.createElement('div');item.setAttribute('role','listitem');item.append(button);$('task-list').append(item)}
 if(!rows.length){const p=document.createElement('p');p.className='hint';p.textContent='没有符合筛选的任务';$('task-list').append(p)}
 if(!state&&selected&&!rows.some(t=>t.task_id===selected.task_id)){selected=null;preview()}
 if(selected){const row=[...document.querySelectorAll('.task-row')].find(b=>b.dataset.task===selected.task_id);if(row){const host=$('task-list');host.scrollTop+=row.getBoundingClientRect().top-host.getBoundingClientRect().top-(host.clientHeight-row.clientHeight)/2}}controls();
}
function render(value,step=null){const changed=selected?.task_id!==value.task.task_id;state=value;selected=value.task;const shown=step===null?value.step:Number(step),trace=value.trajectory.slice(0,shown+1),cell=trace.at(-1).cell,seen=[...new Set(trace.map(t=>t.cell))];context(value.task);$('run-name').value=value.name;
 $('status').textContent=value.readonly?'只读回放':running?'本题自动运行':terminal(value.status)?'已结束':modes[value.controller];$('status').classList.toggle('running',running);$('steps').textContent=shown+' / '+value.budget;$('visited').textContent=seen.length+' / 25';$('model-calls').textContent='0';
 const imageUrl=c=>BASE+'image?'+query({id:value.id,kind:value.kind||'session',cell:c,reveal:$('reveal').checked?'1':'0'});picture('target-image',imageUrl('target'));picture('current-image',imageUrl(cell));board(cell,seen,imageUrl);
 $('current-label').textContent='第'+(Math.floor(cell/5)+1)+'行 / 第'+(cell%5+1)+'列';$('history').replaceChildren();
 for(const t of trace){const li=document.createElement('li');li.textContent=t.step===0?'起点 · 格'+t.cell:`${t.step} · ${{up:'↑',right:'→',down:'↓',left:'←'}[t.action]} · 格${t.cell} · ${t.origin==='rule'?'Frontier':'人工'}${t.out_of_bounds?' · 越界耗步':''}${t.revisited?' · 重访':''} · ${t.reason||''}`;$('history').append(li)}
 for(const e of value.controller_events||[]){const li=document.createElement('li');li.textContent=`第${e.step}步后控制切换：${modes[e.from_controller]} → ${modes[e.to_controller]}`;$('history').append(li)}
 $('save-state').textContent=value.saved?'已保存只读过程 · '+kinds[value.controller_class]:(value.saved_task?'配置已保存，过程仍临时':'临时过程，离页释放');
 if(value.result){const r=value.result;$('result').textContent=`${kinds[value.controller_class]}终局：${r.success?'到达':'未到达'} · SG ${r.SG_cells}格 / ${r.SG_m.toFixed(2)}米 · 实际移动${r.moved_m.toFixed(2)}米 · 人工${r.human_actions??value.step}步 / 规则${r.rule_actions??0}步。${value.observer_revealed?'使用过观察者全图。':''}平台开发记录，非正式验收。`}
 else $('result').textContent=`${kinds[value.controller_class]}过程；目标真值与距离只在终局显示。Frontier仅用位置与访问历史，模型调用0。`;
 $('replay').disabled=!value.readonly;$('replay').max=value.step;$('replay').value=shown;$('replay-label').textContent=value.readonly?`只读回放 ${shown} / ${value.step} 步`:'保存后可回放；暂停不会跳到下一题';controls();
 if(value.saved)history.replaceState(null,'','?run='+value.id+'&kind='+(value.kind||'session'));
 if(changed){$('city').value='';$('role').value='';$('search').value=value.task.task_id;filter()}
}
function options(id,values,placeholder){const old=$(id).value;$(id).replaceChildren(new Option(placeholder,''),...values.map(v=>new Option((v.kind==='legacy'?'旧人工 · ':kinds[v.controller_class]||'任务')+' · '+v.name+' · '+v.task_id,key(v))));if(values.some(v=>key(v)===old))$(id).value=old}
async function refresh(){data=await api('catalog');const city=$('city').value;$('city').replaceChildren(new Option('全部城市',''),...data.cities.map(c=>new Option(c,c)));$('city').value=data.cities.includes(city)?city:'';
 const tasks=[...data.saved_tasks,...data.legacy_saved_tasks],runs=[...data.saved_runs,...data.legacy_saved_runs];options('saved-tasks',tasks,'选择任务配置');options('saved-runs',runs,'选择只读过程');options('compare-a',runs,'选择过程A');options('compare-b',runs,'选择过程B');filter();
 if(data.recovery.broken||data.recovery.pending)notice(`已隔离${data.recovery.broken}条异常记录及${data.recovery.pending}个未完成写盘文件，原件保留。`)
}
async function autoCurrent(){if(running||busy||!state||state.readonly||state.controller!=='frontier')return;running=true;const round=++generation,id=state.id;controls();
 try{while(running&&round===generation&&state?.id===id&&!terminal(state.status)){const ok=await act(async()=>render(await api('rule-step',{id})));if(!ok||!running||terminal(state.status))break;await new Promise(resolve=>setTimeout(resolve,Number($('speed').value)))}
 }finally{if(round===generation){running=false;controls();if(state)render(state)}}
}
for(const id of ['city','role'])$(id).onchange=filter;$('search').oninput=filter;
$('create').onclick=()=>act(async()=>render(await api('create',{name:$('run-name').value,task_id:selected.task_id,controller:$('initial-controller').value})));
$('controller').onchange=()=>{const mode=$('controller').value;pause();act(async()=>render(await api('mode',{id:state.id,controller:mode})))};
document.querySelectorAll('[data-action]').forEach(button=>button.onclick=()=>act(async()=>render(await api('step',{id:state.id,action:button.dataset.action}))));
$('rule-step').onclick=()=>act(async()=>render(await api('rule-step',{id:state.id})));$('auto-current').onclick=autoCurrent;$('pause').onclick=()=>{pause();if(state)render(state)};
$('finish').onclick=()=>act(async()=>render(await api('finish',{id:state.id})));$('save-task').onclick=()=>act(async()=>{render(await api('save-task',{id:state.id}));await refresh()});
$('save-run').onclick=()=>act(async()=>{render(await api('save-run',{id:state.id}));await refresh();$('saved-runs').value=key(state);if(!$('compare-a').value)$('compare-a').value=key(state);else $('compare-b').value=key(state)});
$('new').onclick=()=>act(async()=>{pause();await api('release',{});state=null;preview();filter()});$('refresh').onclick=()=>act(refresh);
async function fork(value){if(state&&!state.saved)throw Error('先保存终局或释放临时过程');const from=selection(value);render(await api('fork',{...from,controller:$('initial-controller').value,name:$('run-name').value+' · 同题'}))}
$('reopen-task').onclick=()=>act(()=>fork($('saved-tasks').value));$('fork').onclick=()=>act(()=>fork(key(state)));
$('open-run').onclick=()=>act(async()=>{if(state&&!state.saved)throw Error('先保存终局或释放临时过程');const from=selection($('saved-runs').value);$('reveal').checked=false;render(await api('state?'+query(from)))});
$('compare').onclick=()=>act(async()=>{const a=selection($('compare-a').value),b=selection($('compare-b').value);const v=await api('compare?'+query({left:a.id,right:b.id,left_kind:a.kind,right_kind:b.kind}));$('comparison').textContent=`同题：${v.task_id}\nA ${kinds[v.A.controller]}：${v.A.result.success?'成功':'失败'}；B ${kinds[v.B.controller]}：${v.B.result.success?'成功':'失败'}\nB−A：SG ${v.SG_cells_delta>=0?'+':''}${v.SG_cells_delta}格，移动 ${v.moved_m_delta.toFixed(2)}米；${v.recovered?'恢复原失败':v.damaged?'损伤原成功':'成功状态一致'}。\n${v.warning}`});
$('reveal').onchange=()=>{if(state)render(state)};$('replay').oninput=()=>{if(state?.readonly)render(state,$('replay').value)};
$('export').onclick=()=>{if(state&&terminal(state.status)){const a=document.createElement('a');a.href=BASE+'export?'+query({id:state.id,kind:state.kind||'session'});a.download='geonav-controller-'+state.id+'.json';a.click()}};
window.addEventListener('keydown',e=>{if(busy||running||!state||state.readonly||state.controller!=='manual'||terminal(state.status)||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;const action={ArrowUp:'up',ArrowRight:'right',ArrowDown:'down',ArrowLeft:'left'}[e.key];if(action){e.preventDefault();act(async()=>render(await api('step',{id:state.id,action})))}});
window.addEventListener('pagehide',()=>{closed=true;pause();fetch(BASE+'release',{method:'POST',headers:headers(),body:'{}',keepalive:true}).catch(()=>{})});
setInterval(()=>{if(!closed&&state&&!state.readonly)api('heartbeat',{}).catch(()=>{})},30000);
const narrow=matchMedia('(max-width:740px)');function libraryLayout(){$('record-library').open=!narrow.matches}narrow.addEventListener('change',libraryLayout);libraryLayout();
act(async()=>{const meta=await (await fetch('/api/meta')).json();token=meta.scene_token;if(!token)throw Error('请用最新唯一入口启动工作台');await refresh();const params=new URLSearchParams(location.search);if(params.get('run'))render(await api('state?'+query({id:params.get('run'),kind:params.get('kind')||'session'})))});
