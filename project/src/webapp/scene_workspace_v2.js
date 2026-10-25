'use strict';
(() => {
  const $=id=>document.getElementById(id), owner=crypto.randomUUID().replaceAll('-','');
  let token='', catalog=null, scene=null, run=null, busy=false, auto=false, runs=new Map(), renderSerial=0;
  const say=(s,error=false)=>{ $('notice').textContent=s; $('notice').classList.toggle('error',error); };
  async function api(path,data){
    const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:{'Content-Type':'application/json','X-GeoNav-Session':owner,'X-GeoNav-Scene-Token':token},...(data===undefined?{}:{body:JSON.stringify(data)})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'操作未完成');return result;
  }
  const post=(verb,data)=>api('/api/scenes/'+verb,data);
  const query=(verb,fields)=>'/api/scenes/'+verb+'?'+new URLSearchParams(fields);
  const options=(select,items,empty)=>{select.replaceChildren();if(empty)select.add(new Option(empty,''));for(const [v,t] of items)select.add(new Option(t,v));};
  const data=()=>catalog.datasets.find(d=>d.id===$('scene-dataset').value);
  function setBase(){
    const d=data(), t=d?.tasks.find(t=>t.key===$('scene-base').value);if(!t)return;
    $('scene-start').value=t.start;$('scene-goal').value=t.goal;$('scene-budget').value=d.budget;
    $('scene-start').max=$('scene-goal').max=d.grid*d.grid-1;$('scene-budget').max=d.budget;
    $('scene-scale').textContent=`${d.grid}×${d.grid} · ${d.cell_size_m==null?'尺度未核实，只报告格数':d.cell_size_m+'米/格'} · 预算上限${d.budget}`;
  }
  function setDataset(){
    const d=data();if(!d)return;const areas=new Set(),items=[];
    for(const t of d.tasks)if(!areas.has(t.area)){areas.add(t.area);items.push([t.key,t.area+' · '+t.key]);}
    options($('scene-base'),items);options($('scene-policy'),d.policies.map(p=>[p,p==='M0'?'M0 · 冻结本地策略':p]));setBase();
  }
  async function refresh(){
    catalog=await api(query('catalog',{}));
    if(!$('scene-dataset').options.length){options($('scene-dataset'),catalog.datasets.map(d=>[d.id,d.name]));$('scene-dataset').value=catalog.datasets.some(d=>d.id==='masa-s3')?'masa-s3':catalog.datasets[0]?.id;setDataset();}
    const selected=scene?.id;options($('saved-scenes'),catalog.scenes.map(s=>[s.id,(s.saved?'已保存 · ':'临时 · ')+s.definition.name]),'选择已创建场景');if(selected)$('saved-scenes').value=selected;
    options($('saved-runs'),catalog.saved_runs.map(r=>[r.id,(r.controller_class==='mixed'?'人机接管':r.controller_class==='human'?'人工':'Agent')+' · '+r.id.slice(0,8)+' · '+r.status]),'选择只读运行记录');
    updateComparisons();
    const recovery=catalog.recovery;
    if(recovery&&(recovery.count||recovery.pending_files))say(`有 ${recovery.count} 条记录待核查，${recovery.pending_files} 个未提交文件已保留；其它有效记录可用。`);
  }
  function updateComparisons(){
    const items=new Map(catalog?.saved_runs.map(r=>[r.id,r])||[]);for(const [id,r] of runs)items.set(id,r);
    const list=[...items.values()].filter(r=>!scene||r.scene_id===scene.id).map(r=>[r.id,(r.controller_class||'未开始')+' · '+r.id.slice(0,8)]);
    const left=$('compare-left').value,right=$('compare-right').value;options($('compare-left'),list);options($('compare-right'),list);
    if(list.some(x=>x[0]===left))$('compare-left').value=left;
    $('compare-right').value=list.some(x=>x[0]===right)?right:(list[1]?.[0]||list[0]?.[0]||'');
  }
  function imageURL(cell){return query('image',{id:run.id,cell,owner,...($('reveal-all').checked?{reveal:'1'}:{})});}
  function render(){
    if(!run)return;const d=run.definition,k=d.grid,seen=new Set(run.trajectory.map(p=>p.cell));runs.set(run.id,run);updateComparisons();
    const terminal=['completed','cancelled','failed'].includes(run.status), readonly=!!run.readonly;
    $('stat-steps').textContent=`${run.events.length} / ${d.budget}`;$('stat-seen').textContent=`${seen.size} / ${k*k}`;
    const labels={human:'人工',mixed:'人机接管',pure_agent:'纯Agent',unstarted:'未开始'};
    $('stat-mode').textContent=labels[run.controller_class]||run.controller_class;
    $('stat-sg').textContent=run.result?run.result.sg_cells+'格':'未终止';
    $('map-title').textContent=d.name;
    $('run-status').textContent=(readonly?'已保存 · 只读回看 · ':run.status==='completed'?(run.result.success?'已到达目标 · ':'预算用尽 · '):run.status==='cancelled'?'已结束 · ':run.status==='failed'?'执行失败 · ':'')+`${d.grid}×${d.grid} / 自由仿真`;
    $('human-mode').classList.toggle('selected',run.mode==='human');$('agent-mode').classList.toggle('selected',run.mode==='agent');
    for(const b of document.querySelectorAll('[data-action]'))b.disabled=busy||terminal||readonly||run.mode!=='human';
    for(const id of ['human-mode','agent-mode','agent-step','agent-auto','cancel-scene'])$(id).disabled=busy||terminal||readonly;
    $('agent-step').disabled=$('agent-auto').disabled=busy||terminal||readonly||run.mode!=='agent';
    $('save-run').disabled=busy||readonly||!terminal;
    $('agent-pause').disabled=readonly||terminal;$('export-scene').disabled=false;
    const dataset=catalog.datasets.find(x=>x.id===d.dataset_id);
    const policyKey=(dataset?.policies||[]).join('|');
    if($('scene-policy').dataset.options!==policyKey){options($('scene-policy'),(dataset?.policies||[]).map(p=>[p,p]));$('scene-policy').dataset.options=policyKey;}
    if(run.policy)$('scene-policy').value=run.policy;
    $('scene-policy').disabled=busy||readonly||run.policy!=null;
    $('controller-note').textContent=run.controller_class==='mixed'?'本次含人工接管，不计入算法SR。策略调用 '+run.policy_calls+' 次，其中合法历史同步 '+run.sync_calls+' 次。':'手动模式不调用模型；Agent只读取当前和合法历史。当前策略调用 '+run.policy_calls+' 次。';
    $('scene-map').style.setProperty('--grid',k);$('scene-map').replaceChildren();
    for(let i=0;i<k*k;i++){
      const cell=document.createElement('div');cell.className='cell'+(seen.has(i)?' seen':'')+(i===run.position?' current':'');
      if(seen.has(i)||$('reveal-all').checked){const im=document.createElement('img');im.src=imageURL(i);im.alt='图块 '+i;cell.append(im);}
      const number=document.createElement('span');number.className='cell-index';number.textContent=i;cell.append(number);$('scene-map').append(cell);
    }
    $('goal-image').src=imageURL('target');$('current-image').src=imageURL(run.position);
    const svg=$('scene-path');svg.setAttribute('viewBox',`0 0 ${k*100} ${k*100}`);svg.replaceChildren();
    const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',run.trajectory.map(p=>`${(p.cell%k+.5)*100},${(Math.floor(p.cell/k)+.5)*100}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#32dbcb');line.setAttribute('stroke-width','5');line.setAttribute('stroke-linejoin','round');svg.append(line);
    $('scene-events').replaceChildren();for(const e of run.events.slice().reverse()){
      const div=document.createElement('div');div.className='event';div.textContent=`${e.step}. ${e.actor==='human'?'人工':'Agent'} · ${{up:'↑',right:'→',down:'↓',left:'←'}[e.action]} · ${e.from} → ${e.to}${e.out_of_bounds?'（越界耗步）':''}`;$('scene-events').append(div);
    }
    if(!run.events.length)$('scene-events').textContent='暂无动作';
  }
  async function action(fn){if(busy)return;busy=true;if(run)render();try{await fn();}catch(e){auto=false;say(e.message,true);}finally{busy=false;if(run)render();}}
  async function begin(){auto=false;$('reveal-all').checked=false;run=await post('start',{scene_id:scene.id});render();}
  $('scene-form').onsubmit=e=>{e.preventDefault();action(async()=>{
    auto=false;scene=await post('create',{name:$('scene-name').value,dataset_id:$('scene-dataset').value,base_task_key:$('scene-base').value,start:Number($('scene-start').value),goal:Number($('scene-goal').value),budget:Number($('scene-budget').value),seed:Number($('scene-seed').value)});
    $('save-scene').disabled=$('start-scene').disabled=false;await refresh();await begin();say(scene.definition.shortest_path_exceeds_budget?'场景已创建；最短路径超过预算，本题必然无法在预算内到达。':'临时场景已创建。可手动探索，或切换Agent；尚未调用模型。');
  });};
  $('scene-dataset').onchange=setDataset;$('scene-base').onchange=setBase;
  $('save-scene').onclick=()=>action(async()=>{scene=await post('save-scene',{scene_id:scene.id});await refresh();say('场景已保存，不复制影像、不运行模型。');});
  $('start-scene').onclick=()=>action(begin);
  $('saved-scenes').onchange=()=>action(async()=>{scene=catalog.scenes.find(s=>s.id===$('saved-scenes').value);if(!scene)return;$('save-scene').disabled=$('start-scene').disabled=false;await begin();say('按原配置另建临时运行；原保存记录未改。');});
  $('saved-runs').onchange=()=>action(async()=>{auto=false;run=await api(query('state',{id:$('saved-runs').value}));scene=catalog.scenes.find(s=>s.id===run.scene_id);render();say('只读打开已保存运行，零模型调用。');});
  for(const b of document.querySelectorAll('[data-action]'))b.onclick=()=>action(async()=>{run=await post('step',{id:run.id,action:b.dataset.action});});
  const mode=async m=>{auto=false;run=await post('mode',{id:run.id,mode:m,policy:$('scene-policy').value});};
  $('human-mode').onclick=()=>action(()=>mode('human'));$('agent-mode').onclick=()=>action(()=>mode('agent'));
  $('agent-step').onclick=()=>action(async()=>{run=await post('step',{id:run.id});});
  $('agent-auto').onclick=async()=>{if(busy||auto||!run)return;auto=true;const id=run.id;while(auto&&run?.id===id&&['ready','running'].includes(run.status)){await action(async()=>{run=await post('step',{id});});if(auto)await new Promise(r=>setTimeout(r,Number($('scene-speed').value)));}auto=false;};
  $('agent-pause').onclick=()=>{auto=false;say('已暂停自动操作；已发出的当前一步完成后停止。');};
  $('cancel-scene').onclick=()=>action(async()=>{auto=false;run=await post('cancel',{id:run.id});});
  $('save-run').onclick=()=>action(async()=>{run=await post('save-run',{id:run.id});await refresh();say('已保存配置与逐步结果；重开只读，未重跑。');});
  $('reveal-all').onchange=render;
  $('refresh-scenes').onclick=()=>action(refresh);
  $('compare-runs').onclick=()=>action(async()=>{const c=await api(query('compare',{left:$('compare-left').value,right:$('compare-right').value}));const describe=r=>`${r.controller_class}：${r.result?(r.result.success?'到达目标':'未到达')+'，SG '+r.result.sg_cells+'格':'未完整结束'}`;$('compare-result').textContent=describe(c.left)+'；'+describe(c.right)+'。'+c.scope+(c.algorithm_pair_eligible?`；恢复 ${Number(c.recovered)} / 损伤 ${Number(c.harmed)}`:'');});
  $('export-scene').onclick=()=>{if(!run)return;const url=URL.createObjectURL(new Blob([JSON.stringify(run,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='geonav-scene-'+run.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  setInterval(()=>{if(token)post('heartbeat',{}).catch(()=>{});},30000);
  window.addEventListener('pagehide',()=>{auto=false;if(token)fetch('/api/scenes/release',{method:'POST',keepalive:true,headers:{'Content-Type':'application/json','X-GeoNav-Session':owner,'X-GeoNav-Scene-Token':token},body:'{}'}).catch(()=>{});});
  for(const id of ['human-mode','agent-mode','agent-step','agent-auto','agent-pause','save-run','cancel-scene','export-scene'])$(id).disabled=true;
  for(const button of document.querySelectorAll('[data-action]'))button.disabled=true;
  api('/api/meta').then(m=>{token=m.scene_token;if(!token)throw Error('当前服务尚未启用场景仿真，请使用4.9入口');return refresh();}).catch(e=>say(e.message,true));
})();
