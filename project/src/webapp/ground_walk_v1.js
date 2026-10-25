'use strict';
const $=id=>document.getElementById(id),base='/api/ground-walk/';
let owner=sessionStorage.getItem('geonav-ground-walk-owner');if(!owner){owner=crypto.randomUUID().replaceAll('-','');sessionStorage.setItem('geonav-ground-walk-owner',owner)}
let token='',id=null,state=null,poll=null,closed=false,available=false;
const headers=()=>({'Content-Type':'application/json','X-GeoNav-Session':owner,'X-GeoNav-Scene-Token':token});
const ended=s=>['completed','failed','cancelled'].includes(s),labels={opening:'编码目标',ready:'等待人工移动',busy:'更新当前观测',completed:'探索结束',failed:'执行失败',cancelled:'已取消'};
async function api(path,value){const r=await fetch(base+path,{headers:headers(),...(value===undefined?{}:{method:'POST',body:JSON.stringify(value)})});const v=await r.json();if(!r.ok)throw Error(v.error||'操作失败');return v}
const query=extra=>new URLSearchParams({owner,...extra}).toString();
function notice(s=''){$('notice').textContent=s}
async function action(fn){try{notice();await fn()}catch(e){notice(e.message)}}
function image(img,url){if(img.getAttribute('src')!==url)img.src=url}
function render(s){id=s.id;state=s;const done=ended(s.status),can=s.status==='ready'&&!s.readonly;
  $('walk-name').value=s.name;$('walk-name').disabled=true;$('walk-budget').disabled=true;$('start').disabled=true;
  $('status').textContent=labels[s.status]||s.status;$('steps').textContent=s.step+' / '+s.budget;$('calls').textContent=s.forward_calls+' / '+s.forward_limit;$('seen').textContent=s.visited.length+' / 25';
  document.querySelectorAll('[data-action]').forEach(b=>b.disabled=!can);$('finish').disabled=!can;$('cancel').disabled=done||s.readonly;$('save').disabled=!done||s.saved;$('export').disabled=!done;$('new').disabled=!done;
  $('reveal').disabled=s.readonly;if(s.readonly)$('reveal').checked=false;
  $('score').textContent='—';$('cache').textContent=done?'本记录无完整当前证据':'正在等待当前观测编码';
  const e=s.current_evidence;if(e){$('score').textContent=e.event.raw_similarity.toFixed(4);$('cache').textContent=e.event.current_cache_hit?'重访：复用本题缓存，新增编码0次':'当前格首次观测：已生成512维特征'}
  $('save-info').textContent=s.saved?'已保存 · 只读重放，不重新编码':'临时过程 · 结束后可显式保存';
  $('cell-label').textContent='第'+(Math.floor(s.position/5)+1)+'行 / 第'+(s.position%5+1)+'列';
  image($('goal'),base+'image?'+query({id,cell:'goal'}));image($('current'),base+'image?'+query({id,cell:s.position}));
  const map=$('walk-map');map.replaceChildren();for(let c=0;c<25;c++){
    const tile=document.createElement('div');tile.className='tile'+(s.visited.includes(c)?' visited':'')+(s.position===c?' current':'');tile.setAttribute('aria-label','图格'+c+(s.position===c?'，当前格':''));
    if(s.visited.includes(c)||$('reveal').checked){const im=document.createElement('img');im.alt='航拍图格'+c;im.src=base+'image?'+query({id,cell:c,reveal:$('reveal').checked?'1':'0'});tile.append(im)}
    const n=document.createElement('span');n.className='num';n.textContent=c;tile.append(n);map.append(tile)
  }
  $('history').replaceChildren();for(const e of s.evidence){const li=document.createElement('li');const move=s.moves.find(m=>m.step===e.step);li.textContent=(e.step===0?'起点':e.step+' · '+({up:'↑',right:'→',down:'↓',left:'←'}[move?.action]||''))+' · 格'+e.cell+' · '+e.event.raw_similarity.toFixed(4)+(e.event.current_cache_hit?' · 缓存':' · 新观测')+(move?.boundary_stay?' · 越界耗步':'');$('history').append(li)}
  if(s.result)$('result').textContent='人工探索终局：'+(s.result.reached_dataset_proxy?'到达数据代理格':'未到达数据代理格')+' · SG '+s.result.SG_cells+'格。该结果不计入Agent SR，距离尺度未核实。';
  else $('result').textContent='目标为数据配对中心代理格；真实目标格与距离不传给编码器。';
  if(s.error)notice(s.error)
}
async function watch(){if(!id||closed)return;try{const s=await api('state?'+query({id}));render(s);if(['opening','busy'].includes(s.status))poll=setTimeout(watch,650)}catch(e){notice(e.message);$('new').disabled=false}}
async function catalog(){const c=await api('catalog');available=c.available;$('scene-info').textContent=c.scene.name+' · 已消费开发来源 · 尺度未核实';$('saved').replaceChildren(new Option('选择记录',''),...c.saved.map(s=>new Option(s.name+' · '+(labels[s.status]||s.status),s.id)));$('start').disabled=Boolean(id)||!available;if(!available)notice('本机未携带配对场景或编码资产，可重放已有过程；不会自动下载。');if(c.recovery.broken||c.recovery.pending)notice('已隔离'+c.recovery.broken+'条异常记录和'+c.recovery.pending+'个未提交目录，原件保留。')}
$('start').onclick=()=>action(async()=>{$('start').disabled=true;try{const s=await api('create',{name:$('walk-name').value,budget:Number($('walk-budget').value)});render(s);watch()}catch(e){$('start').disabled=!available;throw e}});
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>action(async()=>{document.querySelectorAll('[data-action]').forEach(x=>x.disabled=true);try{render(await api('move',{id,action:b.dataset.action}));watch()}catch(e){await watch();throw e}}));
$('finish').onclick=()=>action(async()=>render(await api('finish',{id})));
$('cancel').onclick=()=>action(async()=>{await api('cancel',{id});watch()});
$('save').onclick=()=>action(async()=>{render(await api('save',{id}));await catalog();$('saved').value=id});
$('open').onclick=()=>action(async()=>{const selected=$('saved').value;if(!selected)throw Error('请选择过程记录');if(state&&!ended(state.status))throw Error('先结束或取消当前探索');clearTimeout(poll);render(await api('state?'+query({id:selected})));notice('只读重放，不调用模型。')});
$('new').onclick=()=>action(async()=>{await api('release',{});id=null;state=null;clearTimeout(poll);$('walk-name').disabled=false;$('walk-budget').disabled=false;$('status').textContent='等待开始';$('score').textContent='—';$('cache').textContent='等待观测';$('walk-map').replaceChildren();$('history').replaceChildren();$('goal').removeAttribute('src');$('current').removeAttribute('src');$('steps').textContent='—';$('calls').textContent='0';$('seen').textContent='0 / 25';$('result').textContent='等待开始新的配对场景';$('save-info').textContent='默认临时 · 尚未保存';$('cell-label').textContent='尚未开始';$('reveal').checked=false;$('reveal').disabled=false;for(const k of ['finish','cancel','save','export'])$(k).disabled=true;document.querySelectorAll('[data-action]').forEach(b=>b.disabled=true);await catalog()});
$('reveal').onchange=()=>{if(state)render(state)};
$('refresh').onclick=()=>action(catalog);
$('export').onclick=()=>{if(!id)return;const a=document.createElement('a');a.href=base+'export?'+query({id});a.download='geonav-ground-walk-'+id+'.json';a.click()};
window.addEventListener('pagehide',()=>{closed=true;clearTimeout(poll);fetch(base+'release',{method:'POST',headers:headers(),body:'{}',keepalive:true}).catch(()=>{})});
setInterval(()=>{if(!closed)api('heartbeat',{}).catch(()=>{})},30000);
for(const k of ['finish','cancel','save','export'])$(k).disabled=true;document.querySelectorAll('[data-action]').forEach(b=>b.disabled=true);
action(async()=>{const r=await fetch('/api/meta',{headers:headers()});token=(await r.json()).scene_token;if(!token)throw Error('当前版本不支持人工配对场景');await catalog()});
