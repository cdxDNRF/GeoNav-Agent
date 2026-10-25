'use strict';
(() => {
  const wf=document.createElement('section');wf.className='panel trial-workflow';wf.id='trial-workflow';
  wf.innerHTML='<div class="workflow-heading"><div><span class="eyebrow">EXPERIMENT WORKFLOW</span><h3>保留这次结果，接着做同题对照</h3></div><span class="workflow-tag">试跑 → 保存 → 对照</span></div><p id="workflow-status" role="status"></p><div class="workflow-actions"><button id="save-trial" class="primary" type="button">保存为正式实验</button><label>对照方案<select id="fork-arm" aria-label="同题对照方案"></select></label><button id="fork-trial" type="button">按原题新建对照 →</button></div><p class="muted">保存不会重新运行模型；新对照保留原任务、预算和权重集合，先创建，点击运行后才执行。</p><div id="workflow-comparison" class="workflow-comparison hidden"><div class="workflow-heading"><h3>两次实验的同题结果</h3><button id="load-comparison" type="button">刷新对照</button></div><p id="comparison-status" role="status"></p><div id="comparison-cards" class="cards"></div><p id="comparison-pairs"></p><button id="open-parent" type="button">打开原实验</button></div>';
  $('run-stats').after(wf);
  let requestBusy=false,renderedId=null,comparisonSerial=0;
  async function workflow(path,data){
    const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:{'Content-Type':'application/json','X-GeoNav-Session':pageOwner,'X-GeoNav-Workflow-Token':meta.workflow_token},...(data===undefined?{}:{body:JSON.stringify(data)})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'操作未完成');return result;
  }
  const stable=s=>s&&['ready','completed','cancelled','failed','interrupted'].includes(s.status);
  function renderWorkflow(s){
    const historical=Boolean(s.definition.historical_readonly),temporary=s.definition.storage_mode==='temporary';
    $('save-trial').hidden=!temporary||historical;
    $('save-trial').disabled=requestBusy||!stable(s);
    $('fork-trial').disabled=requestBusy||!stable(s)||historical||temporary;
    $('fork-arm').disabled=requestBusy||historical;
    $('workflow-status').textContent=historical?'这是封存案例；请使用原历史页的新建同题入口。':temporary?(stable(s)?'本次试跑尚未留档。保存后，关闭网页也能在实验记录中重新打开。':'完成或结束实验后可保存；暂停中的模型内存不会被伪装为可恢复记录。'):'正式记录已保存，重新打开不会调用模型。';
    if(renderedId!==s.id){
      renderedId=s.id;comparisonSerial++;
      const dataset=meta.datasets.find(d=>d.id===s.definition.dataset_id);
      $('fork-arm').replaceChildren(...(dataset?.policies||[]).map(arm=>{const option=el('option',arm==='Cloud'?'Cloud · 使用已配置接口':arm);option.value=arm;option.disabled=arm==='Cloud'&&!meta.api;return option;}));
      const alternate=[...$('fork-arm').options].find(o=>!o.disabled&&o.value!==s.definition.arms[0]);
      $('fork-arm').value=alternate?.value||s.definition.arms[0];
      $('workflow-comparison').classList.toggle('hidden',!s.definition.same_task_comparison);
      $('comparison-status').textContent='运行新对照后，刷新查看两次记录的同题结果。';
      $('comparison-cards').replaceChildren();$('comparison-pairs').textContent='';
      if(s.definition.same_task_comparison)loadComparison().catch(e=>{$('comparison-status').textContent=e.message;});
    }
    $('fork-trial').textContent=$('fork-arm').value===s.definition.arms[0]?'按原题新建复跑 →':'按原题新建对照 →';
  }
  const oldRender=renderRun;
  renderRun=function(s){oldRender(s);renderWorkflow(s);};
  $('fork-arm').onchange=()=>{if(current)renderWorkflow(current);};
  $('save-trial').onclick=async()=>{
    if(!current||requestBusy)return;
    requestBusy=true;renderWorkflow(current);
    try{
      const result=await workflow('/api/workflow/save',{id:current.id});
      await refresh();await openExperiment(result.id);
      notice('已保存为正式实验，轨迹和结果原样保留，没有重跑模型。');
    }catch(error){notice(error.message,true);}finally{requestBusy=false;if(current)renderWorkflow(current);}
  };
  $('fork-trial').onclick=async()=>{
    if(!current||requestBusy)return;
    requestBusy=true;renderWorkflow(current);
    try{
      const arm=$('fork-arm').value;
      const result=await workflow('/api/workflow/fork',{id:current.id,arm});
      await refresh();await openExperiment(result.id);
      notice('已按原题新建临时对照，尚未运行。完成后可保存，并与原实验配对查看。');
    }catch(error){notice(error.message,true);}finally{requestBusy=false;if(current)renderWorkflow(current);}
  };
  async function loadComparison(){
    if(!current?.definition.same_task_comparison)return;
    const id=current.id,parent=current.definition.same_task_comparison.parent_id,serial=++comparisonSerial;
    const result=await workflow('/api/workflow/compare?'+new URLSearchParams({left:parent,right:id}));
    if(current?.id!==id||serial!==comparisonSerial)return;
    $('comparison-status').textContent=`${result.completed_pairs}/${result.planned_pairs} 组已完成配对；未运行或未结束的题不会冒充配对结果。`;
    $('comparison-cards').replaceChildren(...['left','right'].map((side)=>{
      const x=result[side],m=x.metrics,c=el('article',undefined,'card');
      c.append(el('h4',side==='left'?'原实验 · '+x.arm:'新实验 · '+x.arm),el('p',x.name),el('strong',`${m.successes}/${m.planned} 成功 · ${(m.sr*100).toFixed(1)}%`),el('p',`完成 ${m.completed}/${m.planned} · SG ${m.sg==null?'等待完整队列':m.sg.toFixed(3)+'格'}`));return c;
    }));
    $('comparison-pairs').textContent=`同题恢复 ${result.recovered}，损伤 ${result.harmed}，净增 ${result.net}。仅比较相同任务和权重集合，不新增独立地区；自由试跑不升级科研默认。`;
  }
  $('load-comparison').onclick=()=>loadComparison().catch(error=>{$('comparison-status').textContent=error.message;});
  $('open-parent').onclick=()=>{const parent=current?.definition.same_task_comparison?.parent_id;if(parent)openExperiment(parent).catch(error=>notice(error.message,true));};
  if(current)renderWorkflow(current);
})();
