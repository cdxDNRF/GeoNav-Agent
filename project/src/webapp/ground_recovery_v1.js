'use strict';
(()=>{
  const panel=document.createElement('section');panel.className='panel';panel.hidden=true;panel.style.overflowWrap='anywhere';
  const boundary=document.querySelector('.boundary');if(!boundary)return;boundary.after(panel);
  async function refreshRecovery(){
    try{
      const response=await fetch('/api/ground/recovery');if(!response.ok)return;
      const report=await response.json();panel.replaceChildren();panel.hidden=!report.count;
      if(!report.count)return;
      const title=document.createElement('h2');title.textContent=`${report.count}条记录待核查`;panel.append(title);
      const note=document.createElement('p');note.textContent=report.scope;panel.append(note);
      const list=document.createElement('ul');
      for(const item of report.records){const row=document.createElement('li');row.textContent=`${item.path}：${item.reason}`;list.append(row)}
      panel.append(list);
    }catch(_){/* Existing task UI remains usable on a transient read failure. */}
  }
  refreshRecovery();setInterval(refreshRecovery,15000);
})();
