'use strict';
(() => {
  const previousNotice=notice;
  notice=function(text,error=false){
    const wording={
      '临时实验完成，退出即释放；需要留存请先导出。':'临时试跑已完成，可保存为正式实验；也可以先导出结果。',
      '临时实验已创建，仅存内存。关闭或刷新本页将释放；可先导出结果。':'临时试跑已创建。完成后可保存为正式实验；关闭或刷新本页会释放未保存试跑。'
    };
    return previousNotice(wording[text]||text,error);
  };
  const previousRender=renderRun;
  renderRun=function(state){
    previousRender(state);
    if(state.status==='paused'&&state.definition.storage_mode==='temporary'){
      $('workflow-status').textContent=state.rows.every(r=>r.status==='completed')?
        '所选题已完成，实验仍在暂停。请点击“结束实验”后保存；已有轨迹会完整保留。':
        '试跑尚未结束，请继续运行或结束实验后保存。';
    }
  };
})();
