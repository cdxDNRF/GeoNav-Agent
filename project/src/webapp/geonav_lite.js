/**
 * GeoNav Lite Frontend Interactive Engine
 * Clean single-screen dashboard with multi-grid custom upload,
 * observer-only full-map preview toggle, and instant task-switch cleanup.
 */
(() => {
  'use strict';

  // --- 状态定义 ---
  let metaData = null;
  let currentState = null;
  let currentGrid = 5;
  let autoTimer = null;
  let playSpeedMs = 750;
  let clickSelectMode = 'start';
  let eyePreview = false;
  let pendingUploadBase64 = null;
  let selectedUploadGrid = 5;

  const GRID_SPECS = {
    3: { budget: 6, note: '3×3 共 9 个图块 · 步数预算 6 步（极速轻量演示）' },
    4: { budget: 8, note: '4×4 共 16 个图块 · 步数预算 8 步（紧凑探索尺度）' },
    5: { budget: 10, note: '5×5 共 25 个图块 · 步数预算 10 步（M0 原生最佳协议）' },
    6: { budget: 14, note: '6×6 共 36 个图块 · 步数预算 14 步（中等粒度探索）' },
    7: { budget: 16, note: '7×7 共 49 个图块 · 步数预算 16 步（扩展视野尺度）' },
    10: { budget: 20, note: '10×10 共 100 个图块 · 步数预算 20 步（大区域前瞻保护）' }
  };

  // --- DOM 元素引用 ---
  const $ = id => document.getElementById(id);
  const toastContainer = $('toast-container');
  const taskSelect = $('task-select');
  const btnRandomAerial = $('btn-random-aerial');
  const btnDeleteUpload = $('btn-delete-upload');
  const footerCatalogStats = $('footer-catalog-stats');
  const startSelect = $('start-select');
  const goalSelect = $('goal-select');
  const policySelect = $('policy-select');
  const btnStartTask = $('btn-start-task');
  const btnReset = $('btn-reset');
  const btnStep = $('btn-step');
  const btnAuto = $('btn-auto');
  const btnPause = $('btn-pause');

  const gridMap = $('grid-map');
  const mapGridLabel = $('map-grid-label');
  const badgeMapMode = $('badge-map-mode');
  const btnToggleEye = $('btn-toggle-eye');
  const eyeIconOpen = $('eye-icon-open');
  const eyeIconClosed = $('eye-icon-closed');
  const eyeBtnText = $('eye-btn-text');

  const apiStatusDot = $('api-status-dot');

  const imgTarget = $('img-target');
  const imgCurrent = $('img-current');
  const placeholderTarget = $('placeholder-target');
  const placeholderCurrent = $('placeholder-current');
  const targetCoords = $('target-coords');
  const currentCoords = $('current-coords');

  const statusBadge = $('status-badge');
  const metricStep = $('metric-step');
  const metricDist = $('metric-dist');
  const metricSuccess = $('metric-success');

  const decisionActionBadge = $('decision-action-badge');
  const decisionBranchBadge = $('decision-branch-badge');
  const decisionHeadline = $('decision-headline');
  const decisionReason = $('decision-reason');
  const barUp = $('bar-up'), valUp = $('val-up'), starUp = $('star-up'), rowUp = $('row-up');
  const barRight = $('bar-right'), valRight = $('val-right'), starRight = $('star-right'), rowRight = $('row-right');
  const barDown = $('bar-down'), valDown = $('val-down'), starDown = $('star-down'), rowDown = $('row-down');
  const barLeft = $('bar-left'), valLeft = $('val-left'), starLeft = $('star-left'), rowLeft = $('row-left');

  // --- 温和非阻塞提示 ---
  function showToast(msg, kind = 'info', durationMs = 3200) {
    if (!toastContainer) return;
    const el = document.createElement('div');
    el.className = `toast toast-${kind}`;
    el.textContent = msg;
    toastContainer.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(-6px)';
      el.style.transition = 'all 0.2s ease';
      setTimeout(() => el.remove(), 220);
    }, durationMs);
  }

  // --- API 客户端封装 ---
  async function api(path, method = 'GET', body = null) {
    const opts = { method, headers: {} };
    if (body !== null) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: '请求失败' }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    return res.json();
  }

  // --- 获取当前选中的任务元数据 ---
  function getSelectedTask() {
    if (!metaData || !metaData.tasks) return null;
    return metaData.tasks.find(t => t.task_key === taskSelect.value) || null;
  }

  // --- 动态构建 k×k 网格 DOM ---
  function buildGridDOM(k) {
    currentGrid = k;
    mapGridLabel.textContent = `${k}×${k}`;
    gridMap.style.setProperty('--grid-n', String(k));
    gridMap.classList.toggle('grid-dense', k >= 7);
    gridMap.classList.toggle('eye-active', eyePreview);
    gridMap.innerHTML = '';

    const total = k * k;
    for (let i = 0; i < total; i++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.index = String(i);
      cell.innerHTML = `
        <span class="cell-index">${i}</span>
        <div class="cell-fog" id="fog-${i}"></div>
        <div class="cell-corner-badges" id="badges-${i}"></div>
      `;
      cell.addEventListener('click', () => handleCellClick(i));
      gridMap.appendChild(cell);
    }
  }

  // --- 填充 0..k*k-1 起终点下拉框 ---
  function populateCoordSelects(k) {
    const prevStart = parseInt(startSelect.value, 10);
    const prevGoal = parseInt(goalSelect.value, 10);
    startSelect.innerHTML = '';
    goalSelect.innerHTML = '';
    const total = k * k;
    for (let i = 0; i < total; i++) {
      const r = Math.floor(i / k);
      const c = i % k;
      const optS = document.createElement('option');
      optS.value = i;
      optS.textContent = `${i} (${r},${c})`;
      startSelect.appendChild(optS);

      const optG = document.createElement('option');
      optG.value = i;
      optG.textContent = `${i} (${r},${c})`;
      goalSelect.appendChild(optG);
    }
    startSelect.value = (!isNaN(prevStart) && prevStart < total) ? prevStart : 0;
    goalSelect.value = (!isNaN(prevGoal) && prevGoal < total) ? prevGoal : (total - 1);
  }

  // --- 清空地图所有图块显示（迷雾复位） ---
  function clearGridTileImages() {
    const total = currentGrid * currentGrid;
    for (let i = 0; i < total; i++) {
      const cell = gridMap.querySelector(`.cell[data-index="${i}"]`);
      if (!cell) continue;
      const img = cell.querySelector('.cell-img');
      if (img) img.remove();
      cell.classList.remove('cell-preview-unvisited');
      const fog = cell.querySelector('.cell-fog');
      if (fog) fog.style.display = 'block';
    }
  }

  // --- 更新小眼睛全图预览（严格只用于人眼观察） ---
  function syncEyePreviewDisplay() {
    const task = getSelectedTask();
    const tkey = (currentState && currentState.task_key) || (task && task.task_key) || '';
    const visitedSet = new Set((currentState && currentState.visited) ? currentState.visited : []);
    const total = currentGrid * currentGrid;
    const stamp = Date.now();

    gridMap.classList.toggle('eye-active', eyePreview);
    btnToggleEye.classList.toggle('active', eyePreview);
    eyeIconOpen.classList.toggle('hidden', eyePreview);
    eyeIconClosed.classList.toggle('hidden', !eyePreview);
    eyeBtnText.textContent = eyePreview ? '关闭预览' : '预览全图';

    for (let i = 0; i < total; i++) {
      const cell = gridMap.querySelector(`.cell[data-index="${i}"]`);
      if (!cell) continue;
      const isVisited = visitedSet.has(i);
      const fog = cell.querySelector('.cell-fog');
      let img = cell.querySelector('.cell-img');

      if (isVisited) {
        // 已探索图块：保持真实图像
        cell.classList.remove('cell-preview-unvisited');
        if (fog) fog.style.display = 'none';
        if ((!img || img.dataset.tkey !== tkey) && tkey) {
          if (img) img.remove();
          img = document.createElement('img');
          img.className = 'cell-img';
          img.dataset.tkey = tkey;
          img.src = `/api/image?task_key=${encodeURIComponent(tkey)}&cell=${i}&t=${stamp}`;
          cell.prepend(img);
        }
      } else if (eyePreview && tkey) {
        // 人类观察全图模式：仅读取静态图块展示给观察者，绝不触碰底层模型状态
        cell.classList.toggle('cell-preview-unvisited', Boolean(currentState));
        if (fog) fog.style.display = 'none';
        if (!img || img.dataset.tkey !== tkey) {
          if (img) img.remove();
          img = document.createElement('img');
          img.className = 'cell-img';
          img.dataset.tkey = tkey;
          img.src = `/api/image?task_key=${encodeURIComponent(tkey)}&cell=${i}&observer=1&t=${stamp}`;
          cell.prepend(img);
        }
      } else {
        // 正常迷雾模式：隐藏未探索图块
        cell.classList.remove('cell-preview-unvisited');
        if (img) img.remove();
        if (fog) fog.style.display = 'block';
      }
    }

    if (eyePreview) {
      badgeMapMode.textContent = '全图预览中（仅人类可见 · 底层导航保持盲视）';
      badgeMapMode.style.borderColor = 'var(--primary)';
    } else if (currentState && !currentState.done) {
      badgeMapMode.textContent = `推演进行中 · 当前位置 [${currentState.coords.join(',')}]`;
      badgeMapMode.style.borderColor = 'var(--border)';
    } else {
      const fixed = task && task.fixed_endpoints;
      badgeMapMode.textContent = fixed ? '地面固定任务起终点（支持方向移动）' : '点击格子设置起点 / 目标';
      badgeMapMode.style.borderColor = 'var(--border)';
    }
  }

  // --- 清空右侧面板到待开始状态 ---
  function resetRightStageUI(budget = 10) {
    imgTarget.src = '';
    imgCurrent.src = '';
    delete imgTarget.dataset.goalKey;
    delete imgCurrent.dataset.currKey;
    imgTarget.classList.add('hidden');
    imgCurrent.classList.add('hidden');
    placeholderTarget.classList.remove('hidden');
    placeholderCurrent.classList.remove('hidden');
    targetCoords.textContent = '目标格子: --';
    currentCoords.textContent = '当前格子: --';

    statusBadge.className = 'val-badge badge-idle';
    statusBadge.textContent = '待开始';
    metricStep.textContent = `0 / ${budget} 步`;
    metricDist.textContent = '-- 格';
    metricSuccess.textContent = '等待执行';
    metricSuccess.style.color = 'var(--text-main)';

    decisionActionBadge.textContent = '等待推演';
    decisionBranchBadge.textContent = '智能网络推演';
    decisionHeadline.textContent = '等待任务启动';
    decisionReason.textContent = '选定起终点后点击【开始任务】，随后可使用【单步推演】或【自动运行】查看每步决策依据。';

    const bars = [
      [barUp, valUp, starUp, rowUp],
      [barRight, valRight, starRight, rowRight],
      [barDown, valDown, starDown, rowDown],
      [barLeft, valLeft, starLeft, rowLeft]
    ];
    for (const [bar, val, star, row] of bars) {
      bar.style.width = '25%';
      val.textContent = '25.0%';
      if (star) star.textContent = '';
      if (row) row.classList.remove('winner');
    }
  }

  // --- 点击格子交互 ---
  function handleCellClick(index) {
    if (currentState && !currentState.done && (currentState.status === 'running' || currentState.status === 'ready')) {
      const k = currentGrid;
      const curr = currentState.position;
      const dr = Math.floor(index / k) - Math.floor(curr / k);
      const dc = (index % k) - (curr % k);
      if (Math.abs(dr) + Math.abs(dc) === 1) {
        const act = dr === -1 ? 'up' : dr === 1 ? 'down' : dc === -1 ? 'left' : 'right';
        step(act);
      }
      return;
    }

    const task = getSelectedTask();
    if (task && task.fixed_endpoints) {
      showToast('地面拍摄点样本采用固定起终点配对，直接点击【开始任务】即可体验。', 'info');
      return;
    }

    if (clickSelectMode === 'start') {
      startSelect.value = index;
      clickSelectMode = 'goal';
      if (!eyePreview) {
        badgeMapMode.textContent = `已选起点 ${index}，再点击格子设置【目标】`;
        badgeMapMode.style.borderColor = 'var(--accent-red)';
      }
    } else {
      goalSelect.value = index;
      clickSelectMode = 'start';
      if (!eyePreview) {
        badgeMapMode.textContent = `已选目标 ${index}，点击格子可重选【起点】`;
        badgeMapMode.style.borderColor = 'var(--accent-green)';
      }
    }
    updateGridHighlights();
  }

  // --- 更新网格起终点及当前角标 ---
  function updateGridHighlights() {
    const sVal = parseInt(startSelect.value, 10);
    const gVal = parseInt(goalSelect.value, 10);
    const currVal = currentState ? currentState.position : -1;
    const total = currentGrid * currentGrid;

    for (let i = 0; i < total; i++) {
      const cell = gridMap.querySelector(`.cell[data-index="${i}"]`);
      if (!cell) continue;

      cell.classList.remove('cell-start', 'cell-goal', 'cell-curr');
      if (i === sVal) cell.classList.add('cell-start');
      if (i === gVal) cell.classList.add('cell-goal');
      if (currentState && i === currVal) cell.classList.add('cell-curr');

      const badges = cell.querySelector('.cell-corner-badges');
      if (badges) {
        let html = '';
        if (i === sVal) html += '<span class="cell-pill pill-start">起</span>';
        if (i === gVal) html += '<span class="cell-pill pill-goal">终</span>';
        badges.innerHTML = html;
      }
    }
  }

  // --- 样本切换时立即清空上一道题的切片与状态 ---
  async function handleTaskChanged({ resetBackend = true } = {}) {
    stopAuto();
    currentState = null;
    btnStep.disabled = true;
    btnAuto.disabled = true;
    btnPause.disabled = true;

    if (resetBackend) {
      api('/api/reset', 'POST').catch(() => {});
    }

    const task = getSelectedTask();
    if (!task) return;

    const k = task.grid || 5;
    if (k !== currentGrid || gridMap.children.length !== k * k) {
      buildGridDOM(k);
      populateCoordSelects(k);
    } else {
      clearGridTileImages();
    }

    const total = k * k;
    const defStart = typeof task.start === 'number' ? task.start : 0;
    const defGoal = typeof task.default_goal === 'number'
      ? task.default_goal
      : ((defStart + Math.floor(total / 2)) % total);

    startSelect.value = defStart;
    goalSelect.value = defGoal === defStart ? (defStart + 1) % total : defGoal;

    const fixed = Boolean(task.fixed_endpoints);
    startSelect.disabled = fixed;
    goalSelect.disabled = fixed;

    btnDeleteUpload.classList.toggle('hidden', task.dataset_id !== 'upload');
    clickSelectMode = 'start';

    resetRightStageUI(task.budget || 10);
    updateGridHighlights();
    syncEyePreviewDisplay();
  }

  // --- 加载 Meta 数据与全部任务样本（含冷启动自动轮询重连，彻底消除早开白屏） ---
  async function loadMeta({ preferTaskKey = null, suppressRestore = false, retryCount = 0 } = {}) {
    try {
      metaData = await api('/api/meta');
      apiStatusDot.classList.toggle('connected', Boolean(metaData.api_configured));
      if (footerCatalogStats && metaData.stats) {
        const st = metaData.stats;
        footerCatalogStats.textContent = `内置 ${st.aerial_images} 张独立航拍底图 (${st.aerial_tasks} 道精选合格题) + ${st.ground_tasks} 地面拍摄点 · 自定义图纯内存运行不污染样本库`;
      }

      taskSelect.innerHTML = '';
      const groups = {};
      for (const t of metaData.tasks) {
        const gName = t.group || '测试样本';
        if (!groups[gName]) groups[gName] = [];
        groups[gName].push(t);
      }

      for (const [gName, tasks] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = gName.includes('题)') ? gName : `${gName} (${tasks.length})`;
        for (const task of tasks) {
          const opt = document.createElement('option');
          opt.value = task.task_key;
          opt.textContent = task.name;
          optgroup.appendChild(opt);
        }
        taskSelect.appendChild(optgroup);
      }

      if (preferTaskKey && metaData.tasks.some(t => t.task_key === preferTaskKey)) {
        taskSelect.value = preferTaskKey;
        await handleTaskChanged({ resetBackend: true });
        return;
      }

      if (!suppressRestore && metaData.active_session) {
        const sess = metaData.active_session;
        if (sess.task_key && metaData.tasks.some(t => t.task_key === sess.task_key)) {
          taskSelect.value = sess.task_key;
        }
        if (sess.grid && sess.grid !== currentGrid) {
          buildGridDOM(sess.grid);
          populateCoordSelects(sess.grid);
        }
        currentState = sess;
        startSelect.value = sess.start;
        goalSelect.value = sess.goal;
        const task = getSelectedTask();
        btnDeleteUpload.classList.toggle('hidden', !task || task.dataset_id !== 'upload');
        renderState(currentState);
        btnStep.disabled = Boolean(currentState.done);
        btnAuto.disabled = Boolean(currentState.done);
      } else if (metaData.tasks.length > 0) {
        await handleTaskChanged({ resetBackend: false });
      }
    } catch (err) {
      console.warn('初始化加载元数据异常:', err);
      if (retryCount < 25) {
        if (retryCount === 0) {
          showToast('正在等待后端预热视觉模型与题库，就绪后自动加载...', 'info', 2600);
        }
        setTimeout(() => loadMeta({ preferTaskKey, suppressRestore, retryCount: retryCount + 1 }), 700);
      } else {
        showToast('未能连接后端服务，请确认启动窗口正常运行', 'warn');
      }
    }
  }

  // --- 心跳保活 ---
  setInterval(() => {
    api('/api/heartbeat').catch(() => {});
  }, 25000);

  // --- 开始新任务 ---
  async function startTask() {
    stopAuto();
    const task_key = taskSelect.value;
    const policy = policySelect.value;
    const start = parseInt(startSelect.value, 10);
    const goal = parseInt(goalSelect.value, 10);

    btnStartTask.disabled = true;
    const oldText = btnStartTask.textContent;
    btnStartTask.textContent = '准备中...';
    try {
      currentState = await api('/api/create', 'POST', {
        task_key,
        policy,
        start,
        goal,
        seed: 0
      });

      if (currentState.grid !== currentGrid) {
        buildGridDOM(currentState.grid);
        populateCoordSelects(currentState.grid);
      }
      clearGridTileImages();
      renderState(currentState);
      btnStep.disabled = false;
      btnAuto.disabled = false;
    } catch (err) {
      showToast(`创建任务失败：${err.message}`, 'error');
    } finally {
      btnStartTask.disabled = false;
      btnStartTask.textContent = oldText;
    }
  }

  // --- 单步推演 ---
  async function step(action = null) {
    if (!currentState || currentState.done) return;
    btnStep.disabled = true;
    try {
      currentState = await api('/api/step', 'POST', { action });
      renderState(currentState);
      if (currentState.done) {
        stopAuto();
      }
    } catch (err) {
      stopAuto();
      showToast(`推演异常：${err.message}`, 'error');
    } finally {
      if (currentState && !currentState.done && !autoTimer) {
        btnStep.disabled = false;
      }
    }
  }

  // --- 自动推演 ---
  function startAuto() {
    if (autoTimer || !currentState || currentState.done) return;
    btnAuto.disabled = true;
    btnPause.disabled = false;
    btnStep.disabled = true;

    const tick = async () => {
      if (!autoTimer || !currentState || currentState.done) {
        stopAuto();
        return;
      }
      await step();
      if (autoTimer && currentState && !currentState.done) {
        autoTimer = setTimeout(tick, playSpeedMs);
      } else {
        stopAuto();
      }
    };
    autoTimer = setTimeout(tick, 100);
  }

  function stopAuto() {
    if (autoTimer) {
      clearTimeout(autoTimer);
      autoTimer = null;
    }
    btnAuto.disabled = !currentState || currentState.done;
    btnPause.disabled = true;
    btnStep.disabled = !currentState || currentState.done;
  }

  // --- 重置当前题目 ---
  async function resetAll() {
    await handleTaskChanged({ resetBackend: true });
    showToast('已重置当前画布与探索状态', 'info', 1800);
  }

  // --- 渲染状态到界面 ---
  function renderState(s) {
    if (!s) return;

    updateGridHighlights();
    syncEyePreviewDisplay();

    // 1. 双视窗图像
    const ts = Date.now();
    placeholderTarget.classList.add('hidden');
    placeholderCurrent.classList.add('hidden');
    imgTarget.classList.remove('hidden');
    imgCurrent.classList.remove('hidden');
    const goalKey = `${s.task_key || ''}_${s.goal}_${s.start}`;
    if (imgTarget.dataset.goalKey !== goalKey) {
      imgTarget.dataset.goalKey = goalKey;
      imgTarget.src = `/api/image?target=1&t=${ts}`;
    }
    const currKey = `${s.task_key || ''}_${s.position}_${s.step}`;
    if (imgCurrent.dataset.currKey !== currKey) {
      imgCurrent.dataset.currKey = currKey;
      imgCurrent.src = `/api/image?cell=${s.position}&t=${ts}`;
    }
    targetCoords.textContent = `目标: [${s.goal_coords.join(',')}] (格${s.goal})`;
    currentCoords.textContent = `当前: [${s.coords.join(',')}] (格${s.position})`;

    // 2. 实时指标
    metricStep.textContent = `${s.step} / ${s.budget} 步`;
    metricDist.textContent = `${s.distance_cells} 格 · ${s.distance_euclidean} 欧氏`;

    if (s.success) {
      statusBadge.className = 'val-badge badge-success';
      statusBadge.textContent = '已到达目标';
      metricSuccess.textContent = '成功定位 (SG=0)';
      metricSuccess.style.color = 'var(--accent-green)';
    } else if (s.done) {
      statusBadge.className = 'val-badge badge-failed';
      statusBadge.textContent = '预算耗尽';
      metricSuccess.textContent = `未到达 (SG=${s.distance_cells})`;
      metricSuccess.style.color = 'var(--accent-red)';
    } else {
      statusBadge.className = 'val-badge badge-running';
      statusBadge.textContent = s.step === 0 ? '已就绪' : '探索中';
      metricSuccess.textContent = '推演进行中';
      metricSuccess.style.color = 'var(--text-main)';
    }

    // 3. 决策与概率透视
    const dec = s.decision || {};
    decisionActionBadge.textContent = dec.action_name || '等待推演';
    decisionBranchBadge.textContent = dec.branch || '智能网络推演';
    decisionHeadline.textContent = dec.reason_headline || '策略步进状态';
    decisionReason.textContent = dec.reason_detail || dec.reason || '就绪';

    const probs = dec.probabilities || [0.25, 0.25, 0.25, 0.25];
    const [pUp, pRight, pDown, pLeft] = [probs[0] ?? 0.25, probs[1] ?? 0.25, probs[2] ?? 0.25, probs[3] ?? 0.25];

    [rowUp, rowRight, rowDown, rowLeft].forEach(r => r && r.classList.remove('winner'));
    [starUp, starRight, starDown, starLeft].forEach(st => st && (st.textContent = ''));

    const setProb = (bar, val, star, row, p, dir) => {
      const pct = Math.max(0, Math.min(100, p * 100));
      bar.style.width = `${Math.max(1.5, pct)}%`;
      val.textContent = `${pct.toFixed(1)}%`;
      if (dec.action === dir) {
        row.classList.add('winner');
        star.textContent = '已采纳';
      }
    };

    setProb(barUp, valUp, starUp, rowUp, pUp, 'up');
    setProb(barRight, valRight, starRight, rowRight, pRight, 'right');
    setProb(barDown, valDown, starDown, rowDown, pDown, 'down');
    setProb(barLeft, valLeft, starLeft, rowLeft, pLeft, 'left');
  }

  // --- 小眼睛切换开关 ---
  btnToggleEye.addEventListener('click', () => {
    eyePreview = !eyePreview;
    syncEyePreviewDisplay();
  });

  // --- 样本库切换监听 ---
  taskSelect.addEventListener('change', () => {
    handleTaskChanged({ resetBackend: true });
  });

  if (btnRandomAerial) {
    btnRandomAerial.addEventListener('click', () => {
      if (!metaData || !metaData.tasks) return;
      const aerialTasks = metaData.tasks.filter(t => t.dataset_id === 'aerial_bank');
      if (!aerialTasks.length) return;
      const curTask = getSelectedTask();
      const curImgKey = curTask ? `${curTask.bank_id}:${curTask.area}` : '';
      const otherImgTasks = aerialTasks.filter(t => `${t.bank_id}:${t.area}` !== curImgKey);
      const pool = otherImgTasks.length ? otherImgTasks : aerialTasks;
      const pick = pool[Math.floor(Math.random() * pool.length)];
      taskSelect.value = pick.task_key;
      handleTaskChanged({ resetBackend: true });
      showToast(`已切换至：${pick.name}`, 'info', 2000);
    });
  }

  startSelect.addEventListener('change', updateGridHighlights);
  goalSelect.addEventListener('change', updateGridHighlights);

  // --- 删除当前临时上传场景 ---
  btnDeleteUpload.addEventListener('click', async () => {
    const task = getSelectedTask();
    if (!task || task.dataset_id !== 'upload') return;
    try {
      await api('/api/upload/delete', 'POST', { task_key: task.task_key });
      showToast(`已移除临时场景「${task.name}」`, 'info');
      await loadMeta({ suppressRestore: true });
    } catch (err) {
      showToast(`移除失败：${err.message}`, 'error');
    }
  });

  // --- 人工方向按钮与键盘监听 ---
  document.querySelectorAll('.pad-btn').forEach(btn => {
    btn.addEventListener('click', () => step(btn.dataset.act));
  });

  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    const keyMap = {
      w: 'up', ArrowUp: 'up',
      s: 'down', ArrowDown: 'down',
      a: 'left', ArrowLeft: 'left',
      d: 'right', ArrowRight: 'right'
    };
    const act = keyMap[e.key];
    if (act) {
      e.preventDefault();
      step(act);
    }
  });

  // --- 速度芯片切换 ---
  document.querySelectorAll('.speed-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.speed-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      playSpeedMs = parseInt(chip.dataset.speed, 10) || 750;
    });
  });

  // --- 模态弹窗控制 ---
  function openModal(id) { $(id).classList.remove('hidden'); }
  function closeModal(id) { $(id).classList.add('hidden'); }
  document.querySelectorAll('[data-close]').forEach(btn => {
    btn.addEventListener('click', () => closeModal(btn.dataset.close));
  });
  $('btn-open-upload').addEventListener('click', () => openModal('modal-upload'));
  $('btn-open-api').addEventListener('click', () => {
    const info = metaData && metaData.api_info;
    if (info) {
      if (!$('api-base-url').value) $('api-base-url').value = info.base_url || '';
      if (!$('api-model').value) $('api-model').value = info.model || '';
      if (info.timeout) $('api-timeout').value = info.timeout;
      const thinkEl = $('api-disable-thinking');
      if (thinkEl && typeof info.disable_thinking === 'boolean') {
        thinkEl.checked = info.disable_thinking;
      }
      $('api-key').placeholder = info.key_configured ? '已内置/保存密钥，留空保持不变' : 'sk-...';
    }
    const statusEl = $('api-test-status');
    if (statusEl) {
      statusEl.className = 'api-test-status hidden';
      statusEl.textContent = '';
    }
    openModal('modal-api');
  });

  // --- 自定义上传与多规格切片预览网格 ---
  const dropzone = $('dropzone');
  const dropPrompt = $('drop-prompt');
  const fileInput = $('file-input');
  const uploadPreviewWrap = $('upload-preview-wrap');
  const uploadPreview = $('upload-preview');
  const uploadGridOverlay = $('upload-grid-overlay');
  const uploadGridHint = $('upload-grid-hint');
  const btnSubmitUpload = $('btn-submit-upload');

  function renderUploadPreviewOverlay(k) {
    selectedUploadGrid = k;
    const spec = GRID_SPECS[k] || GRID_SPECS[5];
    uploadGridHint.textContent = spec.note;
    uploadGridOverlay.style.setProperty('--preview-n', String(k));
    uploadGridOverlay.innerHTML = '';
    for (let i = 0; i < k * k; i++) {
      uploadGridOverlay.appendChild(document.createElement('div'));
    }
  }

  document.querySelectorAll('#upload-grid-selector .grid-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('#upload-grid-selector .grid-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      renderUploadPreviewOverlay(parseInt(pill.dataset.grid, 10));
    });
  });

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });
  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary)';
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'var(--border)';
  });
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--border)';
    const file = e.dataTransfer.files[0];
    if (file) handleImageFile(file);
  });

  function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = e => {
      pendingUploadBase64 = e.target.result;
      uploadPreview.src = pendingUploadBase64;
      dropPrompt.classList.add('hidden');
      uploadPreviewWrap.classList.remove('hidden');
      btnSubmitUpload.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  btnSubmitUpload.addEventListener('click', async () => {
    if (!pendingUploadBase64) return;
    const name = $('upload-name').value.trim() || '临时演示场景';
    btnSubmitUpload.disabled = true;
    const oldText = btnSubmitUpload.textContent;
    btnSubmitUpload.textContent = '切片加载中...';
    try {
      const res = await api('/api/upload', 'POST', {
        image: pendingUploadBase64,
        name,
        grid: selectedUploadGrid
      });
      closeModal('modal-upload');
      showToast(`已生成 ${res.grid}×${res.grid} 临时场景（仅本次服务内存保留）`, 'success');
      await loadMeta({ preferTaskKey: res.task_key });
    } catch (err) {
      showToast(`上传切片失败：${err.message}`, 'error', 4200);
    } finally {
      btnSubmitUpload.disabled = false;
      btnSubmitUpload.textContent = oldText;
    }
  });

  // --- 云端 API 连通性测试 ---
  const btnTestApi = $('btn-test-api');
  if (btnTestApi) {
    btnTestApi.addEventListener('click', async () => {
      const base_url = $('api-base-url').value.trim();
      const model = $('api-model').value.trim();
      const api_key = $('api-key').value.trim();
      const timeout = parseInt($('api-timeout').value, 10) || 20;
      const disable_thinking = $('api-disable-thinking') ? $('api-disable-thinking').checked : true;
      const statusEl = $('api-test-status');
      btnTestApi.disabled = true;
      btnTestApi.textContent = '测试中...';
      statusEl.className = 'api-test-status';
      statusEl.textContent = '正在向云端接口发起轻量握手检测...';
      try {
        const res = await api('/api/config/test', 'POST', { base_url, model, api_key, timeout, disable_thinking });
        statusEl.className = `api-test-status ${res.ok ? 'ok' : 'err'}`;
        statusEl.textContent = res.ok ? `成功：${res.message}` : `未通过：${res.message}`;
      } catch (err) {
        statusEl.className = 'api-test-status err';
        statusEl.textContent = `测试失败：${err.message}`;
      } finally {
        btnTestApi.disabled = false;
        btnTestApi.textContent = '测试连通性';
      }
    });
  }

  // --- 云端 API 保存 ---
  $('btn-save-api').addEventListener('click', async () => {
    const base_url = $('api-base-url').value.trim();
    const model = $('api-model').value.trim();
    const api_key = $('api-key').value.trim();
    const timeout = parseInt($('api-timeout').value, 10) || 30;
    const disable_thinking = $('api-disable-thinking') ? $('api-disable-thinking').checked : true;

    if (!base_url || !model) {
      showToast('请至少填写 Base URL 与 Model 名称', 'warn');
      return;
    }

    try {
      const savedCfg = await api('/api/config', 'POST', { base_url, model, api_key, timeout, disable_thinking });
      if (metaData) {
        metaData.api_configured = true;
        metaData.api_info = savedCfg;
      }
      apiStatusDot.classList.add('connected');
      showToast('云端 API 配置已成功缓存在当前服务内存中', 'success');
      closeModal('modal-api');
    } catch (err) {
      showToast(`保存配置失败：${err.message}`, 'error');
    }
  });

  // --- 主题切换 ---
  const themeToggle = $('btn-theme-toggle');
  const iconMoon = $('icon-moon');
  const iconSun = $('icon-sun');
  let currentTheme = localStorage.getItem('geonav_lite_theme') || 'light';

  function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    iconMoon.classList.toggle('hidden', theme === 'dark');
    iconSun.classList.toggle('hidden', theme !== 'dark');
    localStorage.setItem('geonav_lite_theme', theme);
  }
  applyTheme(currentTheme);
  themeToggle.addEventListener('click', () => applyTheme(currentTheme === 'light' ? 'dark' : 'light'));

  // --- 绑定核心控制按钮 ---
  btnStartTask.addEventListener('click', startTask);
  btnStep.addEventListener('click', () => step());
  btnAuto.addEventListener('click', startAuto);
  btnPause.addEventListener('click', stopAuto);
  btnReset.addEventListener('click', resetAll);

  // --- 启动初始化 ---
  renderUploadPreviewOverlay(5);
  buildGridDOM(5);
  populateCoordSelects(5);
  loadMeta();
})();
