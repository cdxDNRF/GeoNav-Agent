/* GeoNav V5.0 Unified Workbench SPA Controller (Emoji-Free + Curated Interactive Engine) */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const escHtml = (s) =>
    String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");

  const CTRL_NAMES = {
    manual: "人工控制",
    frontier: "Frontier规则",
    M0: "冻结M0",
    api_m0_fallback: "单API＋M0失败回退"
  };

  const CLASS_NAMES = {
    human: "人工",
    rule: "规则",
    model: "M0",
    mixed: "混合接管",
    unstarted: "未执行",
    api: "API方案",
    fallback: "M0回退"
  };

  const RELATION_NAMES = {
    inside: "拍摄点在格内",
    outside: "同源格外",
    unrelated: "异源目标"
  };

  const ARM_LABELS = {
    Full: "真实目标",
    Disabled: "禁用目标",
    Mean: "均值目标",
    Wrong: "错误目标",
    NoTarget: "固定均值 · 无单题目标"
  };

  const METHOD_NAMES = {
    Spatial: "空间小头",
    TrainedMean: "训练过的无目标基线",
    RawCos: "冻结余弦基线"
  };

  const GRID_SPECS = {
    3: { budget: 6, note: "3×3 共 9 个图块 · 步数预算 6 步（极速轻量演示）" },
    4: { budget: 8, note: "4×4 共 16 个图块 · 步数预算 8 步（紧凑探索尺度）" },
    5: { budget: 10, note: "5×5 共 25 个图块 · 步数预算 10 步（M0 原生最佳协议）" },
    6: { budget: 14, note: "6×6 共 36 个图块 · 步数预算 14 步（中等粒度探索）" },
    7: { budget: 16, note: "7×7 共 49 个图块 · 步数预算 16 步（扩展视野尺度）" },
    10: { budget: 20, note: "10×10 共 100 个图块 · 步数预算 20 步（大区域前瞻保护）" }
  };

  const owner = (() => {
    const key = "geonav_v5_hex_owner";
    let v = window.sessionStorage.getItem(key);
    if (!v || !/^[0-9a-f]{32}$/.test(v)) {
      if (window.crypto && window.crypto.randomUUID) {
        v = crypto.randomUUID().replaceAll("-", "").toLowerCase();
      } else {
        v = Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
      }
      window.sessionStorage.setItem(key, v);
    }
    return v;
  })();

  let token = "";
  let apiConfig = null;
  let activeWorkspace = "overview";
  let activeNavTab = "interactive";
  let wizardIndex = 0;

  // Interactive Curated Engine State (/api/v5/interactive/*)
  let ixMeta = null;
  let ixState = null;
  let ixGrid = 5;
  let ixAutoTimer = null;
  let ixSpeedMs = 650;
  let ixClickMode = "start";
  let ixEyePreview = false;

  // Upload & Modal State
  let pendingUploadB64 = null;
  let selectedUploadGrid = 5;

  // Sealed Navigator & Arena State
  let navData = null;
  let navSelected = null;
  let navState = null;
  let navBatch = null;
  let navApiGroup = null;
  let navQueue = [];
  let navBusy = false;
  let navRunning = false;
  let navEpoch = 0;
  const navPreviewLoaded = new Set();

  // Sandbox State
  let sbCatalog = null;
  let sbScene = null;
  let sbRun = null;
  let sbBusy = false;
  let sbAuto = false;
  const sbRuns = new Map();

  // Ground Walk & Live State
  let gwId = null;
  let gwState = null;
  let glId = null;
  let glState = null;
  let glMode = "upload";
  let glImages = {};
  let glReady = false;

  // Evidence State
  let p31Catalog = null;
  let p31Arm = "Full";
  let p31Detail = null;
  let s41Catalog = null;
  let s41Arm = "Full";
  let s41Detail = null;
  let s41LastSeed = "0";

  const isTerminal = (s) => ["completed", "cancelled", "failed"].includes(s);
  const headers = () => ({
    "Content-Type": "application/json",
    "X-GeoNav-Session": owner,
    "X-GeoNav-Scene-Token": token
  });
  const query = (v = {}) => new URLSearchParams({ owner, ...v }).toString();
  const recKey = (v) => (v.kind || "session") + ":" + v.id;

  function showToast(msg, kind = "info", durationMs = 3000) {
    const host = $("v5-toast-container");
    if (!host || !msg) return;
    const el = document.createElement("div");
    el.className = `v5-toast ${kind}`;
    el.textContent = msg;
    host.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(-6px)";
      el.style.transition = "all 0.2s ease";
      setTimeout(() => el.remove(), 220);
    }, durationMs);
  }

  function notice(msg = "", isErr = false) {
    const el = $("global-notice");
    if (el) {
      el.textContent = msg || "";
      el.classList.toggle("error", Boolean(isErr && msg));
    }
    if (msg) showToast(msg, isErr ? "err" : "info", isErr ? 4200 : 2800);
  }

  function downloadable(url) {
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    a.click();
  }

  // ==================== POPOVER (!) EXPLANATIONS ====================
  let activeTipTrigger = null;
  function hidePopover() {
    const pop = $("v5-popover");
    if (pop) pop.classList.add("hidden");
    if (activeTipTrigger) {
      activeTipTrigger.classList.remove("active");
      activeTipTrigger = null;
    }
  }

  function showPopover(trigger) {
    const pop = $("v5-popover");
    if (!pop) return;
    if (activeTipTrigger === trigger && !pop.classList.contains("hidden")) {
      hidePopover();
      return;
    }
    if (activeTipTrigger) activeTipTrigger.classList.remove("active");
    activeTipTrigger = trigger;
    trigger.classList.add("active");
    $("v5-popover-title").textContent = trigger.dataset.tipTitle || "详细说明";
    $("v5-popover-body").textContent = trigger.dataset.tip || "";
    pop.classList.remove("hidden");

    const rect = trigger.getBoundingClientRect();
    const pw = pop.offsetWidth || 330;
    const ph = pop.offsetHeight || 130;
    let left = rect.left + rect.width + 8;
    if (left + pw > window.innerWidth - 12) {
      left = Math.max(12, rect.left - pw - 8);
    }
    let top = rect.top - 6;
    if (top + ph > window.innerHeight - 12) {
      top = Math.max(12, window.innerHeight - ph - 12);
    }
    pop.style.left = `${Math.round(left)}px`;
    pop.style.top = `${Math.round(Math.max(12, top))}px`;
  }

  function bindPopoverAndModals() {
    document.addEventListener("click", (e) => {
      const tipBtn = e.target.closest(".v5-tip-btn");
      if (tipBtn) {
        e.stopPropagation();
        showPopover(tipBtn);
        return;
      }
      const pop = $("v5-popover");
      if (pop && !pop.classList.contains("hidden") && !pop.contains(e.target)) {
        hidePopover();
      }
    });
    $("v5-popover-close")?.addEventListener("click", hidePopover);
    $$("[data-v5-close]").forEach((btn) => {
      btn.addEventListener("click", () => $(btn.dataset.v5Close)?.classList.add("hidden"));
    });
    $$(".v5-modal-backdrop").forEach((bd) => {
      bd.addEventListener("click", (e) => {
        if (e.target === bd) bd.classList.add("hidden");
      });
    });
  }

  // ==================== WORKSPACE, SUBTAB & THEME SWITCHING ====================
  function switchWorkspace(wsId) {
    hidePopover();
    activeWorkspace = wsId;
    $$(".v5-rail-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.workspace === wsId));
    $$(".v5-workspace").forEach((sec) => sec.classList.toggle("active", sec.id === `ws-${wsId}`));
  }

  function switchNavTab(tabId) {
    activeNavTab = tabId;
    $$("[data-nav-tab]").forEach((b) => b.classList.toggle("active", b.dataset.navTab === tabId));
    $$("#ws-navigator .v5-subview").forEach((v) => v.classList.toggle("active", v.id === `nav-view-${tabId}`));
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("v5-icon-moon")?.classList.toggle("hidden", theme === "dark");
    $("v5-icon-sun")?.classList.toggle("hidden", theme !== "dark");
    if ($("v5-theme-label")) $("v5-theme-label").textContent = theme === "dark" ? "浅色" : "暗色";
    localStorage.setItem("geonav_v5_theme", theme);
  }

  function bindGlobalUI() {
    $$(".v5-rail-item").forEach((btn) => btn.addEventListener("click", () => switchWorkspace(btn.dataset.workspace)));
    $$("[data-goto]").forEach((btn) => btn.addEventListener("click", () => switchWorkspace(btn.dataset.goto)));
    $$("[data-nav-tab]").forEach((btn) => btn.addEventListener("click", () => switchNavTab(btn.dataset.navTab)));

    $$("[data-arena-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const t = btn.dataset.arenaTab;
        $$("[data-arena-tab]").forEach((b) => b.classList.toggle("active", b === btn));
        $$("#ws-arena .v5-subview").forEach((v) => v.classList.toggle("active", v.id === `arena-view-${t}`));
      });
    });
    $$("[data-ground-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const t = btn.dataset.groundTab;
        $$("[data-ground-tab]").forEach((b) => b.classList.toggle("active", b === btn));
        $$("#ws-ground .v5-subview").forEach((v) => v.classList.toggle("active", v.id === `ground-view-${t}`));
      });
    });
    $$("[data-ev-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const t = btn.dataset.evTab;
        $$("[data-ev-tab]").forEach((b) => b.classList.toggle("active", b === btn));
        $$("#ws-evidence .v5-subview").forEach((v) => v.classList.toggle("active", v.id === `ev-view-${t}`));
      });
    });

    const savedTheme = localStorage.getItem("geonav_v5_theme") || "light";
    applyTheme(savedTheme);
    $("btn-theme-toggle")?.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
    });
    $("btn-global-refresh")?.addEventListener("click", () => refreshAll());
  }

  // ==================== 5-STEP INTERACTIVE WIZARD ====================
  const WIZARD_STEPS = [
    {
      badge: "向导 1 / 5",
      title: "平台全景与六大核心评测基准 (Overview)",
      desc: "浏览 16,200+ 航拍轨迹、192 张独立航拍底图（960 题）与 192 地面任务的核心指标。",
      run: () => switchWorkspace("overview")
    },
    {
      badge: "向导 2 / 5",
      title: "统一探索控制台：192 图互动与 M0 决策透视 (Navigator)",
      desc: "载入 Masa S2 航拍任务，执行 M0 首步并实时观察小眼睛预览与四向推理倾向度。",
      run: () => runPreset("aerial-m0")
    },
    {
      badge: "向导 3 / 5",
      title: "自由场景与人机平滑接管沙箱 (Sandbox)",
      desc: "创建自由航拍网格场景，演示人工探索与 Agent (M0) 历史同步接管及 SVG 轨迹叠加。",
      run: () => runPreset("sandbox-takeover")
    },
    {
      badge: "向导 4 / 5",
      title: "十题同题打擂与小队列批测 (Arena)",
      desc: "同屏检视固定十题上本地 M0、Frontier 规则与云端 API 的成功率与终局距离对比。",
      run: () => runPreset("arena-compare")
    },
    {
      badge: "向导 5 / 5",
      title: "空间感知与特征证据显微镜 (Evidence)",
      desc: "透视 CV031 (CCVPE 20例) 与 CV041 (三城 150例 7×7 空间小头) 热力图与真值/预测叠合。",
      run: () => runPreset("evidence-demo")
    }
  ];

  function renderWizard() {
    const step = WIZARD_STEPS[wizardIndex];
    $("wizard-step-badge").textContent = step.badge;
    $("wizard-step-title").textContent = step.title;
    $("wizard-step-desc").textContent = step.desc;
    $("wizard-prev").disabled = wizardIndex === 0;
    $("wizard-next").disabled = wizardIndex === WIZARD_STEPS.length - 1;
  }

  function bindWizard() {
    $("btn-start-wizard").addEventListener("click", () => {
      $("wizard-bar").hidden = false;
      wizardIndex = 0;
      renderWizard();
      WIZARD_STEPS[0].run();
    });
    $("wizard-close").addEventListener("click", () => ($("wizard-bar").hidden = true));
    $("wizard-prev").addEventListener("click", () => {
      if (wizardIndex > 0) {
        wizardIndex--;
        renderWizard();
        WIZARD_STEPS[wizardIndex].run();
      }
    });
    $("wizard-next").addEventListener("click", () => {
      if (wizardIndex < WIZARD_STEPS.length - 1) {
        wizardIndex++;
        renderWizard();
        WIZARD_STEPS[wizardIndex].run();
      }
    });
    $("wizard-run-demo").addEventListener("click", () => WIZARD_STEPS[wizardIndex].run());
  }

  // ==================== WORKSPACE 1: OVERVIEW & PRESETS ====================
  async function loadOverview() {
    try {
      const res = await fetch(`/api/v5/overview?${query()}`, { headers: headers() });
      const ov = await res.json();
      if (!res.ok) throw new Error(ov.error || "总览读取失败");

      $("global-pill-kernel").textContent = `内核: ${ov.kernel || "geonav-core-v17"}`;
      const totalIx = ov.counts?.total_interactive_tasks || (960 + 192);
      $("ov-stat-tasks").textContent = Number(totalIx).toLocaleString("en-US");

      const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);
      const tbody = $("overview-benchmark-table").querySelector("tbody");
      tbody.innerHTML = (ov.benchmarks || [])
        .map((b) => {
          const mw =
            b.cue_mean_sr != null && b.cue_wrong_sr != null
              ? `${fmtPct(b.cue_mean_sr)} / ${fmtPct(b.cue_wrong_sr)}`
              : b.cue_mean_sr != null
              ? fmtPct(b.cue_mean_sr)
              : "—";
          return `<tr>
            <td><strong>${escHtml(b.name)}</strong></td>
            <td>${escHtml(b.grid)} · B=${escHtml(b.budget)}</td>
            <td><strong style="color:var(--v5-primary)">${fmtPct(b.m0_sr)}</strong> ${b.m0_sg_cells != null ? `<small class="v5-text-muted">(${b.m0_sg_cells}格)</small>` : ""}</td>
            <td>${fmtPct(b.explorer_only_sr)}</td>
            <td>${mw}</td>
            <td>${fmtPct(b.frontier_sr)}</td>
            <td>
              <button type="button" class="v5-tip-btn" data-tip-title="${escHtml(b.name)} · 核心结论" data-tip="${escHtml(b.detail || b.highlight)}">!</button>
            </td>
          </tr>`;
        })
        .join("");

      const assetList = $("overview-asset-list");
      assetList.innerHTML = (ov.assets || [])
        .map(
          (a) => `<li style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div style="min-width:0;"><strong>${escHtml(a.label)}</strong><div class="v5-text-muted">${escHtml(a.detail)}</div></div>
            <span class="v5-badge ${a.ready ? "v5-badge-green" : "v5-badge-amber"}">${a.ready ? "已就绪" : "待配置"}</span>
          </li>`
        )
        .join("");
    } catch (e) {
      notice(e.message, true);
    }
  }

  async function runPreset(preset) {
    try {
      if (preset === "aerial-m0") {
        switchWorkspace("navigator");
        switchNavTab("interactive");
        if (!ixMeta) await ixLoadMeta();
        const masaTask = ixMeta?.tasks?.find((t) => t.bank_id === "masa-s2") || ixMeta?.tasks?.find((t) => t.dataset_id === "aerial_bank");
        if (masaTask) {
          $("ix-task-select").value = masaTask.task_key;
          await ixHandleTaskChanged({ resetBackend: true });
          $("ix-policy-select").value = "M0";
          await ixStartTask();
          await ixStep();
          showToast("已载入 Masa S2 航拍任务并由 M0 前推第 1 步，可继续单步或自动运行。", "ok");
        }
      } else if (preset === "capturepoint-demo") {
        switchWorkspace("navigator");
        switchNavTab("interactive");
        if (!ixMeta) await ixLoadMeta();
        const camTask = ixMeta?.tasks?.find((t) => t.fixed_endpoints);
        if (camTask) {
          $("ix-task-select").value = camTask.task_key;
          await ixHandleTaskChanged({ resetBackend: true });
          showToast("已切换至 OpenCVL 192 地面拍摄点队列，点击【开始任务】即可体验。", "ok");
        }
      } else if (preset === "sandbox-takeover") {
        switchWorkspace("sandbox");
        if (!sbCatalog) await sbRefresh();
        await sbCreateAndStart();
        showToast("已启动自由航拍沙箱，可用方向键手动探索 1~2 步后切换 Agent 接管。", "ok");
      } else if (preset === "arena-compare") {
        switchWorkspace("arena");
        $$("[data-arena-tab]").forEach((b) => b.classList.toggle("active", b.dataset.arenaTab === "api10"));
        $$("#ws-arena .v5-subview").forEach((v) => v.classList.toggle("active", v.id === "arena-view-api10"));
        if ($("ar-api-saved").options.length > 1) {
          $("ar-api-saved").selectedIndex = 1;
          const id = $("ar-api-saved").value;
          if (id) {
            const v = await navApi("state?" + query({ id }), undefined, "/api/api-comparisons/");
            renderArenaAPI(v);
            showToast("已打开固定十题本地 M0 vs Frontier vs 云端 API 对照账本。", "ok");
          }
        }
      } else if (preset === "evidence-demo") {
        switchWorkspace("evidence");
        $$("[data-ev-tab]").forEach((b) => b.classList.toggle("active", b.dataset.evTab === "cv041"));
        $$("#ws-evidence .v5-subview").forEach((v) => v.classList.toggle("active", v.id === "ev-view-cv041"));
        showToast("已打开 CV041 三城市 150 案例空间小头证据显微镜。", "ok");
      }
    } catch (e) {
      notice(e.message, true);
    }
  }

  function bindPresets() {
    $$("[data-preset]").forEach((btn) => btn.addEventListener("click", () => runPreset(btn.dataset.preset)));
  }

  // ==================== WORKSPACE 2A: INTERACTIVE CURATED ENGINE ====================
  async function ixApi(subpath, method = "GET", body = null) {
    const opts = {
      method,
      headers: {
        "X-GeoNav-Session": owner,
        "X-GeoNav-Scene-Token": token
      }
    };
    if (body !== null) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(`/api/v5/interactive/${subpath}`, opts);
    const data = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  }

  function ixSelectedTask() {
    const key = $("ix-task-select")?.value;
    return ixMeta?.tasks?.find((t) => t.task_key === key) || null;
  }

  function ixBuildGridDOM(k) {
    ixGrid = k;
    $("ix-grid-label").textContent = `${k}×${k}`;
    const board = $("ix-board");
    board.style.setProperty("--grid-n", String(k));
    board.classList.toggle("grid-dense", k >= 7);
    board.innerHTML = "";
    const total = k * k;
    for (let i = 0; i < total; i++) {
      const cell = document.createElement("div");
      cell.className = "ix-cell";
      cell.dataset.idx = String(i);
      cell.innerHTML = `<span class="ix-cell-idx">${i}</span><div class="ix-cell-badges"></div>`;
      cell.addEventListener("click", () => ixHandleCellClick(i));
      board.appendChild(cell);
    }
  }

  function ixPopulateCoords(k) {
    const sSel = $("ix-start-select");
    const gSel = $("ix-goal-select");
    const prevS = parseInt(sSel.value, 10);
    const prevG = parseInt(gSel.value, 10);
    sSel.innerHTML = "";
    gSel.innerHTML = "";
    const total = k * k;
    for (let i = 0; i < total; i++) {
      const r = Math.floor(i / k);
      const c = i % k;
      sSel.appendChild(new Option(`${i} (${r},${c})`, String(i)));
      gSel.appendChild(new Option(`${i} (${r},${c})`, String(i)));
    }
    sSel.value = !isNaN(prevS) && prevS < total ? String(prevS) : "0";
    gSel.value = !isNaN(prevG) && prevG < total ? String(prevG) : String(total - 1);
  }

  function ixClearGridTiles() {
    const board = $("ix-board");
    board.querySelectorAll(".ix-cell").forEach((cell) => {
      cell.querySelector("img")?.remove();
      cell.classList.remove("preview-dim", "is-visited", "is-curr");
    });
  }

  function ixUpdateHighlights() {
    const sVal = parseInt($("ix-start-select").value, 10);
    const gVal = parseInt($("ix-goal-select").value, 10);
    const currVal = ixState ? ixState.position : -1;
    $("ix-board").querySelectorAll(".ix-cell").forEach((cell) => {
      const i = Number(cell.dataset.idx);
      cell.classList.toggle("is-start", i === sVal);
      cell.classList.toggle("is-goal", i === gVal);
      cell.classList.toggle("is-curr", Boolean(ixState && i === currVal));
      const badges = cell.querySelector(".ix-cell-badges");
      if (badges) {
        let html = "";
        if (i === sVal) html += '<span class="ix-pill ix-pill-start">起</span>';
        if (i === gVal) html += '<span class="ix-pill ix-pill-goal">终</span>';
        badges.innerHTML = html;
      }
    });
  }

  function ixSyncEyeDisplay() {
    const task = ixSelectedTask();
    const tkey = (ixState && ixState.task_key) || (task && task.task_key) || "";
    const visited = new Set(ixState?.visited || []);
    const stamp = Date.now();

    $("ix-btn-eye")?.classList.toggle("active", ixEyePreview);
    $("ix-eye-open")?.classList.toggle("hidden", ixEyePreview);
    $("ix-eye-closed")?.classList.toggle("hidden", !ixEyePreview);
    if ($("ix-eye-text")) $("ix-eye-text").textContent = ixEyePreview ? "关闭预览" : "预览全图";

    $("ix-board").querySelectorAll(".ix-cell").forEach((cell) => {
      const i = Number(cell.dataset.idx);
      const isVis = visited.has(i);
      let img = cell.querySelector("img");
      cell.classList.toggle("is-visited", isVis);
      if (isVis && tkey) {
        cell.classList.remove("preview-dim");
        if (!img || img.dataset.tkey !== tkey) {
          img?.remove();
          img = document.createElement("img");
          img.alt = `格${i}`;
          img.dataset.tkey = tkey;
          img.src = `/api/v5/interactive/image?task_key=${encodeURIComponent(tkey)}&cell=${i}&t=${stamp}`;
          cell.prepend(img);
        }
      } else if (ixEyePreview && tkey) {
        cell.classList.toggle("preview-dim", Boolean(ixState));
        if (!img || img.dataset.tkey !== tkey) {
          img?.remove();
          img = document.createElement("img");
          img.alt = `全图预览格${i}`;
          img.dataset.tkey = tkey;
          img.src = `/api/v5/interactive/image?task_key=${encodeURIComponent(tkey)}&cell=${i}&observer=1&t=${stamp}`;
          cell.prepend(img);
        }
      } else {
        cell.classList.remove("preview-dim");
        img?.remove();
      }
    });

    const badge = $("ix-mode-badge");
    if (!badge) return;
    if (ixEyePreview) {
      badge.textContent = "全图预览中 (仅人类可见 · 智能体盲视)";
      badge.className = "v5-badge v5-badge-blue";
    } else if (ixState && !ixState.done) {
      badge.textContent = `推演中 · 当前 [${ixState.coords.join(",")}]`;
      badge.className = "v5-badge v5-badge-green";
    } else {
      badge.textContent = task?.fixed_endpoints ? "地面固定配对 (直接点击开始)" : "点击格子设置起点 / 目标";
      badge.className = "v5-badge";
    }
  }

  function ixResetRightPanel(budget = 10) {
    const imgT = $("ix-target-img");
    const imgC = $("ix-current-img");
    imgT.src = "";
    imgC.src = "";
    delete imgT.dataset.goalKey;
    delete imgC.dataset.currKey;
    imgT.classList.add("hidden");
    imgC.classList.add("hidden");
    $("ix-target-empty").classList.remove("hidden");
    $("ix-current-empty").classList.remove("hidden");
    $("ix-target-coord").textContent = "目标格子: --";
    $("ix-current-coord").textContent = "当前格子: --";

    $("ix-kpi-status").textContent = "待开始";
    $("ix-kpi-step").textContent = `0 / ${budget} 步`;
    $("ix-kpi-dist").textContent = "-- 格";
    $("ix-kpi-result").textContent = "等待执行";
    $("ix-kpi-result").style.color = "";

    $("ix-branch-badge").textContent = "智能网络推演";
    $("ix-action-badge").textContent = "等待推演";
    $("ix-dec-headline").textContent = "等待任务启动";
    $("ix-dec-detail").textContent = "选定起终点后点击【开始任务】，可使用【单步推演】或【自动运行】查看决策依据。";

    for (const dir of ["up", "right", "down", "left"]) {
      $(`ix-bar-${dir}`).style.width = "25%";
      $(`ix-val-${dir}`).textContent = "25.0%";
      $(`ix-win-${dir}`).textContent = "";
      $(`ix-row-${dir}`).classList.remove("winner");
    }
  }

  function ixStopAuto() {
    if (ixAutoTimer) {
      clearTimeout(ixAutoTimer);
      ixAutoTimer = null;
    }
    $("ix-btn-auto").disabled = !ixState || ixState.done;
    $("ix-btn-pause").disabled = true;
    $("ix-btn-step").disabled = !ixState || ixState.done;
  }

  async function ixHandleTaskChanged({ resetBackend = true } = {}) {
    ixStopAuto();
    ixState = null;
    $("ix-btn-step").disabled = true;
    $("ix-btn-auto").disabled = true;
    $("ix-btn-pause").disabled = true;
    if (resetBackend) ixApi("reset", "POST", {}).catch(() => {});

    const task = ixSelectedTask();
    if (!task) return;
    const k = task.grid || 5;
    if (k !== ixGrid || $("ix-board").children.length !== k * k) {
      ixBuildGridDOM(k);
      ixPopulateCoords(k);
    } else {
      ixClearGridTiles();
    }

    const total = k * k;
    const sDef = typeof task.start === "number" ? task.start : 0;
    const gDef = typeof task.default_goal === "number" ? task.default_goal : (sDef + Math.floor(total / 2)) % total;
    $("ix-start-select").value = String(sDef);
    $("ix-goal-select").value = String(gDef === sDef ? (sDef + 1) % total : gDef);

    const fixed = Boolean(task.fixed_endpoints);
    $("ix-start-select").disabled = fixed;
    $("ix-goal-select").disabled = fixed;
    $("ix-btn-delete-upload")?.classList.toggle("hidden", task.dataset_id !== "upload");
    ixClickMode = "start";

    ixResetRightPanel(task.budget || 10);
    ixUpdateHighlights();
    ixSyncEyeDisplay();
    if (navData) navControls();
  }

  function ixRenderState(s) {
    if (!s) return;
    ixUpdateHighlights();
    ixSyncEyeDisplay();

    const ts = Date.now();
    const imgT = $("ix-target-img");
    const imgC = $("ix-current-img");
    $("ix-target-empty").classList.add("hidden");
    $("ix-current-empty").classList.add("hidden");
    imgT.classList.remove("hidden");
    imgC.classList.remove("hidden");
    const gKey = `${s.task_key || ""}_${s.goal}_${s.start}`;
    if (imgT.dataset.goalKey !== gKey) {
      imgT.dataset.goalKey = gKey;
      imgT.src = `/api/v5/interactive/image?target=1&task_key=${encodeURIComponent(s.task_key || "")}&t=${ts}`;
    }
    const cKey = `${s.task_key || ""}_${s.position}_${s.step}`;
    if (imgC.dataset.currKey !== cKey) {
      imgC.dataset.currKey = cKey;
      imgC.src = `/api/v5/interactive/image?cell=${s.position}&task_key=${encodeURIComponent(s.task_key || "")}&t=${ts}`;
    }
    $("ix-target-coord").textContent = `目标: [${s.goal_coords.join(",")}] (格${s.goal})`;
    $("ix-current-coord").textContent = `当前: [${s.coords.join(",")}] (格${s.position})`;

    $("ix-kpi-step").textContent = `${s.step} / ${s.budget} 步`;
    $("ix-kpi-dist").textContent = `${s.distance_cells} 格 · ${s.distance_euclidean}`;
    const resEl = $("ix-kpi-result");
    if (s.success) {
      $("ix-kpi-status").textContent = "已到达目标";
      resEl.textContent = "成功定位 (SG=0)";
      resEl.style.color = "var(--v5-success)";
    } else if (s.done) {
      $("ix-kpi-status").textContent = "预算耗尽";
      resEl.textContent = `未到达 (SG=${s.distance_cells})`;
      resEl.style.color = "var(--v5-danger)";
    } else {
      $("ix-kpi-status").textContent = s.step === 0 ? "已就绪" : "探索中";
      resEl.textContent = "推演进行中";
      resEl.style.color = "";
    }

    const dec = s.decision || {};
    $("ix-branch-badge").textContent = dec.branch || "智能网络推演";
    $("ix-action-badge").textContent = dec.action_name || "等待推演";
    $("ix-dec-headline").textContent = dec.reason_headline || "策略步进状态";
    $("ix-dec-detail").textContent = dec.reason_detail || dec.reason || "就绪";

    const probs = dec.probabilities || [0.25, 0.25, 0.25, 0.25];
    const dirs = ["up", "right", "down", "left"];
    dirs.forEach((dir, idx) => {
      const pct = Math.max(0, Math.min(100, (probs[idx] ?? 0.25) * 100));
      $(`ix-bar-${dir}`).style.width = `${Math.max(1.5, pct)}%`;
      $(`ix-val-${dir}`).textContent = `${pct.toFixed(1)}%`;
      const isWin = dec.action === dir;
      $(`ix-row-${dir}`).classList.toggle("winner", isWin);
      $(`ix-win-${dir}`).textContent = isWin ? "已采纳" : "";
    });
    if (navData) navControls();
  }

  async function ixLoadMeta({ preferTaskKey = null, suppressRestore = false } = {}) {
    ixMeta = await ixApi("meta");
    if (ixMeta.stats && $("ix-footer-stats")) {
      const st = ixMeta.stats;
      $("ix-footer-stats").textContent = `内置 ${st.aerial_images} 张独立航拍底图 (${st.aerial_tasks} 题) + ${st.ground_tasks} 地面题 · 自定义切片纯内存保留不污染标准样本库`;
    }
    const sel = $("ix-task-select");
    sel.innerHTML = "";
    const groups = {};
    for (const t of ixMeta.tasks || []) {
      const g = t.group || "测试样本";
      if (!groups[g]) groups[g] = [];
      groups[g].push(t);
    }
    for (const [gName, rows] of Object.entries(groups)) {
      const og = document.createElement("optgroup");
      og.label = gName.includes("题)") ? gName : `${gName} (${rows.length})`;
      for (const task of rows) og.appendChild(new Option(task.name, task.task_key));
      sel.appendChild(og);
    }

    if (preferTaskKey && ixMeta.tasks.some((t) => t.task_key === preferTaskKey)) {
      sel.value = preferTaskKey;
      await ixHandleTaskChanged({ resetBackend: true });
      return;
    }
    if (!suppressRestore && ixMeta.active_session) {
      const sess = ixMeta.active_session;
      if (sess.task_key && ixMeta.tasks.some((t) => t.task_key === sess.task_key)) sel.value = sess.task_key;
      if (sess.grid && sess.grid !== ixGrid) {
        ixBuildGridDOM(sess.grid);
        ixPopulateCoords(sess.grid);
      }
      ixState = sess;
      $("ix-start-select").value = String(sess.start);
      $("ix-goal-select").value = String(sess.goal);
      $("ix-btn-delete-upload")?.classList.toggle("hidden", ixSelectedTask()?.dataset_id !== "upload");
      ixRenderState(ixState);
      $("ix-btn-step").disabled = Boolean(ixState.done);
      $("ix-btn-auto").disabled = Boolean(ixState.done);
    } else if (ixMeta.tasks.length > 0) {
      await ixHandleTaskChanged({ resetBackend: false });
    }
  }

  async function ixStartTask() {
    ixStopAuto();
    const btn = $("ix-btn-start");
    btn.disabled = true;
    const oldText = btn.textContent;
    btn.textContent = "准备中...";
    try {
      ixState = await ixApi("create", "POST", {
        task_key: $("ix-task-select").value,
        policy: $("ix-policy-select").value,
        start: parseInt($("ix-start-select").value, 10),
        goal: parseInt($("ix-goal-select").value, 10),
        seed: 0
      });
      if (ixState.grid !== ixGrid) {
        ixBuildGridDOM(ixState.grid);
        ixPopulateCoords(ixState.grid);
      }
      ixClearGridTiles();
      ixRenderState(ixState);
      $("ix-btn-step").disabled = false;
      $("ix-btn-auto").disabled = false;
    } catch (err) {
      notice(`创建交互任务失败：${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = oldText;
    }
  }

  async function ixStep(action = null) {
    if (!ixState || ixState.done) return;
    $("ix-btn-step").disabled = true;
    try {
      ixState = await ixApi("step", "POST", { action });
      ixRenderState(ixState);
      if (ixState.done) ixStopAuto();
    } catch (err) {
      ixStopAuto();
      notice(`推演异常：${err.message}`, true);
    } finally {
      if (ixState && !ixState.done && !ixAutoTimer) $("ix-btn-step").disabled = false;
    }
  }

  function ixStartAuto() {
    if (ixAutoTimer || !ixState || ixState.done) return;
    $("ix-btn-auto").disabled = true;
    $("ix-btn-pause").disabled = false;
    $("ix-btn-step").disabled = true;
    const tick = async () => {
      if (!ixAutoTimer || !ixState || ixState.done) {
        ixStopAuto();
        return;
      }
      await ixStep();
      if (ixAutoTimer && ixState && !ixState.done) {
        ixAutoTimer = setTimeout(tick, ixSpeedMs);
      } else {
        ixStopAuto();
      }
    };
    ixAutoTimer = setTimeout(tick, 90);
  }

  function ixHandleCellClick(index) {
    if (ixState && !ixState.done && ["running", "ready"].includes(ixState.status)) {
      const curr = ixState.position;
      const dr = Math.floor(index / ixGrid) - Math.floor(curr / ixGrid);
      const dc = (index % ixGrid) - (curr % ixGrid);
      if (Math.abs(dr) + Math.abs(dc) === 1) {
        const act = dr === -1 ? "up" : dr === 1 ? "down" : dc === -1 ? "left" : "right";
        ixStep(act);
      }
      return;
    }
    const task = ixSelectedTask();
    if (task?.fixed_endpoints) {
      showToast("地面拍摄点样本采用固定起终点配对，直接点击【开始任务】即可体验。", "info");
      return;
    }
    if (ixClickMode === "start") {
      $("ix-start-select").value = String(index);
      ixClickMode = "goal";
      if (!ixEyePreview) $("ix-mode-badge").textContent = `已选起点 ${index}，点击格子设【目标】`;
    } else {
      $("ix-goal-select").value = String(index);
      ixClickMode = "start";
      if (!ixEyePreview) $("ix-mode-badge").textContent = `已选目标 ${index}，点击格子重选【起点】`;
    }
    ixUpdateHighlights();
  }

  function bindInteractiveExplorer() {
    ixBuildGridDOM(5);
    ixPopulateCoords(5);
    $("ix-btn-start").addEventListener("click", ixStartTask);
    $("ix-btn-reset").addEventListener("click", () => ixHandleTaskChanged({ resetBackend: true }));
    $("ix-btn-step").addEventListener("click", () => ixStep());
    $("ix-btn-auto").addEventListener("click", ixStartAuto);
    $("ix-btn-pause").addEventListener("click", ixStopAuto);
    $("ix-btn-eye").addEventListener("click", () => {
      ixEyePreview = !ixEyePreview;
      ixSyncEyeDisplay();
    });
    $("ix-task-select").addEventListener("change", () => ixHandleTaskChanged({ resetBackend: true }));
    $("ix-start-select").addEventListener("change", ixUpdateHighlights);
    $("ix-goal-select").addEventListener("change", ixUpdateHighlights);
    $("ix-policy-select").addEventListener("change", async () => {
      if (ixState && !ixState.done) {
        const prev = ixState.policy;
        try {
          ixState = await ixApi("mode", "POST", { policy: $("ix-policy-select").value });
          ixRenderState(ixState);
          showToast(`已平滑切换策略为 ${ixState.policy}`, "ok");
        } catch (err) {
          $("ix-policy-select").value = prev;
          notice(err.message, true);
        }
      }
    });
    $("ix-btn-random")?.addEventListener("click", () => {
      const aerials = (ixMeta?.tasks || []).filter((t) => t.dataset_id === "aerial_bank");
      if (!aerials.length) return;
      const cur = ixSelectedTask();
      const curArea = cur ? `${cur.bank_id}:${cur.area}` : "";
      const pool = aerials.filter((t) => `${t.bank_id}:${t.area}` !== curArea);
      const pick = (pool.length ? pool : aerials)[Math.floor(Math.random() * (pool.length || aerials.length))];
      $("ix-task-select").value = pick.task_key;
      ixHandleTaskChanged({ resetBackend: true });
      showToast(`已切换底图：${pick.name}`, "info", 2000);
    });
    $("ix-btn-delete-upload")?.addEventListener("click", async () => {
      const task = ixSelectedTask();
      if (!task || task.dataset_id !== "upload") return;
      try {
        await ixApi("upload/delete", "POST", { task_key: task.task_key });
        await ixLoadMeta({ suppressRestore: true });
        await loadOverview();
        showToast("已移除临时上传场景", "info");
      } catch (err) {
        notice(err.message, true);
      }
    });
    $$("[data-ix-act]").forEach((b) => b.addEventListener("click", () => ixStep(b.dataset.ixAct)));
    $$(".ix-speed").forEach((b) => {
      b.addEventListener("click", () => {
        $$(".ix-speed").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        ixSpeedMs = parseInt(b.dataset.speed, 10) || 650;
      });
    });
    setInterval(() => {
      if (ixState && !ixState.done) ixApi("heartbeat").catch(() => {});
    }, 25000);
  }

  // ==================== UPLOAD & CLOUD API MODALS ====================
  function renderUploadGridOverlay(k) {
    selectedUploadGrid = k;
    $("v5-upload-grid-hint").textContent = (GRID_SPECS[k] || GRID_SPECS[5]).note;
    const ov = $("v5-upload-grid-overlay");
    ov.style.setProperty("--preview-n", String(k));
    ov.innerHTML = "";
    for (let i = 0; i < k * k; i++) ov.appendChild(document.createElement("div"));
  }

  function updateApiBadges() {
    const has = Boolean(apiConfig);
    $("v5-api-dot")?.classList.toggle("online", has);
    if ($("global-pill-api")) $("global-pill-api").textContent = has ? `云API: ${apiConfig.model}` : "云API: 本地离线";
    if ($("ar-api-config-info")) {
      $("ar-api-config-info").textContent = has
        ? `${apiConfig.model} · ${apiConfig.base_url} · 超时 ${apiConfig.timeout}s`
        : "尚未配置云端 API 密钥；本地 M0、Frontier 与已存对照记录正常可用。";
    }
  }

  function openApiModal() {
    const info = apiConfig || ixMeta?.api_info;
    if (info) {
      if (!$("v5-api-base-url").value) $("v5-api-base-url").value = info.base_url || "";
      if (!$("v5-api-model").value) $("v5-api-model").value = info.model || "";
      if (info.timeout) $("v5-api-timeout").value = info.timeout;
      if (typeof info.disable_thinking === "boolean") $("v5-api-disable-thinking").checked = info.disable_thinking;
      $("v5-api-key").placeholder = info.key_configured ? "已保存密钥，留空保持不变" : "sk-...";
    }
    const box = $("v5-api-test-status");
    box.className = "v5-api-status-box hidden";
    box.textContent = "";
    $("modal-api").classList.remove("hidden");
  }

  function bindUploadAndApiModals() {
    renderUploadGridOverlay(5);
    $("btn-open-upload").addEventListener("click", () => $("modal-upload").classList.remove("hidden"));
    $("btn-open-api").addEventListener("click", openApiModal);
    $("ar-btn-open-api-cfg")?.addEventListener("click", openApiModal);

    $$("#v5-upload-grid-selector .v5-btn").forEach((pill) => {
      pill.addEventListener("click", () => {
        $$("#v5-upload-grid-selector .v5-btn").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        renderUploadGridOverlay(parseInt(pill.dataset.grid, 10));
      });
    });

    const dz = $("v5-dropzone");
    const fi = $("v5-file-input");
    const readImg = (file) => {
      const r = new FileReader();
      r.onload = (ev) => {
        pendingUploadB64 = ev.target.result;
        $("v5-upload-preview").src = pendingUploadB64;
        $("v5-drop-prompt").classList.add("hidden");
        $("v5-upload-preview-wrap").classList.remove("hidden");
        $("v5-btn-submit-upload").disabled = false;
      };
      r.readAsDataURL(file);
    };
    dz.addEventListener("click", () => fi.click());
    fi.addEventListener("change", (e) => e.target.files[0] && readImg(e.target.files[0]));
    dz.addEventListener("dragover", (e) => e.preventDefault());
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer.files[0]) readImg(e.dataTransfer.files[0]);
    });

    $("v5-btn-submit-upload").addEventListener("click", async () => {
      if (!pendingUploadB64) return;
      const btn = $("v5-btn-submit-upload");
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = "切片中...";
      try {
        const res = await ixApi("upload", "POST", {
          image: pendingUploadB64,
          name: $("v5-upload-name").value.trim() || "V5自定义临时场景",
          grid: selectedUploadGrid
        });
        $("modal-upload").classList.add("hidden");
        switchWorkspace("navigator");
        switchNavTab("interactive");
        await ixLoadMeta({ preferTaskKey: res.task_key });
        await loadOverview();
        showToast(`已生成 ${res.grid}×${res.grid} 纯内存临时场景（关服务自动释放）`, "ok");
      } catch (err) {
        notice(`上传切片失败：${err.message}`, true);
      } finally {
        btn.disabled = false;
        btn.textContent = old;
      }
    });

    $("v5-btn-test-api").addEventListener("click", async () => {
      const box = $("v5-api-test-status");
      const btn = $("v5-btn-test-api");
      btn.disabled = true;
      btn.textContent = "测试中...";
      box.className = "v5-api-status-box";
      box.textContent = "正在发起轻量连通性自检...";
      try {
        const res = await ixApi("config/test", "POST", {
          base_url: $("v5-api-base-url").value.trim(),
          model: $("v5-api-model").value.trim(),
          api_key: $("v5-api-key").value.trim(),
          timeout: parseInt($("v5-api-timeout").value, 10) || 20,
          disable_thinking: $("v5-api-disable-thinking").checked
        });
        box.className = `v5-api-status-box ${res.ok ? "ok" : "err"}`;
        box.textContent = res.ok ? `成功：${res.message}` : `未通过：${res.message}`;
      } catch (err) {
        box.className = "v5-api-status-box err";
        box.textContent = `测试失败：${err.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = "测试连通性";
      }
    });

    $("v5-btn-save-api").addEventListener("click", async () => {
      const base_url = $("v5-api-base-url").value.trim();
      const model = $("v5-api-model").value.trim();
      if (!base_url || !model) {
        notice("请至少填写 Base URL 与 Model 名称", true);
        return;
      }
      try {
        const saved = await ixApi("config", "POST", {
          base_url,
          model,
          api_key: $("v5-api-key").value.trim(),
          timeout: parseInt($("v5-api-timeout").value, 10) || 30,
          disable_thinking: $("v5-api-disable-thinking").checked
        });
        apiConfig = saved;
        if (ixMeta) {
          ixMeta.api_configured = true;
          ixMeta.api_info = saved;
        }
        updateApiBadges();
        navControls();
        $("modal-api").classList.add("hidden");
        showToast("云端 API 配置已生效（纯内存保存）", "ok");
      } catch (err) {
        notice(`保存云端 API 失败：${err.message}`, true);
      }
    });
  }

  // ==================== WORKSPACE 2B & 4: SEALED NAVIGATOR & ARENA ====================
  const NAV_BASE = "/api/controllers/";
  const BATCH_BASE = "/api/controller-batches/";
  const API_BASE = "/api/api-comparisons/";

  async function navApi(path, body, base = NAV_BASE) {
    const r = await fetch(base + path, {
      headers: headers(),
      ...(body === undefined ? {} : { method: "POST", body: JSON.stringify(body) })
    });
    const v = await r.json();
    if (!r.ok) throw new Error(v.error || "操作失败");
    return v;
  }

  function parseSelection(v) {
    const [kind, id] = (v || "").split(":");
    if (!id || !["session", "legacy", "aerial", "aerial-task"].includes(kind)) {
      throw new Error("请先从下拉框选择一条已保存记录");
    }
    return { kind, id };
  }

  function setOptions(id, items, placeholder) {
    const el = $(id);
    if (!el) return;
    const old = el.value;
    el.replaceChildren(
      new Option(placeholder, ""),
      ...items.map((v) => new Option((CLASS_NAMES[v.controller_class] || "任务") + " · " + v.name + " · " + v.task_id, recKey(v)))
    );
    if (items.some((v) => recKey(v) === old)) el.value = old;
  }

  function navDataset() {
    return navData?.datasets.find((d) => d.id === $("nav-dataset").value);
  }

  function navPause() {
    navRunning = false;
    navEpoch++;
    navControls();
  }

  async function navAct(fn) {
    if (navBusy) return false;
    navBusy = true;
    navControls();
    try {
      await fn();
      return true;
    } catch (e) {
      notice(e.message, true);
      navPause();
      return false;
    } finally {
      navBusy = false;
      navControls();
    }
  }

  function navChoices() {
    const d = navDataset();
    if (!d) return;
    for (const id of ["nav-initial-controller", "nav-active-controller"]) {
      const old = $(id).value;
      $(id).replaceChildren(...d.controllers.map((c) => new Option(CTRL_NAMES[c] || c, c)));
      if (d.controllers.includes(old)) $(id).value = old;
    }
    const oldSeed = $("nav-seed").value;
    $("nav-seed").replaceChildren(...d.seed_choices.map((v) => new Option("Seed " + v, v)));
    if (oldSeed !== "" && d.seed_choices.includes(Number(oldSeed))) $("nav-seed").value = oldSeed;

    const rows = navData.tasks.filter((t) => t.dataset_id === d.id);
    const city = $("nav-city").value;
    $("nav-city").replaceChildren(new Option("全部区域", ""), ...[...new Set(rows.map((t) => t.city))].map((c) => new Option(c, c)));
    if (rows.some((t) => t.city === city)) $("nav-city").value = city;
    $("nav-role").value = "";
    navFilter();
    navControls();
  }

  function navControls() {
    const locked = navBusy || navRunning;
    const active = navState && !navState.readonly && !isTerminal(navState.status) && !navBatch && !navApiGroup;
    const canQueue = navBatch && !isTerminal(navBatch.status) && navBatch.controllable;

    $("dot-navigator").hidden = !active && !(ixState && !ixState.done);
    $("dot-arena").hidden = !(canQueue || (navApiGroup && !isTerminal(navApiGroup.status)));
    $("global-pill-session").textContent = active
      ? `审计中: ${navState.task?.task_id || navState.id.slice(0, 6)}`
      : ixState && !ixState.done
      ? `推演中: Step ${ixState.step}`
      : "当前状态: 空闲待命";

    for (const id of ["nav-dataset", "nav-seed", "nav-city", "nav-role", "nav-search", "nav-run-name", "nav-initial-controller"]) {
      $(id).disabled = locked || !!navBatch || !!navApiGroup || !!(navState && !navState.saved);
    }
    $("nav-btn-create").disabled =
      locked ||
      !!navState ||
      !!navBatch ||
      !!navApiGroup ||
      !navSelected ||
      ($("nav-initial-controller").value === "api_m0_fallback" && !$("nav-api-consent").checked);
    $("nav-btn-save-task").disabled = locked || !!navBatch || !!navApiGroup || !navState || navState.readonly || navState.saved_task;
    $("nav-btn-save-run").disabled = locked || !!navBatch || !!navApiGroup || !navState || !isTerminal(navState.status) || navState.saved;
    $("nav-btn-finish").disabled = locked || !active;
    $("nav-btn-export").disabled = locked || !navState || !isTerminal(navState.status);
    $("nav-active-controller").disabled = locked || !active;

    if (navState) {
      $("nav-active-controller").value = navState.controller;
      for (const o of $("nav-active-controller").options) {
        o.disabled = o.value !== "manual" && !!navState.locked_agent && o.value !== navState.locked_agent;
      }
    }
    $$("[data-nav-action]").forEach((b) => (b.disabled = locked || !active || navState.controller !== "manual"));
    $("nav-btn-step").disabled = locked || !active || navState.controller === "manual";
    $("nav-btn-auto").disabled = $("nav-btn-step").disabled;
    $("nav-btn-pause").disabled = !navRunning || !!navBatch || !!navApiGroup;
    $("nav-reveal").disabled = locked || !navState || navState.readonly;
    $("nav-btn-fork").disabled = locked || !navState || !navState.saved || !!navBatch || !!navApiGroup;
    $("nav-btn-queue-add").disabled =
      locked || !!navState || !!navBatch || !!navApiGroup || navSelected?.dataset_id !== "masa-s2" || navQueue.includes(navSelected?.task_key) || navQueue.length >= 10;

    $("ar-queue-clear").disabled = locked || !!navBatch || !!navApiGroup || !navQueue.length;
    $("ar-queue-create").disabled = locked || !!navState || !!navBatch || !!navApiGroup || !navQueue.length;
    for (const id of ["ar-batch-step", "ar-batch-current", "ar-batch-all", "ar-batch-cancel"]) {
      $(id).disabled = locked || !canQueue;
    }
    $("ar-batch-pause").disabled = !navRunning || !navBatch;
    for (const id of ["ar-batch-json", "ar-batch-csv"]) {
      $(id).disabled = locked || !navBatch || navBatch.evidence_complete === false;
    }

    const canAPI = navApiGroup && !isTerminal(navApiGroup.status) && navApiGroup.controllable;
    $("ar-api-create").disabled = locked || !!navState || !!navBatch || !!navApiGroup || !apiConfig;
    for (const id of ["ar-api-step", "ar-api-current", "ar-api-all"]) $(id).disabled = locked || !canAPI;
    $("ar-api-pause").disabled = !navRunning || !navApiGroup;
    $("ar-api-export").disabled = locked || !navApiGroup || !navApiGroup.evidence_complete;
    $("ar-api-csv").disabled = $("ar-api-export").disabled;
  }

  function navPicture(id, url, isPrev = false) {
    const im = $(id);
    if (!im) return;
    if (im.getAttribute("src") === url) {
      if (isPrev && im.complete && im.naturalWidth) {
        navPreviewLoaded.add(id);
        navControls();
      }
      return;
    }
    im.onload = () => {
      if (isPrev && im.getAttribute("src") === url) {
        navPreviewLoaded.add(id);
        navControls();
      }
    };
    im.src = url;
  }

  function navBoard(cell, seen, urlFn) {
    const host = $("nav-board");
    if (host.children.length !== 25) {
      host.replaceChildren();
      for (let c = 0; c < 25; c++) {
        const el = document.createElement("div");
        el.className = "tile";
        el.dataset.cell = c;
        const n = document.createElement("span");
        n.textContent = c;
        el.append(n);
        host.append(el);
      }
    }
    for (const el of host.children) {
      const c = Number(el.dataset.cell);
      const show = seen.includes(c) || (navState && !navState.readonly && $("nav-reveal").checked);
      el.className = "tile" + (seen.includes(c) ? " visited" : "") + (c === cell ? " current" : "");
      let im = el.querySelector("img");
      if (show) {
        if (!im) {
          im = document.createElement("img");
          im.alt = "图格" + c;
          el.prepend(im);
        }
        const u = urlFn(c);
        if (im.getAttribute("src") !== u) im.src = u;
      } else {
        im?.remove();
      }
    }
  }

  function navContext(t) {
    const scale = t.cell_size_m == null ? "尺度未核，仅报告格数" : t.cell_size_m.toFixed(2) + "米/格";
    $("nav-selected-info").textContent = `5×5 · 10步 · 起点格 ${t.start} · ${scale}`;
    $("nav-task-context").textContent = `${t.city} · ${t.role} · ${t.task_id}`;
  }

  function navPreview() {
    navState = null;
    navPreviewLoaded.clear();
    $("nav-reveal").checked = false;
    $("nav-kpi-status").textContent = "待创建";
    $("nav-kpi-steps").textContent = "— / 10";
    $("nav-kpi-visited").textContent = "— / 25";
    $("nav-kpi-calls").textContent = "0";
    $("nav-history").replaceChildren();
    $("nav-result-banner").textContent = "目标真值与 SG 仅终局显示；运行期策略严格盲测。";
    $("nav-replay").disabled = true;
    $("nav-save-state").textContent = "默认临时过程";
    if (!navSelected) {
      $("nav-target-image").removeAttribute("src");
      $("nav-current-image").removeAttribute("src");
      $("nav-board").replaceChildren();
      navControls();
      return;
    }
    navContext(navSelected);
    const url = (cell) => NAV_BASE + "preview-image?" + query({ dataset_id: navSelected.dataset_id, task_key: navSelected.task_key, cell });
    navPicture("nav-target-image", url("target"), true);
    navPicture("nav-current-image", url("start"), true);
    navBoard(navSelected.start, [navSelected.start], () => url("start"));
    $("nav-current-label").textContent = "起点格 " + navSelected.start;
    navControls();
  }

  function navFilter() {
    if (!navData) return;
    const d = navDataset();
    const q = $("nav-search").value.trim().toLowerCase();
    const city = $("nav-city").value;
    const role = $("nav-role").value;
    const all = navData.tasks.filter((t) => t.dataset_id === d?.id);
    const rows = all.filter(
      (t) => (!city || t.city === city) && (!role || t.role === role) && (!q || `${t.task_id} ${t.city} ${t.start}`.toLowerCase().includes(q))
    );
    $("nav-filtered-count").textContent = `${rows.length} / ${all.length}`;
    $("nav-task-list").replaceChildren();
    if (!navSelected && rows.length) navSelected = rows[0];
    for (const task of rows) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "task-row" + (navSelected?.task_key === task.task_key ? " selected" : "");
      const title = document.createElement("strong");
      title.textContent = `${task.city} · 起点格 ${task.start}`;
      const detail = document.createElement("small");
      detail.textContent = `${task.task_id} · ${task.role}`;
      b.append(title, detail);
      b.onclick = () => {
        if (navBusy || navRunning || navBatch) return;
        if (navState && !navState.saved) {
          notice("请先点击「释放/新建」或保存当前临时过程。", true);
          return;
        }
        navSelected = task;
        navPreview();
        navFilter();
      };
      $("nav-task-list").append(b);
    }
    if (navSelected && !navState) navPreview();
    navControls();
  }

  function navRender(v, stepOverride = null) {
    navState = v;
    navSelected = v.task;
    if ($("nav-dataset").value !== v.dataset_id) {
      $("nav-dataset").value = v.dataset_id;
      navChoices();
    }
    $("nav-seed").value = v.seed;
    navContext(v.task);
    $("nav-run-name").value = v.name;
    const n = stepOverride === null ? v.step : Number(stepOverride);
    const trace = v.trajectory.slice(0, n + 1);
    const cell = trace.at(-1).cell;
    const seen = [...new Set(trace.map((t) => t.cell))];

    $("nav-kpi-status").textContent = v.readonly ? "只读回放" : isTerminal(v.status) ? "已结束" : navRunning ? "自动中" : CTRL_NAMES[v.controller] || v.controller;
    $("nav-kpi-steps").textContent = `${n} / ${v.budget}`;
    $("nav-kpi-visited").textContent = `${seen.length} / 25`;
    $("nav-kpi-calls").textContent = v.api_config ? `API ${v.api_requests} / 本地 ${v.model_calls}` : String(v.model_calls ?? 0);

    if (v.api_config) {
      $("nav-api-single-info").hidden = false;
      $("nav-api-single-info").textContent = `${v.api_config.model} · 本题请求 ${v.api_requests}/10，回退 ${v.api_fallbacks} · ${(v.api_usage.latency_ms / 1000).toFixed(2)}s`;
    } else {
      $("nav-api-single-info").hidden = true;
    }

    const url = (c) => NAV_BASE + "image?" + query({ id: v.id, kind: v.kind, cell: c, reveal: $("nav-reveal").checked ? "1" : "0" });
    navPicture("nav-target-image", url("target"));
    navPicture("nav-current-image", url(cell));
    navBoard(cell, seen, url);
    $("nav-current-label").textContent = `格#${cell} (${Math.floor(cell / 5)},${cell % 5})`;

    $("nav-history").replaceChildren();
    for (const t of trace) {
      const li = document.createElement("li");
      li.textContent = t.step
        ? `Step ${t.step} · ${{ up: "↑", right: "→", down: "↓", left: "←" }[t.action]} · 格${t.cell} · ${CLASS_NAMES[t.origin] || t.origin}${t.out_of_bounds ? " · 越界耗步" : ""}${t.revisited ? " · 重访" : ""} · ${t.reason || ""}`
        : `起点 · 格${t.cell}`;
      $("nav-history").append(li);
    }

    $("nav-save-state").textContent = v.saved ? `已保存只读 · ${CLASS_NAMES[v.controller_class]}` : "临时过程";
    if (v.result) {
      const r = v.result;
      $("nav-result-banner").textContent = `${CLASS_NAMES[v.controller_class]}：${r.success ? "已到达目标" : "未到达"} · SG ${r.SG_cells}格${r.SG_m == null ? "" : " / " + r.SG_m.toFixed(2) + "米"} · 移动 ${r.moved_m == null ? r.moved_cells + "格" : r.moved_m.toFixed(2) + "米"}`;
    } else {
      $("nav-result-banner").textContent = "目标真值与 SG 仅终局显示；运行期策略严格盲测。";
    }
    $("nav-replay").disabled = !v.readonly;
    $("nav-replay").max = v.step;
    $("nav-replay").value = n;
    $("nav-replay-label").textContent = v.readonly ? `回放 ${n} / ${v.step}` : "只读回放滑块";
    navControls();
  }

  async function navRefresh() {
    navData = await navApi("catalog");
    const old = $("nav-dataset").value;
    $("nav-dataset").replaceChildren(...navData.datasets.map((d) => new Option(d.name, d.id)));
    $("nav-dataset").value = navData.datasets.some((d) => d.id === old) ? old : (navData.datasets.find((d) => d.id === "masa-s2") || navData.datasets[0])?.id || "";
    if (navData.datasets.length) navChoices();
    setOptions("nav-saved-tasks", navData.saved_tasks, "选择已保存任务配置");
    for (const id of ["nav-saved-runs", "nav-compare-a", "nav-compare-b"]) setOptions(id, navData.saved_runs, "选择已保存终局");

    const c = await navApi("catalog", undefined, BATCH_BASE);
    const prevB = $("ar-saved-batches").value;
    $("ar-saved-batches").replaceChildren(new Option("选择已保存队列", ""), ...c.batches.map((b) => new Option(`${b.name} · ${b.completed}/${b.planned} · ${b.status}`, b.id)));
    $("ar-saved-batches").value = prevB;

    const ac = await navApi("catalog", undefined, API_BASE);
    const prevA = $("ar-api-saved").value;
    $("ar-api-saved").replaceChildren(new Option("选择已保存十题API对照", ""), ...ac.comparisons.map((v) => new Option(`${v.name} · ${v.status}`, v.id)));
    $("ar-api-saved").value = prevA;
  }

  function renderArenaBatch(v) {
    navBatch = v;
    const d = v.definition;
    const m = v.metrics;
    $("ar-batch-status").textContent = `${d.name} · ${v.status} · 动作 ${v.action_attempts} / M0 ${v.model_calls}`;
    $("ar-restart-label").hidden = !v.needs_restart;
    $("ar-batch-summary").textContent =
      v.evidence_complete === false
        ? `证据不完整：${v.evidence_issues.length}条待恢复。`
        : Object.entries(m.arms)
            .map(([c, r]) => `${CTRL_NAMES[c]}：${r.successes}/${r.planned} · SR ${(r.SR * 100).toFixed(1)}% · SG ${r.SG_cells == null ? "待定" : r.SG_cells.toFixed(3) + "格"}`)
            .join("\n") + `\n配对 ${m.paired.completed}/${m.paired.planned}：恢复 ${m.paired.recovered} / 损伤 ${m.paired.damaged} / 净增 ${m.paired.net}。`;

    const body = $("ar-batch-table").tBodies[0];
    body.replaceChildren();
    for (const row of v.rows) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "v5-btn";
      b.textContent = `${row.task_key} / ${CTRL_NAMES[row.controller]}`;
      b.disabled = !row.saved_ref || row.evidence_valid === false || navBusy || navRunning;
      b.onclick = () =>
        navAct(async () => {
          const st = await navApi("state?" + query({ id: v.id, row: row.index }), undefined, BATCH_BASE);
          renderArenaBatch(st);
          if (st.current) {
            switchWorkspace("navigator");
            switchNavTab("sealed");
          }
        });
      td.append(b);
      tr.append(td);
      for (const val of [row.status, row.step, row.result ? (row.result.success ? "已到达" : "未到达") : "—", row.result?.SG_cells ?? "—"]) {
        const cell = document.createElement("td");
        cell.textContent = val;
        tr.append(cell);
      }
      body.append(tr);
    }
    if (v.current) navRender(v.current);
    navControls();
  }

  function renderArenaAPI(v) {
    navApiGroup = v;
    const usage = v.api_usage;
    $("ar-api-restart-label").hidden = !v.needs_restart;
    $("ar-api-summary").textContent =
      Object.entries(v.arms)
        .map(([c, r]) => `${CTRL_NAMES[c]}${r.reused ? "（复用）" : ""}：${r.successes ?? "待核"}/${r.planned} · SR ${r.SR == null ? "待核" : (r.SR * 100).toFixed(1) + "%"} · SG ${r.SG_cells == null ? "待完成" : r.SG_cells.toFixed(3) + "格"}`)
        .join("\n") + `\nAPI请求 ${usage.requests}/${usage.limit} · 回退 ${usage.fallbacks} · ${(usage.latency_ms / 1000).toFixed(2)}s`;

    const body = $("ar-api-table").tBodies[0];
    body.replaceChildren();
    for (const r of v.rows) {
      const tr = document.createElement("tr");
      const base = v.definition.baseline[r.task_key];
      for (const val of [
        r.task_key,
        base.M0.result.success ? "已到达" : "未到达",
        base.frontier.result.success ? "已到达" : "未到达",
        r.status,
        r.result ? `${r.result.success ? "已到达" : "未到达"} / SG=${r.result.SG_cells}` : "—"
      ]) {
        const td = document.createElement("td");
        td.textContent = val;
        tr.append(td);
      }
      body.append(tr);
    }
    if (v.current) navRender(v.current);
    navControls();
  }

  function bindNavigatorAndArena() {
    $("nav-city").onchange = navFilter;
    $("nav-role").onchange = navFilter;
    $("nav-search").oninput = navFilter;
    $("nav-dataset").onchange = () => {
      navSelected = null;
      navChoices();
      navPreview();
    };
    $("nav-api-consent").onchange = navControls;
    $("nav-initial-controller").onchange = navControls;

    $("nav-btn-create").onclick = () =>
      navAct(async () =>
        navRender(
          await navApi("create", {
            name: $("nav-run-name").value,
            dataset_id: navSelected.dataset_id,
            task_key: navSelected.task_key,
            controller: $("nav-initial-controller").value,
            seed: Number($("nav-seed").value)
          })
        )
      );

    $("nav-active-controller").onchange = () =>
      navAct(async () => navRender(await navApi("mode", { id: navState.id, kind: navState.kind, controller: $("nav-active-controller").value })));
    $$("[data-nav-action]").forEach((b) => {
      b.onclick = () => navAct(async () => navRender(await navApi("step", { id: navState.id, kind: navState.kind, action: b.dataset.navAction })));
    });
    $("nav-btn-step").onclick = () => navAct(async () => navRender(await navApi("step", { id: navState.id, kind: navState.kind, action: null })));
    $("nav-btn-auto").onclick = async () => {
      if (navBusy || navRunning || navBatch || navApiGroup || !navState || navState.controller === "manual") return;
      navRunning = true;
      const round = ++navEpoch;
      const id = navState.id;
      navControls();
      try {
        while (navRunning && navEpoch === round && navState?.id === id && !isTerminal(navState.status)) {
          if (!(await navAct(async () => navRender(await navApi("step", { id, kind: navState.kind, action: null }))))) break;
          if (!navRunning || isTerminal(navState.status)) break;
          await new Promise((r) => setTimeout(r, Number($("nav-speed").value)));
        }
      } finally {
        if (round === navEpoch) {
          navRunning = false;
          navControls();
        }
      }
    };
    $("nav-btn-pause").onclick = navPause;

    for (const [btnId, op] of [["nav-btn-finish", "finish"], ["nav-btn-save-task", "save-task"], ["nav-btn-save-run", "save-run"]]) {
      $(btnId).onclick = () =>
        navAct(async () => {
          navRender(await navApi(op, { id: navState.id, kind: navState.kind }));
          if (op !== "finish") await navRefresh();
        });
    }
    $("nav-btn-new").onclick = () =>
      navAct(async () => {
        await navApi("release", {});
        navBatch = navApiGroup = navState = null;
        navPreview();
        await navRefresh();
      });

    const doFork = async (val) => {
      if (navBatch || navApiGroup || (navState && !navState.saved)) throw new Error("请先保存终局或点击「释放/新建」");
      navRender(
        await navApi("fork", {
          ...parseSelection(val),
          controller: $("nav-initial-controller").value,
          name: $("nav-run-name").value + " · 同题",
          seed: Number($("nav-seed").value)
        })
      );
    };
    $("nav-reopen-task").onclick = () => navAct(() => doFork($("nav-saved-tasks").value));
    $("nav-btn-fork").onclick = () => navAct(() => doFork(recKey(navState)));
    $("nav-open-run").onclick = () =>
      navAct(async () => {
        if (navBatch || navApiGroup || (navState && !navState.saved)) throw new Error("请先释放当前临时过程");
        $("nav-reveal").checked = false;
        navRender(await navApi("state?" + query(parseSelection($("nav-saved-runs").value))));
      });
    $("nav-btn-compare").onclick = () =>
      navAct(async () => {
        const a = parseSelection($("nav-compare-a").value);
        const b = parseSelection($("nav-compare-b").value);
        const v = await navApi("compare?" + query({ left: a.id, right: b.id, left_kind: a.kind, right_kind: b.kind }));
        $("nav-comparison-output").textContent = `${v.task_id}\nA (${CLASS_NAMES[v.A.controller]}) ${v.A.result.success ? "到达" : "未达"} vs B (${CLASS_NAMES[v.B.controller]}) ${v.B.result.success ? "到达" : "未达"}\nB−A SG差: ${v.SG_cells_delta}格 · ${v.recovered ? "恢复原失败" : v.damaged ? "损伤原成功" : "状态一致"}`;
      });
    $("nav-reveal").onchange = () => navState && navRender(navState);
    $("nav-replay").oninput = () => navState?.readonly && navRender(navState, $("nav-replay").value);
    $("nav-btn-export").onclick = () => downloadable(NAV_BASE + "export?" + query({ id: navState.id, kind: navState.kind }));

    const updateQueueText = () => {
      $("ar-queue-selection").textContent = navQueue.length
        ? navQueue.map((k, i) => `${i + 1}. ${navData.tasks.find((t) => t.task_key === k)?.task_id || k}`).join("\n")
        : "尚未选择题目";
      navControls();
    };
    $("nav-btn-queue-add").onclick = () => {
      if (navSelected && !navQueue.includes(navSelected.task_key)) {
        navQueue.push(navSelected.task_key);
        updateQueueText();
        showToast(`已加入批测队列 (${navQueue.length} 题)`, "ok");
      }
    };
    $("ar-queue-clear").onclick = () => {
      navQueue = [];
      updateQueueText();
    };
    $("ar-queue-create").onclick = () =>
      navAct(async () => {
        renderArenaBatch(
          await navApi("create", { name: $("nav-run-name").value + " · 同题小队列", dataset_id: "masa-s2", task_keys: navQueue, controllers: ["M0", "frontier"], seed: Number($("nav-seed").value || 0) }, BATCH_BASE)
        );
        await navRefresh();
      });
    $("ar-queue-open").onclick = () =>
      navAct(async () => {
        const id = $("ar-saved-batches").value;
        if (!id) throw new Error("请选择已保存队列");
        await navApi("release", {});
        navState = navApiGroup = null;
        renderArenaBatch(await navApi("state?" + query({ id }), undefined, BATCH_BASE));
      });

    const batchStep = async () => {
      const v = await navApi("step", { id: navBatch.id, restart_incomplete: $("ar-restart-incomplete").checked }, BATCH_BASE);
      $("ar-restart-incomplete").checked = false;
      renderArenaBatch(v);
    };
    $("ar-batch-step").onclick = () => navAct(batchStep);
    const batchAuto = async (all) => {
      if (navBusy || navRunning || !navBatch || isTerminal(navBatch.status)) return;
      navRunning = true;
      const round = ++navEpoch;
      const id = navBatch.id;
      const index = navBatch.rows.find((r) => r.status !== "completed")?.index;
      navControls();
      try {
        while (navRunning && round === navEpoch && navBatch.id === id && !isTerminal(navBatch.status)) {
          if (!(await navAct(batchStep))) break;
          if (!all && navBatch.rows[index].status === "completed") break;
          if (!navRunning || isTerminal(navBatch.status)) break;
          await new Promise((r) => setTimeout(r, 380));
        }
      } finally {
        if (round === navEpoch) {
          navRunning = false;
          if (navBatch?.id === id && !isTerminal(navBatch.status)) await navAct(async () => renderArenaBatch(await navApi("pause", { id }, BATCH_BASE)));
          navControls();
        }
      }
    };
    $("ar-batch-current").onclick = () => batchAuto(false);
    $("ar-batch-all").onclick = () => batchAuto(true);
    $("ar-batch-pause").onclick = navPause;
    $("ar-batch-cancel").onclick = () => navAct(async () => renderArenaBatch(await navApi("cancel", { id: navBatch.id }, BATCH_BASE)));
    $("ar-batch-json").onclick = () => downloadable(BATCH_BASE + "export?" + query({ id: navBatch.id, format: "json" }));
    $("ar-batch-csv").onclick = () => downloadable(BATCH_BASE + "export?" + query({ id: navBatch.id, format: "csv" }));

    $("ar-api-create").onclick = () =>
      navAct(async () => {
        await navApi("release", {});
        navState = navBatch = null;
        renderArenaAPI(await navApi("create", { name: $("nav-run-name").value + " · 十题API对照" }, API_BASE));
        await navRefresh();
      });
    $("ar-api-open").onclick = () =>
      navAct(async () => {
        const id = $("ar-api-saved").value;
        if (!id) throw new Error("请选择已保存十题对照记录");
        await navApi("release", {});
        navState = navBatch = null;
        renderArenaAPI(await navApi("state?" + query({ id }), undefined, API_BASE));
      });
    const apiStep = async () => {
      const v = await navApi("step", { id: navApiGroup.id, restart_incomplete: $("ar-api-restart").checked }, API_BASE);
      $("ar-api-restart").checked = false;
      renderArenaAPI(v);
    };
    $("ar-api-step").onclick = () => navAct(apiStep);
    const apiAuto = async (all) => {
      if (navBusy || navRunning || !navApiGroup || isTerminal(navApiGroup.status)) return;
      navRunning = true;
      const id = navApiGroup.id;
      const round = ++navEpoch;
      const index = navApiGroup.rows.findIndex((r) => r.status !== "completed");
      navControls();
      try {
        while (navRunning && navEpoch === round && navApiGroup.id === id && !isTerminal(navApiGroup.status)) {
          if (!(await navAct(apiStep))) break;
          if (!all && navApiGroup.rows[index].status === "completed") break;
          if (!navRunning || isTerminal(navApiGroup.status)) break;
          await new Promise((r) => setTimeout(r, 450));
        }
      } finally {
        if (round === navEpoch) {
          navRunning = false;
          navControls();
        }
      }
    };
    $("ar-api-current").onclick = () => apiAuto(false);
    $("ar-api-all").onclick = () => apiAuto(true);
    $("ar-api-pause").onclick = navPause;
    $("ar-api-export").onclick = () => downloadable(API_BASE + "export?" + query({ id: navApiGroup.id }));
    $("ar-api-csv").onclick = () => downloadable(API_BASE + "export?" + query({ id: navApiGroup.id, format: "csv" }));
  }

  // ==================== WORKSPACE 3: SANDBOX (/api/scenes/*) ====================
  async function sbApi(verb, data) {
    const url = data === undefined ? `/api/scenes/${verb}?${query()}` : `/api/scenes/${verb}`;
    const r = await fetch(url, { method: data === undefined ? "GET" : "POST", headers: headers(), ...(data === undefined ? {} : { body: JSON.stringify(data) }) });
    const v = await r.json();
    if (!r.ok) throw new Error(v.error || "场景操作未完成");
    return v;
  }

  function sbData() {
    return sbCatalog?.datasets.find((d) => d.id === $("sb-dataset").value);
  }

  function sbSetBase() {
    const d = sbData();
    const t = d?.tasks.find((x) => x.key === $("sb-base").value);
    if (!t) return;
    $("sb-start").value = t.start;
    $("sb-goal").value = t.goal;
    $("sb-budget").value = d.budget;
    $("sb-start").max = $("sb-goal").max = d.grid * d.grid - 1;
    $("sb-budget").max = d.budget;
    $("sb-scale-badge").textContent = `${d.grid}×${d.grid} · B=${d.budget}`;
  }

  function sbSetDataset() {
    const d = sbData();
    if (!d) return;
    const areas = new Set();
    const items = [];
    for (const t of d.tasks) {
      if (!areas.has(t.area)) {
        areas.add(t.area);
        items.push(new Option(`${t.area} · ${t.key}`, t.key));
      }
    }
    $("sb-base").replaceChildren(...items);
    $("sb-policy").replaceChildren(...d.policies.map((p) => new Option(p === "M0" ? "M0 · 冻结本地策略" : p, p)));
    sbSetBase();
  }

  function sbUpdateComparisons() {
    const items = new Map((sbCatalog?.saved_runs || []).map((r) => [r.id, r]));
    for (const [id, r] of sbRuns) items.set(id, r);
    const list = [...items.values()].filter((r) => !sbScene || r.scene_id === sbScene.id);
    for (const id of ["sb-compare-left", "sb-compare-right"]) {
      const old = $(id).value;
      $(id).replaceChildren(...list.map((r) => new Option(`${r.controller_class || "未开始"} · ${r.id.slice(0, 8)}`, r.id)));
      if (list.some((x) => x.id === old)) $(id).value = old;
    }
  }

  async function sbRefresh() {
    sbCatalog = await sbApi("catalog");
    if (!$("sb-dataset").options.length && sbCatalog.datasets.length) {
      $("sb-dataset").replaceChildren(...sbCatalog.datasets.map((d) => new Option(d.name, d.id)));
      $("sb-dataset").value = sbCatalog.datasets.some((d) => d.id === "masa-s3") ? "masa-s3" : sbCatalog.datasets[0].id;
      sbSetDataset();
    }
    $("sb-saved-scenes").replaceChildren(new Option("选择已创建场景", ""), ...sbCatalog.scenes.map((s) => new Option(`${s.saved ? "已保存" : "临时"} · ${s.definition.name}`, s.id)));
    $("sb-saved-runs").replaceChildren(new Option("选择只读运行记录", ""), ...sbCatalog.saved_runs.map((r) => new Option(`${r.controller_class} · ${r.id.slice(0, 8)} · ${r.status}`, r.id)));
    sbUpdateComparisons();
  }

  function sbRender() {
    if (!sbRun) return;
    const d = sbRun.definition;
    const k = d.grid;
    const seen = new Set(sbRun.trajectory.map((p) => p.cell));
    sbRuns.set(sbRun.id, sbRun);
    sbUpdateComparisons();
    const term = isTerminal(sbRun.status);
    const ro = !!sbRun.readonly;

    $("dot-sandbox").hidden = term || ro;
    $("sb-stat-steps").textContent = `${sbRun.events.length} / ${d.budget}`;
    $("sb-stat-seen").textContent = `${seen.size} / ${k * k}`;
    const labels = { human: "人工", mixed: "人机接管", pure_agent: "纯Agent", unstarted: "未开始" };
    $("sb-stat-mode").textContent = labels[sbRun.controller_class] || sbRun.controller_class;
    $("sb-stat-sg").textContent = sbRun.result ? `${sbRun.result.sg_cells}格` : "未终止";
    $("sb-map-title").textContent = d.name;
    $("sb-run-status").textContent = `${ro ? "只读回看 · " : term ? "已结束 · " : "运行中 · "}${k}×${k} 自由仿真`;

    $("sb-human-mode").classList.toggle("selected", sbRun.mode === "human");
    $("sb-agent-mode").classList.toggle("selected", sbRun.mode === "agent");
    $$("[data-sb-action]").forEach((b) => (b.disabled = sbBusy || term || ro || sbRun.mode !== "human"));
    for (const id of ["sb-human-mode", "sb-agent-mode", "sb-cancel-scene"]) $(id).disabled = sbBusy || term || ro;
    $("sb-agent-step").disabled = $("sb-agent-auto").disabled = sbBusy || term || ro || sbRun.mode !== "agent";
    $("sb-save-run").disabled = sbBusy || ro || !term;
    $("sb-agent-pause").disabled = ro || term;
    $("sb-export-scene").disabled = false;

    const imgUrl = (cell) => `/api/scenes/image?${query({ id: sbRun.id, cell, ...($("sb-reveal-all").checked ? { reveal: "1" } : {}) })}`;
    $("sb-map").style.setProperty("--grid", k);
    $("sb-map").replaceChildren();
    for (let i = 0; i < k * k; i++) {
      const cell = document.createElement("div");
      cell.className = "cell" + (seen.has(i) ? " seen" : "") + (i === sbRun.position ? " current" : "");
      if (seen.has(i) || $("sb-reveal-all").checked) {
        const im = document.createElement("img");
        im.src = imgUrl(i);
        im.alt = "图块 " + i;
        cell.append(im);
      }
      const num = document.createElement("span");
      num.className = "cell-index";
      num.textContent = i;
      cell.append(num);
      cell.addEventListener("click", () => {
        if (sbBusy || term || ro || sbRun.mode !== "human") return;
        const curr = sbRun.position;
        const dr = Math.floor(i / k) - Math.floor(curr / k);
        const dc = (i % k) - (curr % k);
        if (Math.abs(dr) + Math.abs(dc) === 1) {
          const act = dr === -1 ? "up" : dr === 1 ? "down" : dc === -1 ? "left" : "right";
          sbApi("step", { id: sbRun.id, action: act }).then((r) => {
            sbRun = r;
            sbRender();
          });
        }
      });
      $("sb-map").append(cell);
    }
    $("sb-goal-image").src = imgUrl("target");
    $("sb-current-image").src = imgUrl(sbRun.position);

    const svg = $("sb-path");
    svg.setAttribute("viewBox", `0 0 ${k * 100} ${k * 100}`);
    svg.replaceChildren();
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", sbRun.trajectory.map((p) => `${((p.cell % k) + 0.5) * 100},${(Math.floor(p.cell / k) + 0.5) * 100}`).join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "#0ea5e9");
    line.setAttribute("stroke-width", "6");
    line.setAttribute("stroke-linejoin", "round");
    svg.append(line);

    $("sb-events").replaceChildren();
    for (const e of sbRun.events.slice().reverse()) {
      const div = document.createElement("div");
      div.textContent = `Step ${e.step} · ${e.actor === "human" ? "人工" : "Agent"} · ${{ up: "↑", right: "→", down: "↓", left: "←" }[e.action]} · ${e.from} → ${e.to}`;
      $("sb-events").append(div);
    }
  }

  async function sbCreateAndStart() {
    sbAuto = false;
    sbScene = await sbApi("create", {
      name: $("sb-name").value,
      dataset_id: $("sb-dataset").value,
      base_task_key: $("sb-base").value,
      start: Number($("sb-start").value),
      goal: Number($("sb-goal").value),
      budget: Number($("sb-budget").value),
      seed: Number($("sb-seed").value)
    });
    $("sb-btn-save-scene").disabled = $("sb-btn-restart").disabled = false;
    await sbRefresh();
    sbRun = await sbApi("start", { scene_id: sbScene.id });
    sbRender();
  }

  function bindSandbox() {
    $("sb-dataset").onchange = sbSetDataset;
    $("sb-base").onchange = sbSetBase;
    $("sb-form").onsubmit = (e) => {
      e.preventDefault();
      sbCreateAndStart().catch((err) => notice(err.message, true));
    };
    $("sb-btn-save-scene").onclick = async () => {
      if (!sbScene) return;
      sbScene = await sbApi("save-scene", { scene_id: sbScene.id });
      await sbRefresh();
      showToast("自定义场景已保存。", "ok");
    };
    $("sb-btn-restart").onclick = async () => {
      if (!sbScene) return;
      sbRun = await sbApi("start", { scene_id: sbScene.id });
      sbRender();
    };
    $("sb-btn-refresh").onclick = () => sbRefresh();
    $("sb-saved-scenes").onchange = async () => {
      const id = $("sb-saved-scenes").value;
      if (!id) return;
      sbScene = sbCatalog.scenes.find((s) => s.id === id);
      sbRun = await sbApi("start", { scene_id: id });
      sbRender();
    };
    $("sb-saved-runs").onchange = async () => {
      const id = $("sb-saved-runs").value;
      if (!id) return;
      sbRun = await (await fetch(`/api/scenes/state?${query({ id })}`, { headers: headers() })).json();
      sbRender();
    };
    $$("[data-sb-action]").forEach((b) => {
      b.onclick = async () => {
        sbRun = await sbApi("step", { id: sbRun.id, action: b.dataset.sbAction });
        sbRender();
      };
    });
    $("sb-human-mode").onclick = async () => {
      sbRun = await sbApi("mode", { id: sbRun.id, mode: "human", policy: $("sb-policy").value });
      sbRender();
    };
    $("sb-agent-mode").onclick = async () => {
      sbRun = await sbApi("mode", { id: sbRun.id, mode: "agent", policy: $("sb-policy").value });
      sbRender();
    };
    $("sb-agent-step").onclick = async () => {
      sbRun = await sbApi("step", { id: sbRun.id });
      sbRender();
    };
    $("sb-agent-auto").onclick = async () => {
      if (sbAuto || !sbRun) return;
      sbAuto = true;
      const id = sbRun.id;
      while (sbAuto && sbRun?.id === id && ["ready", "running"].includes(sbRun.status)) {
        sbRun = await sbApi("step", { id });
        sbRender();
        if (sbAuto) await new Promise((r) => setTimeout(r, Number($("sb-speed").value)));
      }
      sbAuto = false;
    };
    $("sb-agent-pause").onclick = () => (sbAuto = false);
    $("sb-cancel-scene").onclick = async () => {
      sbAuto = false;
      sbRun = await sbApi("cancel", { id: sbRun.id });
      sbRender();
    };
    $("sb-save-run").onclick = async () => {
      sbRun = await sbApi("save-run", { id: sbRun.id });
      await sbRefresh();
      sbRender();
    };
    $("sb-reveal-all").onchange = sbRender;
    $("sb-btn-compare").onclick = async () => {
      const l = $("sb-compare-left").value;
      const r = $("sb-compare-right").value;
      if (!l || !r) return;
      const res = await fetch(`/api/scenes/compare?${query({ left: l, right: r })}`, { headers: headers() });
      const c = await res.json();
      if (!res.ok) return notice(c.error, true);
      $("sb-compare-result").textContent = `左 (${c.left.controller_class}): SG=${c.left.result?.sg_cells ?? "—"}格 vs 右 (${c.right.controller_class}): SG=${c.right.result?.sg_cells ?? "—"}格`;
    };
    $("sb-export-scene").onclick = () => sbRun && downloadable(URL.createObjectURL(new Blob([JSON.stringify(sbRun, null, 2)], { type: "application/json" })));
  }

  // ==================== WORKSPACE 5: GROUND WALK & LIVE ====================
  const GW_BASE = "/api/ground-walk/";
  const GL_BASE = "/api/ground-live/";

  async function gwApi(path, body) {
    const r = await fetch(GW_BASE + path, { headers: headers(), ...(body === undefined ? {} : { method: "POST", body: JSON.stringify(body) }) });
    const v = await r.json();
    if (!r.ok) throw new Error(v.error || "地面漫游操作失败");
    return v;
  }

  function gwRender(s) {
    gwId = s.id;
    gwState = s;
    const done = isTerminal(s.status);
    const can = s.status === "ready" && !s.readonly;
    $("gw-status").textContent = s.status;
    $("gw-steps").textContent = `${s.step} / ${s.budget}`;
    $("gw-calls").textContent = `${s.forward_calls} / ${s.forward_limit}`;
    $("gw-seen").textContent = `${s.visited.length} / 25`;
    $$("[data-gw-action]").forEach((b) => (b.disabled = !can));
    $("gw-finish").disabled = !can;
    $("gw-cancel").disabled = done || s.readonly;
    $("gw-save").disabled = !done || s.saved;
    $("gw-export").disabled = !done;
    const e = s.current_evidence;
    $("gw-score").textContent = e ? e.event.raw_similarity.toFixed(4) : "—";
    $("gw-cache").textContent = e ? (e.event.current_cache_hit ? "缓存复用" : "512维新编码") : "等待观测";
    $("gw-cell-label").textContent = `格#${s.position} (${Math.floor(s.position / 5)},${s.position % 5})`;
    $("gw-goal-img").src = GW_BASE + "image?" + query({ id: gwId, cell: "goal" });
    $("gw-current-img").src = GW_BASE + "image?" + query({ id: gwId, cell: s.position });

    const map = $("gw-map");
    map.replaceChildren();
    for (let c = 0; c < 25; c++) {
      const tile = document.createElement("div");
      tile.className = "tile" + (s.visited.includes(c) ? " visited" : "") + (s.position === c ? " current" : "");
      if (s.visited.includes(c) || $("gw-reveal").checked) {
        const im = document.createElement("img");
        im.src = GW_BASE + "image?" + query({ id: gwId, cell: c, reveal: $("gw-reveal").checked ? "1" : "0" });
        tile.append(im);
      }
      const n = document.createElement("span");
      n.className = "num";
      n.textContent = c;
      tile.append(n);
      map.append(tile);
    }
    $("gw-history").replaceChildren();
    for (const ev of s.evidence || []) {
      const li = document.createElement("li");
      li.textContent = `Step ${ev.step} · 格${ev.cell} · 相似度 ${ev.event.raw_similarity.toFixed(4)}`;
      $("gw-history").append(li);
    }
  }

  async function gwCatalog() {
    const c = await gwApi("catalog");
    $("gw-scene-info").textContent = `${c.scene.name} · 地面-航拍配对连续观察`;
    $("gw-saved").replaceChildren(new Option("选择已保存记录", ""), ...c.saved.map((s) => new Option(`${s.name} · ${s.status}`, s.id)));
  }

  async function glApi(path, body) {
    const r = await fetch(GL_BASE + path, { headers: headers(), ...(body === undefined ? {} : { method: "POST", body: JSON.stringify(body) }) });
    const v = await r.json();
    if (!r.ok) throw new Error(v.error || "实时编码请求失败");
    return v;
  }

  function glRender(s) {
    glState = s;
    glId = s.id;
    const done = isTerminal(s.status);
    $("gl-job-status").textContent = s.status;
    $("gl-save-status").textContent = s.saved ? "已保存 · 只读" : "临时";
    $("gl-call-count").textContent = `${(s.events || []).filter((e) => e.type === "vector_ready").length} / 2`;
    $("gl-save").disabled = !done || s.saved;
    $("gl-export").disabled = !done;
    $("gl-cancel").disabled = done;
    $("gl-target-image").src = GL_BASE + "image?" + query({ id: glId, slot: "target" });
    $("gl-current-image").src = GL_BASE + "image?" + query({ id: glId, slot: "current" });

    $("gl-timeline").replaceChildren();
    for (const e of s.events || []) {
      const li = document.createElement("li");
      li.textContent = `${e.type} (${e.kind || ""})`;
      $("gl-timeline").append(li);
    }
    const r = s.result;
    if (r) {
      $("gl-similarity").textContent = r.event.raw_similarity.toFixed(4);
      $("gl-feature-size").textContent = `${r.packet.target_feature.length} / ${r.packet.current_feature.length} 维`;
      $("gl-frozen-state").textContent = r.frozen_encoder_report.frozen_tensor_checks.every((x) => x.unchanged) ? "通过 · 0参数更新" : "需检查";
      $("gl-resources").textContent = r.resources ? `${r.resources.elapsed_seconds.toFixed(1)} s` : "—";
    }
  }

  async function glCatalog() {
    const c = await glApi("catalog");
    glReady = c.real_encoding_available;
    $("gl-saved").replaceChildren(new Option("选择已保存记录", ""), ...c.saved.map((x) => new Option(`${x.name} · ${x.status}`, x.id)));
  }

  function bindGround() {
    $("gw-start").onclick = async () => gwApi("create", { name: $("gw-name").value, budget: Number($("gw-budget").value) }).then(gwRender).catch((e) => notice(e.message, true));
    $$("[data-gw-action]").forEach((b) => {
      b.onclick = () => gwApi("move", { id: gwId, action: b.dataset.gwAction }).then(gwRender).catch((e) => notice(e.message, true));
    });
    $("gw-finish").onclick = () => gwApi("finish", { id: gwId }).then(gwRender);
    $("gw-cancel").onclick = () => gwApi("cancel", { id: gwId }).then(gwRender);
    $("gw-save").onclick = async () => {
      gwRender(await gwApi("save", { id: gwId }));
      await gwCatalog();
    };
    $("gw-open").onclick = async () => $("gw-saved").value && gwRender(await gwApi("state?" + query({ id: $("gw-saved").value })));
    $("gw-new").onclick = async () => {
      await gwApi("release", {});
      gwId = gwState = null;
      $("gw-map").replaceChildren();
    };
    $("gw-refresh").onclick = gwCatalog;
    $("gw-reveal").onchange = () => gwState && gwRender(gwState);
    $("gw-export").onclick = () => gwId && downloadable(GW_BASE + "export?" + query({ id: gwId }));

    const readAsDataURL = (blob) =>
      new Promise((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(String(r.result));
        r.onerror = () => rej(new Error("读取图像失败"));
        r.readAsDataURL(blob);
      });

    for (const [elId, slot] of [["gl-target-file", "target"], ["gl-current-file", "current"]]) {
      $(elId)?.addEventListener("change", async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const url = await readAsDataURL(file);
        glImages[slot] = url.split(",")[1];
        $(`gl-${slot}-image`).src = url;
        glMode = "upload";
        $("gl-encode").disabled = !glReady || !glImages.target || !glImages.current;
      });
    }

    $("gl-load-sample").onclick = async () => {
      try {
        for (const slot of ["target", "current"]) {
          const r = await fetch(GL_BASE + "sample?" + query({ slot }), { headers: headers() });
          if (!r.ok) throw new Error("示例图未携带，请直接打开右侧已存记录。");
          const url = await readAsDataURL(await r.blob());
          glImages[slot] = url.split(",")[1];
          $(`gl-${slot}-image`).src = url;
        }
        glMode = "sample";
        $("gl-encode").disabled = !glReady;
        showToast("已加载内置开发示例对。", "ok");
      } catch (e) {
        notice(e.message, true);
      }
    };
    $("gl-encode").onclick = async () => {
      try {
        const spec = { name: $("gl-job-name").value, mode: glMode };
        if (glMode === "upload") Object.assign(spec, { target_b64: glImages.target, current_b64: glImages.current });
        glRender(await glApi("create", spec));
      } catch (e) {
        notice(e.message, true);
      }
    };
    $("gl-new-job").onclick = () => {
      glId = glState = null;
      glImages = {};
      $("gl-target-image").removeAttribute("src");
      $("gl-current-image").removeAttribute("src");
      $("gl-encode").disabled = true;
    };
    $("gl-open-saved").onclick = async () => $("gl-saved").value && glRender(await glApi("state?" + query({ id: $("gl-saved").value })));
    $("gl-save").onclick = async () => {
      if (!glId) return;
      glRender(await glApi("save", { id: glId }));
      await glCatalog();
    };
    $("gl-export").onclick = () => glId && downloadable(GL_BASE + "export?" + query({ id: glId }));
    $("gl-refresh").onclick = glCatalog;
  }

  // ==================== WORKSPACE 6: EVIDENCE (CV031 & CV041) ====================
  function drawHeat2D(canvasId, rangeId, heat, range) {
    const canvas = $(canvasId);
    if (!canvas || !heat || !heat.length) return;
    const ctx = canvas.getContext("2d");
    const [lo, hi] = range;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    for (let r = 0; r < heat.length; r++) {
      for (let c = 0; c < heat[0].length; c++) {
        const v = hi === lo ? 0 : (heat[r][c] - lo) / (hi - lo);
        ctx.fillStyle = `rgb(${Math.round(20 + 220 * v)},${Math.round(66 + 117 * v)},${Math.round(88 - 30 * v)})`;
        ctx.fillRect((c * w) / heat[0].length, (r * h) / heat.length, Math.ceil(w / heat[0].length), Math.ceil(h / heat.length));
      }
    }
    $(rangeId).textContent = `数值范围 ${lo.toFixed(3)} ~ ${hi.toFixed(3)}`;
  }

  function drawOverlayMarks(svgId, d) {
    const svg = $(svgId);
    const w = d.bounds[2] - d.bounds[0];
    const h = d.bounds[3] - d.bounds[1];
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.replaceChildren();
    const mark = (pt, color, label) => {
      const c = document.createElementNS(svg.namespaceURI, "circle");
      c.setAttribute("cx", pt[0] - d.bounds[0]);
      c.setAttribute("cy", pt[1] - d.bounds[1]);
      c.setAttribute("r", Math.max(3, w * 0.022));
      c.setAttribute("fill", color);
      c.setAttribute("stroke", "#fff");
      c.setAttribute("stroke-width", Math.max(1, w * 0.006));
      const t = document.createElementNS(svg.namespaceURI, "title");
      t.textContent = label;
      c.append(t);
      svg.append(c);
    };
    if (d.truth_xy) mark(d.truth_xy, "#f59e0b", "真实拍摄点");
    if (d.prediction) mark(d.prediction, "#06b6d4", "模型预测点");
  }

  async function p31Load() {
    const caseId = $("p31-case").value;
    if (!caseId) return;
    $$("#p31-arms [data-p31-arm]").forEach((b) => b.classList.toggle("selected", b.dataset.p31Arm === p31Arm));
    const r = await fetch(`/api/perception/detail?case=${encodeURIComponent(caseId)}&arm=${p31Arm}`);
    const d = await r.json();
    if (!r.ok) return;
    p31Detail = d;
    $("p31-case-title").textContent = `来源 ${d.source} / ${RELATION_NAMES[d.relation]}`;
    $("p31-cell").textContent = `格 ${d.cell} · 20m/格`;
    $("p31-decision").textContent = d.accepted ? "已接受" : "已拒绝";
    $("p31-score").textContent = d.score.toFixed(4);
    $("p31-tau").textContent = d.tau.toFixed(4);
    $("p31-error").textContent = d.error_m === null ? "不适用" : `${d.error_m.toFixed(2)} m`;
    $("p31-interpretation").textContent = d.interpretation;
    $("p31-reason").textContent = d.joint_correct ? "达标：满足该案例判断要求" : "未达标：未满足联合要求或负例被误接受";
    $("p31-binding").textContent = JSON.stringify(d.provenance, null, 2);
    const q = `case=${d.id}&arm=${d.arm}`;
    $("p31-target-image").src = `/api/perception/image?${q}&slot=target`;
    $("p31-current-image").src = `/api/perception/image?${q}&slot=current`;
    drawOverlayMarks("p31-overlay", d);
    drawHeat2D("p31-heat", "p31-heat-range", d.heat, d.heat_range);
    $("p31-download").disabled = false;
  }

  function p31Filter() {
    if (!p31Catalog) return;
    const old = $("p31-case").value;
    const src = $("p31-source").value;
    const rel = $("p31-relation").value;
    const rows = p31Catalog.cases.filter((c) => (src === "all" || String(c.source) === src) && (rel === "all" || c.relation === rel));
    $("p31-case").replaceChildren(...rows.map((c) => new Option(`来源${c.source} → 航片${c.reference} · 格${c.cell} · ${RELATION_NAMES[c.relation]}`, String(c.id))));
    if (rows.some((c) => String(c.id) === old)) $("p31-case").value = old;
    $("p31-case-count").textContent = `${rows.length} / 20 案例`;
    p31Load();
  }

  function s41ConfigureMethod() {
    const m = $("s41-method").value;
    if ($("s41-seed").value !== "none") s41LastSeed = $("s41-seed").value || s41LastSeed;
    $("s41-seed").replaceChildren(...(m === "RawCos" ? [new Option("无种子", "none")] : ["0", "1", "2"].map((s) => new Option("Seed " + s, s))));
    $("s41-seed").disabled = m === "RawCos";
    if (m !== "RawCos") $("s41-seed").value = s41LastSeed;
    const arms = m === "TrainedMean" ? ["NoTarget"] : ["Full", "Disabled", "Mean", "Wrong"];
    if (!arms.includes(s41Arm)) s41Arm = arms[0];
    $("s41-arms").replaceChildren(
      ...arms.map((a) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "v5-btn" + (a === s41Arm ? " selected" : "");
        b.textContent = ARM_LABELS[a];
        b.onclick = () => {
          s41Arm = a;
          s41Load();
        };
        return b;
      })
    );
  }

  async function s41Load() {
    const caseId = $("s41-case").value;
    if (!caseId) return;
    const q = new URLSearchParams({ case: caseId, method: $("s41-method").value, seed: $("s41-seed").value, arm: s41Arm }).toString();
    const r = await fetch(`/api/spatial-evidence/detail?${q}`);
    const d = await r.json();
    if (!r.ok) return;
    s41Detail = d;
    $("s41-case-title").textContent = `${d.city} / ${RELATION_NAMES[d.relation]}`;
    $("s41-cell-note").textContent = `格 ${d.cell} · ${d.metric_cell_m.toFixed(2)}m/格`;
    $("s41-decision").textContent = d.accepted ? "模型接受" : "模型拒绝";
    $("s41-score").textContent = d.score.toFixed(4);
    $("s41-tau").textContent = d.tau.toFixed(4);
    $("s41-error").textContent = d.error_m === null ? "不适用" : `${d.error_m.toFixed(2)} m`;
    $("s41-reason").textContent = d.reason;
    $("s41-binding").textContent = JSON.stringify(d.provenance, null, 2);
    $("s41-target-image").src = `/api/spatial-evidence/image?${q}&slot=target`;
    $("s41-current-image").src = `/api/spatial-evidence/image?${q}&slot=current`;
    drawOverlayMarks("s41-overlay", d);
    drawHeat2D("s41-heat", "s41-heat-range", d.heat, d.heat_range);
    $("s41-download").disabled = false;
  }

  function s41Filter() {
    if (!s41Catalog) return;
    const old = $("s41-case").value;
    const city = $("s41-city").value;
    const rel = $("s41-relation").value;
    const rows = s41Catalog.cases.filter((c) => c.city === city && (rel === "all" || c.relation === rel));
    $("s41-case").replaceChildren(...rows.map((c) => new Option(`格 ${c.cell} · ${RELATION_NAMES[c.relation]}`, String(c.id))));
    if (rows.some((c) => String(c.id) === old)) $("s41-case").value = old;
    $("s41-case-count").textContent = `${rows.length} / 150 案例`;
    s41Load();
  }

  async function initEvidence() {
    const [pr, sr] = await Promise.all([fetch("/api/perception/catalog"), fetch("/api/spatial-evidence/catalog")]);
    p31Catalog = await pr.json();
    s41Catalog = await sr.json();

    $("p31-comparison").replaceChildren();
    for (const a of p31Catalog.arms || []) {
      const m = p31Catalog.metrics[a];
      const tr = document.createElement("tr");
      for (const v of [ARM_LABELS[a], `${m.positive_accepted_and_within5m} / 2`, `${m.positive_MAE_all_predictions_m.toFixed(2)} m`, `${(m.source_macro_balanced_accuracy * 100).toFixed(1)}%`]) {
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(td);
      }
      $("p31-comparison").append(tr);
    }
    p31Filter();

    $("s41-city").replaceChildren(...(s41Catalog.sources || []).map((c) => new Option(c.id, c.id)));
    $("s41-comparison").replaceChildren();
    for (const [k, m] of Object.entries(s41Catalog.metrics || {})) {
      const tr = document.createElement("tr");
      const [method, a] = k.split("/");
      for (const v of [`${METHOD_NAMES[method]} / ${ARM_LABELS[a]}`, `${m.positive_joint} / ${m.positive_n}`, `${(m.source_macro_joint * 100).toFixed(1)}%`, `${m.positive_error_mean_m.toFixed(2)} m`]) {
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(td);
      }
      $("s41-comparison").append(tr);
    }
    s41ConfigureMethod();
    s41Filter();
  }

  function bindEvidence() {
    $("p31-source").onchange = p31Filter;
    $("p31-relation").onchange = p31Filter;
    $("p31-case").onchange = p31Load;
    $$("#p31-arms [data-p31-arm]").forEach((b) => {
      b.onclick = () => {
        p31Arm = b.dataset.p31Arm;
        p31Load();
      };
    });
    $("p31-download").onclick = () => p31Detail && downloadable(`/api/perception/export?case=${p31Detail.id}&arm=${p31Detail.arm}`);

    $("s41-city").onchange = s41Filter;
    $("s41-relation").onchange = s41Filter;
    $("s41-case").onchange = s41Load;
    $("s41-method").onchange = () => {
      s41ConfigureMethod();
      s41Load();
    };
    $("s41-seed").onchange = s41Load;
    $("s41-prev").onclick = () => {
      const s = $("s41-case");
      s.selectedIndex = Math.max(0, s.selectedIndex - 1);
      s41Load();
    };
    $("s41-next").onclick = () => {
      const s = $("s41-case");
      s.selectedIndex = Math.min(s.options.length - 1, s.selectedIndex + 1);
      s41Load();
    };
    $("s41-download").onclick = () => {
      if (s41Detail) {
        const q = new URLSearchParams({ case: $("s41-case").value, method: $("s41-method").value, seed: $("s41-seed").value, arm: s41Arm }).toString();
        downloadable(`/api/spatial-evidence/export?${q}`);
      }
    };
  }

  // ==================== KEYBOARD SHORTCUTS ====================
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hidePopover();
      $$(".v5-modal-backdrop").forEach((m) => m.classList.add("hidden"));
      return;
    }
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target?.tagName)) return;
    const action = { ArrowUp: "up", KeyW: "up", ArrowRight: "right", KeyD: "right", ArrowDown: "down", KeyS: "down", ArrowLeft: "left", KeyA: "left" }[e.code];
    if (!action) return;

    if (activeWorkspace === "navigator" && activeNavTab === "interactive" && ixState && !ixState.done) {
      e.preventDefault();
      ixStep(action);
    } else if (activeWorkspace === "navigator" && activeNavTab === "sealed" && navState && !navState.readonly && navState.controller === "manual" && !isTerminal(navState.status)) {
      e.preventDefault();
      navAct(async () => navRender(await navApi("step", { id: navState.id, kind: navState.kind, action })));
    } else if (activeWorkspace === "sandbox" && sbRun && !sbRun.readonly && sbRun.mode === "human" && !isTerminal(sbRun.status)) {
      e.preventDefault();
      sbApi("step", { id: sbRun.id, action }).then((r) => {
        sbRun = r;
        sbRender();
      });
    }
  });

  // ==================== INITIALIZATION (WITH COLD-START RETRY) ====================
  async function refreshAll(retryCount = 0) {
    try {
      const res = await fetch("/api/meta", { headers: { "X-GeoNav-Session": owner } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const meta = await res.json();
      token = meta.scene_token || "";
      apiConfig = meta.api || null;
      updateApiBadges();
      await Promise.all([loadOverview(), ixLoadMeta(), navRefresh(), sbRefresh(), gwCatalog(), glCatalog(), initEvidence()]);
    } catch (e) {
      if (retryCount < 20) {
        if (retryCount === 0) showToast("正在等待后端预热视觉编码器与题库，就绪后自动加载...", "info", 2400);
        setTimeout(() => refreshAll(retryCount + 1), 700);
      } else {
        notice(e.message || "后端连接失败", true);
      }
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    bindPopoverAndModals();
    bindGlobalUI();
    bindWizard();
    bindPresets();
    bindInteractiveExplorer();
    bindUploadAndApiModals();
    bindNavigatorAndArena();
    bindSandbox();
    bindGround();
    bindEvidence();
    refreshAll();
  });
})();
