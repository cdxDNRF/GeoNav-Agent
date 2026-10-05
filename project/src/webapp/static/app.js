"use strict";
const $ = (id) => document.getElementById(id);
const state = {catalogs: [], rows: [], selected: null, step: 0, frame: null, playing: false, generation: 0, listGeneration: 0, playGeneration: 0};
const direction = {up: "↑ 向上", right: "→ 向右", down: "↓ 向下", left: "← 向左"};
const images = new Map();
async function request(path, params = {}) {
  const response = await fetch(`${path}?${new URLSearchParams(params)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "原件读取失败");
  return data;
}
function error(err) { $("error").hidden = false; $("error").textContent = err.message; pause(); }
function clearError() { $("error").hidden = true; }
function option(value, label) { const node = document.createElement("option"); node.value = value; node.textContent = label; return node; }
function catalog() { return state.catalogs.find((item) => item.id === $("catalog").value); }
function pause() { state.playing = false; ++state.playGeneration; $("play").textContent = "播放轨迹"; }
function imageUrl(frame, cell) { return `/api/image?${new URLSearchParams({id: frame.id, step: frame.step, cell})}`; }
function loadImage(url) {
  if (!images.has(url)) images.set(url, new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = () => { images.delete(url); reject(new Error("图块加载或哈希核验失败")); }; image.src = url;
  }));
  if (images.size > 600) images.delete(images.keys().next().value);
  return images.get(url);
}
function table() {
  const body = $("overview-body"); body.replaceChildren();
  for (const item of state.catalogs) for (const policy of ["M0", "Coverage3Radial"]) {
    const metrics = item.metrics[policy];
    const tr = document.createElement("tr");
    for (const text of [item.label, `${item.grid}×${item.grid} / B${item.budget}`, item.regions, item.planned_tasks,
      policy, `${(metrics.sr * 100).toFixed(2)}%`, metrics.sg_m.toFixed(2), metrics.movement_m.toFixed(2), item.acceptance]) {
      const td = document.createElement("td"); td.textContent = text; tr.append(td);
    }
    body.append(tr);
  }
}
function updateCatalog() {
  pause(); const item = catalog();
  $("area").replaceChildren(option("", "全部区域"), ...item.areas.map((area) => option(area, area)));
  $("acceptance").textContent = item.acceptance; $("acceptance").classList.toggle("failed", !item.accepted);
  $("protocol-badge").textContent = `${item.grid}×${item.grid} · ${item.area_km2} km²`;
  refreshList().catch(error);
}
async function refreshList() {
  pause(); clearError(); ++state.generation; const generation = ++state.listGeneration;
  const payload = await request("/api/episodes", {catalog: $("catalog").value, policy: $("policy").value,
    seed: $("seed").value, area: $("area").value, outcome: $("outcome").value, q: $("query").value});
  if (generation !== state.listGeneration) return;
  state.rows = payload.episodes;
  $("list-count").textContent = `${state.rows.length} 题`;
  if (!state.rows.length) {
    state.selected = null; ++state.generation; state.frame = null; $("episode-list").replaceChildren();
    $("map-loading").hidden = false; $("map-loading").textContent = "没有匹配任务，请调整筛选";
    $("goal-image").removeAttribute("src"); $("current-image").removeAttribute("src");
    for (const id of ["play", "next", "previous", "export", "slider"]) $(id).disabled = true;
    for (const id of ["step-metric", "coverage-metric", "travel-metric", "sg-metric", "decision-status", "explorer", "cue", "controller", "reason", "source", "position", "episode-caption"]) $(id).textContent = "—";
    $("timeline").replaceChildren(); $("probabilities").replaceChildren(); return;
  }
  const preserved = state.rows.find((r) => r.id === state.selected);
  await selectEpisode((preserved || state.rows[0]).id);
}
function list() {
  $("episode-list").replaceChildren();
  for (const row of state.rows) {
    const button = document.createElement("button"); button.className = "episode-item";
    button.classList.toggle("selected", row.id === state.selected);
    button.setAttribute("aria-pressed", String(row.id === state.selected));
    const outcome = document.createElement("b"); outcome.className = `outcome ${row.success ? "" : "failed"}`;
    outcome.textContent = row.success ? "成功" : "失败";
    const label = document.createElement("div"); label.textContent = row.episode_id;
    const info = document.createElement("span"); info.textContent = `${row.area} · C${row.distance} · ${row.steps} 步`;
    button.append(outcome, label, info); button.onclick = () => selectEpisode(row.id).catch(error);
    $("episode-list").append(button);
  }
}
async function selectEpisode(id) {
  pause(); state.selected = id; state.step = 0; $("diagnostic").checked = false; list(); await showFrame();
}
async function draw(frame, generation) {
  const canvas = $("map"), ctx = canvas.getContext("2d"), size = canvas.width, unit = size / frame.grid;
  const tileImages = await Promise.all(frame.visited.map(async (cell) => [cell, await loadImage(imageUrl(frame, cell))]));
  if (generation !== state.generation) return;
  ctx.clearRect(0, 0, size, size);
  for (let cell = 0; cell < frame.grid ** 2; cell++) {
    const x = (cell % frame.grid) * unit, y = Math.floor(cell / frame.grid) * unit;
    ctx.fillStyle = (Math.floor(cell / frame.grid) + cell % frame.grid) % 2 ? "#edf1e9" : "#e8eee2";
    ctx.fillRect(x, y, unit, unit); ctx.strokeStyle = "#d8e1d1"; ctx.lineWidth = .8; ctx.strokeRect(x, y, unit, unit);
    ctx.fillStyle = "#afbfaa"; ctx.font = `${frame.grid === 15 ? 10 : 12}px Segoe UI`; ctx.textAlign = "center";
    ctx.fillText(String(cell), x + unit / 2, y + unit / 2 + 4);
  }
  for (const [cell, image] of tileImages) {
    const x = (cell % frame.grid) * unit, y = Math.floor(cell / frame.grid) * unit;
    ctx.drawImage(image, x + 1, y + 1, unit - 2, unit - 2);
    ctx.fillStyle = "#153c3433"; ctx.fillRect(x, y, unit, unit);
  }
  const point = (cell) => [(cell % frame.grid + .5) * unit, (Math.floor(cell / frame.grid) + .5) * unit];
  for (const [stroke, width] of [["#ffffff", 7], ["#2182e7", 3.6]]) {
    ctx.beginPath(); ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.lineJoin = "round";
    frame.route.forEach((item, i) => { const [x, y] = point(item.patch_id); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
  }
  const start = point(frame.route[0].patch_id); ctx.fillStyle = "white"; ctx.beginPath(); ctx.arc(...start, 6, 0, 2 * Math.PI); ctx.fill();
  const now = point(frame.current); ctx.fillStyle = "#123d34"; ctx.strokeStyle = "white"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(...now, 10, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  if (frame.diagnostic) {
    const goal = frame.diagnostic.goal; ctx.strokeStyle = "#e6a73f"; ctx.lineWidth = 4;
    ctx.strokeRect(goal % frame.grid * unit + 4, Math.floor(goal / frame.grid) * unit + 4, unit - 8, unit - 8);
  }
  $("map-loading").hidden = true;
}
function decision(frame) {
  const d = frame.decision;
  $("decision-status").textContent = d ? direction[d.action] : (frame.result.success ? "✓ 目标已到达" : "行动预算已耗尽");
  $("explorer").textContent = d ? direction[d.explorer_action] : "—";
  $("cue").textContent = d ? (d.cue_action ? direction[d.cue_action] : "未接受方向线索") : "—";
  $("controller").textContent = d ? (d.cue_action ? "已接受目标线索优先" : d.coverage?.radial_guard_blocked ? "径向保护拦截" : d.coverage?.changed ? "覆盖收益接管" : "保留原动作") : "—";
  const reasons = {not_adjacent: "视觉头判为非邻接", accepted: "目标邻接线索获接受", visited: "候选已访问", illegal: "候选不合法", low_confidence: "线索置信度不足"};
  $("reason").textContent = d ? (reasons[d.reason] || d.reason) : (frame.result.success ? "首次到达目标，成功终止" : "行动预算用尽，任务失败");
  $("probabilities").replaceChildren();
  if (d) d.probabilities.forEach((prob, i) => {
    const row = document.createElement("div"); row.className = "prob";
    const label = document.createElement("span"); label.textContent = ["↑ 上", "→ 右", "↓ 下", "← 左", "非邻接"][i];
    const bar = document.createElement("div"); bar.className = "bar"; const fill = document.createElement("div"); fill.style.width = `${Math.max(0, Math.min(1, prob)) * 100}%`; bar.append(fill);
    const score = document.createElement("span"); score.textContent = `${(prob * 100).toFixed(1)}%`; row.append(label, bar, score); $("probabilities").append(row);
  });
}
async function showFrame() {
  if (!state.selected) return;
  clearError(); const generation = ++state.generation;
  const frame = await request("/api/frame", {id: state.selected, step: state.step, diagnostic: $("diagnostic").checked ? "1" : "0"});
  if (generation !== state.generation) return;
  state.frame = frame;
  $("slider").max = frame.total_steps; $("slider").value = frame.step;
  $("step-label").textContent = `${frame.step} / ${frame.total_steps}`;
  $("step-metric").textContent = `${frame.step} / ${frame.budget}`;
  $("coverage-metric").textContent = `${frame.visited.length} / ${frame.grid ** 2}`;
  $("travel-metric").textContent = `${frame.movement_m} m`;
  $("sg-metric").textContent = frame.result ? `${frame.result.sg_m} m` : "未终止";
  $("previous").disabled = frame.step === 0; $("next").disabled = frame.terminal;
  $("play").disabled = false; $("slider").disabled = false; $("export").disabled = false;
  $("goal-image").src = imageUrl(frame, "goal"); $("current-image").src = imageUrl(frame, frame.current);
  $("position").textContent = `第 ${Math.floor(frame.current / frame.grid) + 1} 行 / ${frame.current % frame.grid + 1} 列`;
  $("episode-caption").textContent = frame.episode_id;
  $("source").textContent = `轨迹原件\n${frame.source.trajectory}\n\n文件 SHA256\n${frame.source.sha256}\n\n记录 SHA256\n${frame.source.line_sha256}\n\n原报告\n${frame.source.report}`;
  $("truth-note").textContent = frame.diagnostic ? `事后真值模式：当前至目标 ${frame.diagnostic.distance_cells} 格。黄色框为真值位置；仅用于回放诊断。` : "默认只展示已观察图块。真实目标位置不会作为策略输入。";
  decision(frame);
  $("timeline").replaceChildren();
  for (let i = 0; i <= frame.total_steps; i++) {
    const button = document.createElement("button"); button.title = `第 ${i} 步`; button.setAttribute("aria-label", button.title);
    if (i <= frame.step) button.classList.add("passed"); if (i === frame.step) button.classList.add("cursor");
    button.onclick = () => { pause(); state.step = i; showFrame().catch(error); }; $("timeline").append(button);
  }
  await draw(frame, generation);
  if (frame.terminal) pause();
}
async function playLoop() {
  if (state.playing) { pause(); return; }
  if (!state.frame) return;
  if (state.frame.terminal) { state.step = 0; await showFrame(); }
  state.playing = true; $("play").textContent = "暂停回放";
  const generation = ++state.playGeneration;
  while (state.playing && generation === state.playGeneration) {
    await new Promise((resolve) => setTimeout(resolve, Number($("speed").value)));
    if (!state.playing || generation !== state.playGeneration) break;
    state.step++; await showFrame();
  }
}
function setTab(overview) {
  pause(); $("workspace").hidden = overview; $("overview").hidden = !overview;
  $("replay-tab").classList.toggle("active", !overview); $("overview-tab").classList.toggle("active", overview);
}
async function init() {
  const payload = await request("/api/catalog"); state.catalogs = payload.catalogs;
  $("total-records").textContent = payload.records.toLocaleString();
  $("catalog").replaceChildren(...payload.catalogs.map((item) => option(item.id, item.label)));
  table(); $("catalog").onchange = updateCatalog;
  for (const id of ["policy", "seed", "area", "outcome"]) $(id).onchange = () => refreshList().catch(error);
  let debounce; $("query").oninput = () => { clearTimeout(debounce); debounce = setTimeout(() => refreshList().catch(error), 250); };
  $("slider").oninput = () => { pause(); state.step = Number($("slider").value); showFrame().catch(error); };
  $("diagnostic").onchange = () => showFrame().catch(error);
  $("previous").onclick = () => { pause(); state.step = Math.max(0, state.step - 1); showFrame().catch(error); };
  $("next").onclick = () => { pause(); state.step = Math.min(state.frame.total_steps, state.step + 1); showFrame().catch(error); };
  $("play").onclick = () => playLoop().catch(error);
  $("replay-tab").onclick = () => setTab(false); $("overview-tab").onclick = () => setTab(true);
  $("export").onclick = async () => { try {
    const id = state.selected; const payload = await request("/api/export", {id});
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"}));
    const a = document.createElement("a"); a.href = url; a.download = `GeoNav_${payload.saved_record.episode_id}_${payload.saved_record.policy}_s${payload.saved_record.seed ?? payload.saved_record.local_checkpoint_seed}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (err) { error(err); } };
  document.addEventListener("keydown", (event) => { if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return; if (event.key === "ArrowRight" && !$("next").disabled) $("next").click(); if (event.key === "ArrowLeft" && !$("previous").disabled) $("previous").click(); });
  updateCatalog();
}
init().catch(error);
