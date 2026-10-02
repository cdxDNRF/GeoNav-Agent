// gen-audio.mjs — 本项目的 edge-tts 配音生成器（替代 faceless-explainer 的 HeyGen/Kokoro 管线）。
// 用法: node scripts/gen-audio.mjs
// 1) 解析 SCRIPT.md 的 (Frame N) 段落与缩进台词；
// 2) 每段调用 uvx edge-tts 生成 audio/voice-NN.mp3 与词/句级 SRT；
// 3) 解析 SRT 得到句级时间戳，写入 .hyperframes/audio_meta.json（voices[].frame/path/duration_s/words）。
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const VOICE = "zh-CN-XiaoxiaoNeural";
const RATE = "+0%";
const PAD_S = 0.45; // 每段结尾留白，避免旁白贴着转场切

function parseScript(md) {
  const out = [];
  let cur = null;
  const flush = () => {
    if (cur && cur.text.trim()) out.push({ frame: cur.frame, text: cur.text.trim() });
    cur = null;
  };
  for (const line of md.split(/\r?\n/)) {
    const h = line.match(/^#{2,3}\s+.*?[(（]frame\s*(\d+)[)）]/i);
    if (h) { flush(); cur = { frame: Number(h[1]), text: "" }; continue; }
    if (!cur) continue;
    if (/^\s*\*\*/.test(line)) continue;
    const m = line.match(/^(?: {4,}|\t)(.+)$/);
    if (m) cur.text += (cur.text ? " " : "") + m[1].trim();
  }
  flush();
  return out;
}

function srtTimeToSec(t) {
  const m = t.trim().match(/^(\d{2}):(\d{2}):(\d{2}),(\d{3})$/);
  if (!m) return NaN;
  return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]) + Number(m[4]) / 1000;
}

function parseSrt(md) {
  const cues = [];
  const blocks = md.replace(/\r\n/g, "\n").trim().split(/\n{2,}/);
  for (const b of blocks) {
    const lines = b.split("\n").filter((l) => l.trim() !== "");
    if (lines.length < 2) continue;
    const tm = lines[1].match(/-->/);
    if (!tm) continue;
    const [a, z] = lines[1].split("-->");
    const text = lines.slice(2).join(" ").trim();
    if (!text) continue;
    cues.push({ start: srtTimeToSec(a), end: srtTimeToSec(z), text });
  }
  return cues;
}

const scriptPath = join(ROOT, "SCRIPT.md");
const lines = parseScript(readFileSync(scriptPath, "utf8"));
if (lines.length === 0) { console.error("✗ SCRIPT.md 未解析到台词"); process.exit(1); }

const audioDir = join(ROOT, "audio");
const srtDir = join(ROOT, "audio", "srt");
mkdirSync(audioDir, { recursive: true });
mkdirSync(srtDir, { recursive: true });

const voices = [];
for (const { frame, text } of lines) {
  const nn = String(frame).padStart(2, "0");
  const mp3 = join(audioDir, `voice-${nn}.mp3`);
  const srt = join(srtDir, `voice-${nn}.srt`);
  console.log(`▸ frame ${frame}: edge-tts (${text.length} chars)`);
  const r = spawnSync("uvx", [
    "edge-tts", "--voice", VOICE, `--rate=${RATE}`,
    "--text", text, "--write-media", mp3, "--write-subtitles", srt,
  ], { stdio: ["ignore", "ignore", "pipe"] });
  if (r.status !== 0) {
    console.error(`✗ frame ${frame} TTS 失败: ${r.stderr?.toString().slice(0, 400)}`);
    process.exit(1);
  }
  if (!existsSync(mp3) || !existsSync(srt)) { console.error(`✗ frame ${frame} 输出缺失`); process.exit(1); }
  const cues = parseSrt(readFileSync(srt, "utf8"));
  if (cues.length === 0) { console.error(`✗ frame ${frame} SRT 为空`); process.exit(1); }
  const lastEnd = Math.max(...cues.map((c) => c.end));
  voices.push({
    frame,
    path: `audio/voice-${nn}.mp3`,
    duration_s: Math.round((lastEnd + PAD_S) * 1000) / 1000,
    words: cues.map((c, i) => ({ id: i, text: c.text, start: c.start, end: c.end })),
  });
  console.log(`  ✓ ${cues.length} 句, duration ${(lastEnd + PAD_S).toFixed(2)}s`);
}

const total = voices.reduce((s, v) => s + v.duration_s, 0);
const meta = { bgm: null, bgm_pending: false, voices, sfx: [] };
const outPath = join(ROOT, ".hyperframes", "audio_meta.json");
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(meta, null, 2));
console.log(`✓ audio_meta.json: ${voices.length} 段, 总时长 ${(total / 60).toFixed(1)} 分钟`);
