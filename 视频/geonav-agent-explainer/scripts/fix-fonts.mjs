// fix-fonts.mjs — 为全部合成 HTML 注入本地系统字体的 @font-face(local) 声明，
// 解决 lint 的 font_family_without_font_face 错误（中文栈均为 OS 内置字体，无需字体文件）。
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FAMILIES = [
  "Noto Sans SC", "Microsoft YaHei", "PingFang SC", "Noto Serif SC",
  "SimSun", "Songti SC", "DM Mono", "Newsreader", "Hanken Grotesk",
];
const block = FAMILIES.map((f) => `@font-face{font-family:'${f}';src:local('${f}');}`).join("");

const targets = [
  "compositions/captions.html",
  ...[...Array(10)].map((_, i) => {
    // 从 STORYBOARD.md 的 src 约定直接取帧文件名
    return null;
  }).filter(Boolean),
];
// 帧文件直接枚举（与 compositions/frames 一致）
const fs = await import("node:fs");
const frameFiles = fs.readdirSync(join(ROOT, "compositions", "frames"))
  .filter((f) => f.endsWith(".html")).sort()
  .map((f) => `compositions/frames/${f}`);
targets.length = 0;
targets.push("compositions/captions.html", ...frameFiles);

let patched = 0;
for (const rel of targets) {
  const p = join(ROOT, rel);
  let html = readFileSync(p, "utf8");
  if (html.includes("/*localfont-injected*/")) { console.log(`· skip ${rel}`); continue; }
  const idx = html.indexOf("<style>");
  if (idx < 0) { console.error(`✗ ${rel} 未找到 <style>`); continue; }
  const at = idx + "<style>".length;
  html = html.slice(0, at) + "/*localfont-injected*/" + block + html.slice(at);
  writeFileSync(p, html);
  patched++;
  console.log(`✓ ${rel}`);
}
console.log(`done: ${patched} file(s) patched`);
