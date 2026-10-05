# 视频 — 项目讲解视频（HTML 合成 + TTS 配音）

本目录存放《基于主动探索的视觉地理定位方法设计与实现》的**无出镜讲解视频**工程。视频内容为项目中期讲解：任务定义 → 实验平台 → 云端 VLM 失败 → 四臂对照 → 本地小策略重建 → 目标利用难题 → 边缘线索突破 → S2 正式复验 → 展望，全部数字取自项目实验台账。

- 时长：约 4 分 17 秒（257 秒，10 个场景）
- 形式：HyperFrames HTML 合成（`geonav-agent-explainer/index.html`）+ edge-tts 中文配音（`zh-CN-XiaoxiaoNeural`），无背景音乐
- 不渲染 MP4；在浏览器中播放

## 怎么播放

```bash
cd 视频/geonav-agent-explainer
npm run dev        # 即 npx hyperframes preview，自动打开浏览器播放器
```

播放器支持播放/暂停、拖动进度、逐场景跳转；画面底部为同步字幕。首次打开需联网（GSAP 从 CDN 加载）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `geonav-agent-explainer/` | HyperFrames 视频工程（唯一交付物） |
| `成功案例可视化/case_img3004/` | 单次成功案例逐帧可视化：真实航拍底图 + 20步路线动画 + 步进式 HTML（数据取自径向保护独立确认批次的真实轨迹与源图 TIFF，只读） |
| `实际用途示意/` | 实际需求→无人机命中的示意页：真实需求措辞、目标参考照片与起点第一眼（源图裁块）、四帧全过程、物理量换算（300米/格、6km航程）；数字与径向保护批次轨迹一致 || `geonav-agent-explainer/index.html` | 组装后的可播放入口 |
| `geonav-agent-explainer/STORYBOARD.md` | 分镜（含逐场景镜头时间轴，与旁白逐句对齐） |
| `geonav-agent-explainer/SCRIPT.md` | 旁白脚本（配音的唯一输入） |
| `geonav-agent-explainer/audio/` | edge-tts 生成的 10 段配音 mp3 与句级 SRT |
| `geonav-agent-explainer/compositions/` | 10 个场景合成 + 字幕层 |
| `geonav-agent-explainer/scripts/` | 本地生成脚本：`gen-audio.mjs`（配音）、`captions-local.mjs`（中文字幕分组补丁）、`fix-fonts.mjs`（系统字体声明） |
| `geonav-agent-explainer/snapshots/` | 逐场景快照与拼贴（验收用） |

## 再生成方式

改旁白 → 编辑 `SCRIPT.md` → `node scripts/gen-audio.mjs` 重新配音 → `node scripts/captions-local.mjs build …` 重建字幕 → 重跑 `assemble-index` 与 `transitions inject`。改画面 → 编辑对应 `compositions/frames/NN-*.html`（或改 `STORYBOARD.md` 后重建）→ 重新组装。`npm run check` 可随时验证布局。

本目录由 ZCode 会话创建维护；项目其余文件未改动。
