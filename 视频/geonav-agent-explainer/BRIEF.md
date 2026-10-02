---
workflow: faceless-explainer
flow: automation
storyboard: yes
message: "从云端VLM失败到边缘线索突破：GeoNav-Agent 用可信目标证据在5×5网格里主动找到目标，本地S2正式通过（SR 89.3%）"
destination: embed
aspect: 1920x1080
language: zh
length: 200s
angle: project-story
voice: zh-CN-XiaoxiaoNeural
---

## Intent

为《基于主动探索的视觉地理定位方法设计与实现》课程项目（GeoNav-Agent）制作一部中文中期讲解视频。观众是课程教师与同组同学：他们了解基本机器学习概念，但没读过本项目代码。叙事基调：克制、有据、工程纪律感——用真实实验数字讲故事，不夸大；负结果和正结果同样值得讲。

## Customizations

- 配音使用 edge-tts（`uvx edge-tts`），语音 zh-CN-XiaoxiaoNeural；不用 HeyGen/Kokoro TTS。
- 不渲染 MP4；交付 hyperframes 合成 index.html + 音频文件，浏览器可直接播放。
- 全部工作只在项目根目录的 `视频/geonav-agent-explainer/` 内进行，不触碰工作区其他文件。
- 无背景音乐（music: none），仅旁白。

## Notes

- 所有数字必须与项目实验台账一致：val20 四臂（A 5%/B 10%/C 10%/D Frontier 50%）、本地 PBRS 44.33%→合法动作筛选 66.00%、开发集 Edge 88.57%（基线 72.62%）、S2 正式复验 89.33%（基线 73.00%，+16.33 点，12/12 检查通过）。
- 不要宣称跨数据集或独立源图泛化——S3 未开展，这是当前方法的适用边界。
- 中文显示字体用 Noto Sans SC / Microsoft YaHei 栈；预设的 Latin 字体仅用于拉丁字符与数字。
