# Frame packet: 08-edge

## Project inputs

- Project: D:\桌面\XDUCS-courses\大四\大数据工程\视频\geonav-agent-explainer
- Design tokens: D:\桌面\XDUCS-courses\大四\大数据工程\视频\geonav-agent-explainer\frame.md
- RULES_DIR: C:\Users\DFWJ\.agents\skills\hyperframes-animation\rules

## Assigned storyboard block

## Frame 8 — 边缘突破：接缝里的证据

- status: outline
- src: compositions/frames/08-edge.html
- duration: 32.525s
- transition_in: crossfade
- poster: 17s
- scene: 两个图块自左右对开；四条边接缝段点亮并连线；20维/+2560参数标签；72.62%→88.57%对撞替换
- voiceover: 突破来自一个很细的表征：边缘接缝。把当前图块与目标图块的四条边切成小段，逐段比较颜色误差、梯度误差和相关性，构成二十维特征。只新增两千五百六十个参数，线索头第一次给出可信的邻接判断：目标是否就在相邻格、朝哪个方向。同样的开发集，成功率从百分之七十二点六跃升到百分之八十八点六，最难的短距离档提升最大。
- blueprint: comparison-split

核心方法帧：两图块对开是 comparison-split 的 signature；后接参数/判断小卡与数字对撞。

blueprint: comparison-split (Adapt) — 保留两翼对开 mirrored book-open 与内侧徽章 spring-pop 的 signature；徽章从"能力标签"换为"邻接/方向"判断卡，尾段接数字对撞收束。
focal: 对置的两图块与其接缝连线（前半）；72.62%→88.57% 数字对撞（尾段）
roles: 图块A/B = foreground subject · 接缝段与误差连线 = 次焦点（方法本体）· 20维/+2560/判断卡 = supporting · 数字对与徽章 = 尾段焦点
Scene 1 (0.0–3.9s): 概念名拍："边缘接缝"大字居中，mono 副标 seam features · 20-dim。
Scene 2 (3.9–13.6s): 图解主体：两个 3×3 抽象纹理图块自左右两翼 mirrored book-open 对开并置（split-screen）；随后四条边的接缝小段逐条点亮（1/4/8 像素三种宽度的刻度感），块间误差连线 SVG self-draw；三枚标签按语流弹入："颜色误差""梯度误差""相关性"，汇总徽章"20 维特征"收拢。
Scene 3 (13.6–23.3s): 重排 asymmetric 55/45：图解缩至左侧，右侧两张判断卡依次 spring-pop 打勾："目标在相邻格？✓""朝哪个方向？✓"，上方 mono 小卡"+2,560 参数 · 线索头"。
Scene 4 (23.3–32.5s): 底部数字对撞：72.62% 被 88.57% scale-swap 撞出定格，徽章"+15.95 点"弹入，小注"C4/C5 短距离档提升最大"；hold 定格。

## Selected blueprint: comparison-split

# comparison-split — Comparison Split-Cards

**intent**: Two paired items of equal weight shown side-by-side with mirrored 3D "book-open" tilts — the eye reads them as a balanced comparison, then a pill badge lands at each card's inner edge to punctuate. The motion IS the symmetry: two cards arriving from opposite wings into a held spread.

**roles served**

- Key_Feature (from `comparison-split-cards`): when two complementary features / capabilities of equal weight should be presented **simultaneously, not sequentially** — an A/B, a "X + Y together," paired concepts the viewer must weigh side-by-side. Not for >2 items (use `grid-card-assemble`) or sequential steps.

**duration**: 4–6s

**shot structure** (a `[bg]` canvas carrying two faint ambient glow blooms — `[accent A]` near 30%, `[accent B]` near 70% — so each side owns a color identity across a 50% symmetry axis; equal-width cards under one shared perspective parent)

- **Scene 1 (0.0–~0.8s) — title sets the concept.** A centered `[title line]` with an `[accent keyword]` slides DOWN into place from just above (a short smooth settle). The downward arrival is deliberate: it forms a non-conflicting T-shape against the cards, which arrive from the sides next.
- **Scene 2 (~0.4–1.9s) — the split-tilt entry (signature move).** Two equal-width feature cards arrive from opposite wings — `[left card]` from the left, `[right card]` from the right ~0.2s behind — each carrying a **mirrored 3D `rotateY` tilt** (left faces right, right faces left, opening like a book) and scaling ~0.85→1 as it lands. The entry overlaps the title's tail so the whole thing reads as ONE arrival, not two beats. Each card holds `[image / label / subtitle]`; box-shadows fall **outward** from the tilt (left shadow right, right shadow left).
- **Scene 3 (~1.9–end) — badges punctuate, then hold.** A pill `[badge]` lands at each card's **inner edge** (left then right, ~0.3s apart), overlapping its card ~15% so it reads as attached, not orbiting. This is the lone overshoot in the shot — it earns the punctuation. Settles and holds.

**motion vocabulary**: title slide-down from above; mirrored opposite-wing card entry; static book-open `rotateY` tilt (`+tilt` left, `−tilt` right); tilt-matched outward box-shadow; inner-edge badge spring-pop; gentle phase-opposed idle float (left vs right, never synchronized) registered as subtle jitter; dual side-glow ambient.

**rule mapping**

- two cards entering from opposite wings with mirrored `rotateY` tilts + tilt-matched shadow → `split-tilt-cards` (the signature; keep the two-layer split so the entry `x`/`scale` and the idle never collide on one alias)
- title slide-down settle → `gsap-effects` (translate + opacity on a long-tail `power3`)
- inner-edge pill badge pop (the one overshoot) → `spring-pop-entrance` (overshoot register — earns the punctuation)
- phase-opposed idle float on the pair → `sine-wave-loop` (low-amplitude register — subtle jitter, NOT lazy breathing; left `sin(t)`, right `sin(t+π)` so they never conveyor-belt)
- the two faint side glows behind the cards → `ambient-glow-bloom` (un-triggered soft bloom, one per accent)

**camera modifier**: camera-static by default — the symmetry is the subject and a move would break the balance.
