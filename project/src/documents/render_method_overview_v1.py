"""Create an editable HTML/SVG engineering overview; browser exports PDF/PNG."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / '中期报告相关/中期成果冻结版_v1/图表'

SVG = r'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">
<style>
text{font-family:"Microsoft YaHei","Noto Sans SC",sans-serif;fill:#0f172a}.mono{font-family:Consolas,monospace}.title{font-size:30px;font-weight:700}.sub{font-size:15px;fill:#64748b}.layer{font-size:13px;font-weight:700;letter-spacing:1.5px}.node-title{font-size:18px;font-weight:700}.body{font-size:14px;fill:#425466}.tiny{font-size:12px;fill:#64748b}.box{fill:#fff;stroke:#94a3b8;stroke-width:1.5}.accent{fill:#ecfdf5;stroke:#0f766e;stroke-width:2}.warm{fill:#fffbeb;stroke:#c67c00;stroke-width:1.5}.env{fill:#eff6ff;stroke:#386cb0;stroke-width:1.5}.dash{fill:#f8fafc;stroke:#94a3b8;stroke-dasharray:7 6;stroke-width:1.4}.arrow{stroke:#334155;stroke-width:2;fill:none;marker-end:url(#arrow)}.feedback{stroke:#0f766e;stroke-width:2;fill:none;stroke-dasharray:7 5;marker-end:url(#green)}
</style><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#334155"/></marker><marker id="green" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#0f766e"/></marker></defs>
<rect width="1400" height="820" fill="#f8fafc"/><rect x="22" y="22" width="1356" height="776" rx="2" fill="#fff" stroke="#cbd5e1" stroke-width="2"/>
<text x="55" y="70" class="title">M0单导航智能体：冻结视觉表征、探索提议与可信目标线索</text><text x="55" y="99" class="sub">主动探索视觉地理定位 · 离线训练与在线交互分层 · 评测真值与Agent输入严格隔离</text><line x1="55" y1="122" x2="1345" y2="122" stroke="#cbd5e1"/>
<text x="58" y="160" class="layer">OFFLINE / 冻结准备</text>
<rect x="55" y="178" width="270" height="115" class="box"/><text x="78" y="213" class="node-title">Sat2Cap视觉编码器</text><text x="78" y="242" class="body">全局 512维 + 局部 4×768维</text><text x="78" y="269" class="tiny">预训练后冻结；可复用特征缓存</text>
<path d="M325 235 H390" class="arrow"/><rect x="390" y="178" width="285" height="115" class="box"/><text x="413" y="213" class="node-title">Small256探索器</text><text x="413" y="242" class="body">GRU状态 · Actor / Critic / 预测头</text><text x="413" y="269" class="tiny">PBRS导航训练；目标通道使用拟合均值</text>
<path d="M675 235 H740" class="arrow"/><rect x="740" y="178" width="300" height="115" class="box"/><text x="763" y="213" class="node-title">EdgeTargetCue线索头</text><text x="763" y="242" class="body">语义 + 局部关系 + 20维边缘连续性</text><text x="763" y="269" class="tiny">五类输出：上 / 右 / 下 / 左 / 非邻接</text>
<rect x="1105" y="178" width="240" height="115" class="dash"/><text x="1128" y="213" class="node-title">冻结交付</text><text x="1128" y="242" class="body">3套独立Seed权重</text><text x="1128" y="269" class="tiny mono">config + SHA + means</text><path d="M1040 235 H1105" class="arrow"/>
<text x="58" y="353" class="layer">ONLINE / 每步交互闭环</text>
<rect x="55" y="375" width="220" height="145" class="env"/><text x="78" y="410" class="node-title">局部航拍观测</text><text x="78" y="441" class="body">当前图 + 给定目标图</text><text x="78" y="470" class="body">位置 / 剩余预算 / 访问历史</text><text x="78" y="499" class="tiny">仅公开Observation与两张可见图</text>
<path d="M275 447 H335" class="arrow"/><rect x="335" y="375" width="220" height="145" class="box"/><text x="358" y="410" class="node-title">冻结视觉特征</text><text x="358" y="441" class="body">当前/目标全局与局部向量</text><text x="358" y="470" class="body">当前/目标边缘profile</text><text x="358" y="499" class="tiny">不访问全图或未观察图块</text>
<path d="M555 420 H615" class="arrow"/><rect x="615" y="355" width="235" height="105" class="box"/><text x="638" y="389" class="node-title">探索动作提议</text><text x="638" y="419" class="body">Small256 / GRU</text><text x="638" y="445" class="tiny">覆盖搜索机会</text>
<path d="M555 490 H615" class="arrow"/><rect x="615" y="475" width="235" height="105" class="warm"/><text x="638" y="509" class="node-title">目标邻接证据</text><text x="638" y="539" class="body">EdgeTargetCue</text><text x="638" y="565" class="tiny">可信时输出方向</text>
<path d="M850 407 H910 V450" class="arrow"/><path d="M850 527 H910 V480" class="arrow"/><rect x="910" y="397" width="235" height="135" class="accent"/><text x="933" y="432" class="node-title">动作门控</text><text x="933" y="462" class="body">置信度 ≥ 0.50</text><text x="933" y="490" class="body">合法且未访问 → 线索接管</text><text x="933" y="516" class="tiny">否则执行探索器提议</text>
<path d="M1145 464 H1200" class="arrow"/><rect x="1200" y="397" width="145" height="135" class="env"/><text x="1222" y="432" class="node-title">环境</text><text x="1222" y="462" class="body">上 / 右 / 下 / 左</text><text x="1222" y="490" class="body">越界耗步</text><text x="1222" y="516" class="tiny">首次到达即成功</text>
<path d="M1272 532 V625 H160 V520" class="feedback"/><text x="575" y="649" class="tiny">新观察与已访问前缀返回同一Agent；每个episode重置记忆</text>
<text x="58" y="705" class="layer">EVALUATOR / 评测边界</text><rect x="55" y="722" width="1290" height="52" class="dash"/><text x="78" y="754" class="body">评测器持有真实目标坐标、完整图块库与终局指标；这些信息仅用于环境转移和事后SR/SG/移动距离/恢复损伤，不进入策略输入。</text>
</svg>'''

HTML = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M0方法总览</title><style>html,body{margin:0;background:#f8fafc}body{padding:20px}svg{display:block;width:min(1400px,100%);height:auto;margin:auto;background:#fff}</style></head><body>''' + SVG + '</body></html>'


def main():
    html, svg = FIG / '01_M0方法与评测边界.html', FIG / '01_M0方法与评测边界.svg'
    if html.exists() or svg.exists():raise FileExistsError('method figure already exists')
    html.write_text(HTML, encoding='utf-8');svg.write_text(SVG, encoding='utf-8')
    print(html);print(svg)


if __name__ == '__main__':main()
