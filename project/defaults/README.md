# 按协议加载已确认默认

物理10×10/B20/300米采用Coverage3Radial，固定参考seed0；原5×5及跨数据集脚本仍使用原入口。15×15仅为工程候选，不继承10×10验收。

```python
from agents.area_policy import load_area_policy
from env.area_protocol import GRID10, GRID15
agent = load_area_policy(ROOT, GRID10)  # 本协议默认，seed0
reference = load_area_policy(ROOT, GRID10, reference_M0=True)
engineering = load_area_policy(ROOT, GRID15, allow_experimental=True)
```

策略只接收Observation和当前/给定目标特征；环境与评测器负责图块查找，不将完整特征库、真值位置或源身份交给Agent。配置、模型、均值和独立依据均由加载器校验哈希。新15×15确认前不将工程候选写入defaults。
