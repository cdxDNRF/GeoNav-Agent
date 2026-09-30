# Masa 本地环境数据

本地原始集为 Massachusetts Buildings：train 137、val 4、test 10 张 1500×1500 RGB 影像。本目录用于零训练 AGL 的环境与本地评测。

**重要更正：**论文中的 1188 张规模与本地 151 张规模的来源差异尚未查明。此前“作者内部重切版”的说明无充分依据，已撤回。当前结果不能直接与不同 split / episode 的论文数字比较。

## 内容

| 路径 | 内容 |
|---|---|
| `patches/{train,val,test}/img_<编号>/patch_<0..24>.jpg` | 共3775张300×300 RGB图块，行优先编号；原始数据不改动 |
| `metadata.csv` | split、img_id 与原始切片文件名的对应表，编号在每个 split 内独立 |
| `papr_{train,val,test}_sat_embeds_grid_5.npy` | 每图(25,512)的冻结Sat2Cap特征，float32；保留备用，不是VLM主循环必需输入 |
| `episodes/episodes_{split}.jsonl` | 旧清单，保留历史，不再用于修正版评测 |
| `任务清单_v2/episodes_{split}.jsonl` | 新的本地试卷；仅环境和评测端可读，因为含目标坐标 |
| `任务清单_v2/manifest_{split}.json` | 协议、种子、预算、元数据/图块/episode SHA256、重复抽样数与生成器版本 |
| `评测结果/基础校验_v2/` | 多随机种子基础验证，含逐题轨迹和汇总；仅评测端访问 |

所有新增目录须同步登记工作区根 README；非代码目录优先中文命名。

## 生成规则

- `project/src/data/process_masa.py`：1500×1500图像切5×5，按官方代码的Sat2Cap预处理提取特征。没有本次重切或训练。
- `project/src/data/make_episodes.py`：每图按曼哈顿距离4、5、6、7、8各有放回抽5对起终点，budget=10，seed=42。
- train 3425题、val 100题、test 250题；`train` 在零训练方案中用作开发环境池，不代表进行了参数训练。
- 每个 split/area/distance 使用独立、确定的随机流。单独生成 test 与全量生成 test 内容相同。
- 距离8仅有4种有序对，抽5次必有重复，不能将所有题或多seed运行视作独立地理样本。
- 同名同内容输出允许幂等重跑，不同内容拒绝覆盖。改变参数需另存新版本目录。
- 这借鉴了 GOMAA-Geo 发布代码的采样方式，但随机实现与文件并非其原始清单；也不能直接把 GeoExplorer 的895配置解释成这里的250配置。

## 环境规则与信息边界

四方向动作；越界原地不动且消耗一步；预算内首次到达为成功，最后一步到达仍成功。不提供 stop 动作。

策略只接收 `Observation`：当前图、目标图的无元数据PNG字节、当前位置、预算、已访问记录和四动作列表。不给目标文件名、目标坐标、目标距离、整图或未访问格特征。公开 `StepInfo` 只有越界、重复访问和步数；评测结果终止后才由评测器读取。

这是程序接口隔离，不是防恶意策略读取本机文件的操作系统沙箱。未来的策略运行器只能传公开对象，不能把 Environment 对象、原始 episode 字典或评测日志交给模型。

## 评测

在工作区根目录：

```bash
python -X utf8 project/src/data/make_episodes.py
python -X utf8 -m unittest discover -s project/src/tests -v
python -X utf8 project/src/eval/evaluate.py --splits val test --seeds 0 1 2 3 4
```

SR=成功episode数/总数。SG为终止时到目标的曼哈顿距离，对所有episode求均值。重复访问定义为“动作后落点此前已访问”，含撞墙原地；起始格不计入分母。日志同时给出动作总数加权(micro)和每题平均(macro)两种重复访问率。

多seed随机策略只用于检查结果稳定性与评测闭环，不代表VLM效果或论文复现。后续开发使用train/val，test不可据此调参；跨episode记忆默认清空。
