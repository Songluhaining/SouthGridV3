# g1_button_v5 训练集：补采结果与训练/评测指引

本文接续 [g1_button_v4_dataset_eval.md](g1_button_v4_dataset_eval.md) 与
[g1_button_v4_supplement.md](g1_button_v4_supplement.md)。v4 训出的模型忽略颜色指令
（单色接地 25% ≈ 四选一随机），根因是四键任务占 62% 训练帧。本文记录**补采 300 集单色数据**、
**合并成 v5 训练集**的全过程与实测数字，以及训练侧照做即可的步骤。

---

## 1. 补采是怎么做的（与 v4 的一致性）

在采集 v4 的同一台机器、同一套环境上，用同一条驱动脚本 `auto_collect_windows.py`：

```bash
python auto_collect_windows.py --out <补采目录> --total 300 --chunk 10 \
    --max_buttons 1 --length_weights 1,1,1,1 \
    --extra "--canonical_ratio 0.8 --repo_id local/g1_button_v4_supp"
```

**刻意不传的三个参数**，让它们落回与 v4 相同的默认值：

| 参数 | 默认值 | 为什么不能改 |
|---|---|---|
| `--success_mode` | `official` | 前一版指引写的 `press` 会把 `press_depth` 从 0 改成 0.040，多推 4 厘米。**官方得分曲线顶点在 0.05 m，推深了反而扣分**，且与 v4 的几何口径不一致 |
| `--clock` | `wall` | 与 v4 的帧间时序一致；`--clock sim` 只适用于下一批全新采集 |
| `--fps` | `20` | 与 v4 一致 |

有意与 v4 不同的只有两处，正是这次要修的：`--max_buttons 1`（只采单色）、
`--canonical_ratio 0.8`（说法从 34% 收敛到 80% 用规范句）。

场景与初始位姿同样沿用 v4：布局 `g1_button.json`（按钮是配电柜模型内部的关节，
不是独立 actor，所以场景 actor 列表里看不到；判据是机器人位姿），
机器人基座 `[18.912, -23.007, 0.203]` / 绕 Z 轴 90°。

### 补采结果核对（300 集 / 121,329 帧 / 保留率 100%）

| 指标 | 补采 | v4 基准 | 判定 |
|---|---|---|---|
| 按钮个数 | 全部单色 | — | ✅ |
| 四色均衡 | 绿 74 / 红 79 / 黄 70 / 蓝 77 | — | ✅ |
| 说法种数 | 33 种，规范句 80.7% | 505 种，34% | ✅ |
| `official_score`（每次按压） | 9.320 | 9.280 | ✅ |
| 最佳帧到按钮 site 距离 | 70.7 mm | 71.2 mm | ✅ |
| 瞄准误差 | 15.5 mm | 16.1 mm | ✅ |
| 按对目标按钮 | 300 / 300 | 99.7% | ✅ |
| 基座漂移 | 中位 0.30 / 最大 0.70 mm | < 1 mm | ✅ |
| 首帧右臂位置 | `[0.5363, -0.3067, 0.2871]` | `[0.5390, -0.3110, 0.2855]` | ✅ 差 5.3 mm |

**两处与 v4 的差异，不影响正确性但要知道**：

1. **控制步/帧中位 5.53**（v4 是 7.28，范围 3.94~9.90 与 v4 的 4.46~8.90 大幅重叠）。
   墙钟门控下这个值随机器当时的快慢浮动，**直接决定评测的 `--action_repeat`**，见第 4 节。
2. **首帧姿态几乎零抖动**（跨集标准差 0.01 mm，v4 是 2.65~4.04 mm）。v4 那点抖动来自长时间
   连跑的仿真漂移，不是刻意设计；补采按块重启仿真所以更整齐。初始条件多样性略低，影响很小。

---

## 2. v5 训练集怎么合成的

```bash
python scripts/dataset/build_training_set.py \
    --base <v4目录> --supplement <补采目录> --keep-4button 150 --out <v5目录>
```

> Windows 上加 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`，否则脚本打印 ✅ 时会被 GBK 控制台编码中断。

### 最终分布（按帧，这才是模型看到的比例）

| | 集数 | 帧数 | 占比 | v4 | 目标 |
|---|---|---|---|---|---|
| 单色（1 键） | 431 | 164.1k | **23.7%** | 4.3% | ≥ 13% |
| 2 键 | 171 | 116.3k | 16.8% | 11.7% | — |
| 3 键 | 220 | 218.2k | 31.4% | 21.9% | — |
| 4 键 | 150 | 195.3k | **28.1%** | 62.2% | ≤ 40% |
| **合计** | **972** | **693,792** | | | |

体积 4.0 GB，399 种指令，其中规范单色句 289 集（蓝 78 / 绿 76 / 黄 68 / 红 67）。

### 结构验证（已通过）

- 972 个 parquet、1944 个 mp4、`episodes.jsonl` 972 行，与 `info.json` 三处总数一致
- `index` 列全局连续，`episode_index` 与 `length` 逐集一致
- schema 与 v4 相同：state/action 均 18 维、fps 20、AV1 编码

---

## 3. 训练侧步骤

```bash
# 1) 放到配置里的 repo_id 位置
ln -sfn <v5目录> $HF_LEROBOT_HOME/local/g1_omnipicker_button_combo

# 2) 必须重算归一化统计（数据集换了，动作分布变了）
cd openpi     # 已按 openpi_patches/g1_button/README 打过补丁
uv run scripts/compute_norm_stats.py --config-name pi05_g1_button_lora
#   如果这一步慢得离谱，用 SouthGrid 的 scripts/openpi/compute_norm_stats_fast.py（跳过视频解码）

# 3) 训练
uv run scripts/train.py pi05_g1_button_lora --exp-name g1_button_v5 --overwrite \
    --batch-size <显存能放下的值，24GB 卡建议 16>
```

---

## 4. 评测：两个参数必须按 v5 重设

### `--action_repeat = 6`

评测时每个预测动作保持多少个控制步，应当等于训练数据里"一个记录帧跨多少控制步"：

| 口径 | 中位值 |
|---|---|
| v5 全部帧（按帧加权） | 7.16 |
| **v5 单色帧** | **5.54** |
| v4 全集（旧值） | 7.28 |

**评测走单色分段（见下），所以取单色口径，用 `--action_repeat 6`**，并试 5 和 7 取最好的。
沿用旧的 7 会偏快约 25%；沿用官方指南示例的 `1` 会偏快 6 倍，机器人根本够不到按钮。

### 指令要用规范句

v5 里样本最密集的是这四句，**评测就用它们**：

```
按红色按钮   按绿色按钮   按黄色按钮   按蓝色按钮
```

（不是"请按下红色按钮"——那是模板里的第 3 条，样本少得多。官方指南 4.7.2 节的示例也正是"按红色按钮"。）

### 完整评测命令

**先单色**，确认基础能力（数据上限是每次按压 9.3 分）：

```bash
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <策略服务器> --port 8010 \
    --prompt "按红色按钮" --action_repeat 6 --max_steps 3000
```

**再四色分段**，拿 40 分（`--max_steps` 是每段预算）：

```bash
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <策略服务器> --port 8010 \
    --prompts "按绿色按钮" "按红色按钮" "按黄色按钮" "按蓝色按钮" \
    --action_repeat 6 --max_steps 3000
```

**分段是必须的，不是可选项**：pi0.5 每步只看当前一帧、没有历史，按钮按完弹回、画面不留痕迹，
单条完整顺序的指令它无法判断进行到第几个。脚本内置 `run_segment` 监测按钮关节位移自动切段，
**评分仍由官方 ScorerClient 判定**，`targets` 与单条指令时完全相同。
v5 把训练重心放到单色，正是为了配合这个评测方式——两者是一套组合拳，只做其一拿不到分。

---

## 5. 训练完先做离线验证

不要直接上仿真器。先确认语言接地已修复：

```bash
python scripts/dataset/lang_grounding_probe.py --ckpt <新 checkpoint> --dataset <v5目录> --per_color 6
```

- 单色接地正确率 **≥ 85%** → 可以上仿真器
- 仍 **< 40%** → 分布还不够，加大单色补采 / 再压低 `--keep-4button`，重训

也可以用 `scripts/openpi/offline_eval.py`，它额外给出"四色指令预测目标最大差"
（按钮相距十几厘米，**应为上百毫米**；小于 10 mm 说明仍在忽略语言）。

---

## 6. 已知待修（不阻塞）

- `build_training_set.py` 把所有集写进 `chunk-000` 且 `total_chunks=1`；合并后若超过 1000 集，
  LeRobot 会去找不存在的 `chunk-001`。当前 972 集安全。
- 该脚本不校验补采与 base 的 features / fps / 编码是否一致，直接沿用 base 的 `info.json`。
- 输出不带 `quality.jsonl`，后续再分析 v5 时按钮数只能靠解析中文任务串。

---

## 7. 只传补采数据，训练侧本地合并（推荐）

v5 整包 4.0 GB，而**补采数据只有 699 MB**。训练机上已经有 v4，所以只需传补采包，
在那边执行同一条合并命令即可得到**逐字节等价**的 v5——合并脚本用固定随机种子
（`random.Random(0)`）挑选保留哪 150 个四键集，结果可复现。

补采数据集：

| 项 | 值 |
|---|---|
| 目录 | `g1_button_v4_supp` |
| 体积 | **699 MB** |
| 内容 | 300 集 / 121,329 帧 / 600 个 mp4，`meta/` 五个文件齐全 |
| repo_id | `local/g1_button_v4_supp` |
| 格式 | LeRobot v2.1，state/action 18 维，fps 20，AV1，480×640 两路相机 |

训练机上收到后：

```bash
python scripts/dataset/build_training_set.py \
    --base <v4目录> --supplement <收到的补采目录> --keep-4button 150 --out <v5目录>
```

看到 `972 集 / 693792 帧`、`单色帧 23.7%`、`四键帧 28.1%` 就与本机构建的一致。

---

## 8. 该重训还是在 v4 的 checkpoint 上接着微调？

**建议从 pi0.5 基座重训 v5，不要在 v4 微调结果上继续训，更不要只用补采数据继续训。**

三条理由：

1. **归一化统计变了，旧 checkpoint 的输入编码不再匹配。** 换数据集必须重算 norm stats
   （q01/q99 会变），而 v4 的 checkpoint 是在旧统计下学出来的。喂进不同归一化的状态与动作，
   等于悄悄改变了它的输入分布。
2. **v4 的权重已经塌缩到"忽略语言"那个解。** 这次修的是**训练帧的混合比例**——
   这个机理只有在"从头按新比例学"时才成立。从已经丢掉语言通道的权重出发，
   很可能仍停在原来的局部解里。
3. **只用补采数据继续训会过拟合。** 121k 帧、4 个目标、33 种说法，训久了模型会记住这几句话
   而不是学会读颜色词；四键/三键的画面也全部消失，序列场景下的表现会退化。

重训的代价可控：LoRA 微调本来就是从基座起步的短流程，v5 的帧数（694k）比 v4（997k）还少 30%。

如果时间允许，可以**并行**做一次热启动对照（把 `weight_loader` 指向 v4 checkpoint 的 `params`、
在 v5 混合数据上继续训、步数减半），和重训比一比接地正确率。但**主线走重训**，
因为它可预测、可复现，也是本次分析的前提。
