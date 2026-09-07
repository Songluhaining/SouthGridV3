# g1_button_v4 数据集说明与评测指引

本文写给**在另一台机器上用这批数据做训练与评测的同学**。数据在本机（Windows + RTX 4060 Ti）采集，
评测通常在另一台带 GPU 的机器上跑。文中的数字都是从数据集自带的 `meta/` 统计出来的，
**其中"采集时序"一节直接决定评测命令怎么写，务必先读**。

数据集仓库名：`local/g1_omnipicker_button_combo`（目录名 `g1_button_v4`）。

---

## 1. 数据集概况

| 项 | 值 |
|---|---|
| episode 数 | 1000（采了 1006，按质量门槛留下 1000） |
| 总帧数 | 997,442 |
| 声明帧率 | `info.json` 里 `fps = 20` |
| 相机 | `cam_head`（头部，端口 7090）、`cam_wrist_r`（右腕，端口 7080），480×640 |
| 视频编码 | AV1（NVENC 直出） |
| state / action | 均为 18 维，`action[i] = state[i+1]`（绝对位姿，不是增量） |
| 场景布局 | `src/examples/dataCollection/g1_omnipicker/g1_button.json` |
| 采集脚本 | `g1_omnipicker_collection_scripted_button_combo_lerobot.py`（批量驱动见 `auto_collect_windows.py`） |

### 18 维 state 的构成

```
[0:3]   左手位置 x,y,z          [3:7]   左手四元数 x,y,z,w
[7:10]  右手位置 x,y,z          [10:14] 右手四元数 x,y,z,w
[14:16] 左夹爪两通道            [16:18] 右夹爪两通道
```

**采集时左臂是锁定的**：第 0~6 维在一集之内恒定，跨集之间只有毫米级差异（标准差 0.001~0.013）。
第 14~17 维（两个夹爪）**在整个数据集里恒定不变**（左爪 0.33333，右爪 1.0），标准差为 0。
真正携带信息的只有第 7~13 维（右手位姿）。

右手实际工作范围（1%~99% 分位）：`x 0.543~0.821`、`y -0.368~-0.082`、`z 0.202~0.385`（米）。
评测时如果模型输出的目标落在这个范围之外，多半是链路出了问题。

---

## 2. 采集时序：**决定 `--action_repeat` 该设多少**

采集脚本的 `--clock` 默认是 `wall`（墙钟），即每隔 1/20 **真实**秒记一帧。
但仿真跑得比真实时间快，所以记录帧之间的**仿真时长不是 50 毫秒**：

| 指标 | 中位数 | 范围 |
|---|---|---|
| 每个记录帧对应的**控制步数** | **7.28** | 4.46 ~ 8.90 |
| 按仿真时间算的真实帧率 | 27.5 | 22.5 ~ 44.8 |
| 相邻帧之间的仿真时长 | 36.4 毫秒 | 22 ~ 44 毫秒 |
| 每按一个按钮所需控制步数 | 约 2150 | — |

（算法：`meta/quality.jsonl` 每集的 `metrics.duration_steps` ÷ `meta/episodes.jsonl` 同集的 `length`。）

**推理端的对应关系**：`eval_g1_omnipicker_button_lerobot.py` 里每个预测动作会被保持
`--action_repeat` 个控制步。要复现数据里的运动速度，这个值应当等于上表的 7.28。

- 设成 **1**（官方指南 4.7.2 节示例里的值）→ 相当于要求机器人在 1 步内走完数据里 7 步的距离，
  OSC 控制器追不上，表现为**手臂朝正确方向动但幅度严重不足、够不到按钮**。
- 设成 **7**（推荐）或脚本默认的 **10** → 与数据接近。建议两个都试一次取好的。

`--action_repeat` 数的是控制步（仿真时间），**与评测机器的快慢无关**，所以只要设对，
在 4090 还是别的卡上评测结果都一致。

> **重新采集时的建议**：把 `--clock` 改成 `sim`。上表的"控制步数"波动范围是 4.46~8.90，
> 差了整整两倍——这是采集机器忙闲不均造成的，会让同样的画面对应差两倍的动作位移，
> 模型只能学一个平均值。用仿真时钟门控可以让帧间时长严格恒定。

---

## 3. 任务分布与数据质量

| 按钮个数 | episode 数 | 不同顺序数 | 每种顺序的样本数（中位） |
|---|---|---|---|
| 1 键 | 131 | 4 | 37 |
| 2 键 | 171 | 12 | 14 |
| 3 键 | 220 | 24 | 9 |
| 4 键 | 478 | 24 | 21 |

**指令文本共 505 种，中位数是每种说法只有 1 条样本。** 语言侧监督非常稀疏，
模型对没见过的说法泛化能力有限；相比之下单键指令的样本最密集。

按压质量（`meta/quality.jsonl`，共 3060 次按压）：

| 指标 | 结果 |
|---|---|
| 按中目标按钮 | 3052 / 3060 = 99.7% |
| 每次按压的 `official_score` | 中位 **9.28**，范围 8.85 ~ 9.68 |
| 4 键 episode 的分数合计 | 约 **36.97** |

任务满分 40 分（4 个按钮各 10 分）。**数据本身每次按压是 9.28 分，这就是模仿学习的上限**——
评测拿到 36 分左右即为正常，拿到十几分说明只完成了 1~2 个按钮。

---

## 4. 训练配置

配置名 `pi05_g1_button_lora`，定制代码在本仓库 `openpi_patches/g1_button/`（新机器上起服务前必须先打补丁）。

| 项 | 值 |
|---|---|
| 模型 | pi0.5（`pi05=True`），LoRA 微调 |
| batch_size | 32（**24 GB 的卡放不下，4090 上需要减小**） |
| num_train_steps | 30000 |
| 学习率 | 余弦衰减，预热 1000 步，峰值 5e-5，末值 5e-6 |
| 动作序列长度 | 10 |
| 归一化 | 分位数归一化（`use_quantile_norm=True`，只用 q01/q99） |

> **归一化统计的一个坑**：openpi 用 32 位浮点累加算标准差，对本数据集第 3 维
> （左手四元数 x，均值 −0.673、真实标准差仅 0.0023）会因舍入误差算出 **0**。
> pi0.5 走分位数归一化、不读标准差，所以不受影响；但若换用"减均值除标准差"的配置，
> 必须先用 64 位浮点重算，否则该维会被放大约两千倍。重算脚本见 `scripts/openpi/`。

---

## 5. 评测怎么跑

### 5.1 起策略服务（GPU 机器）

需要 Linux + 至少 10 GB 显存（模型权重进显存约 7.4 GB，加激活约 10 GB）。
openpi 的 JAX GPU 版不支持原生 Windows，Windows 上需要 WSL2。

```bash
cd /path/to/openpi          # 已按 openpi_patches/g1_button/README 打过补丁
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py --port 8010 policy:checkpoint \
    --policy.config=pi05_g1_button_lora \
    --policy.dir=<checkpoint 目录，例如 checkpoints/pi05_g1_button_lora/exp01/10000>
```

等待打印 `server listening on 0.0.0.0:8010`。推理只需 checkpoint，不需要基座权重，
归一化统计已随 checkpoint 的 `assets/` 一起保存。

### 5.2 跑评测（装了 OrcaLab 的机器）

前提：OrcaLab 7.3 已启动并加载 `g1_button.json` 布局，相机端口按采集文档配好（7090 头部 / 7080 右腕），仿真在运行。

**第一步：先测单按钮**，确认基础能力：

```bash
conda activate orcalab_lerobot
cd <SouthGrid>/src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <策略服务器地址> --port 8010 \
    --prompt "请按下红色按钮" \
    --action_repeat 7 \
    --max_steps 3000
```

单按钮应能稳定按中、拿到 9 分左右。**做不到就先别测多按钮**，问题在训练或链路上。

**第二步：多按钮用 `--prompts` 分段**，一段一个颜色：

```bash
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <策略服务器地址> --port 8010 \
    --prompts "请按下绿色按钮" "请按下红色按钮" "请按下黄色按钮" "请按下蓝色按钮" \
    --action_repeat 7 \
    --max_steps 3000
```

**为什么要分段**：pi0.5 每一步只看当前一帧、没有任何历史。给一条完整顺序的指令时，
它无法判断"进行到第几个了"——按钮按完会弹回，画面上没有痕迹，唯一线索只有手臂当前位置，
非常脆弱，典型表现是按完第一个后反复回到第一个或停住。

脚本内置了分段机制（`run_segment`）：监测到当前段的按钮关节位移达标就自动切到下一条指令。
**评分仍然完全由官方 ScorerClient 判定**，传给 `start_attempt` 的 `targets`
与用单条完整指令时一模一样，改变的只是"该按哪个"这个调度放在模型里还是放在脚本里。
指南第 1.1 节允许自行编写控制逻辑。

用 `--prompts` 时 `--max_steps` 是**每段**的步数预算，不是总量。

### 5.3 参数速查

| 参数 | 建议值 | 理由 |
|---|---|---|
| `--action_repeat` | **7**（或试 10） | 数据每记录帧对应 7.28 个控制步 |
| `--max_steps` | **3000** | 数据里每按一个按钮约 2150 个控制步 |
| `--prompt` / `--prompts` | 多按钮任务用 `--prompts` 分段 | 策略无历史，序列任务会卡住 |
| 指令用词 | 用训练数据里出现过的说法 | 见 `meta/tasks.jsonl` |

### 5.4 网络

评测脚本（连 OrcaLab 那台）主动去连策略服务器，方向是单向的。
每次推理上传两张 480×640 图约 1.84 MB，约每秒 2 次，合计约 3.7 MB/s：
千兆局域网可忽略；跨公网需要约 30 Mbps 稳定上行，否则会成为瓶颈，
建议用 ZeroTier / Tailscale 之类的虚拟局域网把两台机器直连。

---

## 6. 不开仿真器也能做的离线检查

`scripts/openpi/offline_eval.py` 用数据集里录好的画面和状态喂模型，比对预测动作与真实动作，
**不需要 OrcaLab**，几分钟出结果：

```bash
uv run python scripts/offline_eval.py \
    --ckpt checkpoints/pi05_g1_button_lora/<exp>/<step> \
    --dataset <HF_LEROBOT_HOME>/local/g1_omnipicker_button_combo
```

输出两组数字：

- **右手位置误差 vs 同期真实位移**：相对误差 20% 以内说明动作学得不错，60% 以上说明基本没学会。
- **语言敏感性**：同一帧画面换四种颜色指令，看预测目标差多少。按钮之间相距十几厘米，
  所以差异应当是**上百毫米**；**小于 10 毫米说明模型在忽略语言**，
  这种情况再训练也救不了，要从数据侧解决（收敛指令说法、加大单键样本量）。

---

## 7. 排查清单

| 现象 | 先查 |
|---|---|
| 手臂朝正确方向动但够不到按钮 | `--action_repeat` 是不是设成了 1 |
| 还没走到按钮就结束 | `--max_steps` 是不是太小（需要 ≥2000/段） |
| 按完第一个就卡住或反复按第一个 | 多按钮任务是否用了 `--prompts` 分段 |
| 完全不动 / 动作里有非有限值 | 看日志的 `[调试] ... 首个动作块` 那行，位移接近 0 或有非有限值即为链路问题 |
| 起服务报找不到配置 | `openpi_patches/g1_button/` 的补丁没打 |
| 动作幅度、方向都乱 | 归一化统计是否与训练时同一份（`assets/` 必须随 checkpoint 走） |
| 画面对不上 | OrcaLab 布局 JSON、相机端口（7090/7080）、分辨率是否与采集时一致 |
