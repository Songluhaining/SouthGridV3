# g1_button_v4 补采与再训练指引（修复"模型忽略语言"）

> ⚠️ **本文所述方案已被取代。当前有效方案见 [g1_button_v6_single_color_plan.md](g1_button_v6_single_color_plan.md)。**
> 本文保留是为了记录当时的分析与实测数字，其中的采集/训练/评测参数**不要再照做**。

写给**在 4060 Ti 机器上采集这批数据的同学**。第一次训练出的模型**完全忽略颜色指令**，
本文说明为什么、要补采什么、以及如何构建下一版训练集**确保有效**。所有结论都有实测数字支撑。

---

## 1. 问题：模型学成了"忽略语言"（实测）

用离线脚本（不开仿真器，直接拿数据集里的画面喂模型，`scripts/dataset/lang_grounding_probe.py`）
测两代模型对颜色指令的响应：

| 指标 | v3（旧数据，能用） | **v4（这批，坏）** |
|---|---|---|
| 单色接地正确率<br>（说"绿色"时预测目标是否真的最靠近绿色按钮） | **96%**（23/24） | **25%（6/24）= 四选一随机** |
| 四色指令预测目标最大差 | 126 mm（语言充分起作用） | **9 mm（< 10 = 完全忽略语言）** |

闭环里的直接后果：v4 模型不管指令说红/绿/黄/蓝，右臂**一律走向红色按钮附近**然后停住，
四个单色只有红色方向对、其余全错；多按钮任务因为读不出顺序也一起垮
（pi0.5 无历史记忆，按钮按完弹回、画面不留痕迹，**顺序的唯一线索就是语言**）。

---

## 2. 根因：训练帧的分布，不是采集质量

采集质量本身很好（按压成功率 99.7%、每次按压官方分中位 9.28）。问题在**按钮数分布**——
模型是按"帧"采样训练的，要看**帧占比**，不是集数占比：

| 按帧占比 | v3（能用） | **v4（坏）** |
|---|---|---|
| 单色（1 键）帧 | 12.8% | **4.3%** |
| 四键（全按）帧 | 37% | **62%** |

**机理**：四键"全按"任务里，"下一步按哪个"光看画面（哪些还没按）就能预测，
**语言是冗余的**。当 62% 的训练帧都是这种"不用读语言也能对"的帧时，
梯度下降会走捷径、把语言彻底丢掉（模仿学习里已知的 causal confusion）。
v3 的单色帧是 v4 的 3 倍、冗余四键帧少得多，所以保住了语言接地。

> **注意**：单色集其实有 131 个，不算少——问题不是"样本太少学不会"，
> 而是**被 62% 的四键冗余帧淹没**。所以修法必须两手抓。

---

## 3. 修法：补采单色 + 下采样四键（缺一不可）

**光补采压不下四键占比**：四键有 62%（62 万帧），补 400 集单色后四键仍占 55%。
必须**同时**在构建训练集时丢掉大部分 v4 的四键集。下面两步一起做。

### 推荐配比（已用真实数据验证达标）

| 方案 | 补采 | 保留 v4 四键 | 结果（单色帧 / 四键帧） |
|---|---|---|---|
| **B（推荐，省采）** | 300 单色集 | 150 / 478 | 20.1% / 29.4% ✅ |
| A（更稳） | 400 单色 + 100 双色 | 200 / 478 | 20.0% / 31.7% ✅ |

目标是单色帧 ≥ 13%、四键帧 ≤ 40%（对齐能工作的 v3）。先用工具规划，不用反复试：

```bash
# 在装了 openpi 的机器上（有 pyarrow 即可），只看分布不落盘：
python scripts/dataset/build_training_set.py --base <v4数据集目录> \
    --keep-4button 150 --sim-single 300 --report-only
# 输出会打印"分布达标 ✅/未达标 ⚠️"，可调 --keep-4button / --sim-single 试到达标
```

---

## 4. 第一步：补采单色数据（4060 Ti 机器上）

用现有的组合采集脚本，**只采单按钮、用规范说法**。关键参数：

```bash
cd src/examples/dataCollection/g1_omnipicker
python g1_omnipicker_collection_scripted_button_combo_lerobot.py \
    --lerobot_out <补采输出目录，例如 ~/datasets/g1_button_v4_supp> \
    --repo_id local/g1_button_v4_supp \
    --episodes 350 \
    --max_buttons 1 \
    --canonical_ratio 0.8 \
    --fps 20 --clock wall \
    --success_mode press
    # 其余参数（轨迹档案/相机/布局）与你们采 v4 时保持一致
```

- `--max_buttons 1`：**只采单色集**（四色会被均衡轮采，各约 1/4）。
- `--canonical_ratio 0.8`：80% 用规范说法"按X按钮"，把语言监督**收敛集中**
  （v4 的 505 种说法太散，每种中位仅 1 条，也削弱了语言学习）。
- `--episodes 350`：目标净留 ~300 集（按 v4 约 85% 保留率留余量）。
- `--clock wall --fps 20`：**与 v4 保持一致**，避免混入两种帧间时序
  （评测时 `--action_repeat 7` 才对得上；`--clock sim` 是下一批全新采集时才换）。

补完确认集数：`cat <补采目录>/meta/info.json` 里 `total_episodes` 应 ≥ 300。

---

## 5. 第二步：构建训练集（补采 + 下采样 v4）

`build_training_set.py` 把"v4 下采样四键"与"补采单色"合并成一份新数据集，
自动重建索引/元数据，并打印最终分布。已用 LeRobotDataset 验证产出可正常加载。

```bash
python scripts/dataset/build_training_set.py \
    --base <v4数据集目录> \
    --supplement <补采目录> \
    --keep-4button 150 \
    --out <训练集输出目录，例如 ~/datasets/g1_button_v5>
```

看到 `>>> 分布达标 ✅` 即可。视频按文件复制、不重编码，几分钟完成。

---

## 6. 第三步：重算 norm stats 并训练

数据集换了，**必须重算归一化统计**（否则动作幅度会错）：

```bash
# 把训练集软链为配置里的 repo_id
ln -sfn <训练集输出目录> <HF_LEROBOT_HOME>/local/g1_omnipicker_button_combo
cd openpi   # 已按 openpi_patches/g1_button/README 打过补丁
uv run scripts/compute_norm_stats.py --config-name pi05_g1_button_lora --max-frames 40000
uv run scripts/train.py pi05_g1_button_lora --exp-name g1_button_v5 \
    --batch-size <你显存能放下的值，24GB 建议 16> --num-train-steps 30000 --overwrite
```

---

## 7. 训练完先做离线验证，再开仿真器评测

**不要**直接上仿真器。先用离线探针确认语言接地已修复（几分钟、不需要 OrcaLab）：

```bash
python scripts/dataset/lang_grounding_probe.py \
    --ckpt <新 checkpoint 目录> --dataset <训练集输出目录> --per_color 6
```

- **单色接地正确率 ≥ 85%** → 语言学回来了，可以上仿真器评测（`--action_repeat 7`，单色先测）。
- 若仍 < 40% → 分布还不够，加大补采单色集数 / 再压低 `--keep-4button`，重训。

---

## 8. 一页速查

| 步骤 | 命令要点 |
|---|---|
| 规划分布 | `build_training_set.py --base v4 --keep-4button 150 --sim-single 300 --report-only` → 要 ✅ |
| 补采 | `..._button_combo_lerobot.py --max_buttons 1 --canonical_ratio 0.8 --episodes 350` |
| 建训练集 | `build_training_set.py --base v4 --supplement 补采 --keep-4button 150 --out v5` → 要 ✅ |
| 重算 stats + 训练 | `compute_norm_stats.py` 后 `train.py ... --exp-name g1_button_v5` |
| 验证 | `lang_grounding_probe.py --ckpt 新ckpt` → 单色接地 ≥ 85% 才上仿真器 |

**一句话**：v4 因为四键任务占 62% 帧、把模型压成"忽略语言"（接地 25%=随机）。
补 ~300 单色集 + 下采样四键到 150，让单色帧回到 20%、四键降到 29%（比能用的 v3 还健康），
重训后离线接地应回到 85%+。修的是**分布**，不是采集质量。
