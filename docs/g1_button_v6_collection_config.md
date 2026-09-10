# g1_button_v6 采集配置速查 + 当前排查状态

给训练/评测侧看的一页：**v6 到底是怎么采出来的**（每个参数、包括没传因而落到默认值的那些），
以及判断"模型有没有学会读颜色"所需的实测基准数据。

方案背景见 [g1_button_v6_single_color_plan.md](g1_button_v6_single_color_plan.md)。

---

## 1. 采集用的确切命令

在采集机（Windows + RTX 4060 Ti，OrcaLab 已加载 `g1_button.json`）上：

```bash
cd src/examples/dataCollection/g1_omnipicker
python auto_collect_windows.py --out <输出目录> --total 800 --chunk 25 \
    --max_buttons 1 --length_weights 1,1,1,1 \
    --extra "--canonical_ratio 1.0 --clock sim \
             --steps_approach 200 --steps_retract 120 --steps_hold 150 \
             --repo_id local/g1_button_v6"
```

`auto_collect_windows.py` 负责分块、块间用 MCP 重启仿真、按官方口径做健康检查、断点续采；
真正采集的是 `g1_omnipicker_collection_scripted_button_combo_lerobot.py`。

### 显式传的参数

| 参数 | 值 | 作用 |
|---|---|---|
| `--max_buttons` | **1** | 只采单色单次按压 |
| `--canonical_ratio` | **1.0** | 100% 用规范句 `按X按钮`，与评测 `--auto_segment` 生成的文本逐字一致 |
| `--clock` | **sim** | 帧由仿真时间门控 → 帧间隔恒定 **10 个控制步**（20fps × 5ms） |
| `--steps_approach` | **200** | 默认 700，压缩以提速 |
| `--steps_retract` | **120** | 默认 600 |
| `--steps_hold` | **150** | 默认 350 |
| `--total` | 800 | 目标保留集数（数的是保留数，不是尝试数） |

### 没传、因而落到默认值的参数（同样重要）

| 参数 | 生效值 | 说明 |
|---|---|---|
| `--success_mode` | **official** | 瞄准按钮 site 前方 `target_site_dist=0.047`，`press_depth=0`。**不要改成 `press`**，那会多推 4 厘米，而官方得分曲线顶点在 0.05m，压深反而扣分 |
| `--fps` | 20 | |
| `--target_site_dist` | 0.047 | |
| `--jitter` | 1.0 | 内置轨迹随机化，默认强度 |
| `--trajectory_archive` | **未使用** | 见第 2 节 |
| `--elite_top_frac` | 1.0 | 同上，未生效 |
| `--pose_candidates` | `pose_g1_button_candidates.yaml` | 与 v4 相同 |
| `--seed` | None | 每次随机 |

### 场景与相机

- 布局 `src/examples/dataCollection/g1_omnipicker/g1_button.json`，机器人基座
  `[18.912, -23.007, 0.203]`、绕 Z 轴 90°。
- **采集全程没有运行 `orcalab_camera_prep.sh`**。腕相机 `camera_right` 的
  `rotate` 保持 **(0, 0, 0)**（布局的 `entity_overrides` 里只有 `NearClipDistance`/`ColorPort`/`IsRecording`，
  不含朝向）。在 (0,0,0) 下腕相机正对面板，面板+夹爪同框。
- **评测时必须加 `--no_camera_prep`**：该脚本会把腕相机转成 `(90,180,0)`，
  那是 v3/v4 时代标定的值，在当前资产状态下会把相机转开、只看到草地和自身手臂。

---

## 2. 没有使用 QD 轨迹档案（已知缺口）

仓库里带着 `button_traj_archive_v2.json`（MAP-Elites 档案）和生成它的
`evolve_button_trajectories.py`，采 v6 时**没有传 `--trajectory_archive`**，
用的是内置随机弯曲。实测两者的覆盖差异：

| 维度 | v6 内置随机（800 集实采） | QD 档案覆盖 |
|---|---|---|
| `bow_y` | −0.060 ~ +0.060（599 个不同值） | **−0.13 ~ +0.13（约 2 倍宽）** |
| `bow_z` | −0.020 ~ +0.070（528 个） | **−0.07 ~ +0.15（约 2 倍宽）** |
| `time_scale` | 1.00 ~ 1.35（256 个） | **0.6 ~ 1.5** |
| `approach_back` | 0.090 ~ 0.160（482 个） | — |
| 形状粒度 | 连续随机 | **每色仅 3 个精英格**（8×8×4=256 格只填了 3 个） |

结论：**内置随机粒度更细，QD 档案幅度约宽一倍**。少用它确实少了约一倍的状态覆盖宽度；
但档案本身也没跑充分（256 格填了 3 格），下次用之前建议先重跑
`evolve_button_trajectories.py` 把档案填满。

另外 v6 的**首帧姿态跨集标准差是 0.00 mm**——每集起点完全相同。这是"每段都从同一预备位姿开始"
的设计所致，但也意味着策略从没见过偏离标称起点的状态。

---

## 3. 判断"语言有没有学会"所需的基准数据

从 v6 的 800 集里取每集**右手探入最深**（x 最大）那一帧，得到四色的真实目标：

| 颜色 | 目标位置 (x, y, z) 米 | 集间标准差 |
|---|---|---|
| red | `0.7872, -0.0826, 0.3680` | ~3 mm |
| green | `0.7867, -0.1642, 0.2922` | ~3 mm |
| blue | `0.7880, -0.2462, 0.2152` | ~3 mm |
| yellow | `0.7849, -0.3292, 0.2930` | ~3 mm |

**两两间距：**

```
red-green   111 mm      green-yellow 165 mm
red-blue    224 mm      green-blue   113 mm
red-yellow  258 mm      yellow-blue  114 mm
```

**这就是判据**：模型在预备位姿下换四种颜色指令，预测目标之间的间距**应当接近 111~258 mm**。
如果只有几十毫米，说明指令没有真正改变它要去的地方。

按 z 排序是 **red 0.368 > yellow 0.293 ≈ green 0.292 > blue 0.215**——最低的是 blue，不是 yellow。

---

## 4. 闭环实测（v6 checkpoint 11999）

三次独立实验，起点都正常（到红色按钮帽约 280 mm）：

| 运行 | `--action_repeat` | 停在（指尖 / 官方口径） |
|---|---|---|
| 4060 Ti 仿真 + 4090 推理 | 10 | 101 mm / 144 mm |
| 4060 Ti 仿真 + 4090 推理 | 30 | 100 mm / 143 mm |
| 4090 本机 | 10 | 99 mm / — |

对照：v6 训练数据是 **指尖 29 mm / 官方口径 70.6 mm**，官方分 9.32。

已排除的原因：

- **相机视角**：评测送给模型的腕相机图与训练首帧逐像素比对，中位差 3，面板/按钮/夹爪同框 ✅
- **提前判定跳出**：`--no_early_stop` 下跑满预算仍停在同一处 ✅
- **执行速率**：`--action_repeat` 10 与 30 结果几乎相同，OSC 滞后不成立 ✅
- **模型不出动作**：首个动作块位移 24→122 mm，末步目标在工作空间内，无非有限值 ✅
- **起点位姿**：起点到按钮 280 mm，与训练一致 ✅

三次实验（两台机器、两种节奏）停在同一点，说明这是**策略闭环的不动点**，不是随机漂移。

---

## 5. 还没定案的那一步：语言接地必须在第 0 帧测

之前用 `scripts/openpi/offline_eval.py` 得到的两个数字要分开看：

- **动作精度 15%（好）** —— 这一项喂的是**中途帧**。走到一半时手臂已经朝某个按钮去了，
  状态本身就决定了下一步，**语言几乎不起作用**。所以它只能说明"模型会把已经开始的动作接着做完"，
  **不能说明它知道该去哪个按钮**。
- **语言敏感性** —— 旧版脚本在**随机抽到的那一帧**上做这个测试，很可能抽到中途帧，
  那种情况下差异小是必然的。**所以旧结果无效，两个方向的结论都不能下。**

已修正：`scripts/openpi/offline_eval.py` 的语言测试现在**固定在第 0 帧（预备位姿）**——
四个按钮都在视野里、只有指令能区分该去哪个——并把四次预测与第 3 节的真实目标逐一比对，
直接输出"最接近哪个按钮 / 距指定按钮多远 / 四次预测的最大间距"。

```bash
python scripts/openpi/offline_eval.py \
    --ckpt <v6 checkpoint 目录> --dataset <v6 数据集目录> \
    --episodes 15 --per_episode 4 --ground_episodes 8
```

判读只有两种结果：

| 输出 | 含义 | 下一步 |
|---|---|---|
| 接地正确率接近 100%、四次预测间距接近 111~258 mm | 语言没问题 | 问题在闭环状态漂移，去补起点扰动 / 纠正数据 |
| 接地正确率接近 25%（随机）、间距只有几十毫米 | **模型没在读颜色** | 补再多纠正数据也没用，得回到语言接地本身 |

**在这个结果出来之前，不要开始下一轮采集。** 两种情况要采的数据完全不同。
