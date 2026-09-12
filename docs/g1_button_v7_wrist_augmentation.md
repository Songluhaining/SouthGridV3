# v7 训练：给腕相机加几何增强

承接 [g1_button_v6_eval_image_ood.md](g1_button_v6_eval_image_ood.md)。那篇定位到「闭环失败全部由腕相机
图像分布外造成」，但没找到代码级的落点。**落点找到了，在 openpi 上游。**

---

## 1. 上游默认跳过腕相机的几何增强

`src/openpi/models/model.py` 的 `preprocess_observation`：

```python
if train:
    transforms = []
    if "wrist" not in key:                      # ← 凡是 key 里含 "wrist" 的相机全部跳过
        transforms += [
            augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
            augmax.Resize(width, height),
            augmax.Rotate((-5, 5)),
        ]
    transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
```

我们的策略变换把相机映射成 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`
（见 `openpi_patches/g1_button/g1_button_policy.py`），所以**右腕相机在整个训练过程中
只做了颜色抖动，一次几何扰动都没见过**。

上游这么做有它的道理：腕相机视角与末端位姿强耦合，几何扰动会破坏图像与状态向量的对应关系。
但对按钮任务，腕相机是唯一能看清按钮的通道，代价是策略对它的视角零容差。

## 2. 与实测完全对上

| 换哪一路图像（同状态、同指令、第 0 帧） | 预测距真值 |
|---|---|
| 换头相机（训练时做过 ±5% 裁剪 + ±5° 旋转） | **6.9 mm** —— 不受影响 |
| 换腕相机（训练时零几何增强） | **55.4 mm** —— 崩 |

单元测试直接量化了「零容差」：同一张棋盘图重复过增强，量归一化后的几何变化幅度

```
腕相机 增强关闭 (1.00, 0°)      0.0000   ← 上游默认 = v6 训练的实际情况
腕相机 本次设置 (0.96, 3°)      0.6834
头相机 上游固定 (0.95, 5°)      0.9797
```

**`0.0000` 是精确的零。** 模型把那一张腕相机图看了 800 遍，没有任何视角冗余。

## 3. 本次改动

`openpi_patches/g1_button/openpi_wrist_augmentation.patch`（在 openpi 仓库里打）：

- `pi0_config.py`：`Pi0Config` 新增 `wrist_aug_crop` / `wrist_aug_rotate_deg`，**默认 1.0 / 0.0
  即完全保持上游行为**，不影响任何其他任务的配置。
- `model.py`：`preprocess_observation` 接受这两个参数；`"wrist" in key` 的分支下按参数加
  RandomCrop + Resize + Rotate。
- `pi0.py`：`Pi0.__init__` 存下来，`compute_loss` 传进去。
- `training/config.py`：`pi05_g1_button_lora` 里设 **`wrist_aug_crop=0.96, wrist_aug_rotate_deg=3.0`**。

强度的取法：模型分辨率是 224×224，crop 0.96 等效随机平移 ±4.5 px（≈ 原始 640 宽下的 ±13 px），
实测评测帧偏 6~7 px，留约 1.9 倍余量；旋转 ±3° 对实测的 ~0.9° 滚转留 3.3 倍余量。
比头相机的 (0.95, 5°) 温和，因为腕相机还要承担精度。

> 这个补丁是在 fork `southgrid-pi05` 上生成的（该版本已把 `Pi0Config` 拆到 `pi0_config.py`）。
> 若在上游 `fdc03f5` 上打，`Pi0Config` 还在 `pi0.py` 里，对应那一段要手工挪位置。

## 4. A800 上的训练

环境：AutoDL 容器，A800 80GB，openpi fork 在 `/root/southgrid-openpi-pi05`（分支 `button-v7-wrist-aug`）。

```bash
# 数据（800 集 / 61621 帧）
/root/autodl-tmp/southgrid/datasets/g1_button_v6

# 归一化统计：跳过 mp4 解码，4 分 13 秒（官方脚本要解码全部视频）
cd /root/southgrid-openpi-pi05
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/compute_norm_stats_button.py --config-name pi05_g1_button_lora

# 训练
export XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled \
       JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/southgrid/cache/jax \
       OPENPI_DATA_HOME=/root/autodl-tmp/southgrid/cache/openpi
setsid nohup .venv/bin/python scripts/train.py pi05_g1_button_lora \
    --exp-name g1_button_v7_wristaug --overwrite --no-wandb-enabled \
    --checkpoint-base-dir /root/autodl-tmp/southgrid/checkpoints \
    --num-workers 16 --num-train-steps 15000 --lr-schedule.decay-steps 15000 \
    --save-interval 2500 --keep-period 15000 \
    > /root/autodl-tmp/southgrid/logs/button_v7_wristaug.log 2>&1 &
```

### 为什么是 15000 步而不是配置里的 30000

batch 32 × 30000 = 96 万样本 = 15.6 个 epoch，对 800 集的单色窄任务是浪费。
4090 那版在 **11999 步（6.2 epoch）** 时开环精度就已到位（第 0 帧 9.5 mm、四色排序正确）——
它失败的原因是腕相机脆弱，不是欠拟合。**`decay_steps` 必须跟着一起改成 15000**，
否则跑到 15000 停时余弦退火还在高位，不如老实按 15000 退火的结果。

### 速度

**3.1 s/it（batch 32），15000 步约 13 小时。** 这是算力下限，不是配置问题：

- LoRA 只省显存和优化器状态，**不省 FLOPs**——反向仍要穿过完整 3B 参数。
- 每张图经 SigLIP 出 256 token，配置喂 3 路（头部 / **全零占位的左腕** / 右腕）= 768 图像 token，
  加 `max_token_len=200` 的文本，单序列约 1000 token。**其中三分之一是那路全零左腕图，纯浪费。**
- 对照：HPC 的 A100 同配置 3.6 s/it，A800 快 14%，符合预期（A800 砍的是卡间带宽，单卡算力与 A100 相当）。

去掉全零左腕图大约能到 2.4 s/it，但会改变输入结构，与 4090 那版就不是同一条件了。
**本次重训的唯一变量必须是腕相机增强**，所以没动。

## 5. 训练完成后怎么验

**先做离线判据，通过了再上闭环。** 这是 v6 的教训——离线只喂训练帧，看不出问题。

1. 把 checkpoint 起成策略服务，在 4060 Ti 上重跑
   [g1_button_v6_eval_image_ood.md](g1_button_v6_eval_image_ood.md) §2 的 A/B 对照：
   **同状态同指令，只换图像来源。** 判据是「评测帧」与「训练帧」两行的误差差距。

   | 结果 | 含义 |
   |---|---|
   | 两者都在 20 mm 内 | 增强起作用了，可以上闭环 |
   | 评测帧仍比训练帧差 40 mm 以上 | 增强强度不够，加到 (0.95, 5°) 重训 |

2. 通过后再跑分色闭环（每色单独 `--prompt`，`--no_camera_prep`，评测前用 MCP 重启仿真）。
3. **在 v7 实测优于 v3 的 23.13 之前不要提交**——官网只显示最新一次成绩，会覆盖。

---

# v7 实测结果与后续（2026-09-12）

## 6. v7 官方口径评测：11.18 / 40，不如 v3 的 23.13，未提交

四钮一次尝试（`--auto_segment`，指令「依次按下红色、绿色、蓝色和黄色按钮」，每段跑满 6000 控制步）：

| 颜色 | 官方口径距离 | 得分率 | P1 | 得分 |
|---|---|---|---|---|
| 红色 | 141 mm | 0.379 | — | 3.79 |
| 绿色 | 139 mm | 0.389 | — | 3.89 |
| 蓝色 | 158 mm | 0.302 | — | 3.02 |
| 黄色 | 205 mm | 0.135 | **×0.35 非最近** | 0.47 |
| | | | **合计** | **11.18 / 40** |

停位与 v6（143 / ~150 / 152 / 216~225 mm）几乎相同——**增强改善了开环瞄准，闭环一步没动。**

## 7. 停滞的真正机理：停滞点的图像同样分布外

在停滞点（`ee=[0.7035,-0.0867,0.3786]`，距红色目标 84 mm，其中 x 方向差 84 mm、横向只差 2~10 mm）
做四格对照，量「模型想在 0.5 秒内朝按钮推进多少」：

| 条件 | 朝按钮推进 |
|---|---|
| 评测状态 + 评测图像（实况） | **+3.8 mm** |
| 评测状态 + 训练图像 | **+27.6 mm** |
| 训练状态 + 训练图像 | +29.3 mm |
| 训练状态 + 评测图像 | **+3.0 mm** |
| 示范在该处实际推进 | +31.3 mm |

**状态向量无关，图像决定一切。** 停滞不是运动学、不是精度不足，是腕相机图像在深入之后依然分布外。

### 顺带否定了「近按钮信噪比崩塌」

按距按钮分层，在**训练帧**上量 v7 的驱动力：

| 距按钮 | 预测误差 | 示范位移 | 相对误差 | 预测朝按钮 |
|---|---|---|---|---|
| 260mm+ | 26.9 | 88.3 | 30% | +73.4 |
| 180-260 | 16.9 | 83.4 | 20% | +74.8 |
| **130-180（实际停在这）** | 11.0 | 70.9 | **16%** | **+61.1** |
| 90-130 | 10.3 | 44.4 | 23% | +34.5 |
| 0-90（接触） | 15.6 | 32.5 | 48% | +5.1 |

训练帧上该位置驱动力 +61 mm、相对误差仅 16%，**没有任何崩塌**。
（本表是动作块口径；4090 文档的单帧口径 4.4 mm/帧 × 10 = 44 mm，两者数值一致，
但闭环用 `action_repeat 10`，手臂被拉向块末目标，所以决定前进的是块口径。）

### 评测侧没有免费修法

在停滞点对腕图扫几何与频域变换，均无法恢复驱动力：

| 变换 | 最好结果 |
|---|---|
| 平移 ±6/±12 px、旋转 ±2/±4°、缩放 0.94~1.06 | +7.8 mm（缩放 1.03） |
| 高斯模糊 σ=0.5~2.5 | +7.6 mm（**训练图模糊反而 +37.8**，说明不依赖高频） |
| 锐化 | 更差（+2.4~4.1） |
| 加噪 σ=20/255 | **+21.4 mm** ← 唯一有效的，恢复约 2/3 |

差异既不是刚性/仿射变换，也不在高频细节。加噪有效说明模型抓住了某种细微外观结构并据此刹车。

## 8. 归一化缺陷：11 个维度在放大浮点噪声

openpi 的分位归一化是 `(x-q01)/(q99-q01+1e-6)*2-1`。左臂 7 维被锁死、夹爪 4 维恒定，
`q99-q01 == 0`，分母只剩 `1e-6`：

```
原始标准差   左臂 [3.4e-6 3.2e-6 1.2e-6 1.2e-7 1.4e-7 7.8e-8 1.2e-7]   ← float32 舍入噪声
归一化后     左臂 [0.113  0.383  0.181  0.075  0.152  0.085  0.073 ]
             右臂 [0.618  0.628  0.592  0.525  0.567  0.526  0.595 ]   ← 真实信号
```

**噪声幅度达真实信号的 12%~61%，7 个动作维度在拟合不可学的噪声。**
这部分的损失下限约 0.0073，而 v7 最终损失是 0.0080——**剩余损失几乎全是它**。
v3/v6 也有同样问题。

修法（`scripts/openpi/fix_degenerate_norm_stats.py`）：不改模型维数、不改评测侧的
`parse_policy_action`，只把 `q99-q01 < 1e-2` 的维度窗口撑到 ±0.5，使其归一化成精确的 0。
阈值取 1e-2 是因为真实维度的窗宽都 ≥0.11，而退化维度 ≤1e-4，中间有三个数量级的空隙。

## 9. 采集与评测的又一处不一致：`env.render()`

| | 采集 | 评测 |
|---|---|---|
| `env.render()` 调用 | **0 次** | 6 处，其中 2 处在每个控制步的内循环 |

实测（`--no_render` 开关）：

| | 与训练首帧的位移 |
|---|---|
| 头相机 有 render | **(0, +5)** |
| 头相机 无 render | **(0, 0)** |

**头相机那 5 像素偏移确实由 `env.render()` 引入。** 但不能直接关闭：关掉后腕相机拿到的是废帧
（实测是完全不同的场景），说明相机流的推进依赖 render。正确修法还需查清采集侧的帧是如何推进的。

## 10. v8：加强增强 + 修正归一化

```bash
# 增强 (0.92, 5°) + 高斯噪声 sigma∈[0,0.08]；单元测试：腕图变化幅度 0.0000 → 1.0265
#（上游默认 0.0000，v7 第一版 0.6666，头相机 0.9061）
.venv/bin/python scripts/train.py pi05_g1_button_lora \
    --exp-name g1_button_v8_strongaug --overwrite --no-wandb-enabled \
    --checkpoint-base-dir /root/autodl-tmp/southgrid/checkpoints \
    --num-workers 16 --num-train-steps 12000 --lr-schedule.decay-steps 12000 \
    --save-interval 2000 --keep-period 12000
```

> `--keep-period` 必须配合磁盘余量：一个 checkpoint 9 GB，`/root/autodl-tmp` 只有 50 GB。
> v7 用 `max_to_keep=1` + `keep_period=15000`，结果中途的 step 10000 被自动删除，
> 无法回头验证「退火末段是否过拟合」（v7 的 @10k 判据 13.7 mm 明显优于 @15k 的 37.6 mm）。
