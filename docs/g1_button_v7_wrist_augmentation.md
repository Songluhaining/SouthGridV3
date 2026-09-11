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
