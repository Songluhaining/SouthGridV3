#!/usr/bin/env python
"""修复零宽度维度的分位归一化。

openpi 的分位归一化是 (x-q01)/(q99-q01+1e-6)*2-1。对于恒定维度（左臂 7 维被锁死、
夹爪 4 维恒定），q99-q01 == 0，分母只剩 1e-6，float32 的舍入噪声被放大一百万倍：
实测左臂各维归一化后标准差 0.07~0.38，而真实的右臂维度是 0.52~0.63——
7 个动作维度在拟合纯噪声，占了训练损失的绝大部分。

修法：把这些维度的分位窗口撑到 ±0.5，使它们归一化成常数 0（噪声降到 1e-6 量级）。
不改模型维数、不改评测侧的 parse_policy_action，反归一化后仍得到正确的常数。
"""
import json, pathlib, sys, shutil
import numpy as np

P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/root/southgrid-openpi-pi05/assets/pi05_g1_button_lora/local/g1_button_v6/norm_stats.json")
WIDTH = 0.5
d = json.loads(P.read_text())
if not P.with_suffix(".json.orig").exists():
    shutil.copy(P, P.with_suffix(".json.orig"))

for key in ("state", "actions"):
    s = d["norm_stats"][key]
    q1 = np.array(s["q01"], float); q9 = np.array(s["q99"], float); mu = np.array(s["mean"], float)
    deg = (q9 - q1) < 1e-6
    print(f"{key}: 退化维度 {np.where(deg)[0].tolist()}")
    q1[deg] = mu[deg] - WIDTH
    q9[deg] = mu[deg] + WIDTH
    s["q01"] = q1.tolist(); s["q99"] = q9.tolist()
    # std 同样保护（z-score 路径用不到，但保持一致）
    sd = np.array(s["std"], float); sd[sd < 1e-6] = WIDTH
    s["std"] = sd.tolist()

P.write_text(json.dumps(d))
print(f"\n已写回 {P}（原件备份为 {P.name}.orig）")

# 自检
d2 = json.loads(P.read_text())["norm_stats"]["state"]
q1 = np.array(d2["q01"]); q9 = np.array(d2["q99"]); mu = np.array(d2["mean"])
out = (mu - q1) / (q9 - q1 + 1e-6) * 2.0 - 1.0
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print("\n修复后把均值代入归一化（恒定维应为 ~0）:")
print(out[:18])
