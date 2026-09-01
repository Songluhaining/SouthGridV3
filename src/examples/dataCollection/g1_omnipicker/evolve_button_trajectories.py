"""MAP-Elites 离线进化按钮按压轨迹库（QD：质量 × 多样性）。

在仓库自带的机器人本地 MuJoCo 模型上（无需 OrcaLab）进化"接近路径"基因型，
产出按行为空间（最大侧偏 × 最大纵偏 × 时长缩放）均匀覆盖的精英轨迹档案，
供 g1_omnipicker_collection_scripted_button_combo_lerobot.py 以
--trajectory_archive 采样执行。

设计要点：
  - 基因型（9 维，走廊归一化）：3 个过路点的 (dy, dz) 偏移、时长缩放、
    预备距离、速度曲线指数——同一条基因可从任意起点实例化（多按钮串接）。
  - 有效性（硬约束）：差分 IK 全程跟踪残差 ≤ 上限（同时涵盖可达性/关节限位/
    奇异位形三类失效）、无新增自碰撞、过路点不穿越面板平面。
  - 鲁棒性：每条基因在 4 个起点上下文（neutral + 其余三色的预备点）上评估，
    取最差上下文的适应度；任一上下文无效则整体无效。
  - 适应度：平滑度(jerk) + 关节速度余量 + 末端到位精度 的加权和（越大越好）。

算法为自含 MAP-Elites（评估接口为"基因批 → (适应度, 描述子, 有效性)批"，
如需换用 QDax/EvoX 等框架只需替换 MapElites 类）。

用法：
  python evolve_button_trajectories.py --out button_traj_archive.json \
      --budget 3000 --workers 12
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import mujoco
import numpy as np
from yaml import safe_load

base_dir = os.path.dirname(os.path.realpath(__file__))
MODEL_XML = os.path.abspath(os.path.join(
    base_dir, "..", "..", "..", "model", "G1_omnipicker", "g1_omnipicker.xml"))

ARM_R_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
ARM_R_NEUTRAL = [1.42, -0.88, -1.54, 1.48, 0, 0, 0]
ARM_L_JOINTS = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
ARM_L_NEUTRAL = [-1.42, 0.88, 1.54, -1.48, 0, 0, 0]
EE_SITE = "ee_center_site_r"
BASE_BODY = "body_link1"

CTRL_DT = 0.005            # 采集执行的控制步长（frame_skip 5 × 0.001s）
BASE_STEPS = 410           # 接近(250)+前推(120)+保压(40)，与采集脚本一致
PRESS_DEPTH = 0.010        # proximity 模式默认前推深度
TRACK_TOL = 0.012          # 有效性：IK 跟踪残差上限（米）
CORRIDOR_MARGIN = 0.02     # 过路点距面板平面的最小余量（米）

# 基因边界：dy1,dz1,dy2,dz2,dy3,dz3, tscale, approach_back, ease
GENE_LO = np.array([-0.12, -0.06, -0.12, -0.06, -0.08, -0.04, 0.60, 0.08, 0.5])
GENE_HI = np.array([+0.12, +0.14, +0.12, +0.14, +0.08, +0.10, 1.50, 0.18, 2.0])
VIA_FRACS = (0.25, 0.50, 0.75)

# 行为描述子网格：最大侧偏(带符号) × 最大纵偏(带符号) × 时长缩放
DESC_LO = np.array([-0.13, -0.07, 0.60])
DESC_HI = np.array([+0.13, +0.15, 1.50])
GRID = (8, 8, 4)


# ---------------------------------------------------------------------------
# 轨迹实例化（采集脚本以同一函数执行档案基因，保证评估与执行一致）
# ---------------------------------------------------------------------------

def instantiate_press_path(start, aim, genes, press_depth=PRESS_DEPTH):
    """从基因实例化一次按压的稠密末端参考路径。

    Returns:
        pts:   (N,3) 每控制步一个末端位置（B 系）
        marks: dict 关键索引 {"approach": i0, "press": i1}
        vias:  3 个过路点（供采集脚本构造分段）
    """
    genes = np.asarray(genes, dtype=np.float64)
    dy = genes[0:6:2]
    dz = genes[1:6:2]
    tscale, aback, ease = genes[6], genes[7], genes[8]

    start = np.asarray(start, dtype=np.float64)
    aim = np.asarray(aim, dtype=np.float64)
    approach = aim.copy()
    approach[0] -= aback
    press = aim.copy()
    press[0] += press_depth

    line = approach - start
    vias = [start + line * f + np.array([0.0, dy[k], dz[k]])
            for k, f in enumerate(VIA_FRACS)]
    knots = [start, *vias, approach, press]

    seg_len = [max(1e-6, float(np.linalg.norm(knots[i + 1] - knots[i])))
               for i in range(len(knots) - 1)]
    n_appr = max(60, int(250 * tscale))
    n_push = max(30, int(120 * tscale))
    appr_len = sum(seg_len[:4])
    seg_steps = [max(8, int(round(n_appr * l / appr_len))) for l in seg_len[:4]]
    seg_steps.append(n_push)

    pts = []
    for i, n in enumerate(seg_steps):
        a, b = knots[i], knots[i + 1]
        for k in range(n):
            u = (k + 1) / n
            u = u ** ease / (u ** ease + (1 - u) ** ease)  # 平滑 S 型速度剖面
            pts.append(a + (b - a) * u)
    i_approach = sum(seg_steps[:4]) - 1
    hold = [pts[-1]] * 40
    pts = np.asarray(pts + hold)
    return pts, {"approach": i_approach, "press": len(pts) - 41}, vias


def descriptor_of(genes, start, aim):
    """行为描述子：路径相对直线的最大侧偏/纵偏（带符号）与时长缩放。"""
    pts, _, _ = instantiate_press_path(start, aim, genes)
    approach = aim.copy()
    approach[0] -= genes[7]
    line = approach - start
    L2 = float(line @ line) or 1e-9
    rel = pts - start[None, :]
    proj = np.clip((rel @ line) / L2, 0, 1)[:, None] * line[None, :]
    dev = rel - proj
    iy = int(np.argmax(np.abs(dev[:, 1])))
    iz = int(np.argmax(np.abs(dev[:, 2])))
    return np.array([dev[iy, 1], dev[iz, 2], genes[6]])


# ---------------------------------------------------------------------------
# 本地 MuJoCo 差分 IK 评估器（每个进程持有一份）
# ---------------------------------------------------------------------------

class LocalEvaluator:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.d = mujoco.MjData(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self.dofs = []
        self.qadr = []
        self.jlim = []
        for jn in ARM_R_JOINTS:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self.qadr.append(int(self.m.jnt_qposadr[j]))
            self.dofs.append(int(self.m.jnt_dofadr[j]))
            lim = self.m.jnt_range[j] if self.m.jnt_limited[j] else (-3.1, 3.1)
            self.jlim.append((float(lim[0]), float(lim[1])))
        self._set_neutral()
        mujoco.mj_forward(self.m, self.d)
        self.baseline_pairs = self._contact_pairs()
        self.q_neutral = self.d.qpos.copy()
        self.ee_neutral_B = self.to_B(self.d.site_xpos[self.site])

    def _set_neutral(self):
        mujoco.mj_resetData(self.m, self.d)
        for names, vals in ((ARM_R_JOINTS, ARM_R_NEUTRAL), (ARM_L_JOINTS, ARM_L_NEUTRAL)):
            for jn, v in zip(names, vals):
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                self.d.qpos[self.m.jnt_qposadr[j]] = v

    def _contact_pairs(self):
        return {(int(self.d.contact.geom1[i]), int(self.d.contact.geom2[i]))
                for i in range(self.d.ncon)}

    def to_B(self, p):
        Rb = self.d.xmat[self.base].reshape(3, 3)
        return Rb.T @ (np.asarray(p) - self.d.xpos[self.base])

    def from_B(self, p):
        Rb = self.d.xmat[self.base].reshape(3, 3)
        return self.d.xpos[self.base] + Rb @ np.asarray(p)

    def _ik_step(self, jacp, target, q, gain=1.0):
        mujoco.mj_jacSite(self.m, self.d, jacp, None, self.site)
        J = jacp[:, self.dofs]
        err = target - self.d.site_xpos[self.site]
        JJt = J @ J.T + 1e-4 * np.eye(3)
        dq = gain * (J.T @ np.linalg.solve(JJt, err))
        step = float(np.linalg.norm(dq))
        if step > 0.2:
            dq *= 0.2 / step
        q = q + dq
        for i, (lo, hi) in enumerate(self.jlim):
            q[i] = min(max(q[i], lo), hi)
        for a, v in zip(self.qadr, q):
            self.d.qpos[a] = v
        mujoco.mj_forward(self.m, self.d)
        return q, step

    def track(self, pts_B, check_every=4):
        """差分 IK 逐点跟踪，返回指标 dict。先做就位迭代（IK 收敛到路径起点）。"""
        self.d.qpos[:] = self.q_neutral
        mujoco.mj_forward(self.m, self.d)
        jacp = np.zeros((3, self.m.nv))
        q = np.array([self.d.qpos[a] for a in self.qadr])

        # 就位：收敛到路径起点（真实执行时手臂本就在起点，此步不计入指标）
        start_w = self.from_B(pts_B[0])
        settled = False
        for _ in range(300):
            q, _ = self._ik_step(jacp, start_w, q)
            if np.linalg.norm(start_w - self.d.site_xpos[self.site]) < 0.005:
                settled = True
                break
        if not settled:
            return {"max_res": 9.9, "max_dq": 0.0, "new_collision": 0, "jerk": 0.0,
                    "settle_fail": True}

        max_res = 0.0
        max_dq = 0.0
        new_collision = 0
        for t, p_B in enumerate(pts_B):
            target = self.from_B(p_B)
            dq_wp = 0.0
            for _ in range(2):
                q, step = self._ik_step(jacp, target, q)
                dq_wp += step
            res = float(np.linalg.norm(target - self.d.site_xpos[self.site]))
            max_res = max(max_res, res)
            max_dq = max(max_dq, dq_wp / CTRL_DT)
            if t % check_every == 0:
                new_collision += len(self._contact_pairs() - self.baseline_pairs)
            if max_res > TRACK_TOL * 3:
                break
        d1 = np.diff(pts_B, axis=0)
        jerk = float(np.abs(np.diff(d1, n=2, axis=0)).mean()) if len(pts_B) > 3 else 0.0
        return {
            "max_res": max_res,
            "max_dq": max_dq,
            "new_collision": new_collision,
            "jerk": jerk,
            "settle_fail": False,
        }

    def evaluate(self, genes, contexts, aim):
        """多起点上下文评估：取最差上下文；任一无效则整体无效。"""
        worst_fit = float("inf")
        agg = None
        approach_plane = aim[0] - CORRIDOR_MARGIN
        for start in contexts:
            pts, marks, vias = instantiate_press_path(start, aim, genes)
            if any(v[0] > approach_plane for v in vias):
                return False, -1e9, {"reason": "corridor"}
            met = self.track(pts)
            valid = (not met["settle_fail"]
                     and met["max_res"] <= TRACK_TOL
                     and met["new_collision"] == 0
                     and met["max_dq"] <= 6.0)
            if not valid:
                return False, -1e9, {"reason": "track", **met}
            fit = (-40.0 * met["jerk"] * 1e3
                   - 0.4 * met["max_dq"]
                   - 200.0 * met["max_res"])
            if fit < worst_fit:
                worst_fit = fit
                agg = met
        return True, worst_fit, agg


_EVAL = None


def _worker_init():
    global _EVAL
    _EVAL = LocalEvaluator()


def _worker_eval(payload):
    genes, contexts, aim = payload
    ok, fit, met = _EVAL.evaluate(np.asarray(genes), [np.asarray(c) for c in contexts],
                                  np.asarray(aim))
    desc = descriptor_of(np.asarray(genes), np.asarray(contexts[0]), np.asarray(aim))
    return ok, fit, desc.tolist(), met


# ---------------------------------------------------------------------------
# MAP-Elites（自含实现；换 QDax/EvoX 时替换本类即可）
# ---------------------------------------------------------------------------

class MapElites:
    def __init__(self, rng):
        self.rng = rng
        self.archive: dict[tuple, dict] = {}

    def bin_of(self, desc):
        idx = []
        for v, lo, hi, n in zip(desc, DESC_LO, DESC_HI, GRID):
            idx.append(min(n - 1, max(0, int((v - lo) / (hi - lo) * n))))
        return tuple(idx)

    def ask(self, batch):
        out = []
        elites = list(self.archive.values())
        for _ in range(batch):
            if not elites or self.rng.random() < 0.15:
                g = GENE_LO + self.rng.random(len(GENE_LO)) * (GENE_HI - GENE_LO)
            else:
                parent = elites[self.rng.integers(len(elites))]
                sigma = 0.10 * (GENE_HI - GENE_LO)
                g = np.clip(np.asarray(parent["genes"]) + self.rng.normal(0, sigma),
                            GENE_LO, GENE_HI)
            out.append(g)
        return out

    def tell(self, genes, ok, fit, desc, met):
        if not ok:
            return False
        key = self.bin_of(desc)
        cur = self.archive.get(key)
        if cur is None or fit > cur["fitness"]:
            self.archive[key] = {
                "genes": [round(float(x), 5) for x in genes],
                "fitness": round(float(fit), 3),
                "descriptor": [round(float(x), 4) for x in desc],
                "metrics": {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in met.items()},
            }
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="MAP-Elites 按钮按压轨迹库进化")
    parser.add_argument("--out", default=os.path.join(base_dir, "button_traj_archive.json"))
    parser.add_argument("--pose_candidates", default=os.path.join(base_dir, "pose_g1_button_candidates.yaml"))
    parser.add_argument("--budget", type=int, default=3000, help="每种颜色的评估次数")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.pose_candidates, encoding="utf-8") as f:
        cand = safe_load(f)
    aims = {}
    quats = {}
    for color, spec in cand["buttons"].items():
        aims[color] = np.mean([c["r_target_b"] for c in spec["candidates"]], axis=0)
        quats[color] = spec["candidates"][0]["r_quat_b"]

    probe = LocalEvaluator()
    # v2：轨迹起点 = ready 预备位姿（与采集脚本 READY_R_POS_B 一致），不再从下垂 neutral 出发
    neutral_ee = np.array([0.540, -0.308, 0.296])
    print(f"ready EE(B) = {neutral_ee.round(4)}（v2 预备位姿起点）", flush=True)

    rng = np.random.default_rng(args.seed)
    archives = {}
    pool = mp.Pool(args.workers, initializer=_worker_init)
    try:
        for color in ("red", "green", "yellow", "blue"):
            aim = aims[color]
            contexts = [neutral_ee] + [
                aims[c] - np.array([0.12, 0.0, 0.0])
                for c in aims if c != color
            ]
            me = MapElites(rng)
            n_eval = 0
            n_valid = 0
            t0 = time.time()
            while n_eval < args.budget:
                batch = me.ask(min(args.batch, args.budget - n_eval))
                payloads = [(g.tolist(), [c.tolist() for c in contexts], aim.tolist())
                            for g in batch]
                for g, (ok, fit, desc, met) in zip(batch, pool.map(_worker_eval, payloads)):
                    me.tell(g, ok, fit, desc, met)
                    n_valid += int(ok)
                n_eval += len(batch)
            occ = len(me.archive)
            total_cells = GRID[0] * GRID[1] * GRID[2]
            print(f"{color:6s}: 评估 {n_eval}，有效率 {n_valid / n_eval:.0%}，"
                  f"档案覆盖 {occ}/{total_cells} 格，用时 {time.time() - t0:.0f}s", flush=True)
            archives[color] = {
                "aim_b": [round(float(x), 5) for x in aim],
                "quat_xyzw": quats[color],
                "elites": list(me.archive.values()),
            }
    finally:
        pool.close()
        pool.join()

    out = {
        "meta": {
            "grid": list(GRID),
            "desc_lo": DESC_LO.tolist(),
            "desc_hi": DESC_HI.tolist(),
            "gene_lo": GENE_LO.tolist(),
            "gene_hi": GENE_HI.tolist(),
            "via_fracs": list(VIA_FRACS),
            "press_depth": PRESS_DEPTH,
            "track_tol": TRACK_TOL,
            "contexts": "neutral + 其余三色预备点，最差上下文",
        },
        "colors": archives,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"档案已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
