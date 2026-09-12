"""G1 OmniPicker 多按钮组合脚本化数据采集（指令多样 + 轨迹多样 + 质量打标）。

相对 g1_omnipicker_collection_scripted_button_lerobot.py 的扩展：
  1. 每集按 1~N 个不重复颜色按钮（组合与顺序均衡覆盖），语言指令从模板库随机改写；
  2. 轨迹不再是固定直线：接触位姿加抖动、接近路径加弯曲过路点、分段时长随机缩放，
     并按 (弯曲量, 抬升量, 时长缩放) 三维分箱做均衡采样，保证多样性覆盖而非纯随机；
  3. 按压成功用按钮滑动关节的真实位移判定（非假设），每集的成功性、按压耗时、
     路径长度等指标写入 <lerobot_out>/meta/quality.jsonl，供后续筛选/加权微调使用；
  4. 默认丢弃任何一次按压失败的集（--keep_failed 保留并在 quality.jsonl 标注）。
"""
# torch 必须在本进程的任何其它原生库之前加载。Windows 实测：在已经建立仿真连接、
# 加载过 MuJoCo 模型的进程里首次导入 torch，会触发 Windows fatal exception
# 0xc0000139（DLL 入口点找不到），进程直接退出，没有 Python 异常也没有 traceback。
# lerobot 本就依赖 torch，这里只是把导入时机提到最前面。
import torch  # noqa: F401  isort:skip

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import NamedTuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

import mujoco
import numpy as np
from yaml import Loader, load, safe_load

import data_collection_scripted as scripted  # noqa: E402

from conf import g1_omnipicker_conf as agent_conf

from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_joint_lock import JointLockController
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
    scratch_dir,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

BUTTON_CAMERA_MAP = {
    "camera_head_color": ("cam_head", 7090),
    "camera_wrist_r_color": ("cam_wrist_r", 7080),
}
ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = scratch_dir("g1_scripted_button_combo_lerobot_stream")

log_dir = os.path.join(base_dir, "logs")
orca_logger = get_orca_logger(
    name="G1ButtonCombo",
    log_file="g1_omnipicker_collection_scripted_button_combo_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_COLOR_ORDER = ["red", "green", "yellow", "blue"]
_COLOR_CN = {"red": "红色", "green": "绿色", "yellow": "黄色", "blue": "蓝色"}

# ---------------------------------------------------------------------------
# 语言指令模板
# ---------------------------------------------------------------------------

_SINGLE_TEMPLATES = [
    "按{a}按钮",            # 与既有单按钮数据集一致的规范表达
    "按下{a}按钮",
    "请按下{a}按钮",
    "按一下{a}的按钮",
    "帮我按{a}按钮",
    "把{a}按钮按下去",
    "麻烦按一下{a}按钮",
    "{a}按钮按一下",
    "去按{a}的按钮",
]

_MULTI_TEMPLATES = [
    "依次按下{lst}按钮",     # 规范表达
    "按顺序按{lst}按钮",
    "请依次按{lst}按钮",
    "按{lst}按钮",
    "把{lst}按钮都按一遍",
    "帮我把{lst}按钮都按下去",
    "麻烦依次按一下{lst}按钮",
    "{chain}",
    "{chain2}",
    "{chain3}",
]


def make_task_phrase(colors: list[str], rng: random.Random, canonical_ratio: float) -> str:
    names = [_COLOR_CN[c] for c in colors]
    if len(names) == 1:
        tpl = _SINGLE_TEMPLATES[0] if rng.random() < canonical_ratio else rng.choice(_SINGLE_TEMPLATES)
        return tpl.format(a=names[0])
    lst = ("、".join(names[:-1]) + "和" + names[-1]) if rng.random() < 0.5 else "、".join(names)
    chain = "先按" + names[0] + "按钮"
    for n in names[1:-1]:
        chain += "，再按" + n + "按钮"
    chain += ("，最后按" if len(names) > 2 else "，再按") + names[-1] + "按钮"
    # 逐步连接式变体：按X按钮，接着再按Y按钮，最后再按Z按钮
    chain2 = "按" + names[0] + "按钮"
    for n in names[1:-1]:
        chain2 += "，接着再按" + n + "按钮"
    chain2 += ("，最后再按" if len(names) > 2 else "，接着再按") + names[-1] + "按钮"
    chain3 = "先按下" + names[0] + "按钮"
    for n in names[1:-1]:
        chain3 += "，然后按下" + n + "按钮"
    chain3 += ("，最后按下" if len(names) > 2 else "，然后按下") + names[-1] + "按钮"
    tpl = _MULTI_TEMPLATES[0] if rng.random() < canonical_ratio else rng.choice(_MULTI_TEMPLATES)
    return tpl.format(lst=lst, chain=chain, chain2=chain2, chain3=chain3)


# ---------------------------------------------------------------------------
# 组合序列均衡采样：长度 1..max 的全部有序不重复颜色序列，取当前覆盖数最少者
# ---------------------------------------------------------------------------

def all_sequences(max_buttons: int) -> list[tuple[str, ...]]:
    seqs = []
    for n in range(1, max_buttons + 1):
        seqs.extend(itertools.permutations(_COLOR_ORDER, n))
    return seqs


class SequenceSampler:
    def __init__(self, max_buttons: int, length_weights: list[float], rng: random.Random):
        self.rng = rng
        self.by_len: dict[int, list[tuple[str, ...]]] = {}
        for s in all_sequences(max_buttons):
            self.by_len.setdefault(len(s), []).append(s)
        self.counts: dict[tuple[str, ...], int] = {s: 0 for s in all_sequences(max_buttons)}
        self.lengths = sorted(self.by_len.keys())
        w = [max(0.0, length_weights[n - 1]) for n in self.lengths]
        total = sum(w) or 1.0
        self.length_p = [x / total for x in w]

    def next(self) -> tuple[str, ...]:
        n = self.rng.choices(self.lengths, weights=self.length_p)[0]
        pool = self.by_len[n]
        m = min(self.counts[s] for s in pool)
        cands = [s for s in pool if self.counts[s] == m]
        seq = self.rng.choice(cands)
        self.counts[seq] += 1
        return seq


# ---------------------------------------------------------------------------
# 多样化轨迹生成：接触抖动 + 弯曲过路点 + 时长缩放，三维分箱均衡采样
# ---------------------------------------------------------------------------

def _quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _jitter_quat_xyzw(q, rng: random.Random, max_deg: float):
    ang = math.radians(min(abs(rng.gauss(0.0, max_deg / 2)), max_deg))
    axis = np.array([rng.gauss(0, 1) for _ in range(3)])
    axis /= (np.linalg.norm(axis) + 1e-9)
    dq = np.array([*(axis * math.sin(ang / 2)), math.cos(ang / 2)])
    out = _quat_mul_xyzw(dq, np.asarray(q, dtype=np.float64))
    return (out / np.linalg.norm(out)).tolist()


class DiversityBins:
    """(bow_y, bow_z, 时长缩放) 三维直方图，采样时优先填充覆盖最少的箱。"""

    BOW_Y = (-0.06, 0.06, 5)
    BOW_Z = (-0.02, 0.07, 4)
    TSCALE = (0.75, 1.35, 3)

    def __init__(self):
        self.hist: dict[tuple[int, int, int], int] = {}

    @staticmethod
    def _bin(v, lo, hi, n):
        return min(n - 1, max(0, int((v - lo) / (hi - lo) * n)))

    def key(self, bow_y, bow_z, tscale):
        return (
            self._bin(bow_y, *self.BOW_Y),
            self._bin(bow_z, *self.BOW_Z),
            self._bin(tscale, *self.TSCALE),
        )

    def pick(self, rng: random.Random, k: int = 8):
        """采 k 组候选参数，返回落在最少覆盖箱里的那组并计数。"""
        best, best_count = None, None
        for _ in range(k):
            p = {
                "bow_y": rng.uniform(*self.BOW_Y[:2]),
                "bow_z": rng.uniform(*self.BOW_Z[:2]),
                "tscale": rng.uniform(*self.TSCALE[:2]),
            }
            c = self.hist.get(self.key(p["bow_y"], p["bow_z"], p["tscale"]), 0)
            if best_count is None or c < best_count:
                best, best_count = p, c
        kk = self.key(best["bow_y"], best["bow_z"], best["tscale"])
        self.hist[kk] = self.hist.get(kk, 0) + 1
        return best


# ---------------------------------------------------------------------------
# 官方计分口径（解码自 orca_scorer_client 26.8.26.1）
# ---------------------------------------------------------------------------
# 判定量：g1_omnipicker_ee_center_site_r 到目标按钮 site 的欧氏距离。
# 评分服务以 10Hz 后台采样，attempt 结束时回溯所有帧、取得分最高的一帧。
#   configs/tasks.yaml           得分曲线（抛物线，顶点 0.05m）与颜色→site 映射
#   orca_competition/conditions.py  抛物线插值与噪声
#   scorer_service/engine.py     最佳帧回溯、P1 最近按钮惩罚、P2 速通惩罚
OFFICIAL_PEAK_M = 0.05        # 抛物线顶点：距离 0.05m 得满分，0m 与 0.10m 均只有 6 分
OFFICIAL_A = 160.0            # ratio = 1 - A*(d-peak)^2，由 (0,0.6)(0.05,1.0)(0.10,0.6) 定出
OFFICIAL_KNEE_M = 0.10        # 抛物线段终点，其后二次衰减
OFFICIAL_TAIL_M = 0.30        # 0.30m 及以外得 0 分
OFFICIAL_KNEE_RATIO = 0.60
OFFICIAL_STEP_SCORE = 10.0    # 每个按钮步满分
OFFICIAL_P1_DISCOUNT = 0.35   # 最佳帧上目标不是最近按钮 → 该步 ×0.35
OFFICIAL_P2_RADIUS = 0.10     # 四钮均进入该半径才可能触发速通惩罚
OFFICIAL_P2_SPAN_S = 20.0     # 四钮最佳帧跨度 ≤ 20s → 全部 ×0.6
OFFICIAL_P2_FAST_S = 5.0      # 跨度 ≤ 5s → 全部 ×0.4
OFFICIAL_P2_DISCOUNT = 0.6
OFFICIAL_P2_FAST_DISCOUNT = 0.4
OFFICIAL_NOISE_AMP = 0.03     # 仅在 d > 0.05m 时施加，故瞄准点取在顶点近侧更稳


def official_ratio(dist_m: float) -> float:
    """官方 score_curve 的得分比率（0~1）。dist_m 为 ee_site 到按钮 site 的距离（米）。

    不含官方那项确定性噪声：噪声只在 d > 0.05m 时施加（幅度 ±0.03），
    瞄准点取在顶点近侧即可完全避开，因此这里按无噪声的下界估分。
    """
    d = float(dist_m)
    if d <= OFFICIAL_KNEE_M:
        return max(0.0, 1.0 - OFFICIAL_A * (d - OFFICIAL_PEAK_M) ** 2)
    if d >= OFFICIAL_TAIL_M:
        return 0.0
    t = (OFFICIAL_TAIL_M - d) / (OFFICIAL_TAIL_M - OFFICIAL_KNEE_M)
    return OFFICIAL_KNEE_RATIO * t * t


def apply_p2_discount(best_times_s: list[float], best_dists_m: list[float]) -> tuple[float, str]:
    """按官方规则计算 P2 速通折扣。返回 (折扣系数, 原因)；不触发时返回 (1.0, "")。

    仅在一次尝试凑满 4 个按钮、且四钮最佳帧距离均 ≤ 0.10m 时才可能触发。
    """
    if len(best_times_s) < 4:
        return 1.0, ""
    if any(d > OFFICIAL_P2_RADIUS + 1e-6 for d in best_dists_m):
        return 1.0, ""
    span = max(best_times_s) - min(best_times_s)
    if span <= OFFICIAL_P2_FAST_S:
        return OFFICIAL_P2_FAST_DISCOUNT, f"四钮最佳帧跨度 {span:.1f}s ≤ {OFFICIAL_P2_FAST_S:.0f}s"
    if span <= OFFICIAL_P2_SPAN_S:
        return OFFICIAL_P2_DISCOUNT, f"四钮最佳帧跨度 {span:.1f}s ≤ {OFFICIAL_P2_SPAN_S:.0f}s"
    return 1.0, ""


def apply_mode_defaults(args):
    """按 success_mode 补齐未显式指定的节奏与瞄准参数（main 与离线测试共用）。

    official 模式的节奏由 P2 速通惩罚倒推：四钮最佳帧跨度必须 > 20s，三个间隔
    即每钮 > 6.7s。最佳帧在窗口内的位置随抖动浮动，实测 8.0s/钮 时相邻间隔会低到
    6.64s（四钮仅 19.9s，刚好踩线），故放宽到 9.5s/钮（1900 控制步 x 5ms），
    最坏间隔仍有 8s 以上，四钮跨度约 24～28s。
    """
    args._contact_dy, args._contact_dz = (float(x) for x in args.contact_offset.split(","))
    if args.press_depth is None:
        # official：瞄准点即目标，不再压入（压入不加分，只会把基座推离电柜）
        args.press_depth = {"proximity": 0.010, "press": 0.040}.get(args.success_mode, 0.0)
    if args.steps_approach is None:
        args.steps_approach = 700 if args.success_mode == "official" else 250
    if args.steps_retract is None:
        args.steps_retract = 600 if args.success_mode == "official" else 150
    if args.steps_push is None:
        args.steps_push = {"official": 250, "proximity": 120}.get(args.success_mode, 400)
    if args.steps_hold is None:
        args.steps_hold = {"official": 350, "proximity": 40}.get(args.success_mode, 200)
    return args


# 右臂 L 型预备位形（定义见 conf.g1_omnipicker_conf.r_arm_ready；推理脚本同源）
READY_R_ARM_Q = agent_conf.r_arm_ready["joint_values"]
READY_R_POS_B = np.array(agent_conf.r_arm_ready["ee_pos_b"], dtype=np.float64)
READY_R_QUAT_B = np.array(agent_conf.r_arm_ready["ee_quat_b"], dtype=np.float64)  # xyzw
READY_STEPS = 150  # 每集起点由瞬移直接设定，此段仅作稳定（不录制）

# 左爪保持值：按钮任务左臂不参与，左爪应停在初始位形而不是被指令张开
LEFT_GRIP_HOLD = float(agent_conf.gripper_l["init_ctrl"][0])

# 左臂锁定增益。纯 PD 压不住这条臂（实测 kp=150 漂 18.3°，kp=800 反而 37.4°，
# 因为力矩已撞驱动器限幅），故改用 hard_lock 运动学锁定，PD 只保留重力前馈。
LEFT_LOCK_KP = 150.0
LEFT_LOCK_KD = 10.0


def build_combo_segments(
    seq_colors: tuple[str, ...],
    buttons: dict,
    cap_provider,
    site_provider,
    r_start: np.ndarray,
    approach_back_base: float,
    g_close: float,
    args,
    rng: random.Random,
    bins: DiversityBins,
    archive: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """为一个颜色序列构建全部轨迹段。

    Returns:
        segments: 传给 build_segmented_trajectory 的段列表
        windows:  每次按压的监测窗口 [{color, t0, t1}]（单位：轨迹步）
        params:   每次按压实际使用的随机参数（写入 quality.jsonl）
    """
    jit = max(0.0, args.jitter)
    segments: list[dict] = []
    windows: list[dict] = []
    params: list[dict] = []
    cursor = np.asarray(r_start, dtype=np.float64)
    t = 0

    # v2 前导段：从当前位置走到 ready 预备位姿（此段不录制，见 pre_roll）
    # 左爪显式钉在 conf 的 init_ctrl：build_segmented_trajectory 的夹爪初值是 g_open，
    # 而本任务左爪全程 "hold"，不显式给值就会被一路指令张开（实测左爪转了 29°）。
    # 预备位姿抖动：v6 的首帧位姿跨集标准差是 0.001mm（等于零），模型因此对起点零容差。
    ready_pos = READY_R_POS_B.copy()
    ready_quat = list(READY_R_QUAT_B)
    _rj_m = float(getattr(args, "ready_jitter_mm", 0.0)) / 1000.0
    _rj_d = float(getattr(args, "ready_jitter_deg", 0.0))
    if _rj_m > 0:
        ready_pos = ready_pos + np.array([rng.gauss(0, _rj_m / 2) for _ in range(3)]).clip(-_rj_m, _rj_m)
    if _rj_d > 0:
        ready_quat = list(_jitter_quat_xyzw(ready_quat, rng, max_deg=_rj_d))

    # 收敛保持段的长度随机化：预备位的画面是一段整定暂态，与训练首帧的距离随该值
    # 呈 V 形（实测 settle=0 时 46、145 时谷底 13、290 时 21）。v6 固定 150 步，
    # 800 集全部卡在同一相位上；评测因推理延迟必然落在别的相位，无法对齐。
    # 随机化该长度即把暂态的多个相位采进训练集。
    _sj = getattr(args, "settle_jitter", "")
    if _sj:
        _lo, _hi = (int(x) for x in str(_sj).split(","))
        settle_steps = rng.randint(min(_lo, _hi), max(_lo, _hi))
    else:
        settle_steps = 150

    segments.append({
        "steps": READY_STEPS, "l_hold": True,
        "r_target_b": ready_pos.tolist(), "r_quat_b": list(ready_quat),
        "gripper_l": LEFT_GRIP_HOLD, "gripper_r": g_close,
    })
    # 收敛保持段：目标钉在 ready，等 OSC 稳态后再开始正式轨迹
    segments.append({
        "steps": settle_steps, "l_hold": True,
        "r_target_b": ready_pos.tolist(), "r_quat_b": list(ready_quat),
        "gripper_l": "hold", "gripper_r": g_close,
    })
    t += READY_STEPS + settle_steps
    pre_roll = t
    cursor = ready_pos.copy()

    for color in seq_colors:
        btn = buttons[color]
        cand_idx = rng.randrange(len(btn["candidates"]))
        r_quat = list(btn["candidates"][cand_idx]["r_quat_b"])
        cap = np.asarray(cap_provider(color), dtype=np.float64)  # 实时帽心（B 系）

        div = bins.pick(rng)
        d_yz = np.array([
            0.0,
            np.clip(rng.gauss(0, 0.003), -0.006, 0.006),
            np.clip(rng.gauss(0, 0.003), -0.006, 0.006),
        ]) * jit
        r_target = cap + d_yz
        r_target[1] += args._contact_dy
        r_target[2] += args._contact_dz
        if args.success_mode == "proximity":
            # 瞄准交付候选位姿（视觉接触点，四色全部可达）；帽心仅作接近度度量
            r_target = np.asarray(btn["candidates"][cand_idx]["r_target_b"], dtype=np.float64) + d_yz
            r_target[1] += args._contact_dy
            r_target[2] += args._contact_dz
        elif args.success_mode == "official":
            # 官方口径：瞄准点取在按钮 site 前方 target_site_dist 处，使 ee_site 到
            # 按钮 site 的距离落在得分抛物线顶点（0.05m）的近侧无噪声区。
            # 面内抖动 dy/dz 会拉长欧氏距离，所以沿法向解出 dx 把总距离钉回目标值。
            site_b = np.asarray(site_provider(color), dtype=np.float64)
            dy = float(d_yz[1] + args._contact_dy)
            dz = float(d_yz[2] + args._contact_dz)
            want = float(args.target_site_dist)
            dx = math.sqrt(max(1e-6, want * want - dy * dy - dz * dz))
            r_target = site_b + np.array([-dx, dy, dz])
        if jit > 0:
            r_quat = _jitter_quat_xyzw(r_quat, rng, max_deg=3.0 * jit)
        elite = None
        if archive is not None:
            elite = rng.choice(archive[color]["elites"])
            genes = np.asarray(elite["genes"], dtype=np.float64)
            approach_back = float(genes[7])
            ts = float(genes[6])
        else:
            approach_back = approach_back_base + jit * rng.uniform(-0.03, 0.04)
            approach_back = float(np.clip(approach_back, 0.08, 0.18))
            ts = 1.0 + (div["tscale"] - 1.0) * jit
        approach = r_target.copy()
        approach[0] -= approach_back

        if args.success_mode == "official":
            # 节奏由 P2 速通惩罚倒推：四钮最佳帧跨度必须 > 20s，因此时长缩放只允许
            # 拉长不允许压缩（TSCALE 下界 0.75 会把 8.0s/钮 压到 6.0s，跨度掉到 18s）。
            ts = max(1.0, ts)
            # 保压段同样不能用固定的 25~55 步：官方以 10Hz 采样，0.13s 的保压只够采到
            # 1~3 帧，很可能整个错过最佳距离。改为按 steps_hold 做 ±15% 抖动。
            n_hold = (int(args.steps_hold * rng.uniform(0.85, 1.15)) if jit > 0
                      else args.steps_hold)
        else:
            n_hold = rng.randint(25, 55) if jit > 0 else args.steps_hold
        n_appr = max(30, int(args.steps_approach * ts))
        n_push = max(30, int(args.steps_push * ts))
        n_retr = max(30, int(args.steps_retract * ts))

        if elite is not None:
            # 档案模式：过路点来自精英基因（走廊归一化，从当前起点实例化）
            line = approach - cursor
            vias = [cursor + line * f + np.array([0.0, genes[2 * k], genes[2 * k + 1]])
                    for k, f in enumerate((0.25, 0.50, 0.75))]
            knots = [*vias, approach]
            lens = [max(1e-6, float(np.linalg.norm(
                knots[i] - (cursor if i == 0 else knots[i - 1])))) for i in range(4)]
            total_len = sum(lens)
            for i, kn in enumerate(knots):
                n_seg = max(8, int(round(n_appr * lens[i] / total_len)))
                segments.append({
                    "steps": n_seg, "l_hold": True,
                    "r_target_b": kn.tolist(), "r_quat_b": r_quat,
                    "gripper_l": "hold", "gripper_r": g_close,
                })
                t += n_seg
        else:
            # 接近段拆成 3 小段，中间两个过路点做垂直于直线方向的弯曲（B 系 y/z 偏移）
            bow = np.array([0.0, div["bow_y"], div["bow_z"]]) * jit
            n1, n2, n3 = n_appr // 3, n_appr // 3, n_appr - 2 * (n_appr // 3)
            for frac, n_seg in ((1 / 3, n1), (2 / 3, n2), (1.0, n3)):
                via = cursor + (approach - cursor) * frac + bow * math.sin(math.pi * frac)
                segments.append({
                    "steps": n_seg, "l_hold": True,
                    "r_target_b": via.tolist(), "r_quat_b": r_quat,
                    "gripper_l": "hold", "gripper_r": g_close,
                })
                t += n_seg
        push_start = t
        press_pt = r_target.copy()
        # official 模式 press_depth 为 0：目标就是瞄准点本身，不再压入帽面。
        # 压入既不加分（按钮 site 随帽滑动，距离不变），又会把基座推离电柜。
        press_pt[0] += max(0.0, args.press_depth)
        segments.append({
            "steps": n_push, "l_hold": True,
            "r_target_b": press_pt.tolist(), "r_quat_b": r_quat,
            "gripper_r": g_close,
        })
        t += n_push
        segments.append({"steps": n_hold, "l_hold": True, "r_hold": True, "gripper_r": g_close})
        t += n_hold
        # 后撤点：official 模式沿法向退到 retract_back 之外再横移到下一个按钮。
        # 贴着柜面平移会让途经按钮在错误时刻刷出高分帧，触发 P1 最近按钮惩罚。
        retreat = r_target.copy()
        retreat[0] -= (float(args.retract_back) if args.success_mode == "official"
                       else approach_back)
        segments.append({
            "steps": n_retr, "l_hold": True,
            "r_target_b": retreat.tolist(), "r_quat_b": r_quat,
            "gripper_r": g_close,
        })
        t += n_retr

        windows.append({"color": color, "t0": push_start, "t1": t,
                        "aim_b": r_target.tolist()})
        params.append({
            "color": color, "cand_idx": cand_idx,
            "cap_B": cap.round(4).tolist(),
            "elite": ({"descriptor": elite["descriptor"], "fitness": elite["fitness"]}
                      if elite is not None else None),
            "contact_jitter_m": d_yz.round(5).tolist(),
            "approach_back_m": round(approach_back, 4),
            "bow_y_m": round(div["bow_y"] * jit, 4),
            "bow_z_m": round(div["bow_z"] * jit, 4),
            "time_scale": round(ts, 3),
            "steps": [n_appr, n_push, n_hold, n_retr],
        })
        cursor = retreat

    return segments, windows, params, pre_roll


# ---------------------------------------------------------------------------
# 带按钮位移监测的轨迹设备
# ---------------------------------------------------------------------------

class MjCtx(NamedTuple):
    """设备层监测用的原生 MuJoCo 句柄。

    用具名字段而非位置元组：这些句柄曾以 7 元组传递，后来新增「全部按钮 site」
    一项时漏改了其中两处解包，ValueError 被 except 吞掉，表现为起点闸门恒为 inf、
    基座漂移采不到数据，且没有任何报错。
    """
    model: object                 # mjModel
    data: object                  # mjData
    base_raw_id: int              # 基座 body 的原生 id
    robot_geom_ids: list          # 机器人所有 geom（算最近距离用）
    window_cap_gids: list         # 每个按压窗口对应的帽 geom
    ee_site_id: int               # 右手 ee_center_site
    window_site_ids: list         # 每个窗口对应的按钮 site
    all_button_site_ids: dict     # {颜色: site_id}，P1 自检用

class MonitoredTrajectoryDevice(AbstractDevice):
    """按预计算轨迹驱动双臂/双夹爪，并在按压窗口内采样按钮关节位移。"""

    SAMPLE_EVERY = 3

    def __init__(self, env, button_joints, windows,
                 l_arm, r_arm, l_grip, r_grip, task_status,
                 l_pos, l_quat, r_pos, r_quat, l_gm, r_gm,
                 mj_ctx=None, pre_roll=0, lock_joints=None):
        super().__init__()
        # 额外做运动学锁定的关节（左爪）。左爪是欠驱动连杆：被驱动的是 joint2，
        # 而 conf 里观测的是 joint1，position 执行器给定 ctrl 后 joint1 会从模型
        # 默认位形滑到该 ctrl 的平衡位形（实测 12°）。左爪不参与按钮任务，钉住即可。
        self._lock_joints = list(lock_joints or [])
        self._lock_qpos = None
        self.pre_roll = int(pre_roll)
        self.env = env
        self.button_joints = list(button_joints)
        self.windows = windows
        # mj_ctx: (mjModel, mjData, base_raw_id, robot_geom_ids, window_cap_gids,
        #          ee_site_id, window_site_ids, all_button_site_ids)
        #          最后一项是 {颜色: site_id}，用于官方 P1「目标必须是最近按钮」自检
        self.mj_ctx = mj_ctx
        self.l_arm, self.r_arm = l_arm, r_arm
        self.l_grip, self.r_grip = l_grip, r_grip
        self.task_status = task_status
        self.l_pos, self.l_quat = l_pos, l_quat
        self.r_pos, self.r_quat = r_pos, r_quat
        self.l_gm, self.r_gm = l_gm, r_gm
        self.t = 0
        self.n_samples = 0
        self.n_query_errors = 0
        self.first_error: str | None = None
        # 每个窗口的监测结果
        self.press_obs = [
            {"max_disp": {j: 0.0 for j in self.button_joints}, "first_cross": {},
             "min_cap_dist": float("inf"), "min_site_dist": float("inf"), "inplane_at_min": None,
             # 官方口径：按得分曲线挑最佳帧（不是最近帧——顶点在 0.05m，过近同样掉分）
             "best_ratio": 0.0, "best_dist": None, "best_t": None,
             "best_p1_ok": None, "best_nearest": None,
             # OSC 跟踪残差：末端离本窗口瞄准点最近时的距离。用于区分
             # 「瞄准点算错」与「末端没走到瞄准点」——两者都会让官方距离偏大。
             "min_aim_dist": float("inf")}
            for _ in windows
        ]

    def _active_windows(self):
        for i, w in enumerate(self.windows):
            if w["t0"] - 5 <= self.t <= w["t1"] + 10:
                yield i

    def update(self):
        if self.t >= len(self.r_pos):
            return
        if self._lock_joints:
            if self._lock_qpos is None:
                qp = self.env.query_joint_qpos(self._lock_joints)
                self._lock_qpos = {j: np.ravel(qp[j]).astype(np.float64).copy()
                                   for j in self._lock_joints}
            self.env.set_joint_qpos(self._lock_qpos)
            self.env.set_joint_qvel({j: np.zeros(1) for j in self._lock_joints})
        if self.t == self.pre_roll:
            self.task_status.update_task_status(True)
            _snap_out = os.environ.get("JOINT_SNAP_OUT")
            if _snap_out:
                try:
                    _names = list(self.env.model.get_joint_dict().keys())
                    _cur = self.env.query_joint_qpos(_names)
                    np.savez(_snap_out, **{n: np.ravel(_cur[n]) for n in _names if n in _cur})
                    orca_logger.info(f"[快照] 录制首帧关节角已存至 {_snap_out}（{len(_names)} 个关节）")
                except Exception as _e:
                    orca_logger.warning(f"[快照] 失败: {_e}")
            # 记录录制首帧时的实际右手位置（B 系），供集末做起点质量判定
            try:
                ctx = self.mj_ctx
                d, base_raw = ctx.data, ctx.base_raw_id
                Rb = d.xmat[base_raw].reshape(3, 3)
                self.ready_actual = Rb.T @ (d.site_xpos[ctx.ee_site_id] - d.xpos[base_raw])
                self.base_w0 = d.xpos[base_raw].copy()  # 基座世界位置：用于集末计算基座漂移
            except Exception:
                self.ready_actual = None
                self.base_w0 = None
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat[self.t])
        self.l_grip.update_ctrl(np.full(len(self.l_grip.ctrl_index), self.l_gm[self.t], dtype=np.float32))
        self.r_grip.update_ctrl(np.full(len(self.r_grip.ctrl_index), self.r_gm[self.t], dtype=np.float32))

        active = list(self._active_windows())
        if active and self.t % self.SAMPLE_EVERY == 0:
            try:
                qpos = self.env.query_joint_qpos(self.button_joints)
                self.n_samples += 1
            except Exception as e:
                qpos = {}
                self.n_query_errors += 1
                if self.first_error is None:
                    self.first_error = repr(e)
            for i in active:
                rec = self.press_obs[i]
                for j in self.button_joints:
                    v = qpos.get(j)
                    if v is None:
                        continue
                    disp = float(np.abs(np.asarray(v)).max())
                    if disp > rec["max_disp"][j]:
                        rec["max_disp"][j] = disp
                    if disp > 1e-4 and j not in rec["first_cross"]:
                        rec["first_cross"][j] = self.t

        if active and self.t % self.SAMPLE_EVERY == 0 and self.mj_ctx is not None:
            ctx = self.mj_ctx
            d, base_raw = ctx.data, ctx.base_raw_id
            robot_geoms, win_caps = ctx.robot_geom_ids, ctx.window_cap_gids
            ee_sid, win_sites, all_sites = ctx.ee_site_id, ctx.window_site_ids, ctx.all_button_site_ids
            pb = d.xpos[base_raw]
            Rb = d.xmat[base_raw].reshape(3, 3)
            robot_pts = np.array([Rb.T @ (d.geom_xpos[g] - pb) for g in robot_geoms])
            ee_B = Rb.T @ (d.site_xpos[ee_sid] - pb)
            for i in active:
                cap = Rb.T @ (d.geom_xpos[win_caps[i]] - pb)
                dist = float(np.linalg.norm(robot_pts - cap[None, :], axis=1).min())
                if dist < self.press_obs[i]["min_cap_dist"]:
                    self.press_obs[i]["min_cap_dist"] = dist
                # 官方计分口径：ee_site 到按钮 site（site 随帽滑动）的距离
                if win_sites[i] is not None:
                    off = ee_B - Rb.T @ (d.site_xpos[win_sites[i]] - pb)
                    sd = float(np.linalg.norm(off))
                    rec_i = self.press_obs[i]
                    aim = self.windows[i].get("aim_b")
                    if aim is not None:
                        ad = float(np.linalg.norm(ee_B - np.asarray(aim)))
                        if ad < rec_i["min_aim_dist"]:
                            rec_i["min_aim_dist"] = ad
                    if sd < rec_i["min_site_dist"]:
                        rec_i["min_site_dist"] = sd
                        rec_i["inplane_at_min"] = float(np.linalg.norm(off[1:]))
                    # 复刻评分服务的最佳帧回溯：逐帧按得分曲线取比率最高的一帧，
                    # 并在该帧上做 P1 自检（目标按钮必须是离末端最近的按钮）。
                    ratio = official_ratio(sd)
                    if ratio > rec_i["best_ratio"]:
                        rec_i["best_ratio"] = ratio
                        rec_i["best_dist"] = sd
                        rec_i["best_t"] = self.t
                        ee_w = d.site_xpos[ee_sid]
                        dists = {c: float(np.linalg.norm(ee_w - d.site_xpos[sid]))
                                 for c, sid in all_sites.items() if sid is not None}
                        if dists:
                            nearest = min(dists, key=dists.get)
                            tgt = self.windows[i]["color"]
                            rec_i["best_nearest"] = nearest
                            rec_i["best_p1_ok"] = bool(
                                dists.get(tgt, float("inf")) <= min(dists.values()) + 1e-6)

        if self.t == len(self.r_pos) - 1:
            # 按压反作用力会把基座推离电柜：记录本集基座世界位移（场景漂移信号，state 基座系随之漂移）
            if self.mj_ctx is not None and getattr(self, "base_w0", None) is not None:
                ctx = self.mj_ctx
                self.base_drift = float(
                    np.linalg.norm(ctx.data.xpos[ctx.base_raw_id] - self.base_w0))
            self.task_status.update_task_status(True)
        self.t += 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker 多按钮组合采集（指令多样 + 轨迹多样 + 质量打标）"
    )
    parser.add_argument("--level", type=str, default="default")
    parser.add_argument("--task_config", type=str, default="../common/example.yaml")
    parser.add_argument("--lerobot_out", type=str, required=True)
    parser.add_argument("--repo_id", default="local/g1_omnipicker_button_combo")
    parser.add_argument("--episodes", type=int, default=64, help="采集总集数")
    parser.add_argument("--max_buttons", type=int, default=4, choices=(1, 2, 3, 4),
                        help="单集最多按几个按钮（有序不重复颜色组合共 4/16/40/64 种）")
    parser.add_argument("--length_weights", type=str, default="1,1,1,1",
                        help="1~4 个按钮的集数权重，逗号分隔")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--pose_candidates", type=str,
                        default=os.path.join(base_dir, "pose_g1_button_candidates.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trajectory_archive", type=str, default=None,
                        help="MAP-Elites 轨迹档案 JSON（evolve_button_trajectories.py 产出）。"
                             "给出后按压路径从档案精英均匀采样，替代内置弯曲随机")
    parser.add_argument("--elite_top_frac", type=float, default=1.0,
                        help="每色只保留 fitness 最优的前 N 比例精英（0~1，1 为全部）")
    parser.add_argument("--settle_jitter", type=str, default="",
                        help="收敛保持段步数的随机范围，如 60,250。留空则固定 150（v6 行为）。"
                             "用于把预备位整定暂态的多个相位采进训练集")
    parser.add_argument("--ready_jitter_mm", type=float, default=0.0,
                        help="预备位姿位置抖动上限（毫米）；v6 为 0，导致首帧位姿跨集标准差 0.001mm")
    parser.add_argument("--ready_jitter_deg", type=float, default=0.0,
                        help="预备位姿姿态抖动上限（度）")
    parser.add_argument("--jitter", type=float, default=1.0,
                        help="轨迹随机化总强度，0 关闭全部随机化，1 为默认强度")
    parser.add_argument("--canonical_ratio", type=float, default=0.34,
                        help="使用规范表达（与既有数据一致）的概率")
    parser.add_argument("--success_mode", choices=("proximity", "press", "official"),
                        default="official",
                        help="瞄准与判定口径：official=复刻官方评分（瞄准按钮 site 前方 "
                             "target_site_dist，按得分曲线逐帧取最佳帧并自检 P1/P2，默认）；"
                             "proximity=指尖接近按钮帽；press=按钮关节真实位移（物理按下）。"
                             "官方评分不看按钮是否被按下，只看末端到按钮 site 的距离曲线")
    parser.add_argument("--target_site_dist", type=float, default=0.047,
                        help="official 模式瞄准距离(米)：ee_site 到按钮 site 的目标距离。"
                             "得分曲线顶点在 0.05m，且噪声只在 d>0.05m 时施加，"
                             "故取在顶点近侧（默认 0.047）以避开噪声。"
                             "注意：末端受按钮阻挡，实际能到的最近距离约 70mm；把本值继续调小"
                             "并不能让末端更靠近，只会把 OSC 残差转成接触力、令基座漂移翻倍（实测）")
    parser.add_argument("--retract_back", type=float, default=0.14,
                        help="official 模式后撤距离(米)：按完一个按钮沿法向退出该距离再横移，"
                             "避免贴柜面平移让途经按钮刷出高分帧而触发 P1")
    parser.add_argument("--min_step_score", type=float, default=8.5,
                        help="official 模式闸门：每个按钮按官方口径估出的得分(满分10)低于该值即整集判废。"
                             "实测（Windows/RTX4060Ti）：不深压按钮时 OSC 能稳定到达的官方距离下限约 70mm，"
                             "对应估分 8.8~9.6；阈值取 8.5 只拦真正的异常（OSC 完全没到位、按错、"
                             "P1 触发），调高会显著提高丢弃率而对训练数据质量帮助有限")
    parser.add_argument("--proximity_threshold", type=float, default=0.045,
                        help="proximity 模式：按压窗口内指尖 geom 中心到帽心的最小距离(米)低于该值，"
                             "或按钮出现任何真实位移，视为成功（geom 中心距指尖表面约有 15~20mm 固有偏置）")
    parser.add_argument("--press_threshold", type=float, default=0.001,
                        help="按钮关节位移超过该值(米)视为按压成功。位移>0 即证明发生了真实物理接触；"
                             "本场景手臂在可达边界附近，实测位移 0.9~5mm 波动")
    parser.add_argument("--press_depth", type=float, default=None,
                        help="前推目标越过瞄准点的深度(米)。默认：proximity 模式 0.010，press 模式 0.040")
    parser.add_argument("--max_consec_fail", type=int, default=3,
                        help="连续多少集按压全败即停止（场景漂移信号：按压反作用力会累计推远基座与电柜，需重启仿真）")
    parser.add_argument("--contact_offset", type=str, default="0,0",
                        help="调试/标定用：给所有接触位姿加固定偏移 dy,dz（米），如 -0.008,0")
    parser.add_argument("--max_site_dist", type=float, default=0.080,
                        help="press 模式：按压窗口内 ee_site 到按钮 site 的最小距离(米)须 ≤ 该值，否则整集判废。"
                             "这是官方计分口径（0.05 内满分，0.068≈9.5 分，0.077≈9.0 分）。指尖顶按实测约 0.067~0.073，"
                             "红色偏斜时 0.073~0.076，默认 0.080 只剔除明显偏斜的按压；0 关闭")
    parser.add_argument("--max_cam_gap_s", type=float, default=0.5,
                        help="本集内任一相机相邻两帧接收间隔的最大值(秒)超过该值即判废——相机接收线程被主循环"
                             "挤占会让图像相对 state 滞后。试采实测每集最大间隔常态 0.30~0.36s，v3 数据里最坏约 1s，"
                             "默认 0.5 只剔除明显滞后的集；0 关闭")
    parser.add_argument("--keep_failed", action="store_true",
                        help="保留按压失败的集（默认丢弃，均在 quality.jsonl 记录）")
    parser.add_argument("--no_recenter", action="store_true",
                        help="不用按钮刚体位置修正候选位姿的 y/z（默认修正；候选原值偏离按钮中心 6~32mm）")
    parser.add_argument("--steps_approach", type=int, default=None,
                        help="接近段步数。默认：official 模式 700，其余 250")
    parser.add_argument("--steps_push", type=int, default=None,
                        help="前推段步数。默认：proximity 模式 120；press 模式 400（实测慢推才能可靠压下按钮）")
    parser.add_argument("--steps_hold", type=int, default=None,
                        help="保压段步数。默认：proximity 模式 40；press 模式 200")
    parser.add_argument("--steps_retract", type=int, default=None,
                        help="后撤段步数。默认：official 模式 600，其余 150")
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    apply_mode_defaults(args)
    length_weights = [float(x) for x in args.length_weights.split(",")]
    assert len(length_weights) >= args.max_buttons, "--length_weights 至少要给出 max_buttons 个权重"

    cand_path = os.path.abspath(os.path.expanduser(args.pose_candidates))
    with open(cand_path, "r", encoding="utf-8") as f:
        cand_spec = safe_load(f)
    g_open = float(cand_spec.get("gripper_open", -0.8561))
    g_close = float(cand_spec.get("gripper_close", 2.0))
    approach_back = float(cand_spec.get("approach_back", 0.12))
    buttons: dict = cand_spec["buttons"]

    sampler = SequenceSampler(args.max_buttons, length_weights, rng)
    bins = DiversityBins()

    traj_archive = None
    if args.trajectory_archive:
        with open(os.path.abspath(os.path.expanduser(args.trajectory_archive)), encoding="utf-8") as f:
            traj_archive = json.load(f)["colors"]
        if args.elite_top_frac < 1.0:
            # 只保留每色 fitness 最优的前 N%（快而平滑的轨迹），砍掉迂回精英
            for c, v in traj_archive.items():
                es = sorted(v["elites"], key=lambda e: e["fitness"], reverse=True)
                keep = max(8, int(len(es) * args.elite_top_frac))
                v["elites"] = es[:keep]
        orca_logger.info("[档案] 载入轨迹档案: " + ", ".join(
            f"{c}={len(v['elites'])}精英" for c, v in traj_archive.items()))

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))
    quality_path = os.path.join(lerobot_out, "meta", "quality.jsonl")

    default_joint_values: dict = {}
    for arm in (agent_conf.l_arm, agent_conf.r_arm):
        for jn, v in zip(arm["joint_names"], arm["neutral_joint_values"]):
            default_joint_values[jn] = v

    orca_logger.info("Creating scene manager")
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.show_ui_message(1, "脚本控制：G1 多按钮组合采集", "0xffff00", showtime=5)

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker_button_combo", args.level)
    # 左臂通道写首帧常数（左臂已 PD 锁定），推理侧同样开启
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir, lock_left=True)
    _n_motor = len(agent_conf.gripper_l["actuator_names"]) + len(agent_conf.gripper_r["actuator_names"])

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        orca_logger.error("场景初始化失败，退出")
        env.close()
        return
    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # 发现按钮关节（按压成功的真值来源）
    jdict = env.model.get_joint_dict()
    button_joints = sorted(
        n for n in jdict if "ElectricalCabinet" in n and "utton" in n
    )
    if not button_joints:
        orca_logger.error("场景中没有找到按钮关节，无法做按压判定，退出")
        env.close()
        return
    orca_logger.info(f"按钮关节: {button_joints}")

    # 颜色→按钮关节/帽 geom 标定：以原生 MuJoCo 数据中帽 geom 的实时位置为瞄准真值。
    # 背景（实测）：候选位姿的 x 比真实帽面浅约 5cm，按原候选按压从未真正压下过按钮。
    _mj_m, _mj_d = env.gym._mjModel, env.gym._mjData
    _base_raw = mujoco.mj_name2id(_mj_m, mujoco.mjtObj.mjOBJ_BODY, env.body(agent_conf.base_body))

    def _cap_pos_B(cap_gid: int) -> np.ndarray:
        mujoco.mj_forward(_mj_m, _mj_d)
        Rb = _mj_d.xmat[_base_raw].reshape(3, 3)
        return Rb.T @ (_mj_d.geom_xpos[cap_gid] - _mj_d.xpos[_base_raw])

    _joint_caps: dict[str, tuple[int, np.ndarray]] = {}
    for _jn in button_joints:
        _bname = env.model.body_id2name(int(jdict[_jn]["BodyID"]))
        _braw = mujoco.mj_name2id(_mj_m, mujoco.mjtObj.mjOBJ_BODY, _bname)
        _gid = next(g for g in range(_mj_m.ngeom) if _mj_m.geom_bodyid[g] == _braw)
        _joint_caps[_jn] = (_gid, _cap_pos_B(_gid))
    color2joint: dict[str, str] = {}
    color2cap: dict[str, int] = {}
    for _color, _spec in buttons.items():
        _mean = np.mean([c["r_target_b"] for c in _spec["candidates"]], axis=0)
        _jn = min(_joint_caps,
                  key=lambda k: float(np.linalg.norm(_joint_caps[k][1][1:] - _mean[1:])))
        color2joint[_color] = _jn
        color2cap[_color] = _joint_caps[_jn][0]
        orca_logger.info(
            f"[标定] {_COLOR_CN[_color]} → {_jn.split('_')[-2]} "
            f"帽心 B={_joint_caps[_jn][1].round(4).tolist()}")
    _robot_geom_ids = [
        g for g in range(_mj_m.ngeom)
        if "g1_omnipicker" in (mujoco.mj_id2name(_mj_m, mujoco.mjtObj.mjOBJ_BODY,
                                                 int(_mj_m.geom_bodyid[g])) or "")
    ]
    # 官方计分口径所需 site：右手 ee_center_site 与各颜色按钮 site（与评分服务同名）
    _ee_sid = mujoco.mj_name2id(_mj_m, mujoco.mjtObj.mjOBJ_SITE, env.site(agent_conf.r_arm["ee_site_name"]))
    _site_names = [mujoco.mj_id2name(_mj_m, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(_mj_m.nsite)]
    # 场景里关节名与 site 名的大小写不一致（Button03_joint vs button03_site），按小写匹配
    color2site = {
        c: next((i for i, n in enumerate(_site_names)
                 if n.lower().endswith(f"electricalcabinet_{color2joint[c].split('_')[-2].lower()}_site")), None)
        for c in color2joint
    }
    if any(v is None for v in color2site.values()):
        orca_logger.warning(f"[标定] 按钮 site 缺失: {color2site}，官方口径闸门将不生效")

    def _site_pos_B(color: str) -> np.ndarray:
        """按钮 site 的实时 B 系位置（官方判定就用这个 site）。site 缺失时回退到帽心。"""
        sid = color2site.get(color)
        if sid is None:
            return _cap_pos_B(color2cap[color])
        mujoco.mj_forward(_mj_m, _mj_d)
        Rb = _mj_d.xmat[_base_raw].reshape(3, 3)
        return Rb.T @ (_mj_d.site_xpos[sid] - _mj_d.xpos[_base_raw])

    if args.success_mode == "official":
        if any(v is None for v in color2site.values()):
            orca_logger.error("official 模式需要全部四个按钮 site，当前缺失，退出")
            env.close()
            return
        for _c in _COLOR_ORDER:
            if _c in color2site:
                orca_logger.info(
                    f"[标定] {_COLOR_CN[_c]} site B={_site_pos_B(_c).round(4).tolist()}")

    # 双臂 OSC + 双夹爪控制器。做成可重建：控制器在创建时绑定 model 对象，
    # 场景异步 publish 会替换 model，持有过期引用的 OSC 表现为 ~10 倍跟踪迟滞
    # （进程级随机病态的根因嫌疑）。每集重建以确保绑定当前 model。
    def build_controllers():
        ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
        ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
        # v2：左臂用关节 PD 锁定（重力前馈），彻底消除左臂噪声进入数据
        l_arm = JointLockController(env, agent_conf.l_arm, ctrl_l_name,
                                    kp=LEFT_LOCK_KP, kd=LEFT_LOCK_KD, hard_lock=True)
        r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name,
                                          dict(zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])))
        l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
        r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
        l_grip = create_gripper_2f85_reverse_controller(
            env, agent_conf.gripper_l, agent_conf.base_body, l_gname,
            dict(zip(l_gname, agent_conf.gripper_l["init_ctrl"])), Controller2F85Reverse.ControllerType.DATA)
        r_grip = create_gripper_2f85_reverse_controller(
            env, agent_conf.gripper_r, agent_conf.base_body, r_gname,
            dict(zip(r_gname, agent_conf.gripper_r["init_ctrl"])), Controller2F85Reverse.ControllerType.DATA)
        manager.controllers = []
        for c in (l_arm, r_arm, l_grip, r_grip):
            manager.add_controller(c)
        return l_arm, r_arm, l_grip, r_grip

    l_arm, r_arm, l_grip, r_grip = build_controllers()
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # 相机解码线程一旦跑起来，再首次导入 lerobot/datasets 会让进程直接崩溃，
    # 所以在拉起相机之前先把这些导入做完（详见 LeRobotDatasetWriter.preload）。
    LeRobotDatasetWriter.preload()
    orca_logger.info("LeRobot 依赖已预加载")

    # 相机
    cameras: dict = {}
    cam_hw = DEFAULT_HW
    camera_map = dict(BUTTON_CAMERA_MAP)
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")
    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return
    cam_shape = (3, cam_hw[0], cam_hw[1])

    if args.resume and not os.path.exists(os.path.join(lerobot_out, "meta", "info.json")):
        orca_logger.warning("--resume 指定但数据集尚不存在（meta/info.json 缺失），改为全新创建")
        args.resume = False

    # NVENC 会话在此创建。相机解码线程此时已在运行，二者同时调用 avcodec_open2
    # 曾在 Linux 上触发 libnvcuvid 崩溃；这两行日志用于区分「崩在建会话」与「崩在别处」。
    orca_logger.info(
        f"创建 LeRobot 写入器: root={lerobot_out} fps={args.fps} "
        f"相机={list(camera_map)} 分辨率={cam_hw}")
    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=lerobot_out,
        fps=args.fps,
        camera_map=camera_map,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="g1_omnipicker",
    )
    orca_logger.info("LeRobot 写入器就绪")
    storage.configure_lerobot(
        fps=args.fps, cameras=cameras, camera_map=camera_map, target_hw=cam_hw,
        writer=writer, task="g1 button combo", clock=args.clock,
    )

    os.makedirs(os.path.dirname(quality_path), exist_ok=True)
    n_kept = 0
    n_dropped = 0
    consec_fail = 0
    ctrl_dt = 0.005  # frame_skip(5) x time_step(0.001)，控制步时长

    def _query_r_start() -> np.ndarray:
        ee = env.site(agent_conf.r_arm["ee_site_name"])
        b = env.body(agent_conf.base_body)
        return env.query_site_pos_and_quat_B([ee], [b])[ee]["xpos"].astype(np.float64)

    orca_logger.info(
        f"开始采集：{args.episodes} 集，max_buttons={args.max_buttons}，"
        f"jitter={args.jitter}，输出 {lerobot_out}"
    )

    try:
        with writer:
            for ep in range(args.episodes):
                seq = sampler.next()
                phrase = make_task_phrase(list(seq), rng, args.canonical_ratio)
                orca_logger.info(f"\n=== Episode {ep + 1}/{args.episodes} | {'→'.join(_COLOR_CN[c] for c in seq)} | “{phrase}” ===")
                print(f"\n>>> 第 {ep + 1}/{args.episodes} 集 | {'→'.join(_COLOR_CN[c] for c in seq)} | 指令: {phrase}", flush=True)

                storage.set_task(phrase)
                try:
                    scene_manager.show_ui_message(
                        1, f"采集中: {phrase} ({ep + 1}/{args.episodes})", "0x00ff88", showtime=0)
                except Exception:
                    pass

                env.reset()
                time.sleep(0.05)
                if not manager.update_scene():
                    orca_logger.info("场景更新失败，停止采集")
                    break
                env.set_default_joint_values(default_joint_values)

                # 每集直接瞬移设定初始位形（用户确认的 L 型；物理稳态、零接触），
                # 替代"走过去"——起点与 OSC 执行状态完全解耦
                _init_q = {env.joint(j): np.array([v]) for j, v in
                           zip(agent_conf.r_arm["joint_names"], READY_R_ARM_Q)}
                _init_q.update({env.joint(j): np.array([v]) for j, v in
                                zip(agent_conf.l_arm["joint_names"],
                                    agent_conf.l_arm["neutral_joint_values"])})
                env.set_joint_qpos(_init_q)
                env.mj_forward()
                try:
                    _ee_after = storage.build_state(storage.obs_callback(env))[7:10]
                    orca_logger.info(f"[瞬移] 设定后右手 ee={np.round(_ee_after,3).tolist()}")
                except Exception as _e:
                    orca_logger.warning(f"[瞬移] FK 读取失败: {_e}")
                l_arm, r_arm, l_grip, r_grip = build_controllers()
                segments, windows, press_params, pre_roll = build_combo_segments(
                    seq, buttons, lambda c: _cap_pos_B(color2cap[c]), _site_pos_B,
                    _query_r_start(), approach_back, g_close, args, rng, bins,
                    archive=traj_archive)
                l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm = scripted.build_segmented_trajectory(
                    env, agent_conf, segments, g_open, g_close)

                win_caps = [color2cap[w["color"]] for w in windows]
                device = MonitoredTrajectoryDevice(
                    env, button_joints, windows,
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm,
                    mj_ctx=MjCtx(
                        model=_mj_m, data=_mj_d, base_raw_id=_base_raw,
                        robot_geom_ids=_robot_geom_ids, window_cap_gids=win_caps,
                        ee_site_id=_ee_sid,
                        window_site_ids=[color2site[w["color"]] for w in windows],
                        all_button_site_ids=color2site),
                    pre_roll=pre_roll,
                    lock_joints=[env.joint(j) for j in agent_conf.gripper_l["joint_names"]])
                manager.set_device(device)
                # 防御：位标志会被任何 model 重载冲掉（竞态随机出现），每集强制重设并验证。
                # position 伺服组若复活会与 OSC 力矩对抗，手速降 ~10 倍停在力平衡点。
                env.disable_actuator([agent_conf.positions_group])
                _dis = int(env.gym._mjModel.opt.disableactuator)
                if not (_dis >> agent_conf.positions_group) & 1:
                    orca_logger.warning(f"[防御] positions 禁用位设置失败: {_dis:#x}")
                _t_ep0 = time.time()
                manager.run_episode()
                cam_gap = max((cam.max_gap_since(_t_ep0) for cam in cameras.values()
                               if hasattr(cam, "max_gap_since")), default=0.0)

                # ── 质量评估 ────────────────────────────────────────────────
                if device.n_query_errors:
                    orca_logger.warning(
                        f"[监测] 按钮位移采样失败 {device.n_query_errors} 次"
                        f"（成功 {device.n_samples} 次），首个错误: {device.first_error}")
                else:
                    orca_logger.info(f"[监测] 按钮位移采样 {device.n_samples} 次")
                presses = []
                all_success = True
                # 起点质量闸门：录制首帧必须在设计 ready 位 ±3cm 内，否则整集丢弃
                _ra = getattr(device, "ready_actual", None)
                _rd = float(np.linalg.norm(np.asarray(_ra) - READY_R_POS_B)) if _ra is not None else float("inf")
                if _rd > 0.03:
                    all_success = False
                    orca_logger.warning(f"[起点] 录制首帧偏离 ready {_rd*1000:.0f}mm > 30mm，本集判废")
                for w, obs_rec, pp in zip(windows, device.press_obs, press_params):
                    disp = obs_rec["max_disp"]
                    target_joint = color2joint[w["color"]]
                    target_disp = disp.get(target_joint, 0.0)
                    pressed_joint = max(disp, key=disp.get) if disp else None
                    min_dist = obs_rec.get("min_cap_dist", float("inf"))
                    min_site = obs_rec.get("min_site_dist", float("inf"))
                    best_ratio = float(obs_rec.get("best_ratio") or 0.0)
                    best_dist = obs_rec.get("best_dist")
                    best_t = obs_rec.get("best_t")
                    best_t_s = round(best_t * ctrl_dt, 3) if best_t is not None else None
                    p1_ok = obs_rec.get("best_p1_ok")
                    step_score = None
                    if args.success_mode == "official":
                        # 官方口径：最佳帧比率 × 满分，再按 P1 打折（P2 在集末统一算）
                        step_score = OFFICIAL_STEP_SCORE * best_ratio
                        if p1_ok is False:
                            step_score *= OFFICIAL_P1_DISCOUNT
                            orca_logger.warning(
                                f"[P1] {w['color']} 最佳帧上离 {obs_rec.get('best_nearest')} 更近，"
                                f"该步 ×{OFFICIAL_P1_DISCOUNT}")
                        success = step_score >= args.min_step_score
                        if not success:
                            orca_logger.warning(
                                f"[官方口径] {w['color']} 估分 {step_score:.2f} < "
                                f"{args.min_step_score:.2f}（最佳帧距离 "
                                f"{(best_dist or float('nan')) * 1000:.1f}mm），本集判废")
                    elif args.success_mode == "proximity":
                        success = (min_dist <= args.proximity_threshold
                                   or target_disp >= args.press_threshold)
                    else:
                        success = target_disp >= args.press_threshold
                        if args.max_site_dist > 0 and min_site != float("inf") and min_site > args.max_site_dist:
                            success = False
                            orca_logger.warning(
                                f"[官方口径] {w['color']} ee_site 最小距离 {min_site*1000:.0f}mm"
                                f" > {args.max_site_dist*1000:.0f}mm（面内偏移 "
                                f"{(obs_rec.get('inplane_at_min') or 0)*1000:.0f}mm），本集判废")
                    wrong_button = (
                        pressed_joint != target_joint
                        and disp.get(pressed_joint, 0.0) >= args.press_threshold
                    )
                    if wrong_button:
                        orca_logger.warning(
                            f"[按错] 目标 {target_joint}，实际位移最大的是 {pressed_joint}")
                    press_t = obs_rec["first_cross"].get(target_joint)
                    presses.append({
                        "color": w["color"],
                        "target_joint": target_joint.split("_")[-2],
                        "pressed_joint": pressed_joint.split("_")[-2] if pressed_joint else None,
                        "max_disp_m": round(target_disp, 5),
                        "min_cap_dist_m": round(min_dist, 4) if min_dist != float("inf") else None,
                        "min_site_dist_m": round(min_site, 4) if min_site != float("inf") else None,
                        "inplane_at_min_m": (round(obs_rec["inplane_at_min"], 4)
                                             if obs_rec.get("inplane_at_min") is not None else None),
                        # 官方口径估分所需字段（最佳帧 = 得分曲线上比率最高的一帧）
                        "official_score": round(step_score, 2) if step_score is not None else None,
                        "best_site_dist_m": round(best_dist, 4) if best_dist is not None else None,
                        "best_t_s": best_t_s,
                        "aim_err_m": (round(obs_rec["min_aim_dist"], 4)
                                      if obs_rec.get("min_aim_dist") not in (None, float("inf"))
                                      else None),
                        "p1_ok": p1_ok,
                        "nearest_at_best": obs_rec.get("best_nearest"),
                        "success": bool(success),
                        "wrong_button": bool(wrong_button),
                        "press_time_s": round((press_t - w["t0"]) * ctrl_dt, 3) if press_t is not None else None,
                        **pp,
                    })
                    all_success = all_success and success and not wrong_button

                # P2 速通自检：四钮最佳帧跨度 ≤20s 会让四个按钮的得分全部打折。
                # 官方仅在凑满 4 个按钮时应用，故 1~3 钮的集只记录跨度、不判废。
                p2_discount, p2_reason = 1.0, ""
                if args.success_mode == "official":
                    _bt = [p["best_t_s"] for p in presses if p.get("best_t_s") is not None]
                    _bd = [p["best_site_dist_m"] for p in presses
                           if p.get("best_site_dist_m") is not None]
                    p2_discount, p2_reason = apply_p2_discount(_bt, _bd)
                    if p2_discount < 1.0:
                        all_success = False
                        orca_logger.warning(
                            f"[P2 速通] {p2_reason}，四钮得分将被 ×{p2_discount}，本集判废")
                    for p in presses:
                        if p.get("official_score") is not None:
                            p["official_score_final"] = round(
                                p["official_score"] * p2_discount, 2)

                if args.max_cam_gap_s > 0 and cam_gap > args.max_cam_gap_s:
                    all_success = False
                    orca_logger.warning(f"[相机] 最大帧间隔 {cam_gap:.2f}s > {args.max_cam_gap_s:.2f}s，图像滞后，本集判废")
                dr = np.diff(np.asarray(r_pos, dtype=np.float64), axis=0)
                metrics = {
                    "duration_steps": int(len(r_pos)),
                    "duration_s": round(len(r_pos) * ctrl_dt, 2),
                    "ee_path_len_m": round(float(np.linalg.norm(dr, axis=1).sum()), 3),
                    "mean_jerk": round(float(np.abs(np.diff(dr, n=2, axis=0)).mean()), 8),
                    "cam_max_gap_s": round(float(cam_gap), 3),
                    "base_drift_m": round(float(getattr(device, "base_drift", float("nan"))), 4),
                }
                if args.success_mode == "official":
                    _scores = [p.get("official_score_final") for p in presses
                               if p.get("official_score_final") is not None]
                    _times = [p["best_t_s"] for p in presses if p.get("best_t_s") is not None]
                    metrics["official_score_sum"] = round(sum(_scores), 2) if _scores else None
                    metrics["official_score_min"] = round(min(_scores), 2) if _scores else None
                    metrics["best_frame_span_s"] = (round(max(_times) - min(_times), 2)
                                                    if len(_times) >= 2 else 0.0)
                    metrics["p2_discount"] = p2_discount
                    if p2_reason:
                        metrics["p2_reason"] = p2_reason

                keep = all_success or args.keep_failed
                if keep:
                    storage.save_data(
                        task_info=manager.task.get_task_info(),
                        scene_info=manager.scene_manager.get_scene_info(),
                        task_description=manager.task.get_task_description(),
                    )
                    n_kept += 1
                    ep_index = writer.num_episodes - 1
                    tag = "✓" if all_success else "✗(保留)"
                else:
                    storage.clear_data()
                    n_dropped += 1
                    ep_index = None
                    tag = "✗ 丢弃"

                with open(quality_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "episode_index": ep_index,
                        "kept": bool(keep),
                        "all_success": bool(all_success),
                        "sequence": list(seq),
                        "task": phrase,
                        "presses": presses,
                        "metrics": metrics,
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }, ensure_ascii=False) + "\n")

                ok_str = " ".join(
                    f"{_COLOR_CN[p['color']]}:{'✓' if p['success'] else '✗'}"
                    + (f"(距{p['min_cap_dist_m'] * 1000:.0f}mm" if p["min_cap_dist_m"] is not None else "(")
                    + f",压{p['max_disp_m'] * 1000:.1f}mm)"
                    for p in presses
                )
                orca_logger.info(f"[{tag}] {ok_str}")
                print(f">>> [{tag}] {ok_str}", flush=True)

                consec_fail = 0 if all_success else consec_fail + 1
                if consec_fail >= args.max_consec_fail:
                    orca_logger.warning(
                        f"连续 {consec_fail} 集按压失败——按压反作用力累计会把基座推离电柜"
                        f"（本集帽心 x={press_params[0]['cap_B'][0]}，超过约 0.87 即接近可达边界）。"
                        f"请重启仿真（orcalab-cli stop_simulation + start_simulation）后加 --resume 续采。")
                    print("\n[漂移保护] 连续按压失败，已停止。请重启仿真后用 --resume 续采。", flush=True)
                    break

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = (f"采集结束：保留 {writer.num_episodes} 集 / {writer.num_frames} 帧，"
                   f"丢弃 {n_dropped} 集，质量记录: {quality_path}")
        orca_logger.info(summary)
        print(f"\n{'=' * 62}\n  {summary}\n{'=' * 62}", flush=True)


if __name__ == "__main__":
    main()
