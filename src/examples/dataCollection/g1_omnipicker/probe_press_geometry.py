"""按压几何探测：量化不同夹爪朝向按下按钮时 ee_center_site 到按钮 site 的距离（官方计分口径）。

背景（v3 数据实测）：指尖顶按时 ee_center_site 在指尖后方约 50 mm，且按钮 site 随帽一起滑动，
所以指尖顶按的官方距离恒 ≈50–55 mm，压深无用；要进 50 mm 满分区必须换接触部位。
本脚本在冷启动的 OrcaLab 上，从 L 型预备位姿出发，依次尝试若干夹爪朝向：走到帽前方，
沿柜面法向 +x_B 逐步推进，记录按钮开始位移（接触）与压到 8 mm 时的官方距离。

用法（先 bash orcalab_restart.sh；每次运行前都要冷重启）：
  python probe_press_geometry.py --color green --modes tip,tilt_y_35_35,tilt_y_35_-35,side_yneg_y,...
  模式（可加后缀 +comp 启用接近点残差补偿）：tip=指尖顶按（基线）；align_<0~1>=把候选姿态的夹爪轴线向柜面法向扶正的比例；side_<轴>_<y|z>=夹爪轴线放平；tilt_<y|z>_<度>_<横向偏移mm>=倾斜+横向偏移
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from yaml import Loader, load, safe_load

_here = os.path.dirname(os.path.realpath(__file__))
for p in (os.path.abspath(os.path.join(_here, "../../..")),
          os.path.abspath(os.path.join(_here, "../../inference/g1_omnipicker"))):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco  # noqa: E402
import eval_g1_omnipicker_button_lerobot as ev  # noqa: E402
from conf import g1_omnipicker_conf as agent_conf  # noqa: E402
from dataCollectionManager.data_collection_manager import DataCollectionManager  # noqa: E402
from dataStorage.lerobot_camera import scratch_dir  # noqa: E402
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402
from task.abstract_task import EmptyTask  # noqa: E402

X_B = np.array([1.0, 0.0, 0.0])
AXES = {"xpos": X_B, "xneg": -X_B, "ypos": np.array([0.0, 1.0, 0.0]), "yneg": np.array([0.0, -1.0, 0.0]),
        "zpos": np.array([0.0, 0.0, 1.0]), "zneg": np.array([0.0, 0.0, -1.0])}


def side_quat(axis_dir: np.ndarray, slab_axis: str) -> np.ndarray:
    """夹爪轴线（ee x 轴）指向 axis_dir，手指侧面法向（ee y 或 z 轴）指向柜面 +x_B。返回 xyzw。"""
    a = axis_dir / np.linalg.norm(axis_dir)
    if slab_axis == "y":
        ey = X_B; ez = np.cross(a, ey)
    else:
        ez = X_B; ey = np.cross(ez, a)
    M = np.stack([a, ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)], axis=1)
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1
    return R.from_matrix(M).as_quat()


def align_quat(q_xyzw: np.ndarray, frac: float) -> np.ndarray:
    """把姿态 q 的 x 轴向 +x_B 旋转 frac 比例（绕两者叉积轴的最小旋转）。返回 xyzw。"""
    Rq = R.from_quat(q_xyzw)
    ax = Rq.as_matrix()[:, 0]
    axis = np.cross(ax, X_B); s_ = np.linalg.norm(axis)
    if s_ < 1e-6:
        return np.asarray(q_xyzw)
    ang = np.arctan2(s_, float(np.dot(ax, X_B)))
    return (R.from_rotvec(axis / s_ * ang * frac) * Rq).as_quat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="green")
    ap.add_argument("--modes", default="tip,side_yneg_y,side_yneg_z,side_ypos_y,side_ypos_z,side_zneg_y,side_zneg_z,side_zpos_y,side_zpos_z")
    ap.add_argument("--standoff", type=float, default=0.06, help="接近点距帽面（m）")
    ap.add_argument("--press_depth", type=float, default=0.04, help="与采集脚本 press 模式相同：目标压入帽心以里的深度（m）")
    ap.add_argument("--task_config", default="../common/example.yaml")
    ap.add_argument("--pose_candidates", default=os.path.join(_here, "pose_g1_button_candidates.yaml"))
    args = ap.parse_args()

    cfg = load(open(os.path.join(_here, args.task_config), encoding="utf-8"), Loader=Loader)
    cand = safe_load(open(args.pose_candidates, encoding="utf-8"))
    tip_quat = np.asarray(cand["buttons"][args.color]["candidates"][0]["r_quat_b"], dtype=np.float64)
    g_close = float(cand.get("gripper_close", 2.0))

    sm = SceneManager("localhost:50051", config=cfg)
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir("_probe_scratch"), lock_left=True)

    def obs_safe(env):
        if env.model.nu == 0:
            return {"/action/end/position": np.zeros((2, 3), np.float32),
                    "/action/end/orientation": np.zeros((2, 4), np.float32),
                    "/action/effector/motor": np.zeros(4, np.float32)}
        return storage.obs_callback(env)

    manager = DataCollectionManager(agent_name="g1_omnipicker", env_name="DataCollection", entry_point=ev.ENTRY_POINT,
                                    default_joint_values={}, obs_callback=obs_safe, env_index=0, device=None,
                                    scene_manager=sm, frame_skip=5, orcagym_addr="localhost:50051")
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])
    manager.set_task(EmptyTask(env))
    manager.mode = DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset(); time.sleep(0.1); manager.update_scene()
    env.set_default_joint_values(ev.build_default_joint_values())
    ev.teleport_to_ready(env)
    l_arm, r_arm, l_grip, r_grip = ev.build_controllers(manager, env)
    manager.set_init_ctrl(); env.set_ctrl(manager.ctrl); env.mj_forward()
    for c in manager.controllers:
        c.reset()
    env.disable_actuator([agent_conf.positions_group])
    mon = ev.ButtonPressMonitor(env, 0.001, 0.0)
    mon.start_episode([args.color])
    m_, d_ = env.gym._mjModel, env.gym._mjData
    base_bid = mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_BODY, env.body(agent_conf.base_body))

    def to_B(p_w):
        return d_.xmat[base_bid].reshape(3, 3).T @ (np.asarray(p_w) - d_.xpos[base_bid])

    def state():
        return storage.build_state(storage.obs_callback(env))

    def cur_apply():
        return ev.action_dict_for_apply(ev.parse_policy_action(state()))

    device = ev.EEFDevice(l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip, **cur_apply())
    manager.set_device(device)
    ev.drive_to_ready(manager, env, device, cur_apply(), 150, 150)

    def move_to(pos, quat, steps):
        st = cur_apply()
        p0 = np.asarray(st["r_pos_b"], np.float64); sl = Slerp([0, 1], R.from_quat([st["r_quat_b"], quat]))
        tgt = dict(st, r_grip_ctrl=np.array([g_close] * 2, np.float32))
        for i in range(steps):
            a = (i + 1) / steps
            tgt["r_pos_b"] = (p0 + (np.asarray(pos) - p0) * a).astype(np.float32)
            tgt["r_quat_b"] = sl(a).as_quat().astype(np.float32)
            ev.hold_target(manager, env, device, tgt, 1)
        ev.hold_target(manager, env, device, tgt, 100)
        return tgt

    cap_B = to_B(d_.site_xpos[mon.color2site[args.color]])
    print(f"目标 {args.color} 按钮 site（B 系）: {np.round(cap_B, 3).tolist()}", flush=True)
    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        lateral = np.zeros(3)
        compensate = mode.endswith("+comp")
        mode = mode.replace("+comp", "")
        if mode == "tip":
            quat = tip_quat
        elif mode.startswith("side_"):
            _, ax, slab = mode.split("_")
            quat = side_quat(AXES[ax], slab)
        elif mode.startswith("align_"):
            # align_<frac>：把候选姿态的 ee x 轴向柜面法向 +x_B 扶正 frac（1.0=完全对齐），
            # 使指尖与 ee_center_site 沿法向共线，接触时 site 无面内偏移
            frac = float(mode.split("_")[1])
            quat = align_quat(tip_quat, frac)
        else:
            # tilt_<y|z>_<deg>_<offset_mm>：在指尖朝向基础上绕 ee 的 y/z 轴倾斜 deg，
            # 并把接近点沿 B 系 z/y 横向偏移 offset_mm（让指尖擦过帽边、由手指侧面近端接触）
            _, ax, deg, off = mode.split("_")
            rot = R.from_rotvec((np.array([0, 1, 0]) if ax == "y" else np.array([0, 0, 1])) * np.radians(float(deg)))
            quat = (R.from_quat(tip_quat) * rot).as_quat()
            lateral = (np.array([0, 0, 1.0]) if ax == "y" else np.array([0, 1.0, 0])) * float(off) / 1000.0
        mon.start_episode([args.color])
        base = mon.obs[args.color]["base"]
        appr = cap_B - np.array([args.standoff, 0.0, 0.0]) + lateral
        tgt = move_to(appr, quat, 400)
        st = state()
        ang = np.degrees((R.from_quat(st[10:14]) * R.from_quat(quat).inv()).magnitude())
        perr = np.linalg.norm(st[7:10] - appr) * 1000
        rec = {"mode": mode, "appr_pos_err_mm": round(float(perr)), "appr_ang_err_deg": round(float(ang)),
               "contact_site_mm": None, "max_disp_mm": 0.0}
        rec.update({"min_site_mm": None, "inplane_at_min_mm": None})
        if ang < 25 and perr < 60:
            # 与采集脚本相同的推压：目标 = 帽心 + press_depth（法向），推 400 步、保持 200 步；
            # comp=1 时把接近点的 OSC 稳态残差加到推压目标上（面内偏移补偿）
            comp = (appr - st[7:10]) if compensate else np.zeros(3)
            comp[0] = 0.0
            press_pt = cap_B + np.array([args.press_depth, 0, 0]) + lateral + comp
            p0 = np.asarray(state()[7:10], np.float64)
            best = (1e9, None, None)
            for i in range(600):
                a = min(1.0, (i + 1) / 400)
                tgt["r_pos_b"] = (p0 + (press_pt - p0) * a).astype(np.float32)
                ev.hold_target(manager, env, device, tgt, 1)
                if i % 5 == 0:
                    q = env.query_joint_qpos([mon.color2joint[args.color]])[mon.color2joint[args.color]]
                    disp = abs(float(np.ravel(q)[0]) - base)
                    ee = d_.site_xpos[mon.ee_sid]; site = d_.site_xpos[mon.color2site[args.color]]
                    dist = float(np.linalg.norm(ee - site)) * 1000
                    rec["max_disp_mm"] = max(rec["max_disp_mm"], disp * 1000)
                    if disp >= 0.001 and rec["contact_site_mm"] is None:
                        rec["contact_site_mm"] = round(dist)
                    if dist < best[0]:
                        off = to_B(ee) - to_B(site)
                        best = (dist, round(float(np.linalg.norm(off[1:]) * 1000)), round(float(off[0] * 1000)))
            rec["min_site_mm"] = round(best[0]); rec["inplane_at_min_mm"] = best[1]; rec["normal_at_min_mm"] = best[2]
        print(f"  {mode:14s} comp={int(compensate)} 接近误差 {rec['appr_pos_err_mm']}mm/{rec['appr_ang_err_deg']}°  "
              f"初触 {rec['contact_site_mm']}mm  全程最小官方距离 {rec['min_site_mm']}mm"
              f"（面内 {rec['inplane_at_min_mm']} / 法向 {rec.get('normal_at_min_mm')}）  最大压深 {rec['max_disp_mm']:.1f}mm", flush=True)
        results.append(rec)
        move_to(appr - np.array([0.06, 0, 0]), quat, 150)
        ev.drive_to_ready(manager, env, device, cur_apply(), 200, 100)
    print("\n汇总:", flush=True)
    for r_ in sorted(results, key=lambda r_: (r_["min_site_mm"] is None, r_["min_site_mm"] or 999)):
        print("  ", r_, flush=True)
    env.close()


if __name__ == "__main__":
    main()
