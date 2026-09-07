"""在 OrcaLab 中运行 G1 OmniPicker OpenPI 远程策略推理。"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import traceback

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from yaml import Loader, load

# 首次中断请求正常结束；再次中断会立即退出。
_interrupt = threading.Event()


def _install_interrupt_handlers() -> None:
    def _handler(signum, frame):
        if _interrupt.is_set():
            print("\n[退出] 再次收到中断信号，立即退出", flush=True)
            os._exit(130)
        _interrupt.set()
        print("\n[退出] Ctrl+C 收到，正在结束当前评估...", flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.log.orca_log import OrcaLog, get_orca_logger

from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_joint_lock import JointLockController
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_reverse_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    omnipicker_camera_map,
    probe_camera_hw,
    scratch_dir,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = scratch_dir("eval_g1_lerobot_stream")

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="EvalG1OmnipickerButton",
    log_file="eval_g1_omnipicker_button_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# State/Action：左臂位置 3、左臂四元数 4、右臂位置 3、
# 右臂四元数 4、左右夹爪归一化值各 2；四元数顺序为 xyzw。
_L_GRIP_RANGES = agent_conf.gripper_l["actuator_ranges"]
_R_GRIP_RANGES = agent_conf.gripper_r["actuator_ranges"]


def _denorm_grip(norm_val: float, grip_range: tuple[float, float]) -> float:
    """将 [0,1] 归一化值反归一化回电机量程内的绝对值。"""
    lo, hi = float(grip_range[0]), float(grip_range[1])
    return float(np.clip(norm_val, 0.0, 1.0)) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# EEFDevice：将策略输出的末端动作实时转发给 OSC 控制器
# ---------------------------------------------------------------------------

class EEFDevice(AbstractDevice):
    """将策略输出的 18 维 action 实时转发给 OSC 双臂与 2F85Reverse 夹爪控制器。

    l_grip_ctrl / r_grip_ctrl 均为长度 2 的数组 [inner, outer]，
    传入 Controller2F85Reverse.update_ctrl()。
    """

    def __init__(
        self,
        l_arm=None,
        r_arm=None,
        l_grip=None,
        r_grip=None,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.l_pos_b = None if l_pos_b is None else np.asarray(l_pos_b, dtype=np.float32)
        self.l_quat_b = None if l_quat_b is None else np.asarray(l_quat_b, dtype=np.float32)
        self.r_pos_b = None if r_pos_b is None else np.asarray(r_pos_b, dtype=np.float32)
        self.r_quat_b = None if r_quat_b is None else np.asarray(r_quat_b, dtype=np.float32)
        # 长度 2：[inner, outer]，已是绝对电机值
        self.l_grip_ctrl = None if l_grip_ctrl is None else np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2)
        self.r_grip_ctrl = None if r_grip_ctrl is None else np.asarray(r_grip_ctrl, dtype=np.float32).reshape(2)

    def set_target(
        self,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        if l_pos_b is not None:
            self.l_pos_b = np.asarray(l_pos_b, dtype=np.float32)
        if l_quat_b is not None:
            self.l_quat_b = np.asarray(l_quat_b, dtype=np.float32)
        if r_pos_b is not None:
            self.r_pos_b = np.asarray(r_pos_b, dtype=np.float32)
        if r_quat_b is not None:
            self.r_quat_b = np.asarray(r_quat_b, dtype=np.float32)
        if l_grip_ctrl is not None:
            self.l_grip_ctrl = np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2)
        if r_grip_ctrl is not None:
            self.r_grip_ctrl = np.asarray(r_grip_ctrl, dtype=np.float32).reshape(2)

    def update(self):
        if self.l_arm is not None and self.l_pos_b is not None and self.l_quat_b is not None:
            self.l_arm.update_action_position(self.l_pos_b)
            self.l_arm.update_action_axisangle(self.l_quat_b)
        if self.r_arm is not None and self.r_pos_b is not None and self.r_quat_b is not None:
            self.r_arm.update_action_position(self.r_pos_b)
            self.r_arm.update_action_axisangle(self.r_quat_b)
        if self.l_grip is not None and self.l_grip_ctrl is not None:
            self.l_grip.update_ctrl(self.l_grip_ctrl)
        if self.r_grip is not None and self.r_grip_ctrl is not None:
            self.r_grip.update_ctrl(self.r_grip_ctrl)


# ---------------------------------------------------------------------------
# 按压成功判定（真值来自本进程 MuJoCo 的按钮滑动关节）
# ---------------------------------------------------------------------------

_COLOR_CN = {"red": "红色", "green": "绿色", "blue": "蓝色", "yellow": "黄色"}
# 颜色→按钮编号映射（采集阶段以帽 geom 位置实测标定得到，场景固定不变）
_COLOR_TOKEN = {"red": "button01", "green": "button02", "blue": "button03", "yellow": "button04"}
# 官方评分口径（scorer_service.engine._TASK2_BUTTON_SITES）：末端 ee_site 到按钮 site 的最小距离，
# 0.05 m 内满分；且最佳帧上目标按钮必须是离末端最近的按钮，否则该步得分 ×0.35（P1）
_OFFICIAL_SITE_SUFFIX = "ElectricalCabinet_{tok}_site"
_OFFICIAL_FULL_DIST = 0.05


def colors_from_prompt(prompt: str) -> list[str]:
    """按指令中出现的先后顺序提取目标按钮颜色。"""
    hits = [(prompt.index(cn), c) for c, cn in _COLOR_CN.items() if cn in prompt]
    return [c for _, c in sorted(hits)]


class ButtonPressMonitor:
    """监测目标按钮：滑动关节位移（按下真值）与机器人-帽面最小距离（接近判据）。"""

    def __init__(self, env, success_disp: float, success_dist: float):
        self.env = env
        self.success_disp = success_disp
        self.success_dist = success_dist
        jdict = env.model.get_joint_dict()
        button_joints = [n for n in jdict if "ElectricalCabinet" in n and "utton" in n]
        self.color2joint = {
            c: next((j for j in button_joints if tok in j.lower()), None)
            for c, tok in _COLOR_TOKEN.items()
        }
        m, d = env.gym._mjModel, env.gym._mjData
        self._m, self._d = m, d
        self.color2cap = {}
        for c, jn in self.color2joint.items():
            if jn is None:
                continue
            braw = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, env.model.body_id2name(int(jdict[jn]["BodyID"])))
            self.color2cap[c] = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == braw)
        self.robot_geoms = [
            g for g in range(m.ngeom)
            if "g1_omnipicker" in (mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
        ]
        # 官方口径所需 site：右手 ee_site 与四个按钮 site（按名称后缀匹配，缺失则不算）
        site_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(m.nsite)]
        self.ee_sid = next((i for i, n in enumerate(site_names)
                            if n.endswith(agent_conf.r_arm["ee_site_name"])), None)
        self.color2site = {
            c: next((i for i, n in enumerate(site_names)
                     if n.endswith(_OFFICIAL_SITE_SUFFIX.format(tok=tok))), None)
            for c, tok in _COLOR_TOKEN.items()
        }
        self.targets: list[str] = []
        self.obs: dict[str, dict] = {}

    def start_episode(self, targets: list[str]) -> None:
        self.targets = [c for c in targets if self.color2joint.get(c)]
        base = self.env.query_joint_qpos([self.color2joint[c] for c in self.targets])
        self.obs = {
            c: {"base": float(np.ravel(base[self.color2joint[c]])[0]),
                "max_disp": 0.0, "min_dist": float("inf"), "hit_step": None,
                "site_dist": float("inf"), "site_nearest_ok": None}
            for c in self.targets
        }

    def check(self, step: int) -> bool:
        """每步调用；返回是否所有目标均已达成（位移或接近判据）。"""
        if not self.targets:
            return False
        qpos = self.env.query_joint_qpos([self.color2joint[c] for c in self.targets])
        for c in self.targets:
            rec = self.obs[c]
            disp = abs(float(np.ravel(qpos[self.color2joint[c]])[0]) - rec["base"])
            rec["max_disp"] = max(rec["max_disp"], disp)
            cap_pos = self._d.geom_xpos[self.color2cap[c]]
            dist = float(np.linalg.norm(
                self._d.geom_xpos[self.robot_geoms] - cap_pos[None, :], axis=1).min())
            rec["min_dist"] = min(rec["min_dist"], dist)
            if self.ee_sid is not None and self.color2site.get(c) is not None:
                ee = self._d.site_xpos[self.ee_sid]
                d_all = {cc: float(np.linalg.norm(ee - self._d.site_xpos[sid]))
                         for cc, sid in self.color2site.items() if sid is not None}
                if d_all[c] < rec["site_dist"]:
                    rec["site_dist"] = d_all[c]
                    rec["site_nearest_ok"] = d_all[c] <= min(d_all.values()) + 1e-6
            if rec["hit_step"] is None and (
                    disp >= self.success_disp or dist <= self.success_dist):
                rec["hit_step"] = step
        return all(r["hit_step"] is not None for r in self.obs.values())

    def summary(self) -> dict:
        hits = [(c, r["hit_step"]) for c, r in self.obs.items() if r["hit_step"] is not None]
        order_ok = [c for c, _ in sorted(hits, key=lambda x: x[1])] == self.targets
        return {
            "targets": self.targets,
            "success": len(hits) == len(self.targets) and len(self.targets) > 0,
            "order_ok": order_ok if hits else False,
            "detail": {c: dict(r) for c, r in self.obs.items()},
        }


# ---------------------------------------------------------------------------
# Action 工具
# ---------------------------------------------------------------------------

def parse_policy_action(raw_action: np.ndarray) -> dict:
    """将 18 维策略输出拆分为末端位姿 + 归一化夹爪 dict。

    夹爪保留归一化 [0,1]，施加给电机时再反归一化（见 action_dict_for_apply）。
    """
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.size < 18:
        raise ValueError(f"Expected at least 18 action dims, got {action.size}")
    return {
        "l_pos_b":          action[0:3],
        "l_quat_b":         action[3:7],
        "r_pos_b":          action[7:10],
        "r_quat_b":         action[10:14],
        "l_grip_inner_norm": float(np.clip(action[14], 0.0, 1.0)),
        "l_grip_outer_norm": float(np.clip(action[15], 0.0, 1.0)),
        "r_grip_inner_norm": float(np.clip(action[16], 0.0, 1.0)),
        "r_grip_outer_norm": float(np.clip(action[17], 0.0, 1.0)),
    }


def action_dict_for_apply(action_dict: dict) -> dict:
    """把归一化 [0,1] 的夹爪值反归一化为电机绝对值，位姿原样透传。

    夹爪反归一化公式（与 G1OmniPickerLeRobotStorage.build_state 正向归一化一致）：
        val = norm * (hi - lo) + lo
    默认量程 (-1, 2)，即 val = norm * 3 - 1。
    """
    l_inner = _denorm_grip(action_dict["l_grip_inner_norm"], _L_GRIP_RANGES[0])
    l_outer = _denorm_grip(action_dict["l_grip_outer_norm"], _L_GRIP_RANGES[1])
    r_inner = _denorm_grip(action_dict["r_grip_inner_norm"], _R_GRIP_RANGES[0])
    r_outer = _denorm_grip(action_dict["r_grip_outer_norm"], _R_GRIP_RANGES[1])
    return {
        "l_pos_b":    np.asarray(action_dict["l_pos_b"],  dtype=np.float32).copy(),
        "l_quat_b":   np.asarray(action_dict["l_quat_b"], dtype=np.float32).copy(),
        "r_pos_b":    np.asarray(action_dict["r_pos_b"],  dtype=np.float32).copy(),
        "r_quat_b":   np.asarray(action_dict["r_quat_b"], dtype=np.float32).copy(),
        "l_grip_ctrl": np.array([l_inner, l_outer], dtype=np.float32),
        "r_grip_ctrl": np.array([r_inner, r_outer], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# 相机观测构建器与策略运行器
# ---------------------------------------------------------------------------

class CameraObservationBuilder:
    """从已配置的相机数据流构建策略图像观测。"""

    def __init__(
        self,
        cameras: dict,
        camera_name_map: dict[str, str],
        target_hw: tuple = (480, 640),
    ):
        self.cameras = cameras
        self.camera_name_map = camera_name_map
        self.target_hw = target_hw

    def build_images(self) -> dict:
        H, W = self.target_hw
        images = {}
        for env_camera_name, policy_camera_name in self.camera_name_map.items():
            cam = self.cameras.get(env_camera_name)
            if cam is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                try:
                    frame, _ = cam.get_frame(format="rgb24")
                    if frame is None or frame.size == 0:
                        rgb = np.zeros((H, W, 3), dtype=np.uint8)
                    else:
                        if frame.shape[0] != H or frame.shape[1] != W:
                            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
                except Exception:
                    rgb = np.zeros((H, W, 3), dtype=np.uint8)
            images[policy_camera_name] = np.transpose(rgb, (2, 0, 1))
        return images


class OpenPIPolicyRunner:
    """封装 openpi_client WebSocket 策略调用。"""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        camera_name_map: dict[str, str],
        cameras: dict,
        target_hw: tuple = (480, 640),
        use_images: bool = True,
    ):
        from openpi_client import websocket_client_policy

        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.prompt = prompt
        self.use_images = use_images
        self.cam_builder = (
            CameraObservationBuilder(
                cameras=cameras,
                camera_name_map=camera_name_map,
                target_hw=target_hw,
            )
            if use_images
            else None
        )

    def build_observation(self, state: np.ndarray) -> dict:
        images = self.cam_builder.build_images() if self.use_images else {}
        return {"state": state, "images": images, "prompt": self.prompt}

    def infer_action_chunk(self, state: np.ndarray) -> np.ndarray:
        observation = self.build_observation(state)
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] < 18:
            raise ValueError(f"Expected policy action dim >= 18, got {actions.shape}")
        return actions


# ---------------------------------------------------------------------------
# 与采集脚本一致的起点 / 控制器构建（g1_omnipicker_collection_scripted_button_combo_lerobot.py）
# ---------------------------------------------------------------------------

CAMERA_PREP_SCRIPT = os.path.abspath(os.path.join(
    base_dir, "../../dataCollection/g1_omnipicker/orcalab_camera_prep.sh"))


def prepare_orcalab_cameras(mcp_url: str) -> bool:
    """腕相机朝向 (90,180,0) + IsRecording 重建，与采集共用同一脚本；须在连接相机之前执行。"""
    env_ = dict(os.environ, ORCALAB_CLI=os.path.join(os.path.dirname(sys.executable), "orcalab-cli"))
    proc = subprocess.run(["bash", CAMERA_PREP_SCRIPT, mcp_url], env=env_,
                          capture_output=True, text=True)
    log = orca_logger.info if proc.returncode == 0 else orca_logger.warning
    for line in (proc.stdout + proc.stderr).strip().splitlines():
        log(line)
    return proc.returncode == 0


def hold_target(manager, env, device, action: dict, steps: int, sleep_s: float = 0.0):
    """保持末端目标运行 steps 个控制步（起点稳定 / 相机预热）。"""
    for _ in range(max(0, steps)):
        device.set_target(**action)
        env.step(manager.run_controllers())
        env.render()
        if sleep_s > 0:
            time.sleep(sleep_s)


# 采集前导段的右爪指令（候选点 gripper_close；数据首帧右爪归一化值 1.0 ⇔ ctrl 2.0）
READY_R_GRIP_CTRL = 2.0


def drive_to_ready(manager, env, device, start: dict, move_steps: int, hold_steps: int) -> dict:
    """复现采集前导段：右手从当前位姿按位置线性 / 姿态 slerp 插值驶向 L 型预备位姿并闭合右爪，
    再钉住目标等 OSC 收敛。阶跃目标会让 OSC 走到另一条路上卡住（实测停在 129mm 外），
    必须与采集一样平滑插值。返回最终目标 action。
    """
    ready_pos = np.asarray(agent_conf.r_arm_ready["ee_pos_b"], dtype=np.float32)
    ready_quat = np.asarray(agent_conf.r_arm_ready["ee_quat_b"], dtype=np.float32)
    p0 = np.asarray(start["r_pos_b"], dtype=np.float32)
    slerp = Slerp([0.0, 1.0], R.from_quat([start["r_quat_b"], ready_quat]))
    target = dict(start, r_grip_ctrl=np.array([READY_R_GRIP_CTRL] * 2, dtype=np.float32))
    for i in range(max(1, move_steps)):
        a = (i + 1) / max(1, move_steps)
        target["r_pos_b"] = p0 + (ready_pos - p0) * a
        target["r_quat_b"] = slerp(a).as_quat().astype(np.float32)
        hold_target(manager, env, device, target, 1)
    target["r_pos_b"], target["r_quat_b"] = ready_pos, ready_quat
    hold_target(manager, env, device, target, hold_steps)
    return target


def build_default_joint_values() -> dict:
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


def teleport_to_ready(env) -> None:
    """每集起点：右臂瞬移到 L 型预备位形、左臂回中立位（与采集数据首帧一致）。"""
    q = {env.joint(j): np.array([v]) for j, v in
         zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm_ready["joint_values"])}
    q.update({env.joint(j): np.array([v]) for j, v in
              zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"])})
    env.set_joint_qpos(q)
    env.mj_forward()


def create_gripper(env, grip_conf):
    ctrl_names = [env.actuator(name) for name in grip_conf["actuator_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_reverse_controller(
        env, grip_conf, agent_conf.base_body, ctrl_names, init_ctrl,
        Controller2F85Reverse.ControllerType.DATA,
    )


def build_controllers(manager, env):
    """每集重建控制器（env.reset() 重载模型，关节锁缓存的原生索引会失效）。

    左臂用关节 PD 锁定（与采集一致，策略输出的左臂通道被忽略），右臂 OSC。
    """
    ctrl_l = [env.actuator(n) for n in agent_conf.l_arm["motors_names"]]
    ctrl_r = [env.actuator(n) for n in agent_conf.r_arm["motors_names"]]
    l_arm = JointLockController(env, agent_conf.l_arm, ctrl_l)
    r_arm = create_arm_osc_controller(
        env, agent_conf.r_arm, agent_conf.base_body, ctrl_r,
        dict(zip(ctrl_r, agent_conf.r_arm["motors_init_ctrl"])))
    l_grip = create_gripper(env, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf.gripper_r)
    manager.controllers = []
    for c in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(c)
    return l_arm, r_arm, l_grip, r_grip


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker OpenPI 远程策略推理评估"
    )
    parser.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml",
                        help="场景配置 YAML（默认 example.yaml）")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--host", type=str, default="localhost", help="策略服务器主机")
    parser.add_argument("--port", type=int, default=8010, help="策略服务器端口")
    parser.add_argument("--prompt", type=str, default="按红色按钮",
                        help="任务语言描述（必须与训练时一致）")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="同一集（同一 attempt）内依次执行的多条指令，每条之间先回到预备位姿；"
                             "给出时忽略 --prompt，--max_steps 为每条指令的步数预算")
    parser.add_argument("--sleep", action="store_true", help="按 real_time_step 节奏运行")
    parser.add_argument("--max_steps", type=int, default=6000,
                        help="每集最大控制步数（5 ms/步；目标全部按下会提前结束）")
    parser.add_argument("--action_repeat", type=int, default=10,
                        help="每个推理 action 重复执行的控制步数（训练 20 fps ↔ 控制 200 Hz，须为 10）")
    parser.add_argument("--episodes", type=int, default=1, help="评估集数")
    parser.add_argument("--camera_warmup_steps", type=int, default=10,
                        help="每集推理前相机预热步数（默认 10）")
    parser.add_argument("--start_pose", choices=("ready", "neutral"), default="ready",
                        help="起点：ready=复现 v3 采集的 L 型预备位姿前导段；neutral=沿用场景默认位姿"
                             "（评估 v2 及更早、以默认位姿起步采集的模型时使用）")
    parser.add_argument("--ready_move_steps", type=int, default=150,
                        help="每集开始前右手插值驶向 L 型预备位姿的控制步数（与采集前导段一致）")
    parser.add_argument("--settle_steps", type=int, default=150,
                        help="到达预备位姿后钉住目标等 OSC 收敛的控制步数（与采集一致；不计入 max_steps）")
    parser.add_argument("--no_camera_prep", action="store_true",
                        help="不经 MCP 设置腕相机朝向与重建 IsRecording（已手动设置时使用）")
    parser.add_argument("--mcp_url", type=str, default="http://127.0.0.1:12345/mcp",
                        help="OrcaLab MCP 地址（相机准备用）")
    parser.add_argument("--dump_dir", type=str, default="",
                        help="调试：把每集首个观测的图像存为 PNG，并记录首个动作块相对 state 的位移")
    parser.add_argument("--no_images", action="store_true",
                        help="跳过相机采图，发送空图（仅用 state 的策略）")
    parser.add_argument("--no_preview", action="store_true", help="不显示相机实时预览小窗口")
    parser.add_argument("--success_disp", type=float, default=0.001,
                        help="按钮滑动关节位移成功阈值（米）")
    parser.add_argument("--success_dist", type=float, default=0.045,
                        help="机器人与按钮帽面最小距离成功阈值（米）")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="目标全部达成后不提前结束，仍跑满 max_steps")
    parser.add_argument("--team_id", type=str, default=os.environ.get("ORCA_SCORING_TEAM_ID", ""),
                        help="官方评分：队伍 ID（不传则不接入评分）")
    parser.add_argument("--team_token", type=str,
                        default=os.environ.get("ORCA_SCORING_TEAM_TOKEN", ""),
                        help="官方评分：队伍 Token")
    parser.add_argument("--robot_id", type=str, default="g1_omnipicker",
                        help="官方评分：机器人形态 ID")
    parser.add_argument("--task_id", type=str, default="task2_button_press",
                        help="官方评分：任务 ID")
    parser.add_argument("--scoring_targets", type=str, nargs="*", default=None,
                        help="官方评分：目标颜色英文名（默认从 prompt 解析）")
    parser.add_argument("--video_wait", type=float, default=90.0,
                        help="官方评分：结束后等待任务视频上传的秒数")
    parser.add_argument(
        "--enable_wrist_l",
        action="store_true",
        help="可选启用左腕相机 camera_wrist_l_color:7070（默认关闭）",
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    # 相机准备必须先于任何相机连接（连接后再翻转 IsRecording 会打断推流）
    if not args.no_images and not args.no_camera_prep:
        if not prepare_orcalab_cameras(args.mcp_url):
            orca_logger.warning("相机准备失败：腕相机朝向可能与训练数据不一致，评估结果不可信")

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    # storage 仅用于 obs_callback 与 build_state，不落盘；左臂通道写常数与采集一致。
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir("_eval_g1_scratch"), lock_left=True)

    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=build_default_joint_values(),
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])
    manager.set_task(EmptyTask(env))
    manager.mode = DataCollectionManager.DataCollectionMode.INFERENCE
    # DataCollectionManager 构造完成后注册，确保使用本入口的中断处理。
    _install_interrupt_handlers()

    camera_map = omnipicker_camera_map(enable_wrist_l=args.enable_wrist_l)
    # 环境相机传感器名到策略观测键的映射
    camera_name_map: dict[str, str] = {
        env_name: lerobot_key
        for env_name, (lerobot_key, _port) in camera_map.items()
    }
    orca_logger.info("推理相机配置已加载")

    _need_cameras = (not args.no_images) or (not args.no_preview)
    _shared_cameras: dict = {}
    _target_hw: tuple = DEFAULT_HW
    _preview_ready: bool = False
    _PREVIEW_W, _PREVIEW_H = 320, 240
    _PREVIEW_CAMS = list(camera_map.keys())
    policy_runner: OpenPIPolicyRunner | None = None
    device: EEFDevice | None = None

    _TPROF = {"ctrl": 0.0, "step": 0.0, "render": 0.0, "preview": 0.0, "n": 0}

    try:
        _video_started = False
        episode_results: list[bool] = []
        button_monitor: ButtonPressMonitor | None = None
        press_results: list[dict] = []

        # 官方评分接入（指南第 7 章）：ScorerClient 自动拉起本地评分服务，
        # 每集一次 attempt；评分服务经 gRPC 读 ORCA 真值自行判定，与本地判定互不影响。
        scorer = None
        _attempt_active = False
        if args.team_id:
            from orca_scorer_client import ScorerClient
            scorer = ScorerClient(
                team_id=args.team_id,
                team_token=args.team_token,
                robot_id=args.robot_id,
                timeout=60.0,
            )
            orca_logger.info(f"官方评分已启用: task_id={args.task_id}")

        for episode_index in range(args.episodes):
            if _interrupt.is_set():
                orca_logger.info("收到中断，跳过后续 episode")
                break

            orca_logger.info(f"=== Episode {episode_index + 1}/{args.episodes} ===")

            env.reset()
            time.sleep(0.1)

            if not manager.update_scene():
                orca_logger.error("场景更新失败，退出")
                return

            # 与采集一致：瞬移到 L 型预备位形，再重建控制器（左臂关节锁以当前位形为目标）
            if args.start_pose == "ready":
                teleport_to_ready(env)
            l_arm, r_arm, l_grip, r_grip = build_controllers(manager, env)
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            env.mj_forward()
            for controller in manager.controllers:
                controller.reset()
            env.render()
            # 位标志会被 model 重载冲掉：每集强制重设并验证（position 伺服组复活会与 OSC 对抗）
            env.disable_actuator([agent_conf.positions_group])
            _dis = int(env.gym._mjModel.opt.disableactuator)
            if not (_dis >> agent_conf.positions_group) & 1:
                orca_logger.warning(f"positions 禁用位设置失败: {_dis:#x}")

            # 瞬移只是粗定位：OrcaLab 导出模型的关节角语义随导出瞬间姿态变化（实测同一组
            # 关节角在不同进程对应不同物理姿态），真正的起点由下面的笛卡尔稳定段决定。
            _init_state = storage.build_state(storage.obs_callback(env))
            _init_action_apply = action_dict_for_apply(parse_policy_action(_init_state))

            # 按压成功判定：从 prompt 解析目标颜色，监测按钮关节位移与接近距离。
            # env.reset() 会重新加载 MuJoCo 模型，mjModel/mjData 句柄失效，因此每集重建。
            button_monitor = ButtonPressMonitor(env, args.success_disp, args.success_dist)
            _prompts = args.prompts or [args.prompt]
            _seg_targets = [colors_from_prompt(p) for p in _prompts]
            _targets = [c for seg in _seg_targets for c in seg]
            button_monitor.start_episode(_targets)
            if _targets:
                orca_logger.info(
                    "目标按钮: " + "→".join(_COLOR_CN[c] for c in _targets))
            else:
                orca_logger.warning("prompt 中未识别到按钮颜色，本集只跑不判定")

            device = EEFDevice(l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                               **_init_action_apply)
            manager.set_device(device)

            # 首集：场景就绪后启动相机内存流并连接策略服务器
            if episode_index == 0:
                if _need_cameras:
                    try:
                        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                        env.begin_save_video(STREAM_TRIGGER_PATH)
                        _video_started = True
                        _shared_cameras = bring_up_cameras(
                            camera_map, port_timeout=30.0, frame_timeout=30.0
                        )
                        _target_hw = probe_camera_hw(_shared_cameras, camera_map)
                        orca_logger.info(
                            f"内存流相机已就绪（{len(_shared_cameras)} 路），分辨率={_target_hw}"
                        )
                        if not args.no_preview:
                            _n_cams = len(_shared_cameras)
                            cv2.namedWindow("eval-preview", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("eval-preview", _PREVIEW_W * max(_n_cams, 1), _PREVIEW_H)
                            _preview_ready = True
                            orca_logger.info("预览窗口已创建，按 q 提前结束当前 episode")
                    except Exception as _e:
                        orca_logger.warning(
                            "相机启动失败，已启用占位图像；本次推理结果不可用于评估"
                        )
                        _shared_cameras = {}

                policy_runner = OpenPIPolicyRunner(
                    host=args.host,
                    port=args.port,
                    prompt=_prompts[0],
                    camera_name_map=camera_name_map,
                    cameras=_shared_cameras,
                    target_hw=_target_hw,
                    use_images=not args.no_images,
                )
                orca_logger.info(f"已连接策略服务器: {args.host}:{args.port}")
                orca_logger.info("策略服务已就绪")
                orca_logger.info("Prompt: " + " | ".join(_prompts))

            # 起点（不计入评分）：与采集前导段一致地驶向 L 型预备位姿，使首帧观测与训练数据同分布；
            # 随后以稳定后的实测位姿作为策略的初始末端目标。
            if args.start_pose == "ready":
                _ready_apply = drive_to_ready(manager, env, device, _init_action_apply,
                                              args.ready_move_steps, args.settle_steps)
                _start_state = storage.build_state(storage.obs_callback(env))
                _start_err = float(np.linalg.norm(_start_state[7:10] - _ready_apply["r_pos_b"]))
                _start_ang = float(np.degrees((R.from_quat(_start_state[10:14])
                                               * R.from_quat(_ready_apply["r_quat_b"]).inv()).magnitude()))
                _start_bad = _start_err > 0.03 or _start_ang > 15.0
                (orca_logger.warning if _start_bad else orca_logger.info)(
                    f"起点右手 ee={np.round(_start_state[7:10], 3).tolist()} 距预备位 {_start_err * 1000:.0f}mm"
                    f" 姿态偏差 {_start_ang:.0f}° quat={np.round(_start_state[10:14], 3).tolist()}"
                    f" 左手 ee={np.round(_start_state[0:3], 3).tolist()}"
                    + ("（超出训练首帧分布，本集结果不可信）" if _start_bad else ""))
            else:
                hold_target(manager, env, device, _init_action_apply, args.settle_steps)
                _start_state = storage.build_state(storage.obs_callback(env))
                orca_logger.info(f"起点右手 ee={np.round(_start_state[7:10], 3).tolist()}（默认位姿）")
            _init_action_apply = action_dict_for_apply(parse_policy_action(_start_state))
            device.set_target(**_init_action_apply)
            if _targets:
                button_monitor.check(0)
                orca_logger.info("起点到目标按钮帽距离: " + " ".join(
                    f"{_COLOR_CN[c]} {button_monitor.obs[c]['min_dist'] * 1000:.0f}mm" for c in _targets))
            if not args.no_images:
                hold_target(manager, env, device, _init_action_apply,
                            args.camera_warmup_steps, sleep_s=0.05)

            if scorer is not None:
                _sc_targets = (args.scoring_targets
                               if args.scoring_targets is not None else _targets)
                _sc_info = scorer.start_attempt(
                    args.task_id, targets=_sc_targets, prompt=" ".join(_prompts))
                _attempt_active = True
                orca_logger.info(
                    f"[scorer] attempt started: {_sc_info.get('attempt_id')}"
                    f" targets={_sc_targets}")

            step = 0
            truncated = False

            def run_segment(prompt: str, seg_targets: list[str], budget: int) -> None:
                """执行一条指令直到预算用尽 / 该段目标全部达成 / 中断。"""
                nonlocal step, truncated
                policy_runner.prompt = prompt
                seg_end = step + budget
                seg_done = False
                while (step < seg_end and not truncated and not seg_done
                       and not _interrupt.is_set()):
                    # 按模型输入 schema 构造机器人状态观测。
                    state = storage.build_state(storage.obs_callback(env))
                    action_chunk = policy_runner.infer_action_chunk(state)
                    if step == seg_end - budget:
                        _d_first = np.linalg.norm(action_chunk[0, 7:10] - state[7:10]) * 1000
                        _d_last = np.linalg.norm(action_chunk[-1, 7:10] - state[7:10]) * 1000
                        orca_logger.info(
                            f"[调试] 「{prompt}」首个动作块 {action_chunk.shape}: 右手目标相对 state 位移"
                            f" 首步 {_d_first:.0f}mm 末步 {_d_last:.0f}mm；"
                            f"末步目标 {np.round(action_chunk[-1, 7:10], 3).tolist()}"
                            f" 右爪 {np.round(action_chunk[-1, 16:18], 2).tolist()}"
                            f" 非有限值 {int((~np.isfinite(action_chunk)).sum())}")
                        if args.dump_dir:
                            os.makedirs(args.dump_dir, exist_ok=True)
                            for _k, _img in policy_runner.build_observation(state)["images"].items():
                                _hwc = np.transpose(_img, (1, 2, 0))
                                cv2.imwrite(os.path.join(args.dump_dir, f"ep{episode_index + 1}_{_k}.png"),
                                            cv2.cvtColor(_hwc, cv2.COLOR_RGB2BGR))
                                orca_logger.info(f"[调试] {_k}: shape={_img.shape} mean={_img.mean():.1f}")

                    for model_action in action_chunk:
                        if step >= seg_end or truncated or seg_done or _interrupt.is_set():
                            break

                        parsed_action = parse_policy_action(model_action)
                        _apply = action_dict_for_apply(parsed_action)
                        device.set_target(**_apply)

                        for _ in range(args.action_repeat):
                            if step >= seg_end or truncated or seg_done or _interrupt.is_set():
                                break

                            start_time = time.time()
                            _pt0 = time.perf_counter()
                            action = manager.run_controllers()
                            _pt1 = time.perf_counter()
                            _, _, _, truncated, _ = env.step(action)
                            button_monitor.check(step)
                            if not args.no_early_stop and all(
                                    button_monitor.obs[c]["hit_step"] is not None for c in seg_targets):
                                seg_done = True
                            _pt2 = time.perf_counter()
                            env.render()
                            _pt3 = time.perf_counter()

                            # 实时预览（复用同一套内存流相机）
                            if _shared_cameras and _preview_ready:
                                try:
                                    frames = []
                                    for _cn in _PREVIEW_CAMS:
                                        _cam = _shared_cameras.get(_cn)
                                        if _cam is not None:
                                            _f, _ = _cam.get_frame(format="rgb24")
                                            if _f is not None and _f.size > 0:
                                                _f = cv2.resize(_f, (_PREVIEW_W, _PREVIEW_H))
                                                frames.append(cv2.cvtColor(_f, cv2.COLOR_RGB2BGR))
                                    if frames:
                                        cv2.imshow("eval-preview", np.concatenate(frames, axis=1))
                                        if cv2.waitKey(1) & 0xFF == ord("q"):
                                            truncated = True
                                except Exception:
                                    pass

                            _pt4 = time.perf_counter()
                            _TPROF["ctrl"]    += _pt1 - _pt0
                            _TPROF["step"]    += _pt2 - _pt1
                            _TPROF["render"]  += _pt3 - _pt2
                            _TPROF["preview"] += _pt4 - _pt3
                            _TPROF["n"] += 1

                            if step % 200 == 0:
                                _n = max(1, _TPROF["n"])
                                orca_logger.info(
                                    f"[运行] 推理进度 {step}/{seg_end}"
                                    f"（每步 ctrl {_TPROF['ctrl'] / _n * 1e3:.1f}ms"
                                    f" step {_TPROF['step'] / _n * 1e3:.1f}ms"
                                    f" render {_TPROF['render'] / _n * 1e3:.1f}ms）")

                            step += 1
                            if truncated:
                                break

                            if args.sleep:
                                remain = manager.real_time_step - (time.time() - start_time)
                                if remain > 0:
                                    time.sleep(remain)

            for _si, (_p, _st) in enumerate(zip(_prompts, _seg_targets)):
                if truncated or _interrupt.is_set():
                    break
                if _si > 0 and args.start_pose == "ready":
                    # 下一条指令前回到预备位姿（与每集起点同一前导段；此段仍在 attempt 内）
                    _cur = action_dict_for_apply(parse_policy_action(
                        storage.build_state(storage.obs_callback(env))))
                    _ready_apply = drive_to_ready(manager, env, device, _cur,
                                                  args.ready_move_steps, args.settle_steps)
                    device.set_target(**action_dict_for_apply(parse_policy_action(
                        storage.build_state(storage.obs_callback(env)))))
                orca_logger.info(f"=== 指令 {_si + 1}/{len(_prompts)}: {_p} ===")
                run_segment(_p, _st, args.max_steps)

            if _interrupt.is_set():
                truncated = True
            completed = not truncated
            episode_results.append(completed)
            _res = button_monitor.summary()
            _res["steps"] = step
            press_results.append(_res)
            if _res["targets"]:
                _parts = []
                for c in _res["targets"]:
                    r = _res["detail"][c]
                    _official = ""
                    if r["site_dist"] != float("inf"):
                        _ok = r["site_dist"] <= _OFFICIAL_FULL_DIST
                        _official = (f",官方口径{'✓' if _ok else '✗'}{r['site_dist'] * 1000:.0f}mm"
                                     + ("" if r["site_nearest_ok"] else "/非最近×0.35"))
                    _parts.append(
                        f"{_COLOR_CN[c]}:{'✓' if r['hit_step'] is not None else '✗'}"
                        f"(压{r['max_disp'] * 1000:.1f}mm,距{r['min_dist'] * 1000:.0f}mm{_official})")
                orca_logger.info(
                    f"第 {episode_index + 1} 集判定: "
                    + ("成功" if _res["success"] else "失败") + " "
                    + " ".join(_parts)
                    + (f" 顺序{'正确' if _res['order_ok'] else '错误'}"
                       if _res["success"] and len(_res["targets"]) > 1 else "")
                    + f" 用时 {step} 步")
            if scorer is not None and _attempt_active:
                _attempt_active = False
                try:
                    _sc_sum = scorer.finish()
                    _sc_line = (
                        f"[scorer] 官方得分: {_sc_sum.get('score')}"
                        f"/{_sc_sum.get('max_score')}")
                    for _st in _sc_sum.get("step_results", []):
                        _sc_line += (
                            f" {_st.get('step_id')}:"
                            + ("PASS" if _st.get("success") else "FAIL"))
                    orca_logger.info(_sc_line)
                    print(_sc_line, flush=True)
                    if args.video_wait > 0 and os.environ.get(
                            "ORCA_SCORING_VIDEO_ENABLED", "1") != "0":
                        orca_logger.info(
                            f"[scorer] 等待任务视频上传 {args.video_wait:.0f}s...")
                        time.sleep(args.video_wait)
                except Exception as _sc_err:
                    orca_logger.warning(f"[scorer] finish 失败: {_sc_err}")
            orca_logger.info(
                f"第 {episode_index + 1} 集"
                + ("推理完成" if completed else "推理已中断")
            )
            if completed:
                scene_manager.show_ui_message(1, "推理完成", "0x00ff00", showtime=0)
            else:
                scene_manager.show_ui_message(1, "推理中断", "0xff8800", showtime=0)
            if _interrupt.is_set():
                orca_logger.info("用户中断，结束评估")
                break

        done_count = sum(1 for ok in episode_results if ok)
        orca_logger.info(f"全部 {len(episode_results)} 集完成: {done_count} 集完整跑完")
        _judged = [r for r in press_results if r["targets"]]
        if _judged:
            _ok = sum(1 for r in _judged if r["success"])
            _summary = (
                f"按压判定成功率: {_ok}/{len(_judged)}"
                + f"（判定阈值: 位移≥{args.success_disp * 1000:.1f}mm"
                + f" 或 距离≤{args.success_dist * 1000:.0f}mm）")
            orca_logger.info(_summary)
            print(_summary, flush=True)

        if not _interrupt.is_set():
            scene_manager.show_ui_message(1, "推理完成", "0x00ff00", showtime=0)
            orca_logger.info("推理完成，场景保持打开，按 Ctrl+C 退出")
            print("推理完成，场景保持打开，按 Ctrl+C 退出", flush=True)
            while not _interrupt.is_set():
                if device is not None:
                    action = manager.run_controllers()
                    env.step(action)
                env.render()
                time.sleep(0.05)

    finally:
        if scorer is not None and _attempt_active:
            try:
                scorer.finish()
                orca_logger.info("[scorer] 未完成的评分尝试已在退出时结束")
            except Exception:
                pass
        if _shared_cameras:
            close_cameras(_shared_cameras)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if _video_started:
            try:
                env.stop_save_video()
            except Exception:
                pass
        try:
            scene_manager.show_ui_message(1, "", showtime=0)
            env.render()
        except Exception:
            orca_logger.warning("界面状态清理未完成")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("已收到中断请求")
    except Exception as e:
        OrcaLog.get_instance().error(f"推理异常: {e}")
    finally:
        orca_logger.info("程序已退出")
        os._exit(0)
