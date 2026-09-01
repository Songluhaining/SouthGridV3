"""关节 PD 锁定控制器：把一条手臂刚性钉在当前关节位形。"""
import mujoco
import numpy as np


class JointLockController:
    """关节 PD 锁定：tau = kp*(q_lock - q) - kd*qv + 重力/科氏前馈（qfrc_bias）。

    G1 OmniPicker 左臂在采集与推理中都用它替代 OSC，使左臂完全静止
    （实测：重力前馈 + 分执行器限幅下锁定误差 <0.1 mrad）。
    鸭子类型兼容 arm OSC 控制器接口：设备下发的末端目标一律忽略。

    注意：内部缓存原生 mjModel 的关节/自由度索引，env.reset() 重载模型后必须重建。
    """

    def __init__(self, env, arm_conf, ctrl_names, kp=150.0, kd=10.0):
        self.env = env
        self.joint_names = [env.joint(j) for j in arm_conf["joint_names"]]
        self.ctrl_name = list(ctrl_names)
        self.init_ctrl = {n: 0.0 for n in self.ctrl_name}
        self.kp, self.kd = kp, kd
        self.q_lock = None
        self.ctrl_index: list[int] = []
        m_ = env.gym._mjModel
        self._jids = [mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_JOINT, j)
                      for j in self.joint_names]
        self._dadr = [m_.jnt_dofadr[j] for j in self._jids]
        self._gear = None
        self._lim = None

    def init_ctrl_index(self):
        self.ctrl_index = [self.env.model.actuator_name2id(n) for n in self.ctrl_name]
        return self.ctrl_index

    def get_init_ctrl(self):
        return {self.env.model.actuator_name2id(n): v for n, v in self.init_ctrl.items()}

    def reset(self):
        """以当前关节位形为锁定目标。"""
        qpos = self.env.query_joint_qpos(self.joint_names)
        self.q_lock = np.array([float(np.ravel(qpos[j])[0]) for j in self.joint_names])

    def update_action_position(self, *_):
        pass

    def update_action_axisangle(self, *_):
        pass

    def run_controller(self) -> dict:
        if self.q_lock is None:
            self.reset()
        m_, d_ = self.env.gym._mjModel, self.env.gym._mjData
        if self._gear is None:
            self._gear = [max(1e-6, abs(float(m_.actuator_gear[i][0]))) for i in self.ctrl_index]
            self._lim = [0.97 * float(m_.actuator_ctrlrange[i][1]) for i in self.ctrl_index]
        qpos = self.env.query_joint_qpos(self.joint_names)
        qvel = self.env.query_joint_qvel(self.joint_names)
        out = {}
        for k, (idx, j, ql) in enumerate(zip(self.ctrl_index, self.joint_names, self.q_lock)):
            q = float(np.ravel(qpos[j])[0])
            qv = float(np.ravel(qvel[j])[0])
            tau = self.kp * (ql - q) - self.kd * qv + float(d_.qfrc_bias[self._dadr[k]])
            out[idx] = float(np.clip(tau / self._gear[k], -self._lim[k], self._lim[k]))
        return out
