"""SouthGrid G1 OmniPicker 按钮任务的 openpi 数据变换。

数据集：LeRobot v2.1，state/action 均为 18 维
  [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4), l_grip(2), r_grip(2)]
action[i] = state[i+1]（绝对下一步位姿，无需 delta 变换）。
相机：cam_head（头部第三人称）→ base_0_rgb；cam_wrist_r（右腕）→ right_wrist_0_rgb。
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

G1_BUTTON_ACTION_DIM = 18


def make_g1_button_example() -> dict:
    return {
        "observation/state": np.random.rand(G1_BUTTON_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "按红色按钮",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1ButtonInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "observation/image" in data:
            # 训练路径：数据集经 RepackTransform 后的键
            base_image = _parse_image(data["observation/image"])
            wrist_image = _parse_image(data["observation/wrist_image"])
            state = data["observation/state"]
        else:
            # 推理路径：交付 eval 客户端发送 {"state", "images": {cam_head, cam_wrist_r}, "prompt"}
            imgs = data.get("images", {})
            base_image = _parse_image(imgs["cam_head"])
            wrist_image = _parse_image(imgs["cam_wrist_r"])
            state = data["state"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # 无左腕相机：置零图并按模型类型决定 mask（与官方 libero 模板一致）
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1ButtonOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :G1_BUTTON_ACTION_DIM])}
