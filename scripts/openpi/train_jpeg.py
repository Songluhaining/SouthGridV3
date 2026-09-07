#!/usr/bin/env python
"""训练时不解码视频，直接读 extract_frames.py 导出的 224x224 JPEG。

替换 LeRobot 唯一的取帧入口：把「打开 mp4 + 跳转 + 解码」换成「读一张小 JPEG」。
补丁写在模块顶层，spawn 出来的数据读取子进程重新导入本文件时同样生效。

    FRAMES_ROOT=/tmp/frames python scripts/train_jpeg.py <配置名> [其它参数...]
"""
import importlib
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

FRAMES_ROOT = pathlib.Path(os.environ.get("FRAMES_ROOT", "/tmp/frames"))
FPS = float(os.environ.get("FRAMES_FPS", "20"))


def _patch_decode() -> None:
    mods = []
    for name in ("lerobot.common.datasets.video_utils",
                 "lerobot.datasets.video_utils",
                 "lerobot.common.datasets.lerobot_dataset",
                 "lerobot.datasets.lerobot_dataset"):
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass

    def decode(video_path, timestamps, tolerance_s=None, backend=None, **kwargs):
        p = pathlib.Path(video_path)
        base = FRAMES_ROOT / p.parent.name / p.stem
        out = []
        for ts in timestamps:
            idx = int(round(float(ts) * FPS))
            f = base / f"frame_{idx:06d}.jpg"
            if not f.exists():
                raise FileNotFoundError(f"缺少帧文件 {f}（时间戳 {ts}），请先跑 extract_frames.py")
            arr = np.asarray(Image.open(f).convert("RGB"))
            out.append(torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0))
        return torch.stack(out)

    n = 0
    for m in mods:
        for attr in ("decode_video_frames_torchvision", "decode_video_frames"):
            if hasattr(m, attr):
                setattr(m, attr, decode)
                n += 1
    print(f"[jpeg] pid={os.getpid()} 已替换 {n} 处取帧入口，帧目录 {FRAMES_ROOT}", file=sys.stderr)


_patch_decode()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import openpi.training.config as _config

    importlib.import_module("train").main(_config.cli())
