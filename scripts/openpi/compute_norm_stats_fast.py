#!/usr/bin/env python
"""与 openpi 官方 scripts/compute_norm_stats.py 完全等价，但不解码 mp4。

归一化统计只用到 state / actions（都在 parquet 里），图像不参与，
因此把 LeRobot 唯一的视频解码入口换成"返回全零占位张量"。

三个要点：
1. 补丁写在模块顶层执行 —— 数据加载用 spawn 方式起子进程时，子进程会重新
   导入本文件（名字变成 __mp_main__），顶层代码照样跑，补丁才能落到真正
   干活的子进程上。写在 main() 里只有父进程生效，等于没改。
2. 官方脚本以正常模块名导入（不是 __main__），它里面定义的变换类才能被
   子进程按模块名还原回来，否则报 "Can't get attribute 'XXX' on __main__"。
3. 占位图默认缩到 32x32，避免每批白白分配和缩放几百 MB 的零张量。

放到 openpi 仓库的 scripts/ 目录下，用法与原脚本一致：
    python scripts/compute_norm_stats_fast.py --config-name pi05_g1_button_lora

两个应急开关（出问题时才用）：
    NORM_STATS_FAKE_FULL=1      占位图用数据集声明的原始尺寸（变换链挑尺寸时）
    NORM_STATS_NUM_WORKERS=0    强制单进程读数据（子进程再出岔子时）
"""

import importlib
import os
import pathlib
import sys

import torch

_FAKE_SIZE = 32


def _patch_lerobot_video_decode() -> None:
    """把 LeRobotDataset._query_videos 换成全零占位，跳过 mp4 解码。"""
    mod = None
    for path in ("lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"):
        try:
            mod = importlib.import_module(path)
            break
        except ImportError:
            continue
    if mod is None:
        raise RuntimeError("找不到 lerobot 的 lerobot_dataset 模块，请检查 lerobot 版本")

    use_full = os.environ.get("NORM_STATS_FAKE_FULL") == "1"

    def _query_videos(self, query_timestamps, ep_idx):  # noqa: ANN001
        item = {}
        for key, timestamps in query_timestamps.items():
            feature = self.meta.features[key]
            declared = tuple(feature["shape"])
            if use_full:
                shape = declared
            else:
                # 只把高宽换成极小值，通道维保持原样，保证"通道在前"的布局判断不受影响
                names = feature.get("names") or ["channels", "height", "width"]
                shape = tuple(
                    _FAKE_SIZE if n in ("height", "width") else declared[i]
                    for i, n in enumerate(names)
                )
            frames = torch.zeros((len(timestamps), *shape), dtype=torch.float32)
            item[key] = frames.squeeze(0) if len(timestamps) == 1 else frames
        return item

    mod.LeRobotDataset._query_videos = _query_videos
    size = "原尺寸" if use_full else f"{_FAKE_SIZE}x{_FAKE_SIZE}"
    print(f"[fast] pid={os.getpid()} 视频解码已关闭（占位图 {size}）", file=sys.stderr)


def _maybe_force_num_workers() -> None:
    """按需强制数据加载的子进程数（默认不干预，沿用官方脚本的设置）。"""
    raw = os.environ.get("NORM_STATS_NUM_WORKERS")
    if raw is None:
        return
    n = int(raw)
    import torch.utils.data as tud

    original_init = tud.DataLoader.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001
        args = list(args)
        if len(args) >= 6:  # num_workers 是第 6 个位置参数
            args[5] = n
        else:
            kwargs["num_workers"] = n
        if n == 0:
            for key in ("prefetch_factor", "persistent_workers", "multiprocessing_context"):
                kwargs.pop(key, None)
        original_init(self, *args, **kwargs)

    tud.DataLoader.__init__ = patched_init
    print(f"[fast] 强制 num_workers={n}", file=sys.stderr)


def _load_original():
    """以正常模块名导入官方脚本，保证它定义的类可以被子进程还原。"""
    scripts_dir = pathlib.Path(__file__).resolve().parent
    path = scripts_dir / "compute_norm_stats.py"
    env = os.environ.get("OPENPI_NORM_SCRIPT")
    if env:
        path = pathlib.Path(env).resolve()
        scripts_dir = path.parent
    if not path.exists():
        raise FileNotFoundError(
            "找不到官方 compute_norm_stats.py：请把本脚本放进 openpi 的 scripts/ 目录，"
            "或用环境变量 OPENPI_NORM_SCRIPT 指定它的路径"
        )
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(path.stem)


# 顶层执行：父进程和每个 spawn 出来的子进程都会走到这里
_patch_lerobot_video_decode()
_maybe_force_num_workers()


if __name__ == "__main__":
    import tyro

    tyro.cli(_load_original().main)
