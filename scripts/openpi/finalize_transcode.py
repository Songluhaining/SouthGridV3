#!/usr/bin/env python
"""转码后的校验与切换。

先逐条比对新视频的帧数与 meta/episodes.jsonl 记录的长度（帧数对不上会让
LeRobot 按时间戳取帧时错位，是转码唯一真正危险的失败方式）；全部通过后，
加 --apply 才会把 videos 换成新目录，并同步改写 meta/info.json 的分辨率与编码。

用法:
    python finalize_transcode.py --dataset ~/wfn/lerobot_datasets/<repo_id>          # 只校验
    python finalize_transcode.py --dataset ~/wfn/lerobot_datasets/<repo_id> --apply  # 校验并切换
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys


def probe_frame_count(path: pathlib.Path) -> int:
    """先读容器里的帧数字段，读不到再数一遍包，避免整段解码。"""
    for args, field in (
        (["-show_entries", "stream=nb_frames"], "nb_frames"),
        (["-count_packets", "-show_entries", "stream=nb_read_packets"], "nb_read_packets"),
    ):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *args, "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        ).stdout.strip()
        if out and out != "N/A":
            return int(out)
    raise RuntimeError(f"无法读出帧数: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot 数据集根目录")
    ap.add_argument("--new-dir", default="videos_h264", help="转码产物目录名")
    ap.add_argument("--backup-dir", default="videos_av1_backup", help="原视频备份目录名")
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--codec", default="h264")
    ap.add_argument("--limit", type=int, default=0, help="只抽查前 N 条 episode（0 = 全查）")
    ap.add_argument("--apply", action="store_true", help="校验通过后真正切换目录并改 info.json")
    args = ap.parse_args()

    root = pathlib.Path(args.dataset).expanduser()
    new_root = root / args.new_dir
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    chunks_size = int(info.get("chunks_size", 1000))
    template = info["video_path"]

    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        episodes = episodes[: args.limit]

    print(f"校验 {len(episodes)} 条 episode × {len(video_keys)} 路相机 ...")
    bad, missing = [], []
    for n, ep in enumerate(episodes, 1):
        idx, length = ep["episode_index"], ep["length"]
        for key in video_keys:
            rel = template.format(episode_chunk=idx // chunks_size, video_key=key, episode_index=idx)
            path = new_root / pathlib.Path(rel).relative_to("videos")
            if not path.exists():
                missing.append(str(path))
                continue
            got = probe_frame_count(path)
            if got != length:
                bad.append((str(path), length, got))
        if n % 100 == 0:
            print(f"  已校验 {n}/{len(episodes)}")

    if missing:
        print(f"\n缺少 {len(missing)} 个转码产物，例如: {missing[:3]}")
    if bad:
        print(f"\n帧数不一致 {len(bad)} 个，例如: {bad[:3]}（格式: 文件, 应有帧数, 实得帧数）")
    if missing or bad:
        print("\n未通过校验，没有做任何改动。重跑 transcode_videos.sh 补齐后再试。")
        return 1

    print("\n校验通过：所有视频帧数与 episodes.jsonl 完全一致。")
    if not args.apply:
        print("这是只读检查。确认无误后加 --apply 执行切换。")
        return 0

    backup = root / args.backup_dir
    if backup.exists():
        print(f"备份目录已存在，先删掉或改名: {backup}")
        return 1
    shutil.copy2(info_path, info_path.with_suffix(".json.bak"))
    (root / "videos").rename(backup)
    new_root.rename(root / "videos")

    for key in video_keys:
        feat = info["features"][key]
        names = feat.get("names") or ["channels", "height", "width"]
        shape = list(feat["shape"])
        shape[names.index("height")] = args.height
        shape[names.index("width")] = args.width
        feat["shape"] = shape
        feat["info"]["video.height"] = args.height
        feat["info"]["video.width"] = args.width
        feat["info"]["video.codec"] = args.codec
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"已切换: 原视频 -> {backup.name}，新视频 -> videos")
    print(f"已更新 meta/info.json（{args.width}x{args.height}, {args.codec}），旧版备份为 info.json.bak")
    print("确认训练跑通后，再删除备份目录以释放空间。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
