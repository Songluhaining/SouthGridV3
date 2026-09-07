#!/usr/bin/env python
"""顺序解码 LeRobot 数据集的视频，导出 224x224 JPEG，供训练时直接读取。

缩放逻辑与 openpi_client.image_tools.resize_with_pad 完全一致（等比缩放 + 居中补黑边），
所以训练侧的 ResizeImages(224,224) 会命中"尺寸已对，原样返回"的短路分支。

  抽帧: python extract_frames.py --dataset <数据集根> --out /tmp/frames --jobs 3
  校验: python extract_frames.py --dataset <数据集根> --out /tmp/frames --verify 20
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import random

import av
import numpy as np
from PIL import Image

H = W = 224


def resize_with_pad(im):
    cw, ch = im.size
    if (cw, ch) == (W, H):
        return im
    ratio = max(cw / W, ch / H)
    rw, rh = int(cw / ratio), int(ch / ratio)
    r = im.resize((rw, rh), resample=Image.BILINEAR)
    canvas = Image.new(r.mode, (W, H), 0)
    canvas.paste(r, (max(0, (W - rw) // 2), max(0, (H - rh) // 2)))
    return canvas


def extract_one(job):
    src, dst, expected = job
    dst.mkdir(parents=True, exist_ok=True)
    if len(list(dst.glob("frame_*.jpg"))) == expected:
        return (str(src), expected, "skip")
    n = 0
    with av.open(str(src)) as c:
        for frame in c.decode(c.streams.video[0]):
            resize_with_pad(frame.to_image().convert("RGB")).save(
                dst / f"frame_{n:06d}.jpg", quality=92, subsampling=0)
            n += 1
    return (str(src), n, "ok" if n == expected else f"帧数不符(应为 {expected})")


def build_jobs(dataset, out):
    root, out = pathlib.Path(dataset), pathlib.Path(out)
    info = json.loads((root / "meta/info.json").read_text())
    keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    cs, tpl = int(info.get("chunks_size", 1000)), info["video_path"]
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for e in eps:
        i = e["episode_index"]
        for k in keys:
            jobs.append((root / tpl.format(episode_chunk=i // cs, video_key=k, episode_index=i),
                         out / k / f"episode_{i:06d}", e["length"]))
    return jobs


def verify(jobs, n):
    random.Random(0).shuffle(jobs)
    worst = 0.0
    for src, dst, length in jobs[:n]:
        idx = random.randrange(length)
        ref = None
        with av.open(str(src)) as c:
            for i, fr in enumerate(c.decode(c.streams.video[0])):
                if i == idx:
                    ref = resize_with_pad(fr.to_image().convert("RGB"))
                    break
        got = Image.open(dst / f"frame_{idx:06d}.jpg").convert("RGB")
        d = float(np.abs(np.asarray(ref, float) - np.asarray(got, float)).mean())
        worst = max(worst, d)
        print(f"  {dst.parent.name}/{dst.name} 第 {idx} 帧: 平均像素差 {d:.2f}")
    print(f"最大平均像素差 {worst:.2f}（JPEG 压缩所致，小于 5 正常；大于 20 说明帧对不上）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--verify", type=int, default=0)
    a = ap.parse_args()
    jobs = build_jobs(a.dataset, a.out)
    if a.verify:
        return verify(jobs, a.verify)
    print(f"待处理 {len(jobs)} 个视频，并行 {a.jobs}")
    bad = 0
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for n, (src, cnt, st) in enumerate(ex.map(extract_one, jobs, chunksize=4), 1):
            if st not in ("ok", "skip"):
                bad += 1
                print("!!", src, st, flush=True)
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print(f"完成，异常 {bad} 个")


if __name__ == "__main__":
    main()
