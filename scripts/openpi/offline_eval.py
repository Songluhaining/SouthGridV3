#!/usr/bin/env python
"""Offline replay eval: no simulator. Feed recorded observations to the policy
and compare predicted action chunks against the recorded ground-truth actions.

  python offline_eval.py --ckpt <checkpoint dir> --dataset <lerobot dataset root>
"""
import argparse
import json
import pathlib
import random

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

CAMS = ("observation.images.cam_head", "observation.images.cam_wrist_r")
COLORS = ("red", "green", "yellow", "blue")


def color_prompts(ds):
    """Pick one real single-button Chinese instruction per color from quality.jsonl."""
    out = {}
    for line in (ds / "meta/quality.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seq = r.get("sequence") or []
        if len(seq) == 1 and seq[0] not in out:
            out[seq[0]] = r["task"]
    return [out[c] for c in COLORS if c in out]


def read_frames(path, wanted):
    """Sequentially decode one video, keep only the frames we need."""
    out, want = {}, set(wanted)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                out[i] = np.asarray(fr.to_image().convert("RGB"))
                if len(out) == len(want):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="pi05_g1_button_lora")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--per_episode", type=int, default=4)
    ap.add_argument("--frames", default=None, help="optional pre-extracted jpeg root")
    a = ap.parse_args()

    ds = pathlib.Path(a.dataset)
    info = json.loads((ds / "meta/info.json").read_text())
    cs, vtpl = int(info.get("chunks_size", 1000)), info["video_path"]
    eps = [json.loads(l) for l in (ds / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rnd = random.Random(0)
    rnd.shuffle(eps)

    policy = _policy_config.create_trained_policy(_config.get_config(a.config), a.ckpt)
    errs, moves, sample, h = [], [], None, 0

    for e in eps[:a.episodes]:
        i, n = e["episode_index"], e["length"]
        ch = i // cs
        tb = pq.read_table(ds / f"data/chunk-{ch:03d}/episode_{i:06d}.parquet",
                           columns=["observation.state", "action"])
        st = np.stack(tb.column("observation.state").to_pylist()).astype(np.float32)
        act = np.stack(tb.column("action").to_pylist()).astype(np.float32)
        ts = sorted(rnd.sample(range(0, max(1, n - 16)), a.per_episode))
        pics = {}
        for cam in CAMS:
            if a.frames:
                root = pathlib.Path(a.frames) / cam / f"episode_{i:06d}"
                pics[cam] = {t: np.asarray(Image.open(root / f"frame_{t:06d}.jpg").convert("RGB")) for t in ts}
            else:
                pics[cam] = read_frames(ds / vtpl.format(episode_chunk=ch, video_key=cam, episode_index=i), ts)
        prompt = e["tasks"][0]
        for t in ts:
            obs = {"state": st[t], "prompt": prompt,
                   "images": {c.split(".")[-1]: pics[c][t] for c in CAMS}}
            pred = np.asarray(policy.infer(obs)["actions"])
            h = pred.shape[0]
            gt = act[t:t + h]
            errs.append(np.abs(pred[:len(gt)] - gt).mean(0))
            moves.append(np.abs(gt[-1] - st[t]))
            if sample is None:
                sample = (i, t, st[t], prompt, {c.split(".")[-1]: pics[c][t] for c in CAMS})

    E, M = np.stack(errs), np.stack(moves)
    np.set_printoptions(precision=4, suppress=True)
    err_mm = np.linalg.norm(E.mean(0)[7:10]) * 1000
    mov_mm = np.linalg.norm(M.mean(0)[7:10]) * 1000
    print(f"\nsamples={len(E)}  action_chunk={h}")
    print("per-dim MAE:", E.mean(0))
    print(f"right-hand position: error {err_mm:.1f} mm   true motion {mov_mm:.1f} mm"
          f"   relative {err_mm / max(mov_mm, 1e-9) * 100:.0f}%")

    i, t, s0, prompt, imgs = sample
    print(f"\n--- language sensitivity (episode {i} frame {t}, original task: {prompt})")
    tips = []
    for p in color_prompts(ds):
        pr = np.asarray(policy.infer({"state": s0, "images": imgs, "prompt": p})["actions"])
        tips.append(pr[-1, 7:10])
        print(f"  {p}: last-step target {np.round(pr[-1, 7:10], 4)}")
    d = max(float(np.linalg.norm(x - y)) for x in tips for y in tips)
    print(f"max spread across the 4 prompts: {d * 1000:.1f} mm"
          f"   (<10 mm = language ignored, >50 mm = language works)")


if __name__ == "__main__":
    main()
