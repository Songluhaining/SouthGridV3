#!/usr/bin/env python
"""Offline probe: no simulator needed.

Two independent measurements, because they answer different questions:

1. Action accuracy -- feed recorded observations, compare the predicted action chunk
   against the recorded ground truth. This says whether the policy can CONTINUE a
   motion that is already under way. It barely exercises language: mid-trajectory the
   arm is already committed, so the state alone predicts the next step. A good score
   here does NOT mean the policy knows which button to go to.

2. Language grounding at frame 0 -- at the ready pose, where all four buttons are in
   view and ONLY the instruction says which one to press, run all four colour prompts
   and check which button each prediction actually lands on. This is the measurement
   that decides the closed-loop task.

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
    """One real single-colour instruction per colour, taken from the dataset itself."""
    out = {}
    for line in (ds / "meta/quality.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seq = r.get("sequence") or []
        if len(seq) == 1 and seq[0] not in out:
            out[seq[0]] = r["task"]
    return {c: out[c] for c in COLORS if c in out}


def color_targets(ds, chunks_size, limit=30):
    """Ground-truth target per colour: mean right-hand position at the deepest reach
    (the frame with the largest x) over up to `limit` single-colour episodes."""
    per = {}
    for line in (ds / "meta/quality.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seq = r.get("sequence") or []
        if not r.get("kept") or len(seq) != 1:
            continue
        c = seq[0]
        per.setdefault(c, [])
        if len(per[c]) >= limit:
            continue
        i = r["episode_index"]
        tb = pq.read_table(ds / f"data/chunk-{i // chunks_size:03d}/episode_{i:06d}.parquet",
                           columns=["observation.state"])
        S = np.stack(tb.column("observation.state").to_pylist()).astype(np.float64)
        per[c].append(S[int(np.argmax(S[:, 7])), 7:10])
    return {c: np.stack(v).mean(0) for c, v in per.items() if v}


def read_frames(path, wanted):
    out, want = {}, set(wanted)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                out[i] = np.asarray(fr.to_image().convert("RGB"))
                if len(out) == len(want):
                    break
    return out


def load_obs(ds, info, idx, ts, frames_root=None):
    """Observations, states and actions for one episode at the given frame indices."""
    cs = int(info.get("chunks_size", 1000))
    ch = idx // cs
    tb = pq.read_table(ds / f"data/chunk-{ch:03d}/episode_{idx:06d}.parquet",
                       columns=["observation.state", "action"])
    st = np.stack(tb.column("observation.state").to_pylist()).astype(np.float32)
    act = np.stack(tb.column("action").to_pylist()).astype(np.float32)
    pics = {}
    for cam in CAMS:
        if frames_root:
            root = pathlib.Path(frames_root) / cam / f"episode_{idx:06d}"
            pics[cam] = {t: np.asarray(Image.open(root / f"frame_{t:06d}.jpg").convert("RGB")) for t in ts}
        else:
            pics[cam] = read_frames(
                ds / info["video_path"].format(episode_chunk=ch, video_key=cam, episode_index=idx), ts)
    obs = {t: {"state": st[t], "images": {c.split(".")[-1]: pics[c][t] for c in CAMS}} for t in ts}
    return obs, st, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="pi05_g1_button_lora")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--per_episode", type=int, default=4)
    ap.add_argument("--ground_episodes", type=int, default=8,
                    help="episodes used for the frame-0 language grounding test")
    ap.add_argument("--frames", default=None, help="optional pre-extracted jpeg root")
    args = ap.parse_args()

    ds = pathlib.Path(args.dataset)
    info = json.loads((ds / "meta/info.json").read_text())
    cs = int(info.get("chunks_size", 1000))
    eps = [json.loads(l) for l in (ds / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rnd = random.Random(0)
    rnd.shuffle(eps)

    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt)
    np.set_printoptions(precision=4, suppress=True)

    # ---- 1. action accuracy: can it continue a motion already under way ----
    errs, moves, horizon = [], [], 0
    for e in eps[:args.episodes]:
        idx, n = e["episode_index"], e["length"]
        ts = sorted(rnd.sample(range(0, max(1, n - 16)), args.per_episode))
        obs, st, act = load_obs(ds, info, idx, ts, args.frames)
        prompt = e["tasks"][0]
        for t in ts:
            pred = np.asarray(policy.infer({**obs[t], "prompt": prompt})["actions"])
            horizon = pred.shape[0]
            gt = act[t:t + horizon]
            errs.append(np.abs(pred[:len(gt)] - gt).mean(0))
            moves.append(np.abs(gt[-1] - st[t]))
    E, M = np.stack(errs), np.stack(moves)
    err_mm = float(np.linalg.norm(E.mean(0)[7:10])) * 1000
    mov_mm = float(np.linalg.norm(M.mean(0)[7:10])) * 1000
    print(f"\n[1] action accuracy   samples={len(E)}  chunk={horizon}")
    print("    per-dim MAE:", E.mean(0))
    print(f"    right hand: error {err_mm:.1f} mm   true motion {mov_mm:.1f} mm"
          f"   relative {err_mm / max(mov_mm, 1e-9) * 100:.0f}%")
    print("    NOTE: this measures continuation, not language. See the module docstring.")

    # ---- 2. language grounding at frame 0 ----
    prompts = color_prompts(ds)
    tgt = color_targets(ds, cs)
    print("\n[2] language grounding at frame 0 (ready pose)")
    print("    ground-truth target per colour (mean right-hand position at deepest reach):")
    for c in COLORS:
        if c in tgt:
            print(f"      {c:7s} {np.round(tgt[c], 4)}")
    pairs = [(x, y, float(np.linalg.norm(tgt[x] - tgt[y])) * 1000)
             for k, x in enumerate(COLORS) for y in COLORS[k + 1:] if x in tgt and y in tgt]
    print("    true pairwise separation: " + ", ".join(f"{x}-{y} {d:.0f}mm" for x, y, d in pairs))

    hit = tot = 0
    spreads, dists = [], []
    for n_ep, e in enumerate(eps[:args.ground_episodes]):
        idx = e["episode_index"]
        obs, _, _ = load_obs(ds, info, idx, [0], args.frames)
        o0 = obs[0]
        preds = {}
        for c, p in prompts.items():
            pr = np.asarray(policy.infer({**o0, "prompt": p})["actions"])
            preds[c] = pr[-1, 7:10]
        for c, v in preds.items():
            near = min(tgt, key=lambda k: float(np.linalg.norm(v - tgt[k])))
            dists.append(float(np.linalg.norm(v - tgt[c])) * 1000)
            tot += 1
            hit += int(near == c)
            if n_ep == 0:
                print(f"      [{prompts[c]}] -> {np.round(v, 4)}  nearest={near}"
                      f"  dist to {c} target {float(np.linalg.norm(v - tgt[c])) * 1000:.0f} mm")
        spreads.append(max(float(np.linalg.norm(preds[x] - preds[y])) for x in preds for y in preds) * 1000)

    print(f"    grounding accuracy {hit}/{tot} = {hit / max(tot, 1) * 100:.0f}%   (25% = chance)")
    print(f"    median distance to the commanded target: {np.median(dists):.0f} mm")
    print(f"    median spread across the four prompts: {np.median(spreads):.0f} mm")
    print("    Read it like this: the spread should approach the true separation printed above.")
    print("    Under ~30 mm means the instruction barely moves the target -- the policy is")
    print("    heading to the same place whatever colour you ask for.")


if __name__ == "__main__":
    main()
