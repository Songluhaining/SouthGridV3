"""把每集裁到「右手探入最深」那一帧，丢掉之后的保持段与后撤段。

动机（实测，见 docs/g1_button_v11_score_ceiling.md）：
- 评测里段间是脚本回位（eval_..._button_lerobot.py 的 drive_to_ready），
  策略永远不需要自己后撤，但 v6 每集有 25.5% 的帧在教后撤。
- 开环探测显示 v9 在距最深点还有 9.6mm 时就已经预测后退（真值仍在前进），
  与闭环实测的 9~13mm 官方距离欠冲吻合。

只改 parquet 行数与 meta 计数；视频原样复用（解码按时间戳取帧，多余尾帧不会被读到）。
"""
import argparse
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

R_POS_X = 7          # observation.state 里右手位置 x 的下标（接近方向）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep_after", type=int, default=0,
                    help="最深帧之后额外保留的帧数（默认 0 = 裁到最深帧为止）")
    ap.add_argument("--videos", choices=("symlink", "copy", "skip"), default="symlink")
    args = ap.parse_args()

    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    eps = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    info = json.loads((src / "meta/info.json").read_text(encoding="utf-8"))

    running = 0
    new_eps, kept_ratio = [], []
    for e in eps:
        idx = e["episode_index"]
        rel = f"data/chunk-{idx // info['chunks_size']:03d}/episode_{idx:06d}.parquet"
        t = pq.read_table(src / rel)
        st = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
        n = len(st)
        cut = min(n, int(np.argmax(st[:, R_POS_X])) + 1 + args.keep_after)
        cut = max(cut, 2)                       # 至少留两帧，动作配对才成立

        t = t.slice(0, cut)
        cols = {name: t.column(name) for name in t.schema.names}
        cols["index"] = pa.array(np.arange(running, running + cut, dtype=np.int64))
        cols["frame_index"] = pa.array(np.arange(cut, dtype=np.int64))
        out = pa.table(cols, schema=t.schema)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, dst / rel)

        running += cut
        kept_ratio.append(cut / n)
        new_eps.append({**e, "length": cut})

    (dst / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in new_eps), encoding="utf-8")
    info = {**info, "total_frames": running}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    for name in ("tasks.jsonl", "episodes_stats.jsonl", "quality.jsonl"):
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, dst / "meta" / name)

    if args.videos == "symlink":
        tgt = dst / "videos"
        if not tgt.exists():
            tgt.symlink_to(src.resolve() / "videos", target_is_directory=True)
    elif args.videos == "copy":
        shutil.copytree(src / "videos", dst / "videos", dirs_exist_ok=True)

    print(f"{len(eps)} 集：{sum(e['length'] for e in eps)} 帧 → {running} 帧 "
          f"（保留 {100 * running / max(1, sum(e['length'] for e in eps)):.1f}%，"
          f"每集平均保留 {100 * float(np.mean(kept_ratio)):.1f}%）")
    print(f"输出：{dst}")


if __name__ == "__main__":
    main()
