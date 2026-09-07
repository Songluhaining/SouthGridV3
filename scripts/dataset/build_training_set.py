#!/usr/bin/env python
"""把 v4 主数据集(下采样四按钮集)与补采的单色数据集合并成一份训练集，并打印最终分布。

背景：v4 的四按钮集占了 62% 的训练帧，把模型压成"忽略语言"(单色接地正确率 25%≈随机)。
修法是让"单目标(需要读语言才知道去哪)"的帧占比回到能工作的水平(v3 为 12.8%、接地 96%)。
光补采压不下四按钮占比，必须同时下采样 v4 的四按钮集——本脚本一步完成。

用法:
  # 只看分布、不落盘(先规划补多少)
  python build_training_set.py --base <v4目录> --keep-4button 200 --report-only
  # 正式构建(补采数据集就绪后)
  python build_training_set.py --base <v4目录> --supplement <补采目录> \
      --keep-4button 200 --out <输出目录>

依赖 pyarrow（LeRobot 自带）。视频直接按文件复制，不重新编码。
"""
import argparse, json, pathlib, shutil, collections, random
import pyarrow.parquet as pq
import pyarrow as pa


def load_meta(root: pathlib.Path):
    info = json.loads((root / "meta/info.json").read_text())
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = {json.loads(l)["episode_index"]: json.loads(l)
             for l in (root / "meta/episodes_stats.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    # 每集按钮数：优先用 quality.jsonl 的 sequence，退化到解析 task 文本里的颜色数
    nbtn = {}
    qp = root / "meta/quality.jsonl"
    if qp.exists():
        for l in qp.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("episode_index") is not None and r.get("sequence"):
                nbtn[r["episode_index"]] = len(r["sequence"])
    for e in eps:
        if e["episode_index"] not in nbtn:
            t = e["tasks"][0]
            nbtn[e["episode_index"]] = sum(c in t for c in ("红", "绿", "蓝", "黄")) or 1
    return info, eps, stats, nbtn


def pick(root, eps, nbtn, keep4, rng, is_base):
    """返回要保留的 episode_index 列表。base: 全留 1/2/3 键 + 下采样 4 键；补采: 全留。"""
    if not is_base:
        return [e["episode_index"] for e in eps]
    keep, four = [], []
    for e in eps:
        i = e["episode_index"]
        (four if nbtn[i] == 4 else keep).append(i)
    rng.shuffle(four)
    keep += four[:keep4]
    return sorted(keep)


def build(args):
    rng = random.Random(0)
    base = pathlib.Path(args.base)
    bi, beps, bstats, bnb = load_meta(base)
    sources = [("base", base, bi, {e["episode_index"]: e for e in beps}, bstats, bnb,
                pick(base, beps, bnb, args.keep_4button, rng, True))]
    if args.supplement:
        sup = pathlib.Path(args.supplement)
        si, seps, sstats, snb = load_meta(sup)
        sources.append(("sup", sup, si, {e["episode_index"]: e for e in seps}, sstats, snb,
                        pick(sup, seps, snb, 0, rng, False)))

    # 汇总分布
    fr = collections.Counter(); ep = collections.Counter()
    plan = []  # (src_root, old_idx, epmeta, statmeta, nbtn, vtpl_old, data_old)
    for name, root, info, epmap, stats, nb, keep in sources:
        for i in keep:
            L = epmap[i]["length"]; n = nb[i]
            fr[n] += L; ep[n] += 1
            plan.append((root, i, epmap[i], stats.get(i), n, info))
    if getattr(args, "sim_single", 0):
        fr[1] += args.sim_single * 302; ep[1] += args.sim_single
    if getattr(args, "sim_two", 0):
        fr[2] += args.sim_two * 609; ep[2] += args.sim_two
    tot = sum(fr.values())
    tag = "（含模拟补采）" if (getattr(args,"sim_single",0) or getattr(args,"sim_two",0)) else ""
    print(f"\n=== 合并后训练集分布(按帧){tag} ===")
    for n in (1, 2, 3, 4):
        pct = 100 * fr[n] / tot if tot else 0
        print(f"  {n}键: {ep[n]:4d}集  {fr[n] / 1000:6.1f}k帧 ({pct:4.1f}%)")
    print(f"  合计 {ep.total()}集 / {tot / 1000:.0f}k帧")
    print(f"  单色帧占比 {100 * fr[1] / tot:.1f}%（目标≥13%，v3=12.8%接地96%）")
    print(f"  四键帧占比 {100 * fr[4] / tot:.1f}%（目标≤40%，v3=37%）")
    ok = (100 * fr[1] / tot >= 13) and (100 * fr[4] / tot <= 40)
    print("  >>> 分布达标 ✅" if ok else "  >>> 分布未达标 ⚠️ 调整 --keep-4button 或补采集数")

    if args.report_only:
        return
    if not args.supplement:
        raise SystemExit("正式构建需要 --supplement（补采数据集）；仅规划请加 --report-only")

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在，请先删除或改名: {out}")
    (out / "meta").mkdir(parents=True); (out / "data/chunk-000").mkdir(parents=True)
    cams = [k for k in bi["features"] if k.startswith("observation.images.")]
    for c in cams:
        (out / f"videos/chunk-000/{c}").mkdir(parents=True)

    # 任务表去重重编号
    task2idx = {}
    def tidx(t):
        if t not in task2idx:
            task2idx[t] = len(task2idx)
        return task2idx[t]

    new_eps, new_stats = [], []
    gidx = 0
    for new_i, (root, old_i, epm, stm, n, info) in enumerate(plan):
        t = epm["tasks"][0]; ti = tidx(t)
        # parquet：改写 episode_index / task_index / index(全局累加)，frame_index 不变
        src_pq = root / info["data_path"].format(episode_chunk=old_i // info["chunks_size"], episode_index=old_i)
        tb = pq.read_table(src_pq)
        L = tb.num_rows
        cols = {name: tb.column(name) for name in tb.column_names}
        cols["episode_index"] = pa.array([new_i] * L, pa.int64())
        cols["task_index"] = pa.array([ti] * L, pa.int64())
        cols["index"] = pa.array(list(range(gidx, gidx + L)), pa.int64())
        pq.write_table(pa.table(cols), out / f"data/chunk-000/episode_{new_i:06d}.parquet")
        gidx += L
        # 视频复制
        for c in cams:
            src_v = root / info["video_path"].format(episode_chunk=old_i // info["chunks_size"], video_key=c, episode_index=old_i)
            shutil.copy2(src_v, out / f"videos/chunk-000/{c}/episode_{new_i:06d}.mp4")
        new_eps.append({"episode_index": new_i, "tasks": [t], "length": L})
        if stm is not None:
            s = dict(stm); s["episode_index"] = new_i; new_stats.append(s)

    # 写元数据
    info_out = dict(bi)
    info_out["total_episodes"] = len(new_eps)
    info_out["total_frames"] = gidx
    info_out["total_videos"] = len(new_eps) * len(cams)
    info_out["total_tasks"] = len(task2idx)
    info_out["total_chunks"] = 1
    info_out["splits"] = {"train": f"0:{len(new_eps)}"}
    (out / "meta/info.json").write_text(json.dumps(info_out, ensure_ascii=False, indent=4))
    with (out / "meta/episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in new_eps:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (out / "meta/episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for s in new_stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (out / "meta/tasks.jsonl").open("w", encoding="utf-8") as f:
        for t, i in sorted(task2idx.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")
    print(f"\n已写出训练集: {out}  （{len(new_eps)} 集 / {gidx} 帧 / {len(task2idx)} 种指令）")
    print("下一步：重算 norm stats 后训练（见 docs/g1_button_v4_supplement.md）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="v4 主数据集目录")
    ap.add_argument("--supplement", default="", help="补采(单色)数据集目录；--report-only 时可省")
    ap.add_argument("--keep-4button", type=int, default=200, help="从 base 保留多少个四按钮集(其余丢弃)")
    ap.add_argument("--out", default="", help="输出目录")
    ap.add_argument("--report-only", action="store_true", help="只打印分布不落盘")
    ap.add_argument("--sim-single", type=int, default=0, help="规划用：模拟再补 N 个单色集(中位302帧)看分布")
    ap.add_argument("--sim-two", type=int, default=0, help="规划用：模拟再补 M 个双色集(中位609帧)看分布")
    args = ap.parse_args()
    build(args)
