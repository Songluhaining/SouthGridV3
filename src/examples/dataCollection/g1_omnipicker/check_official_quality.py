"""按官方口径检查已采数据集的质量记录。

读取 <dataset>/meta/quality.jsonl，逐集列出每个按钮的官方估分、最佳帧距离与
P1/P2 自检结果，并汇总最容易失分的几项。仅读文件，不连 OrcaLab。

用法:
    python check_official_quality.py <dataset_dir> [--all]

    --all  连被丢弃的集一并列出（默认只列保留下来的集）
"""
import argparse
import json
import os
import sys

# 官方口径的关键阈值（见 orca_scorer_client 的 configs/tasks.yaml 与 engine.py）
PEAK_MM = 50.0        # 得分曲线顶点
P2_SPAN_S = 20.0      # 四钮最佳帧跨度低于该值触发速通惩罚
CAM_GAP_WARN = 0.5    # 相机帧间隔告警阈值
DRIFT_WARN = 0.005    # 基座漂移告警阈值（米）

_CN = {"red": "红", "green": "绿", "yellow": "黄", "blue": "蓝"}


def main():
    ap = argparse.ArgumentParser(description="按官方口径检查采集质量")
    ap.add_argument("dataset_dir", help="数据集目录（含 meta/quality.jsonl）")
    ap.add_argument("--all", action="store_true", help="连被丢弃的集一并列出")
    args = ap.parse_args()

    path = os.path.join(os.path.abspath(os.path.expanduser(args.dataset_dir)),
                        "meta", "quality.jsonl")
    if not os.path.exists(path):
        print(f"找不到质量记录: {path}")
        print("（该文件在第一集采完后才会生成）")
        return 1

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print(f"{path} 为空")
        return 1

    kept = [r for r in records if r.get("kept")]
    print("=" * 74)
    print(f"  质量记录: {path}")
    print(f"  共 {len(records)} 集，保留 {len(kept)} 集，丢弃 {len(records) - len(kept)} 集")
    print("=" * 74)

    all_scores, all_dists, spans, drifts, gaps = [], [], [], [], []
    n_p1, n_p2 = 0, 0

    for r in records:
        if not r.get("kept") and not args.all:
            continue
        m = r.get("metrics") or {}
        tag = "保留" if r.get("kept") else "丢弃"
        ep = r.get("episode_index")
        seq = "→".join(_CN.get(c, c) for c in r.get("sequence", []))
        print(f"\n[{tag}] 第 {ep if ep is not None else '-'} 集  {seq}  「{r.get('task','')}」")

        for p in r.get("presses", []):
            color = _CN.get(p.get("color"), p.get("color"))
            d = p.get("best_site_dist_m")
            s = p.get("official_score_final", p.get("official_score"))
            t = p.get("best_t_s")
            p1 = p.get("p1_ok")
            bits = []
            if d is not None:
                bits.append(f"最佳帧距离 {d * 1000:6.1f}mm（离顶点 {abs(d * 1000 - PEAK_MM):4.1f}mm）")
                all_dists.append(d * 1000)
            if s is not None:
                bits.append(f"估分 {s:5.2f}")
                all_scores.append(s)
            if t is not None:
                bits.append(f"t={t:6.2f}s")
            ae = p.get("aim_err_m")
            if ae is not None:
                bits.append(f"OSC残差 {ae * 1000:5.1f}mm")
            if p1 is False:
                bits.append(f"P1 触发！离 {_CN.get(p.get('nearest_at_best'), p.get('nearest_at_best'))} 更近")
                n_p1 += 1
            print(f"    {color}  " + "  ".join(bits))

        span = m.get("best_frame_span_s")
        n_press = len(r.get("presses", []))
        if span is not None:
            spans.append(span)
            if n_press >= 4:
                # P2 只在凑满四钮时生效，少于四钮的集不受该门槛约束
                flag = "  <<< 触发 P2，四钮全部打折" if 0 < span <= P2_SPAN_S else ""
                print(f"    四钮最佳帧跨度 {span:.2f}s（需 > {P2_SPAN_S:.0f}s）{flag}")
                if flag:
                    n_p2 += 1
            else:
                per = span / (n_press - 1) if n_press > 1 else 0.0
                print(f"    {n_press} 钮跨度 {span:.2f}s（单钮间隔 {per:.2f}s；"
                      f"P2 只在四钮时生效，按此间隔四钮可达 {per * 3:.1f}s）")
        if m.get("p2_discount", 1.0) < 1.0:
            print(f"    P2 折扣 ×{m['p2_discount']}  {m.get('p2_reason','')}")
        drift = m.get("base_drift_m")
        if drift is not None and drift == drift:  # 排除 NaN
            drifts.append(drift)
            print(f"    基座漂移 {drift * 1000:.1f}mm" + ("  <<< 偏大" if drift > DRIFT_WARN else ""))
        gap = m.get("cam_max_gap_s")
        if gap is not None:
            gaps.append(gap)
            print(f"    相机最大帧间隔 {gap:.2f}s" + ("  <<< 图像滞后" if gap > CAM_GAP_WARN else ""))
        if m.get("official_score_sum") is not None:
            print(f"    本集官方估分合计 {m['official_score_sum']:.2f}")

    print("\n" + "=" * 74)
    print("  汇总")
    print("=" * 74)

    def stat(name, vals, unit="", fmt="%.2f"):
        if not vals:
            print(f"  {name:<22} 无数据")
            return
        lo, hi = min(vals), max(vals)
        avg = sum(vals) / len(vals)
        print(("  %-22s 均值 " + fmt + unit + "   范围 " + fmt + " ~ " + fmt + unit
               + "   n=%d") % (name, avg, lo, hi, len(vals)))

    stat("每钮官方估分", all_scores)
    stat("最佳帧距离", all_dists, "mm", "%.1f")
    stat("四钮跨度", spans, "s")
    stat("基座漂移", [d * 1000 for d in drifts], "mm", "%.1f")
    stat("相机最大帧间隔", gaps, "s")

    print()
    if all_dists:
        off = sum(abs(d - PEAK_MM) for d in all_dists) / len(all_dists)
        verdict = "好" if off <= 10 else ("偏差偏大，检查 OSC 跟踪" if off <= 25 else "严重偏离，需排查")
        print(f"  距顶点平均偏差 {off:.1f}mm —— {verdict}")
    print(f"  P1 触发次数 {n_p1}   P2 触发集数 {n_p2}")
    if n_p1 == 0 and n_p2 == 0 and all_scores and min(all_scores) >= 9.0:
        print("  所有保留集均无 P1/P2 惩罚，每钮估分不低于 9.0 —— 可以放量采集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
