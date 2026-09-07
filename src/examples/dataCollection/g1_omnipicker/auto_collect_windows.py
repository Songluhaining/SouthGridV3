"""Windows 自动采集：分块跑采集脚本，块间用 MCP 重启仿真，按官方口径做健康检查。

对应 Linux 的 run_button_combo_collection.sh，差别在于：
  - 用 OrcaLab 的 MCP 接口（orcalab-cli）控制仿真启停，不需要冷重启整个 OrcaLab；
    start_simulation 的 program_name="external" 就是界面里的「无仿真程序（手动启动）」。
  - 每块采完读 meta/quality.jsonl，按官方计分口径判断是否健康；发现基座漂移累积或
    末端距离系统性变大时重启仿真再继续。

前置条件：OrcaLab 已启动并加载 g1_button.json 布局（本脚本只控制仿真启停，不启动 OrcaLab）。

用法:
    python auto_collect_windows.py --out G:\\datasets\\g1_button_official --total 200
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
COLLECT = os.path.join(BASE_DIR, "g1_omnipicker_collection_scripted_button_combo_lerobot.py")

DEFAULT_CLI = os.path.join(os.path.dirname(sys.executable), "Scripts", "orcalab-cli.exe")
if not os.path.exists(DEFAULT_CLI):
    DEFAULT_CLI = os.path.join(os.path.dirname(sys.executable), "orcalab-cli.exe")

# 健康阈值（依据实测：正常集基座漂移 <1mm、最佳帧距离 65~78mm）
DRIFT_WARN_M = 0.004        # 单集基座漂移超过该值视为接触异常
DIST_CREEP_MM = 8.0         # 后段平均距离比前段大出该值视为场景漂移累积
MIN_KEEP_RATIO = 0.35       # 单块保留率低于该值说明这一块状态不好


def log(msg):
    print(f"[auto] {msg}", flush=True)


def mcp(cli, tool, payload, url):
    """调用一个 MCP 工具，返回 (成功, 文本)。"""
    env = dict(os.environ, no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"):
        env.pop(var, None)
    r = subprocess.run([cli, tool, "--json", json.dumps(payload), "--url", url],
                       capture_output=True, text=True, env=env, timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    return ('"isError": false' in out), out


def port_open(port, host="127.0.0.1", timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def ensure_simulation(cli, url, wait_s=90):
    """确保仿真以 external 模式运行且 gRPC 就绪。返回 True/False。"""
    ok, out = mcp(cli, "get_simulation_state", {}, url)
    running = '\\"running\\": true' in out or '"running": true' in out
    if running and port_open(50051):
        return True
    if running:
        log("仿真在运行但 gRPC 未就绪，先停止")
        mcp(cli, "stop_simulation", {}, url)
        time.sleep(5)
    log("以 external（无仿真程序）模式启动仿真...")
    ok, out = mcp(cli, "start_simulation", {"program_name": "external"}, url)
    if not ok:
        log(f"start_simulation 调用失败: {out.strip()[:200]}")
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if port_open(50051):
            time.sleep(3)  # 给相机端口一点时间
            log("仿真已就绪（50051 开放）")
            return True
        time.sleep(2)
    log("等待 gRPC 超时")
    return False


def wait_for_simulation(cli, url, max_wait_min):
    """反复尝试把仿真拉起来。用于长跑任务：OrcaLab 偶发无响应时等待其恢复，
    而不是直接放弃整轮采集。仍然无法恢复时返回 False，由调用方决定去留。"""
    deadline = time.time() + max_wait_min * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        if ensure_simulation(cli, url):
            if attempt > 1:
                log(f"仿真已恢复（第 {attempt} 次尝试）")
            return True
        left = int((deadline - time.time()) / 60)
        log(f"仿真未就绪（第 {attempt} 次尝试），{left} 分钟内继续重试。"
            f"若 OrcaLab 已退出，请手动启动并加载 g1_button.json")
        time.sleep(30)
    return False


def restart_simulation(cli, url):
    log("重启仿真以消除累积漂移...")
    mcp(cli, "stop_simulation", {}, url)
    time.sleep(8)
    return ensure_simulation(cli, url)


def read_quality(out_dir):
    path = os.path.join(out_dir, "meta", "quality.jsonl")
    if not os.path.exists(path):
        return []
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def episodes_done(out_dir):
    info = os.path.join(out_dir, "meta", "info.json")
    if not os.path.exists(info):
        return 0
    try:
        with open(info, encoding="utf-8") as f:
            return int(json.load(f).get("total_episodes", 0))
    except Exception:
        return 0


def health_report(recs, chunk_n):
    """对最近一块做健康判断，返回 (是否健康, 说明文本)。"""
    if not recs:
        return True, "无记录"
    recent = recs[-chunk_n:] if chunk_n < len(recs) else recs
    kept = [r for r in recent if r.get("kept")]
    keep_ratio = len(kept) / max(1, len(recent))

    drifts, dists, scores = [], [], []
    n_p1 = n_p2 = 0
    for r in recent:
        m = r.get("metrics") or {}
        d = m.get("base_drift_m")
        if isinstance(d, (int, float)) and d == d:
            drifts.append(d)
        if m.get("p2_discount", 1.0) < 1.0:
            n_p2 += 1
        for p in r.get("presses", []):
            if p.get("best_site_dist_m") is not None:
                dists.append(p["best_site_dist_m"] * 1000)
            s = p.get("official_score_final", p.get("official_score"))
            if s is not None:
                scores.append(s)
            if p.get("p1_ok") is False:
                n_p1 += 1

    bits = [f"保留率 {keep_ratio:.0%}"]
    if scores:
        bits.append(f"每钮估分均值 {sum(scores) / len(scores):.2f}")
    if dists:
        bits.append(f"最佳帧距离均值 {sum(dists) / len(dists):.1f}mm")
    if drifts:
        bits.append(f"基座漂移最大 {max(drifts) * 1000:.1f}mm")
    bits.append(f"P1×{n_p1} P2×{n_p2}")
    text = "，".join(bits)

    problems = []
    if drifts and max(drifts) > DRIFT_WARN_M:
        problems.append(f"基座漂移 {max(drifts) * 1000:.1f}mm 超阈值")
    if keep_ratio < MIN_KEEP_RATIO:
        problems.append(f"保留率 {keep_ratio:.0%} 过低")
    # 距离随集数系统性变大 = 场景漂移累积
    all_dists = []
    for r in recs:
        for p in r.get("presses", []):
            if p.get("best_site_dist_m") is not None:
                all_dists.append(p["best_site_dist_m"] * 1000)
    if len(all_dists) >= 20:
        head = all_dists[:len(all_dists) // 3]
        tail = all_dists[-len(all_dists) // 3:]
        creep = sum(tail) / len(tail) - sum(head) / len(head)
        text += f"，距离漂移 {creep:+.1f}mm"
        if creep > DIST_CREEP_MM:
            problems.append(f"末端距离较开采时增大 {creep:.1f}mm")
    return (not problems), text + ("；问题: " + "；".join(problems) if problems else "")


def main():
    ap = argparse.ArgumentParser(description="Windows 自动采集（MCP 控制仿真 + 官方口径健康检查）")
    ap.add_argument("--out", required=True, help="数据集输出目录")
    ap.add_argument("--total", type=int, required=True, help="目标保留集数")
    ap.add_argument("--chunk", type=int, default=10, help="每块采集集数（默认 10）")
    ap.add_argument("--max_buttons", type=int, default=4)
    ap.add_argument("--length_weights", default="0.6,0.8,1,2",
                    help="1~4 钮的集数权重。评测按四钮计分，故四钮占比最高（约 45%%）；"
                         "同时保留 1~3 钮以覆盖更多指令措辞与组合")
    ap.add_argument("--mcp_url", default="http://127.0.0.1:12345/mcp")
    ap.add_argument("--cli", default=DEFAULT_CLI, help="orcalab-cli 路径")
    ap.add_argument("--max_rounds", type=int, default=2000, help="最多跑多少块（防死循环）")
    ap.add_argument("--max_wait_minutes", type=int, default=30,
                    help="仿真不可用时最长等待分钟数（长跑任务用于熬过 OrcaLab 偶发无响应）")
    ap.add_argument("--min_free_gb", type=float, default=5.0,
                    help="剩余磁盘低于该值即停止，避免写坏数据集")
    ap.add_argument("--extra", default="", help="透传给采集脚本的额外参数")
    args = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(args.out))
    start_have = episodes_done(out)
    if not os.path.exists(args.cli):
        log(f"找不到 orcalab-cli: {args.cli}")
        return 2

    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    rounds = 0
    consec_bad = 0
    t_start = time.time()
    while rounds < args.max_rounds:
        have = episodes_done(out)
        elapsed = time.time() - t_start
        eta = ""
        if rounds > 0 and have > 0:
            per = elapsed / max(1, have - start_have)
            eta = f"，预计剩余 {per * (args.total - have) / 3600:.1f} 小时"
        log(f"已保留 {have}/{args.total} 集{eta}")
        if have >= args.total:
            break
        rounds += 1

        free_gb = shutil.disk_usage(os.path.dirname(out) or out).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            log(f"剩余磁盘 {free_gb:.1f}GB 低于 {args.min_free_gb}GB，停止采集")
            return 1

        if not wait_for_simulation(args.cli, args.mcp_url, args.max_wait_minutes):
            log(f"仿真在 {args.max_wait_minutes} 分钟内未能就绪，停止。"
                f"已采数据完好，恢复后用同样命令即可续采")
            return 1

        want = min(args.chunk, args.total - have)
        cmd = [sys.executable, "-u", COLLECT, "--lerobot_out", out,
               "--episodes", str(want), "--max_buttons", str(args.max_buttons),
               "--length_weights", args.length_weights]
        if have > 0 or os.path.exists(os.path.join(out, "meta", "info.json")):
            cmd.append("--resume")
        if args.extra:
            cmd += args.extra.split()
        log(f"第 {rounds} 块：采集 {want} 集 ...")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=BASE_DIR, env=env)
        got = episodes_done(out) - have
        log(f"第 {rounds} 块结束（退出码 {r.returncode}，新增 {got} 集，"
            f"耗时 {time.time() - t0:.0f}s）")
        if r.returncode != 0 or got == 0:
            # 采集进程异常退出或一集没产出：多半是仿真侧状态坏了，重启后再来一块
            consec_bad += 1
            log(f"本块异常（连续第 {consec_bad} 次），重启仿真后重试")
            restart_simulation(args.cli, args.mcp_url)
            if consec_bad >= 5:
                log("连续 5 块异常，停止以免空转。已采数据完好，可用同样命令续采")
                return 1
            continue
        consec_bad = 0

        recs = read_quality(out)
        healthy, text = health_report(recs, want)
        log(f"健康检查：{text}")
        if not healthy:
            if not restart_simulation(args.cli, args.mcp_url):
                log("重启后仿真仍未就绪，退出")
                return 1

    final = episodes_done(out)
    recs = read_quality(out)
    healthy, text = health_report(recs, min(args.chunk, len(recs)))
    log(f"完成：共保留 {final} 集")
    log(f"最终健康状态：{text}")
    log(f"详细质量报告：python check_official_quality.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
