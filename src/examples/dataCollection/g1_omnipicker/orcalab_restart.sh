#!/usr/bin/env bash
# 冷重启 OrcaLab 并把按钮任务场景准备到"可连接"状态（采集与推理共用）：
# 杀旧实例 → 无头启动（按钮布局，NVIDIA 适配器）→ 等 MCP → 启动仿真 → 等 gRPC → 相机准备。
# 为什么每个采集/推理进程之前都要冷重启、且每个实例只连一个客户端：
#   - stop/start 仿真后相机推流退化为 ~1fps 且无法恢复，只有冷启动后的首次推流健康；
#   - OrcaLab 导出给本地 MuJoCo 的模型会把导出瞬间的手臂姿态烘焙进连杆坐标系，第二个连接的
#     客户端拿到的关节角语义/限位已被上一客户端留下的姿态平移，起点前导段推不到位（实测）。
# 用法: bash orcalab_restart.sh
# 可用环境变量覆盖: ORCALAB_DIR ORCALAB_PY ORCALAB_CLI LAYOUT MCP_URL DISPLAY XAUTHORITY
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCALAB_DIR="${ORCALAB_DIR:-/data/whn/orcalab}"
ORCALAB_PY="${ORCALAB_PY:-/data/whn/miniconda3/envs/orcalab/bin/python3.12}"
ORCALAB_CLI="${ORCALAB_CLI:-/data/whn/miniconda3/envs/orcalab_lerobot/bin/orcalab-cli}"
LAYOUT="${LAYOUT:-$here/g1_button.json}"
MCP_URL="${MCP_URL:-http://127.0.0.1:12345/mcp}"
export DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-/run/user/1004/gdm/Xauthority}"

echo "[OrcaLab] 冷重启..."
P=$(pgrep -f "orcalab[.]ma[i]n")
[ -n "$P" ] && { kill $P 2>/dev/null; sleep 6; kill -0 $P 2>/dev/null && kill -9 $P; }
pkill -f "run_sim_lo[o]p" 2>/dev/null
sleep 4
( cd "$ORCALAB_DIR" && env PATH="$(dirname "$ORCALAB_PY"):$PATH" \
    nohup "$ORCALAB_PY" -m orcalab.main --scene CSG_RobotCompetition_2026 \
    --layout "$LAYOUT" --force-adapter NVIDIA >/tmp/orcalab_wrapper.log 2>&1 </dev/null & )
for _ in $(seq 1 40); do
    sleep 5
    (echo >/dev/tcp/127.0.0.1/12345) 2>/dev/null && break
done
timeout 30 env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy \
    "$ORCALAB_CLI" start_simulation --json '{"program_name":"run_sim_loop"}' --url "$MCP_URL" >/dev/null 2>&1
for _ in $(seq 1 30); do
    sleep 2
    (echo >/dev/tcp/127.0.0.1/50051) 2>/dev/null && break
done
(echo >/dev/tcp/127.0.0.1/50051) 2>/dev/null || { echo "[OrcaLab] gRPC 50051 未就绪（见 /tmp/orcalab_wrapper.log）" >&2; exit 1; }
sleep 8
ORCALAB_CLI="$ORCALAB_CLI" bash "$here/orcalab_camera_prep.sh" "$MCP_URL"
