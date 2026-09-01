#!/usr/bin/env bash
# OrcaLab 相机准备（按钮任务采集与推理共用），必须在采集/推理进程连接相机之前执行：
# 1) 腕相机朝向修正：官方布局里 camera_right 背向夹爪（画面只有背景），转为与夹爪同向，
#    按钮 + 夹爪同框（手眼协调视角）。实测定标 rotate=(90,180,0)。
# 2) OrcaLab 冷启动后 7080 端口常缺失：翻转 IsRecording 重建监听。
# 用法: [ORCALAB_CLI=<orcalab-cli 路径>] bash orcalab_camera_prep.sh [mcp_url]
set -uo pipefail
mcp_url="${1:-${MCP_URL:-http://127.0.0.1:12345/mcp}}"
cli="${ORCALAB_CLI:-$(command -v orcalab-cli || true)}"
[ -x "$cli" ] || { echo "[相机准备] 找不到 orcalab-cli（设置 ORCALAB_CLI）" >&2; exit 1; }

fail=0
mcp() {
    env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy \
        "$cli" set_actor_properties --json "$1" --url "$mcp_url" >/dev/null 2>&1 || fail=1
}
for kv in rotate.x:90.0 rotate.y:180.0 rotate.z:0.0; do
    mcp "{\"asset_path\":\"/g1_omnipicker/camera_right\",\"property_name\":\"${kv%%:*}\",\"property_value\":${kv##*:}}"
done
sleep 2
for cam in camera_right camera_head; do
    for v in false true; do
        mcp "{\"asset_path\":\"/g1_omnipicker/$cam\",\"property_name\":\"IsRecording\",\"property_value\":$v}"
        sleep 2
    done
done
sleep 4
if [ "$fail" = 1 ]; then
    echo "[相机准备] 有 MCP 调用失败：确认 OrcaLab 已启动且 MCP ($mcp_url) 可达" >&2
    exit 1
fi
echo "[相机准备] 腕相机朝向 (90,180,0) 已设置，IsRecording 已重建"
