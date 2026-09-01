#!/usr/bin/env bash
# 多按钮组合采集包装脚本：分块采集，块间重启仿真（实测：重启后第一个连接的采集进程状态最好，故不做热身）。
# 用法: bash run_button_combo_collection.sh <lerobot_out> <总集数> [每块集数=10]
set -uo pipefail

out="${1:?用法: bash run_button_combo_collection.sh <lerobot_out> <总集数> [每块集数]}"
total="${2:?缺少总集数}"
chunk="${3:-10}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

restart_sim() {
    # 每块冷重启 OrcaLab（原因见 orcalab_restart.sh 头部注释）
    bash "$here/orcalab_restart.sh" || echo "[包装] OrcaLab 未就绪，继续尝试（由质量闸门兜底）"
}

kept() {
    python3 - "$out" <<'PY'
import json, os, sys
p = os.path.join(os.path.expanduser(sys.argv[1]), "meta", "info.json")
print(json.load(open(p)).get("total_episodes", 0) if os.path.exists(p) else 0)
PY
}

while true; do
    have="$(kept)"
    echo "[包装] 已保留 ${have}/${total} 集"
    [ "$have" -ge "$total" ] && break
    n=$(( total - have < chunk ? total - have : chunk ))
    # 冷启动后前几集常处于病态期；配额太小会在病态期内全部损耗，卡住收尾
    [ "$n" -lt 6 ] && n=6
    restart_sim
    # 健康门与正式采集合并：进程健康状态不跨进程传递（实测），
    # 由正式进程自身的三重质量闸门+max_consec_fail 完成筛选——
    # 病态进程产出全废后自动退出重抽，健康进程持续产出。
    resume_flag=""
    out_abs="$(eval echo "$out")"
    if [ "$(kept)" -gt 0 ]; then
        resume_flag="--resume"
    elif [ -d "$out_abs" ]; then
        # 结构存在但 0 集（首块全失败）同样按残目录处理，避免 resume 撞 401
        # 目录存在但 meta 不完整（上一块失败集全被丢弃）：残目录会让脚本误走
        # hub 加载路径导致 401，直接清掉从零开始
        echo "[包装] 清理无效残目录 $out_abs"
        rm -rf "$out_abs"
    fi
    env HF_HUB_OFFLINE=1 /data/whn/miniconda3/envs/orcalab_lerobot/bin/python "$here/g1_omnipicker_collection_scripted_button_combo_lerobot.py" \
        --lerobot_out "$out" --episodes "$n" $resume_flag "${@:4}"
done
echo "[包装] 完成：$(kept) 集，质量记录见 $out/meta/quality.jsonl"
