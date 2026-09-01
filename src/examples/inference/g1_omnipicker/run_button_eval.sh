#!/usr/bin/env bash
# 按钮任务评测包装：每个评测进程之前冷重启 OrcaLab，再运行 eval（策略服务需已在运行）。
# 实测同一 OrcaLab 实例上第二个连接的客户端起点前导段推不到位、策略输出随之失真，
# 必须像采集包装一样"一次冷重启只连一个进程"（原因见 dataCollection/g1_omnipicker/orcalab_restart.sh）。
# 用法: bash run_button_eval.sh --prompt "按红色按钮" [eval_g1_omnipicker_button_lerobot.py 的其它参数...]
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/../../dataCollection/g1_omnipicker/orcalab_restart.sh" || exit 1
cd "$here" && exec env HF_HUB_OFFLINE=1 python eval_g1_omnipicker_button_lerobot.py --no_camera_prep "$@"
