#!/usr/bin/env bash
# 把 LeRobot 数据集的视频从 AV1/480x640 转成 H.264/240x320、关键帧间隔 10。
#
# 为什么这么转：
#   1) AV1 软解比 H.264 慢数倍；
#   2) 关键帧间隔太大时，随机取一帧要从上一个关键帧顺序解到目标帧；
#   3) pi0.5 最终把图缩到 224x224（等比缩放后补边），存 480x640 是白解 4 倍像素。
#      240x320 与原图长宽比一致，缩放补边的结果与原来几乎相同，训练/推理不会错位。
#
# 用法: bash transcode_videos.sh <数据集根目录> [并行数]
set -euo pipefail

DATASET="${1:?用法: bash transcode_videos.sh <数据集根目录> [并行数]}"
JOBS="${2:-$(nproc)}"
export SRC="${DATASET%/}/videos"
export DST="${DATASET%/}/videos_h264"

command -v ffmpeg >/dev/null 2>&1 || { echo "找不到 ffmpeg，先装：conda install -y -c conda-forge ffmpeg"; exit 1; }
[ -d "$SRC" ] || { echo "找不到目录 $SRC"; exit 1; }

total=$(find "$SRC" -name '*.mp4' | wc -l)
echo "待转码 ${total} 个文件，并行度 ${JOBS}，输出到 ${DST}"
start=$(date +%s)

find "$SRC" -name '*.mp4' -print0 | xargs -0 -P "$JOBS" -I{} bash -c '
  in="$1"
  out="${DST}/${in#${SRC}/}"
  # 已完成的跳过，实现断点续跑；临时文件写完再改名，避免半截文件被当成已完成
  [ -s "$out" ] && exit 0
  mkdir -p "$(dirname "$out")"
  ffmpeg -nostdin -v error -y -i "$in" \
      -vf scale=320:240 \
      -c:v libx264 -preset veryfast -crf 23 \
      -g 10 -keyint_min 10 -sc_threshold 0 \
      -pix_fmt yuv420p -an "$out.tmp.mp4" && mv -f "$out.tmp.mp4" "$out"
' _ {}

done_n=$(find "$DST" -name '*.mp4' | wc -l)
echo "转码结束：${done_n}/${total} 个文件，用时 $(( $(date +%s) - start )) 秒"
echo "原始体积: $(du -sh "$SRC" | cut -f1)   新体积: $(du -sh "$DST" | cut -f1)"
[ "$done_n" -eq "$total" ] || { echo "！有文件没转成功，重跑本脚本即可续跑"; exit 1; }
