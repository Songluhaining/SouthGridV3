# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

SouthGrid 为南方电网竞赛提供人形机器人数据采集、回放与在线推理工具，运行在 OrcaLab/OrcaGym 仿真器之上（OrcaLab 是外部进程，通过 gRPC `localhost:50051` 通信；本仓库不含物理引擎）。两条平台线：

- **智元 G1 OmniPicker**（交付主线）：`src/examples/dataCollection/g1_omnipicker/`（采集/回放）+ `src/examples/inference/g1_omnipicker/`（在线推理）。
- **宇树 G1**：`src/examples/dataCollection/unitree_g1/`（`g1_pick_osc_*`），与 OmniPicker 平行的骨架，换用 `g1_pick_osc_conf` 与 `G1PickOscLeRobotStorage`（28 维 state、Δq action，区别于 OmniPicker 的 18 维绝对位姿 state）。

数据输出为 LeRobot v2.1 数据集（字段以数据集内 `meta/info.json` 为准）。推理通过 WebSocket 连接独立部署的 OpenPI 策略服务（默认端口 8010，独立 uv 环境，见 `docs/openpi_deployment.md`）。

## 环境与命令

```bash
# 安装：必须新建 conda 环境（Python 固定 3.12.13），不要复用旧环境
conda env create -f environment-unitree.yml
conda activate orcalab_lerobot
bash scripts/install_runtime.sh

# 环境校验（fail-fast 比对所有 pin 的版本）
python scripts/verify_environment.py
```

依赖管理是强 pin 的：`requirements.in` 为直接依赖清单，`requirements.txt` 是 `uv pip compile` 生成的带 hash 锁定文件（再生成命令见 `requirements.in` 头部注释）。`install_runtime.sh` 全程 `--no-deps` 安装：先装锁定集，再装 `orca-gym`/`orca-lab==26.7.3`，最后源码安装 `third_party/{lerobot,televuer,openpi-client}`。NumPy/SciPy 由 Conda 提供，不能被 pip 替换。改动依赖时需同步更新 `requirements.in`、`constraints.txt`、重新生成 `requirements.txt`，并更新 `scripts/verify_environment.py` 里的版本表。

本仓库无测试、无 CI。ruff 仅配置忽略 E402（入口脚本先做 sys.path 注入再 import，是有意为之）。

## 运行入口脚本

**所有入口脚本必须 `cd` 到脚本所在目录运行**——`--task_config` 默认值、路点 YAML、日志目录等均相对脚本目录解析。运行前提：OrcaLab 7.3 已启动并加载了对应布局 JSON（与脚本同目录）、相机已按 docs 配置（Color Port 7080 右腕 / 7090 头部；7070 左腕仅共享代码保留）、仿真已运行。

代表性命令（完整参数表见 `docs/`）：

```bash
# OmniPicker 脚本化采集（工具整理；按钮任务用 ..._scripted_button_lerobot.py + --counts）
cd src/examples/dataCollection/g1_omnipicker
python g1_omnipicker_collection_scripted_tool_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_tool_scripted --repo_id local/g1_omnipicker_tool \
    --num_episodes 20 --fps 20

# 回放（--episode 从 1 开始；不传则播全部）
python g1_omnipicker_replay_lerobot.py --dataset_dir <dataset> \
    --task_config ../common/example.yaml --episode 1

# Pico 遥操作采集：先 adb reverse tcp:8001 tcp:8001，再跑 ..._tele_lerobot.py

# 在线推理（先按 docs/openpi_deployment.md 启动策略服务；工具任务用 eval_g1_omnipicker_tool_lerobot.py）
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost --port 8010 --prompt "按红色按钮"
```

宇树 G1 同构，位于 `unitree_g1/`：需加 `OMP_NUM_THREADS=1` 前缀与 `--agent_name g1_pick`；采集示例必须显式 `--joint_strip on`（脚本默认为 `off`，与文档示例不同）。断点续采统一用 `--resume`。

## 架构

分层：OrcaLab（外部仿真/渲染，gRPC）← `orca-gym`（pip 包，提供 `OrcaGymLocalEnv`、robosuite 控制器适配、Pico 手柄、RGBD 相机客户端）← `src/` 各层 ← `src/examples/` 入口脚本。

**入口脚本是组合根，`src/` 靠 sys.path 注入。** 每个入口脚本开头把 `src/` 插入 `sys.path`（脚本里的变量名 `project_root` 实际指向 `src/` 目录，不是仓库根），随后 `from conf import ...`、`from dataStorage.lerobot_data_storage import ...`。本仓库自身从不被 pip 安装。脚本按序构建：`SceneManager` → storage → `DataCollectionManager`（传入 `storage.obs_callback`）→ `env.reset()`/`update_scene()` → 创建 arm OSC / gripper 控制器 → task → 相机（`env.begin_save_video` 触发推流后 `bring_up_cameras`）→ `LeRobotDatasetWriter.create` + `storage.configure_lerobot`。

| 路径 | 职责 |
|---|---|
| `src/envs/dataCollection/dataCollection_env.py` | 唯一 env。`step()` = `do_simulation` + 注入的 `obs_callback`；reward 恒 0、永不 terminate——是仿真驱动器，不是 RL env。观测内容完全由 storage 的回调决定 |
| `src/dataCollectionManager/data_collection_manager.py` | 编排器。`create_env()` 用字符串入口 `"envs.dataCollection.dataCollection_env:DataCollectionEnv"` 走 `gym.register`/`gym.make`（因此 `src/` 必须已在 sys.path），并把 SceneManager 回接到 env |
| `src/conf/*.py` | 纯数据模块（只有 dict）：语义角色 → MuJoCo/OrcaStudio 实体名（关节、torque/position actuator 及分组、`ee_site`、`base_body` 基座标系等）。gripper 的 `actuator_ranges` 同时是 LeRobot 夹爪通道归一化的依据 |
| `src/controllers/` | 包装 `orca_gym.adapters.robosuite` 的 OSC/IK；设备事件 → `{actuator_index: ctrl}`。`controllers.py` 的 `install_osc_patches` 把 robosuite 的 opspace 数值 monkey-patch 成阻尼最小二乘（DLS）变体 |
| `src/devices/` | 输入源：Pico VR 手柄、HDF5 回放（`data_device.py`）；推理脚本内联一个 `EEFDevice` 接策略输出 |
| `src/task/` | 成功判据 + 语言指令；G1 OmniPicker 全部用 `EmptyTask` |
| `src/scene/scene_manager.py` | 持有独立的 gRPC 连接：spawn/随机化/publish 场景、场景内 UI 提示。场景 publish 之前 env 的 model 为空（`obs_callback` 有对应保护） |
| `src/dataStorage/` | obs 提取 + LeRobot v2.1 写入：相机 WebSocket 客户端（`lerobot_camera.py`）、NVENC 编码后端（默认 in-process，另有 subproc/lerobot 两个备选后端） |

**每步流程**（`manager.run_episode()`）：`run_controllers()`（device.update → 各 controller）→ `env.step(ctrl)` → task status 判定 → RUNNING 时 `storage.collection_data(obs, env)`。写帧按 `1/fps` 时钟门控（`--clock sim` 用仿真时间，`wall` 用墙钟）；OmniPicker 的 **action = 下一帧的绝对 state**（one-frame-lag 配对），不是增量。

**推理侧**复用整套采集栈：把输入设备换成 `openpi_client.WebsocketClientPolicy`，发送 `{"state", "images"(CHW uint8), "prompt"(中文指令，须与训练数据一致)}`；state 用一个临时 `G1OmniPickerLeRobotStorage` 的同一 `build_state` 路径生成，保证与训练 schema 一致。改 storage 的 state/action 定义时，推理侧的 `parse_policy_action`/归一化反变换要同步改。

**遗留/平行结构**（改共享代码时注意）：

- `lerobot_data_storage.py` 在模块级 import tiangong/openloong 两个 legacy storage，它们因此是所有入口脚本的硬依赖，尽管没有入口脚本使用它们。
- G1 OmniPicker 的入口脚本绕过 `DataCollectionManager.run()`，自己驱动 episode 循环（为实现 Grip 丢弃、失败重试、mp4 相机模式）；`run()` 及其 AUGMENTATION/HDF5 路径（`data_device.py`、`Interpolator/`）只有 legacy 通路可达。
- `src/conf/d12_conf.py`、`src/utils/g1_pick_ee_pose_log.py`、`src/utils/g1_pick_weighted_moving_filter.py` 当前无任何引用方。
- `src/model/` 下的 MJCF/网格资产由 OrcaStudio 侧加载，Python 代码不读取。

## 文档与约定

`docs/` 四篇是使用说明的权威来源：`g1_omnipicker_collection.md`（含 Pico 按键映射、故障排查）、`g1_omnipicker_inference.md`、`unitree_g1_collection.md`、`openpi_deployment.md`。文档为中文；commit message 为英文祈使句短句（以句号结尾）。
