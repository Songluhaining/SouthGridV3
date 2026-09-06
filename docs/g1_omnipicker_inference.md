# 智元 G1 OmniPicker · 在线推理

本文说明按钮/工具任务的 OpenPI 在线推理。数据采集见 [g1_omnipicker_collection.md](g1_omnipicker_collection.md)。

---

## 架构概览

推理由两个独立进程组成，通过 WebSocket 通信：

```
┌──────────────────────────────────────────┐      WebSocket :8010
│  策略服务器（openpi uv 环境）              │ ◄──────────────────────
│  serve_policy.py      │
│  需要 GPU（≥8 GB 显存）                   │ ──────────────────────►
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  eval 推理脚本（orcalab_lerobot Conda 环境）│
│  eval_g1_omnipicker_button_lerobot.py    │
│  连接 OrcaLab 仿真 + 策略服务器           │
└──────────────────────────────────────────┘
```

两个进程所用的 Python 环境**完全独立**：eval 脚本用本仓库的 `orcalab_lerobot`（Conda），策略服务器用 openpi 的 `uv` 环境。

---

## 前置条件

1. 请在运行本项目的主机上启动 OrcaLab 7.3，并在 OrcaLab 的加载布局对话框中选择与任务对应的布局文件（`src/examples/dataCollection/g1_omnipicker/g1_button.json` 或 `src/examples/dataCollection/g1_omnipicker/g1_tool.json`）。
2. 请按采集文档配置相机端口并启动仿真（`localhost:50051`）。
3. 请确认已按仓库根目录 README 的「环境安装」一节执行 `bash scripts/install_runtime.sh`。
4. 策略服务器需要独立的 **openpi uv 环境**。请先按 [策略服务部署](openpi_deployment.md) 创建独立的 OpenPI 环境，并使用交付任务对应的策略配置和 checkpoint 启动服务。

---

## 场景一：本地推理

策略服务器与 eval 脚本运行在同一台机器上。该机器需要同时具备 GPU 和已安装的 OrcaLab。

### 1. 启动策略服务器

请在运行本项目的主机上打开一个独立终端，进入 openpi 工作目录（完整说明见 [openpi_deployment.md § 8](openpi_deployment.md#8-启动推理服务)）：

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=<your_config_name> \
    --policy.dir=checkpoints/<your_config_name>/<exp_name>/<step>
```

请等待策略服务器终端打印 `server listening on 0.0.0.0:8010` 后，再继续下一步。

### 2. 运行 eval 脚本

**按钮任务请通过包装脚本运行**（每个评测进程之前冷重启 OrcaLab）：

```bash
cd src/examples/inference/g1_omnipicker
bash run_button_eval.sh --prompt "按红色按钮" --episodes 1 \
    --team_id <team_id> --team_token <token>   # 官方评分参数按需追加
```

原因（实测）：同一个 OrcaLab 实例上第二个连接进来的客户端，起点前导段推不到 L 型预备位姿（OrcaLab 导出给本地 MuJoCo 的模型把导出瞬间的手臂姿态烘焙进连杆坐标系，关节角语义与限位随之平移），策略随即输出失真；采集包装脚本也是因此每块冷重启。包装脚本调用 `dataCollection/g1_omnipicker/orcalab_restart.sh`（无头启动 + 开仿真 + 相机准备），机器相关路径可用环境变量覆盖（见脚本头部）。

官方评分看的是 `ee_site` 到按钮 site 的最小距离（0.05 m 内满分、逐帧回溯取最优），而本地判定的提前结束条件是机器人几何体到按钮帽 ≤ `--success_dist`（默认 45 mm）；正式评分建议加 `--no_early_stop`，让策略跑满 `max_steps` 以免在临界距离处过早结束。

按钮任务 eval 会自动复现采集时的起始条件，保证观测分布与训练数据一致：每集先按关节角瞬移粗定位，再按采集前导段的方式（150 步插值 + 150 步保持）把右手驶向 L 型预备位姿（`conf/g1_omnipicker_conf.py` 的 `r_arm_ready`）并闭合右爪、左臂关节 PD 锁定（策略输出的左臂通道被忽略、state 左臂通道写常数），并在连接相机前通过 OrcaLab MCP 把腕相机 `camera_right` 转到 (90,180,0)、重建 `IsRecording`（与采集包装脚本共用 `dataCollection/g1_omnipicker/orcalab_camera_prep.sh`）。因此 OrcaLab 的 MCP 服务（默认 `http://127.0.0.1:12345/mcp`）需要可达；若已手动设置相机，可加 `--no_camera_prep`。

请在运行本项目的主机上打开另一个终端，从仓库根目录进入推理脚本所在目录。本地推理时，请将 `--host` 设为 `localhost`（该参数的默认值即为 `localhost`）：

**按钮任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "按红色按钮" \
    --episodes 3
```

**工具任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_tool_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

---

## 场景二：远程服务器推理

策略服务器运行在远程 GPU 服务器上，eval 脚本在本地（OrcaLab 所在机器）运行。

### 1. 在 GPU 服务器上启动策略服务

请先通过 SSH 登录远程 GPU 服务器，再在该服务器的终端中进入 openpi 工作目录（参考 [openpi_deployment.md § 8](openpi_deployment.md#8-启动推理服务)）：

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=<your_config_name> \
    --policy.dir=checkpoints/<your_config_name>/<exp_name>/<step>
```

> **网络要求**：服务器的 8010 端口需对本地机器可达。若两台机器不在同一内网，可通过 SSH 隧道转发：
> ```bash
> # 在本地机器执行，将本地 8010 映射到远端服务器 8010
> ssh -L 8010:localhost:8010 user@<server_ip>
> ```
> 使用 SSH 隧道时，eval 脚本仍填 `--host localhost --port 8010`。

### 2. 在本地运行 eval 脚本

请在运行 OrcaLab 的本地机器上打开终端，从仓库根目录进入推理脚本所在目录。不使用 SSH 隧道时，请将 `--host` 替换为服务器 IP 或主机名：

**按钮任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_button_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <server_ip_or_hostname> \
    --port 8010 \
    --prompt "按红色按钮" \
    --episodes 3
```

**工具任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_tool_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <server_ip_or_hostname> \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

---

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task_config` | 任务配置文件路径 | `../../dataCollection/common/example.yaml` |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` |
| `--host` | 策略服务器主机（本地填 `localhost`，远程填 IP 或主机名） | `localhost` |
| `--port` | 策略服务器 WebSocket 端口 | `8010` |
| `--prompt` | 任务语言指令，须与训练数据中的描述完全一致 | 按钮任务为 `按红色按钮`，工具任务为 `整理工具` |
| `--max_steps` | 每集最大控制步数（按钮任务 5 ms/步，目标全部按下会提前结束） | 按钮 6000，工具 10000 |
| `--action_repeat` | 每个推理 action 重复执行的控制步数。按钮任务训练数据 20 fps、控制 200 Hz，必须为 `10`，否则轨迹被压缩 10 倍、手几乎不动 | 按钮 `10`，工具 `1` |
| `--episodes` | 评估集数 | `1` |
| `--camera_warmup_steps` | 每集推理前相机预热步数 | `10` |
| `--prompts` | 按钮任务：同一集（同一 attempt）内依次执行的多条指令，每条之间先回到预备位姿；给出时忽略 `--prompt`，`--max_steps` 变为每条指令的步数预算。配合 `--success_dist 0` 可让每条指令在真实压下后立即切换下一条 | 未启用 |
| `--left_pose` | 按钮任务：左臂锁定位姿 `x,y,z,qx,qy,qz,qw`（base 系，xyzw）。训练数据的左臂不在本机中立位时必须传；前导段用 OSC 把左臂驶到该位姿后再交给关节锁 | 空（左臂锁在中立位） |
| `--start_pose` | 按钮任务：`ready`（默认）复现 v3 采集的 L 型预备位姿前导段；`neutral` 沿用场景默认位姿，用于评估以默认位姿起步采集的旧模型 | `ready` |
| `--ready_move_steps` | 按钮任务：每集开始前右手按采集前导段的方式插值驶向 L 型预备位姿的步数 | `150` |
| `--settle_steps` | 按钮任务：到达预备位姿后钉住目标等 OSC 收敛的步数（不计入 `max_steps`） | `150` |
| `--no_camera_prep` | 按钮任务：跳过经 MCP 设置腕相机朝向 (90,180,0) 与重建 IsRecording | 未启用（默认自动执行） |
| `--mcp_url` | 按钮任务：OrcaLab MCP 地址（相机准备用） | `http://127.0.0.1:12345/mcp` |
| `--sleep` | 按实时步长节奏运行 | 未启用 |
| `--no_images` | 跳过相机采图，发送空图 | 未启用 |
| `--no_preview` | 按钮任务：不显示相机实时预览小窗口 | 未启用（按钮任务默认显示预览） |
| `--preview` | 工具任务：显示相机实时预览 | 未启用（工具任务默认不显示预览） |
| `--kp` | 工具任务：阻抗刚度 | `150.0` |

工具任务的近桌外环积分等冷门参数见下方「高级调参参数」。

---

## 故障排查

**现象**：找不到 `openpi_client`。**处理**：请在仓库根目录重新执行 `bash scripts/install_runtime.sh`，不要添加外部源码路径。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口，并确认 IsRecording 已勾选。

**现象**：WebSocket 连接失败（远程场景）。**处理**：确认服务器防火墙已放行 8010 端口，或改用 SSH 隧道方案。

**现象**：策略服务器 OOM 或响应慢。**处理**：确认所用 GPU 满足策略服务的显存要求，并检查是否有其他进程占用显存。
