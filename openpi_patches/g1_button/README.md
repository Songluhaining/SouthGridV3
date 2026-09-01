# openpi 定制：G1 OmniPicker 按钮任务（配置 `pi05_g1_button_lora`）

在**另一台机器**上用训练好的 checkpoint 起策略服务时，openpi 源码需要包含本目录的定制，否则
`--policy.config=pi05_g1_button_lora` 无法解析。

| 文件 | 用途 |
|---|---|
| `g1_button_policy.py` | 输入/输出变换：18 维 state/action；`cam_head`→`base_0_rgb`、`cam_wrist_r`→`right_wrist_0_rgb`；同时兼容训练键与 eval 客户端键 |
| `openpi_training_config.patch` | `config.py` 新增 `LeRobotG1ButtonDataConfig` 与 `pi05_g1_button_lora` TrainConfig；`data_loader.py` 指定 `video_backend="pyav"`（数据集视频为 AV1） |
| `OPENPI_BASE_COMMIT` | 生成补丁时 openpi 上游的 commit（`fdc03f5`），在该版本上打补丁最稳 |

## 安装到新机器

```bash
git clone https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout $(cat <SouthGrid>/openpi_patches/g1_button/OPENPI_BASE_COMMIT)
git apply <SouthGrid>/openpi_patches/g1_button/openpi_training_config.patch
cp <SouthGrid>/openpi_patches/g1_button/g1_button_policy.py src/openpi/policies/
GIT_LFS_SKIP_SMUDGE=1 uv sync          # 其余环境步骤见 docs/openpi_deployment.md
```

## 起服务（推理只需 checkpoint，不需要基座权重；norm stats 已随 checkpoint 的 `assets/` 一起保存）

```bash
uv run scripts/serve_policy.py --port 8010 policy:checkpoint \
    --policy.config=pi05_g1_button_lora \
    --policy.dir=<checkpoint 目录，例如 .../g1_button_local_v3/29999>
```

首次运行会下载 PaliGemma tokenizer（`gs://big_vision/paligemma_tokenizer.model`）到 `OPENPI_DATA_HOME`
（默认 `~/.cache/openpi`），需要能访问 GCS；离线机器可从训练机拷贝
`$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` 到同样的相对路径。
