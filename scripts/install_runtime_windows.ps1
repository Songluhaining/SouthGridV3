# SouthGrid 运行时安装脚本（Windows x64）。
#
# 对应 Linux 的 scripts/install_runtime.sh，步骤与安装顺序完全一致，
# 差别只在依赖锁文件：Windows 使用 requirements-windows.txt
# （由 requirements.in 针对 x86_64-pc-windows-msvc 重新解析，生成命令见该文件头部）。
#
# 用法（先激活 environment-windows.yml 创建的环境）：
#   conda activate orcalab_lerobot
#   powershell -ExecutionPolicy Bypass -File scripts\install_runtime_windows.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $env:CONDA_PREFIX) {
    Write-Error "error: activate the environment created from environment-windows.yml first"
    exit 2
}

$pythonVersion = (python -c "import sys; print('.'.join(str(p) for p in sys.version_info[:3]))").Trim()
if ($pythonVersion -ne "3.12.13") {
    Write-Error "error: expected Python 3.12.13, got $pythonVersion; create a fresh environment from environment-windows.yml"
    exit 2
}

# NumPy / SciPy 必须用 PyPI 版本，不能用 conda 的：conda-forge 的构建链接 LLVM 的
# libomp.dll，与 torch 自带的 Intel libiomp5md.dll 在同一进程里冲突，会让采集脚本
# 在调用 numpy.linalg 时直接 abort（详见 environment-windows.yml 的注释）。
python -m pip install --no-deps --timeout 120 --retries 10 "numpy==2.2.6" "scipy==1.16.2"
if (-not $?) { exit 1 }

# 安装解析好的 pip 依赖集。锁文件中不含 numpy / scipy（上一步已单独安装），
# Python 3.12 使用标准库的 argparse 模块，同样已排除。
python -m pip install --no-deps --require-hashes --timeout 120 --retries 10 -r requirements-windows.txt
if (-not $?) { exit 1 }

# OrcaLab 与 OrcaGym 使用同一个已验证的发布版本。
python -m pip install --no-deps --timeout 120 --retries 10 "orca-gym==26.7.3" "orca-lab==26.7.3"
if (-not $?) { exit 1 }

# 安装本仓库自带的三个源码包。
python -m pip install --no-deps --no-build-isolation .\third_party\lerobot .\third_party\televuer .\third_party\openpi-client
if (-not $?) { exit 1 }

python scripts\verify_environment.py
