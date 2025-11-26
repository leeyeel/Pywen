# Pywen Evaluation for SWE-bench

本目录包含 Pywen 在 SWE-bench 上的评测代码，采用与 Trae Agent 相同的 Docker 注入架构。

## 📦 安装

```bash
cd Pywen
# 安装评测依赖
uv sync --extra evaluation
# 或使用 pip
pip install -e ".[evaluation]"
```

## 🚀 快速开始

### 1. 配置 API Key

确保环境变量中有以下 Key（根据您使用的 Agent）：
```bash
export QWEN_API_KEY="your-key"
export QWEN_BASE_URL="https://..."
# 或
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### 2. 准备配置文件

在 Pywen 根目录创建 `pywen_config.json`（参考 `pywen_config.json.example`）。

### 3. 运行评测

```bash
cd Pywen

# 运行单个 Instance
python evaluation/run_evaluation.py \
  --instance-ids django__django-11001 \
  --dataset SWE-bench_Verified \
  --config-file pywen_config.json

# 运行多个 Instance（并行）
python evaluation/run_evaluation.py \
  --instance-ids django__django-11001 astropy__astropy-14365 \
  --max-workers 2 \
  --dataset SWE-bench_Lite

# 强制重建环境（如果 Pywen 代码更新了）
python evaluation/run_evaluation.py \
  --instance-ids django__django-11001 \
  --force-rebuild
```

## 📂 输出结构

```
Pywen/
├── evaluation/
│   ├── pywen_workspace/
│   │   └── pywen_env.tar        # 预构建的环境包（可复用）
│   └── results/
│       └── SWE-bench_SWE-bench_Verified_pywen-agent/
│           ├── django__django-11001/
│           │   ├── problem_statement.txt
│           │   ├── django__django-11001.patch  ⭐ 生成的补丁
│           │   └── run.log
│           └── ...
```

## 🏗️ 工作原理

1. **环境预构建**：
   - 首次运行时，在 `python:3.11-slim` 容器中构建 Pywen 环境
   - 打包为 `pywen_env.tar`（约 100-200MB）
   - 后续运行直接复用，无需重建

2. **镜像管理**：
   - 自动拉取 SWE-bench 官方镜像（如 `swebench/sweb.eval.x86_64.django_1776_django-11001:latest`）
   - 每个 Instance 对应一个专用镜像，包含完整的运行环境

3. **Agent 注入**：
   - 启动 SWE-bench 容器
   - 解压 `pywen_env.tar` 到 `/opt/pywen_env`
   - 挂载配置文件和结果目录
   - 执行 `pywen` CLI 命令

4. **Patch 收集**：
   - Agent 完成后，通过 `git diff` 提取修改
   - 保存为 `{instance_id}.patch`

## 🔧 参数说明

- `--instance-ids`: 要运行的 Instance ID（可多个）
- `--dataset`: 数据集名称（`SWE-bench`, `SWE-bench_Lite`, `SWE-bench_Verified`）
- `--max-workers`: 并行度（默认 1）
- `--force-rebuild`: 强制重建环境包
- `--config-file`: Pywen 配置文件路径

## 📤 提交评分

生成的 Patch 可以直接提交给 `swe-cli` 或其他云端评分服务。

## 🆚 与 Trae Agent 的对比

| 特性 | Trae Agent | Pywen |
|------|-----------|-------|
| 架构 | Docker 注入 | Docker 注入 ✅ |
| 环境构建 | `uv` | `pip` + `venv` |
| CLI 调用 | `trae-cli run --file ...` | `pywen "prompt"` |
| Patch 收集 | `--patch-path` 参数 | `git diff` |
| 依赖管理 | `[project.optional-dependencies]` | `[project.optional-dependencies]` ✅ |

核心流程完全一致，细节上根据各自 CLI 的特性略有调整。

