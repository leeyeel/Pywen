
# Pywen × SWE-bench：容器化运行指南

---

## 一、先决条件

* 操作系统：Linux / macOS（Windows 需 WSL2）
* Docker ≥ 20.10
* 网络能访问 Docker Hub（拉取 SWE-bench 镜像）
* 已准备好 **Pywen 配置文件** `pywen_config.yaml`

  运行时通过 `--config` 指向自定义路径,比如自己的`~/.pywen/pywen_config.yaml`

---

## 二、目录结构（关键路径）

```
Pywen/
├─ Dockerfile.pywen-agent      # 构建 Pywen Agent 镜像
├─ pyproject.toml
├─ uv.lock
├─ .python-version             # 指定 Python 版本（3.12.x）
├─ pywen/                      # Pywen 源码
├─ pywen_config.example.yaml   # 配置模板
└─ evaluation/
   └─ run_evaluation.py        # 评测脚本（SWE-bench）
```

运行后脚本会在工作目录下生成缓存与结果：

```
pywen_workspace/pywen_agent_cache/
├─ Pywen/                # 映射到容器 /opt/Pywen   （含 .venv）
├─ uv_bin/uv             # 映射到容器 /root/.local/bin/uv
└─ uv_share/uv/...       # 映射到容器 /root/.local/share/uv（含托管 CPython）
results/
└─ SWE-bench_.../        # 每个实例的日志、patch、predictions.json 等
```

---

## 三、构建镜像（一次性）

在项目根目录执行：

```bash
docker build -f Dockerfile.pywen-agent -t pywen/agent:0.1 .
```

镜像中包含：

* `/opt/Pywen/.venv`（venv；其 python 为 uv 的 shim）
* `/root/.local/bin/uv`
* `/root/.local/share/uv`（**托管的 CPython 3.12 + 轮子缓存**）

> 镜像构建时 **不会 apt 安装 python**。uv 会根据 `.python-version` 自动下载 manylinux 预编译的 CPython 3.12。

---

## 四、一次性导出运行时缓存（自动完成）

`evaluation/run_evaluation.py` 在首次运行时会自动：

* 启动一个短暂的 `pywen/agent:0.1` 容器
* 将镜像内的 `/opt/Pywen`、`/root/.local/bin/uv`、`/root/.local/share/uv` **拷贝到宿主机**
* 缓存路径：`pywen_workspace/pywen_agent_cache/`

后续所有 SWE 实例容器只需挂载这些目录，无需在每个容器中再次解压或安装。

---

## 五、运行方式

### 1）单实例运行（推荐先试）

```bash
python evaluation/run_evaluation.py \
  --config ~/.pywen/pywen_config.yaml \
  --limit 2
```

说明：

* `--config`：你的真实配置文件。未指定时将依次查找：

  1. `~/.pywen/pywen_config.yaml`
  2. 项目内 `pywen_config.example.yaml`（没法用，会报错）
* `--limit 2`：只下载两个镜像处理。

---

## 六、参数说明（`run_evaluation.py`）

* `--benchmark`：默认 `SWE-bench`
* `--dataset`：`SWE-bench` / `SWE-bench_Lite` / `SWE-bench_Verified`
* `--working-dir`：工作区（缓存与中间文件）
* `--config`：Pywen 配置 YAML 路径（强烈建议显式传入）
* `--agent`：`qwen` / `codex` / `claude`
* `--instance_ids`：指定实例（空则跑全量）
* `--pattern`：用正则匹配实例 id
* `--limit`：最多跑多少个实例
* `--max_workers`：并发度
* `--mode`：

  * `expr`：只生成补丁
  * `eval`：只跑评测（需要先有 predictions.json）
  * `e2e`：生成补丁 + 评测

---

## 七、工作原理（简述）

1. **构建阶段**：用 uv 根据 `.python-version` 下载 **CPython 3.12（manylinux 预编译）**，创建 `.venv` 并安装依赖。
2. **导出缓存**：从构建镜像把 `/opt/Pywen`、`/root/.local/bin/uv`、`/root/.local/share/uv` 拷贝到宿主 `pywen_agent_cache/`。
3. **运行阶段**：每个 SWE 实例容器挂载三处缓存；在容器内以 **`/opt/Pywen/.venv/bin/pywen ...`** 直接运行（不需要 `source activate`）。
4. `.venv/bin/python` 是 uv 的 **shim**，会解析到挂载的 `~/.local/share/uv/python/.../bin/python`，避免依赖实例容器的系统 Python/`libpython3.12.so`。

---

## 八、手动调试（可选）

想进入某个 SWE 实例容器手动跑 Pywen，可参考：

```bash
docker run --rm -it \
  -v "$(pwd)/pywen_workspace/pywen_agent_cache/Pywen":/opt/Pywen:ro \
  -v "$(pwd)/pywen_workspace/pywen_agent_cache/uv_bin":/root/.local/bin:ro \
  -v "$(pwd)/pywen_workspace/pywen_agent_cache/uv_share":/root/.local/share:ro \
  -v "$(pwd)/results/demo":/results:rw \
  --workdir /testbed \
  swebench/sweb.eval.x86_64.<instance_id>:latest \
  bash

# 容器里：
/opt/Pywen/.venv/bin/pywen --config /results/pywen_config.yaml --agent qwen --permission-mode yolo
```

---

## 九、清理

```bash
# 删除导出的缓存（会在下次运行时自动重新导出）
rm -rf pywen_workspace/pywen_agent_cache

# 删除评测结果
rm -rf results
```
