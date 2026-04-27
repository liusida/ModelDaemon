[English](README.md)

# ModelDaemon（演示）

一个进程把模型留在内存里；另一个终端在同一进程里执行 `task.py`。守护进程会包装 `AutoModelForCausalLM.from_pretrained`，因此**同一份任务脚本**既可以单独运行（每次正常完整加载），也可以在守护进程下运行（按模型 id 首次加载后缓存，直到进程退出）。`task.py` 不需要 import 或提及守护进程。

这不是生产级方案，只用来演示思路。

## 环境（uv）

用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境与锁文件。

```bash
uv sync
```

只跑任务（无守护进程）：

```bash
uv run python task.py --model Qwen/Qwen3-0.6B
```

配合守护进程：

```bash
uv run model_daemon.py                    # 启动 serve（等价于显式写 `serve`）
uv run model_daemon.py run task.py --model Qwen/Qwen3-0.6B
```

也可以激活虚拟环境：Unix 下 `source .venv/bin/activate`，再照常使用 `python`。

### 不用 uv

任意虚拟环境安装相同依赖即可，例如：

```bash
pip install torch "transformers>=4.51.0"
```

## 运行

```text
# 终端 A
python model_daemon.py serve

# 终端 B — 示例为 Qwen3-0.6B（名称在 task.py / --model 中，不在服务端写死）
python model_daemon.py run task.py --model Qwen/Qwen3-0.6B
```

再次使用相同 `--model` 会复用已缓存的权重（无需重新下载/加载）。换另一个 `--model` 会在首次使用时加载一次，之后也会一直缓存。

### 耗时示例（`task.py` 在 stderr 上打印 `wall time`）

两种情况下 `task.py` 相同，只是进程不同。请**在另一个终端保持 `model_daemon.py serve` 运行**，这样第二条命令会命中内存里的模型（以下为某台带 CUDA 的机器上的数据，你的环境会不同）。

```text
$ python task.py
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████| 311/311 [00:00<00:00, 3634.76it/s]
Qwen/Qwen3-0.6B: 596.05M params, device=cuda:0
[transformers] The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
sample: "Hello Answer! I'm a bit confused about"
wall time: 3.680s

$ python model_daemon.py run task.py
Qwen/Qwen3-0.6B: 596.05M params, device=cuda:0
sample: "Hello Answer! I'm a bit confused about"
wall time: 1.792s
```

可选预热：`python model_daemon.py serve my_loader.py`，其中 `my_loader.py` 定义 `load_models() -> dict`（在跑任务前合并进缓存）。

端口默认 `8765`。要改端口，**两个终端**都要设置相同的 `MODEL_DAEMON_PORT`。

## 文件

| 文件 | 说明 |
|------|------|
| `model_daemon.py` | TCP + `runpy`，包装 `AutoModelForCausalLM.from_pretrained`；可选 `serve 某路径.py` 预热 |
| `task.py` | 示例：`--model`，默认 Qwen3-0.6B |
| `pyproject.toml` | 仅列出依赖供 `uv` 使用（不作为可安装包发布） |
| `uv.lock` | 固定版本（由 `uv lock` / `uv sync` 生成） |

## 注意

访客代码在守护进程内执行（等同于任意代码执行）。监听地址仅为 `127.0.0.1`。
