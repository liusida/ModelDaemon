[English](README.md)

# ModelDaemon

调试时经常反复跑同一段脚本，可默认每次新开 Python 进程都会重新加载模型，等待时间长。ModelDaemon 让一个服务进程常驻，把同一份权重留在 GPU 显存或 CPU 内存里，直到你关掉它为止。

## 运行

```bash
uv sync   # 首次或依赖变更后
```

**终端 A — 常驻进程，模型留在内存**

```bash
uv run model_daemon.py
```

**终端 B — 与普通 `task.py` 相同，但在 A 的进程里执行**

```bash
uv run model_daemon.py run task.py --model Qwen/Qwen3-0.6B
```

若已激活虚拟环境：`python model_daemon.py` / `python model_daemon.py run task.py …`

默认端口 `8765`。若修改端口，**两个终端**使用相同的 `MODEL_DAEMON_PORT`。

## 耗时（`task.py` 在 stderr 打印）

同一份 `task.py`；差别在于 `from_pretrained` 是在新进程里完整跑一遍，还是复用守护进程里已在显存/内存中的权重。某台 CUDA 机器上的示例：

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

## 作用说明

- 常驻进程按 Hugging Face 模型 id 加载并缓存权重（通过包装 `AutoModelForCausalLM.from_pretrained`）。
- `task.py` 只用常规 Transformers 写法，不 import 守护进程。
- 单独运行 `task.py` 每次冷启动；通过 `model_daemon.py run …` 执行则直到服务端退出前都可复用权重。
- 相同 `--model` 会跳过重复加载；新 id 首次加载后也会保留在缓存中。

可选：`python model_daemon.py serve my_loader.py`，`my_loader.py` 中定义 `load_models() -> dict`，在跑任务前预填缓存。

## 环境

用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境与 `uv.lock`。

只跑任务（无守护进程）：

```bash
uv run python task.py --model Qwen/Qwen3-0.6B
```

不用 uv：

```bash
pip install torch "transformers>=4.51.0"
```

虚拟环境已同步时可用 `uv run --no-sync …` 减少重复安装提示。

## 文件

| 文件 | 说明 |
|------|------|
| `model_daemon.py` | TCP + `runpy`，包装 `AutoModelForCausalLM.from_pretrained`；可选 `serve 某路径.py` 预热 |
| `task.py` | 示例：`--model`，默认 Qwen3-0.6B |
| `pyproject.toml` | 供 `uv` 使用的依赖列表（虚拟项目，不作为库发布） |
| `uv.lock` | 固定版本 |

访客代码在守护进程内执行。服务仅监听 `127.0.0.1`。
