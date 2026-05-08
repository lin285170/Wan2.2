# Wan2.2 集群推理与 DashScope 风格 HTTP 服务部署指南

本文说明如何在 **2 台 × 4×A100 40GB**（或任意 `nnodes × nproc_per_node = WORLD_SIZE`）上运行本仓库自带的 **异步任务 API**（兼容 DashScope 习惯的 `Bearer` 鉴权、`POST` 提交、`GET` 轮询任务状态）。

---

## 1. 架构说明

| 组件 | 职责 |
|------|------|
| **Redis** | 任务队列 `WAN_QUEUE_NAME`、任务元数据 `WAN_TASK_KEY_PREFIX*`、集群互斥锁 `WAN_CLUSTER_LOCK_KEY`（同一时刻只跑一个 `torchrun` 作业）。 |
| **`run_api_server.py` + `serve.api`** | FastAPI：提交任务、查询状态、下载 MP4。 |
| **`python -m serve.worker_main`** | 从队列取 `task_id`，写 `job.json`，调用 `torchrun … generate_job.py`。 |
| **`generate_job.py`** | 读取 JSON，调用 `generate.args_from_job_dict` + `generate.generate`。 |

**重要**：默认实现假设 **只有一个 worker 进程** 在消费队列（全局 GPU 锁）。若启动多个 worker 会抢锁并反复 requeue；多任务并发需改造为每套 GPU 独立队列或 Kubernetes Job。

---

## 2. 环境准备（两台 GPU 机 + 一台 API 机可选）

1. **Python**：与官方 README 一致，`torch>=2.4`，安装 `requirements.txt` + 推理所需依赖。  
2. **服务依赖**（跑 API / worker 的机器）：  
   ```bash
   pip install -r requirements_serve.txt
   ```  
3. **Redis**：可部署在 API 同机或独立 VM；两台 GPU 机与 API 均需能访问该地址。  
4. **共享存储（强烈推荐）**：NFS 等，两台 GPU 上 **相同绝对路径** 挂载：  
   - 模型目录 `WAN_CKPT_DIR`  
   - 任务 JSON 目录 `WAN_JOB_DIR`  
   - 输出视频目录 `WAN_OUTPUT_DIR`  

5. **NCCL 双机**：设置 `NCCL_SOCKET_IFNAME`、主机名解析、防火墙放行 `WAN_MASTER_PORT` 及 PyTorch 分布式端口；有 RDMA 时按机房文档配置 IB。

---

## 3. 环境变量参考

### 通用 / API / Worker

| 变量 | 说明 | 示例 |
|------|------|------|
| `WAN_SERVE_API_KEYS` | 逗号分隔的 API Key（`Authorization: Bearer <key>`） | `sk-local-xxx,sk-local-yyy` |
| `WAN_REDIS_URL` | Redis 连接串 | `redis://10.0.0.5:6379/0` |
| `WAN_REPO_ROOT` | 本仓库绝对路径 | `/data/Wan2.2` |
| `WAN_JOB_DIR` | 任务 JSON 目录（需共享） | `/mnt/wan/jobs` |
| `WAN_OUTPUT_DIR` | 输出 MP4（需共享） | `/mnt/wan/out` |
| `WAN_CKPT_DIR` | 默认 checkpoint 根目录 | `/mnt/wan/Wan2.2-T2V-A14B` |

### 多机 torchrun（Worker 所在机应能 `ssh` 到第二台时）

| 变量 | 说明 |
|------|------|
| `WAN_NNODES` | 节点数，例如 `2` |
| `WAN_NPROC_PER_NODE` | 每节点进程数，例如 `4`（总 8 卡） |
| `WAN_MASTER_ADDR` | rank0 所在机 IP（**第一**台 GPU 机） |
| `WAN_MASTER_PORT` | rendezvous 端口，如 `29500` |
| `WAN_RDZV_PREFIX` | rendezvous id 前缀（会再拼 `task_id`） |
| `WAN_SSH_SECOND_NODE` | 第二台登录串，如 `ubuntu@192.168.1.12` |
| `WAN_SSH_TORCHRUN_PREFIX` | SSH 远端 shell 前缀，默认 `cd {repo_root} && export PYTHONPATH={repo_root}:$PYTHONPATH && ` |
| `WAN_PYTHON` / `WAN_TORCHRUN` | 可选，覆盖可执行文件路径 |

### Prompt 扩展（可选）

若任务 JSON 里 `use_prompt_extend=true` 且 `prompt_extend_method=dashscope`：

- `DASH_API_KEY`  
- 国际站可设 `DASH_API_URL=https://dashscope-intl.aliyuncs.com/api/v1`

---

## 4. 单机 8 卡（单节点测试）

```bash
export WAN_SERVE_API_KEYS="sk-dev"
export WAN_REDIS_URL="redis://127.0.0.1:6379/0"
export WAN_REPO_ROOT="/data/Wan2.2"
export WAN_CKPT_DIR="/data/Wan2.2-T2V-A14B"
export WAN_JOB_DIR="/tmp/wan_jobs"
export WAN_OUTPUT_DIR="/tmp/wan_out"
export WAN_NNODES=1
export WAN_NPROC_PER_NODE=8
export PYTHONPATH="/data/Wan2.2:$PYTHONPATH"

# 终端 1
redis-server &
python run_api_server.py

# 终端 2（与 API 同机或能访问 Redis 的 GPU 机）
python -m serve.worker_main
```

提交示例：

```bash
curl -sS -X POST "http://127.0.0.1:8008/api/v1/video/generation" \
  -H "Authorization: Bearer sk-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-t2v-a14b",
    "input": { "prompt": "A cat walking on grass." },
    "parameters": {
      "size": "1280*720",
      "dit_fsdp": true,
      "t5_fsdp": true,
      "ulysses_size": 8,
      "offload_model": false,
      "convert_model_dtype": true
    }
  }'
```

查询与下载（将 `TASK_ID` 换成响应里的 `task_id`）：

```bash
curl -sS -H "Authorization: Bearer sk-dev" \
  "http://127.0.0.1:8008/api/v1/tasks/TASK_ID"

curl -L -o out.mp4 -H "Authorization: Bearer sk-dev" \
  "http://127.0.0.1:8008/api/v1/files/by-task/TASK_ID"
```

---

## 5. 双机 2×4 卡 A100（推荐生产形态）

1. **节点 0**：跑 `worker_main`（或单独调度机通过 SSH 触发；默认 worker 在节点 0 本机起 `torchrun`，并对节点 1 执行 SSH）。  
2. **节点 0 / 1**：`WAN_REPO_ROOT`、`WAN_JOB_DIR`、`WAN_OUTPUT_DIR`、`WAN_CKPT_DIR` 在 NFS 上路径一致。  
3. **免密 SSH**：节点 0 → 节点 1 建议配置公钥登录；生产环境请将 `serve/launcher.py` 中 `StrictHostKeyChecking=no` 改为受控 known_hosts。  

```bash
# 两机相同（或通过 systemd 注入）
export WAN_NNODES=2
export WAN_NPROC_PER_NODE=4
export WAN_MASTER_ADDR=10.0.0.10      # 节点0 内网 IP
export WAN_MASTER_PORT=29500
export WAN_SSH_SECOND_NODE=ubuntu@10.0.0.11
export WAN_REPO_ROOT=/mnt/wan/Wan2.2
export PYTHONPATH=/mnt/wan/Wan2.2:$PYTHONPATH
export WAN_CKPT_DIR=/mnt/wan/Wan2.2-T2V-A14B
export WAN_JOB_DIR=/mnt/wan/jobs
export WAN_OUTPUT_DIR=/mnt/wan/out
```

在 **节点 0** 启动 worker：

```bash
python -m serve.worker_main
```

Worker 会：

1. 在节点 0 执行 `torchrun --nnodes=2 --nproc_per_node=4 … generate_job.py --job_json <共享路径>`；  
2. 通过 SSH 在节点 1 启动 **同一条** `torchrun` 命令（依赖 PyTorch c10d rendezvous 自动分配 rank）。

API 服务可放在任意能访问 Redis 的机器上（不必有 GPU）。

---

## 6. 任务 JSON 与 `model` 别名

HTTP 请求体会被合并为 `generate.args_from_job_dict` 可接受的字典：

- 顶层 **`model`** 可为：`wan2.2-t2v-a14b`、`wan2.2-i2v-a14b`、`wan2.2-ti2v-5b`、`wan2.2-s2v-14b`、`wan2.2-animate-14b`，或直接 `WAN_CONFIGS` 里的 `task` 字符串。  
- 其余字段与 `generate.py` 命令行一致，嵌套在 `input` / `parameters` 中亦可。  
- `sample_guide_scale` 可为 **单个 float** 或 **两个 float 的数组**（低/高噪声专家）。  

直接调用 `generate_job.py`（不经 HTTP）示例：

```bash
cat > /mnt/wan/jobs/manual.json <<'EOF'
{
  "model": "wan2.2-t2v-a14b",
  "ckpt_dir": "/mnt/wan/Wan2.2-T2V-A14B",
  "save_file": "/mnt/wan/out/manual.mp4",
  "prompt": "Two cats boxing on stage.",
  "size": "1280*720",
  "dit_fsdp": true,
  "t5_fsdp": true,
  "ulysses_size": 8,
  "convert_model_dtype": true,
  "offload_model": false
}
EOF

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29501 --rdzv_id=manual1 \
  /mnt/wan/Wan2.2/generate_job.py --job_json /mnt/wan/jobs/manual.json
```

---

## 7. systemd 示例（API）

`/etc/systemd/system/wan-api.service`：

```ini
[Unit]
Description=Wan2.2 HTTP API
After=network.target

[Service]
User=wan
WorkingDirectory=/mnt/wan/Wan2.2
Environment=PYTHONPATH=/mnt/wan/Wan2.2
Environment=WAN_SERVE_API_KEYS=sk-prod-xxx
Environment=WAN_REDIS_URL=redis://127.0.0.1:6379/0
Environment=WAN_CKPT_DIR=/mnt/wan/Wan2.2-T2V-A14B
Environment=WAN_JOB_DIR=/mnt/wan/jobs
Environment=WAN_OUTPUT_DIR=/mnt/wan/out
Environment=WAN_REPO_ROOT=/mnt/wan/Wan2.2
ExecStart=/mnt/wan/venv/bin/python /mnt/wan/Wan2.2/run_api_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Worker 类似，将 `ExecStart` 改为 `python -m serve.worker_main`，并在 GPU 节点 0 上运行。

---

## 8. 安全与运维建议

- 仅内网暴露 API，或前置 mTLS / 零信任网关。  
- 定期轮换 `WAN_SERVE_API_KEYS`。  
- 大模型与生成结果路径做磁盘配额与清理任务。  
- 监控 Redis 队列长度、worker 日志、`torchrun` 退出码。  

---

## 9. 容器化部署（Docker Compose）

仓库提供 **CPU 版 API 镜像** 与 **GPU Worker 镜像**，由 `docker-compose.yml` 编排 Redis、API、Worker。

### 9.1 前置条件

- 已安装 [Docker](https://docs.docker.com/engine/install/) 与 [Docker Compose V2](https://docs.docker.com/compose/)。  
- **Worker 所在宿主机** 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，并可用 `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 验证。  
- 将官方权重下载到宿主机目录，例如 `/data/Wan2.2-T2V-A14B`，供 **只读** 挂载到 Worker 容器的 `/ckpt`。

### 9.2 配置与启动

```bash
cd /path/to/Wan2.2
cp docker/compose.env.example .env
# 编辑 .env：至少设置 WAN_SERVE_API_KEYS、WAN_CKPT_HOST_PATH
```

仅启动 **Redis + API**（开发机无 GPU时）：

```bash
docker compose up -d --build redis api
```

在 **带 NVIDIA GPU 的机器** 上启动完整栈（含 Worker，使用 Compose `gpu` profile）：

```bash
docker compose --profile gpu up -d --build
```

常用命令：

```bash
docker compose logs -f api worker
docker compose ps
```

API 默认映射到宿主机 `WAN_API_PORT`（默认 `8008`）。健康检查：`GET http://<host>:8008/healthz`。

### 9.3 数据卷说明

| 卷名 | 挂载点 | 说明 |
|------|--------|------|
| `wan_shared` | 容器内 `/data` | `jobs` → `/data/jobs`，`outputs` → `/data/outputs`；API 与 Worker 共享，用于任务 JSON 与生成视频。 |
| 绑定挂载 | `/ckpt` | 来自 `.env` 的 `WAN_CKPT_HOST_PATH`，只读挂载到 Worker。 |

### 9.4 镜像构建参数（Worker）

| 构建参数 | 默认 | 说明 |
|----------|------|------|
| `BASE_IMAGE` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | 可按机房 CUDA 版本替换为官方 PyTorch 标签。 |
| `INSTALL_FLASH_ATTN` | `0` | 设为 `1` 时尝试安装 `flash_attn`（需与基础镜像 CUDA 匹配，失败时构建仍可能继续）。 |

示例：

```bash
docker build -f docker/Dockerfile.worker \
  --build-arg INSTALL_FLASH_ATTN=1 \
  -t wan2-worker:latest .
```

### 9.5 双机 GPU 与 Compose

`docker-compose.yml` 描述的是 **单机上的多卡容器**。若要在 **两台物理机** 各跑 4 卡并沿用现有 `serve.launcher` 的 SSH 双机 `torchrun`：

1. 两台机器安装 Docker + NVIDIA Toolkit，**同一 NFS** 挂载到相同路径（含代码、权重、`WAN_JOB_DIR` / `WAN_OUTPUT_DIR`）。  
2. 在 **节点 0** 上可仍用 Compose 起 Redis（或外置托管 Redis），API 与 Worker 容器；在 **节点 1** 仅起 **Worker 容器**（或不用 Compose，直接 `docker run`），两台 Worker 不要同时消费同一队列——当前设计为 **单 worker 消费**；双机多卡推荐 **只在节点 0 起一个 Worker 容器**，并在 `.env` 中配置 `WAN_NNODES=2`、`WAN_NPROC_PER_NODE=4`、`WAN_MASTER_ADDR`、`WAN_SSH_SECOND_NODE`，由容器内 `torchrun` + SSH 拉起第二台进程（需节点 0 容器能 SSH 到节点 1，且节点 1 已安装相同镜像或具备相同 Python/torch 环境）。  

更稳妥的生产方式是将 **Redis + API** 托管在控制面，**每台 GPU 机** 用 `docker run` 或 Kubernetes Job 只跑 `wan2-worker`，并改造队列分区；超出本文范围时可单独扩展。

### 9.6 相关文件

| 路径 | 说明 |
|------|------|
| `docker-compose.yml` | Redis、api、worker 服务定义 |
| `docker/Dockerfile.api` | 仅 FastAPI 依赖的轻量 API 镜像 |
| `docker/Dockerfile.worker` | CUDA + Wan 推理 + `serve.worker` |
| `docker/entrypoint-worker.sh` | Worker 入口 |
| `docker/compose.env.example` | 复制为仓库根目录 `.env` 的模板 |
| `.dockerignore` | 减小构建上下文 |
| `README.md` | 部署与 HTTP 服务主文档（本文件） |
| `DEPLOY_SERVE.md` | 历史/外链兼容：仅指向 `README.md` |

---

## 10. 代码变更摘要

| 路径 | 说明 |
|------|------|
| `README.md` | 部署、DashScope 风格 API、Docker 主文档 |
| `generate.py` | `_build_parser` / `parse_args` / `args_from_job_dict` / `JOB_MODEL_ALIASES` |
| `generate_job.py` | `torchrun` 入口，读 `--job_json` |
| `serve/` | FastAPI、Redis、launcher、worker |
| `run_api_server.py` | 开发用 uvicorn 启动 |
| `requirements_serve.txt` | API 额外依赖 |
| `docker-compose.yml` / `docker/*` | 容器化编排与镜像 |

若需 **HTTPS、限流、多队列、回调 Webhook**，可在 `serve/api.py` 外再包一层网关或扩展本模块。
