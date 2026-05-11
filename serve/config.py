from __future__ import annotations

import os
from dataclasses import dataclass


def _b(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default


def _i(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration (environment variables)."""
    
    redis_url: str
    api_keys: frozenset[str]
    repo_root: str
    job_dir: str
    output_dir: str
    ckpt_dir: str
    # torchrun / multi-node
    nnodes: int
    nproc_per_node: int
    master_addr: str
    master_port: int
    rdzv_id_prefix: str
    # SSH second node (optional). Example: user@192.168.1.2
    ssh_second_node: str
    ssh_torchrun_prefix: str
    python_bin: str
    torchrun_bin: str
    cluster_lock_ttl_sec: int
    queue_name: str
    task_key_prefix: str
    lock_key: str
    conda_env: str = ""
    conda_exe: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        keys_raw = _b("WAN_SERVE_API_KEYS", _b("WAN_SERVE_API_KEY", ""))
        keys = frozenset(k.strip() for k in keys_raw.split(",") if k.strip())
        return cls(
            redis_url=_b("WAN_REDIS_URL", "redis://127.0.0.1:6379/0"),
            api_keys=keys,
            repo_root=_b("WAN_REPO_ROOT", os.getcwd()),
            job_dir=_b("WAN_JOB_DIR", "/tmp/wan2_jobs"),
            output_dir=_b("WAN_OUTPUT_DIR", "/tmp/wan2_outputs"),
            ckpt_dir=_b("WAN_CKPT_DIR", ""),
            nnodes=_i("WAN_NNODES", 1),
            nproc_per_node=_i("WAN_NPROC_PER_NODE", 1),
            master_addr=_b("WAN_MASTER_ADDR", "127.0.0.1"),
            master_port=_i("WAN_MASTER_PORT", 29500),
            rdzv_id_prefix=_b("WAN_RDZV_PREFIX", "wan"),
            ssh_second_node=_b("WAN_SSH_SECOND_NODE", ""),
            ssh_torchrun_prefix=_b(
                "WAN_SSH_TORCHRUN_PREFIX",
                "cd {repo_root} && export PYTHONPATH={repo_root}:$PYTHONPATH && ",
            ),
            python_bin=_b("WAN_PYTHON", "python3"),
            torchrun_bin=_b("WAN_TORCHRUN", "torchrun"),
            cluster_lock_ttl_sec=_i("WAN_CLUSTER_LOCK_TTL_SEC", 86400),
            queue_name=_b("WAN_QUEUE_NAME", "wan:queue"),
            task_key_prefix=_b("WAN_TASK_KEY_PREFIX", "wan:task:"),
            lock_key=_b("WAN_CLUSTER_LOCK_KEY", "wan:cluster_lock"),
            conda_env=_b("WAN_CONDA_ENV", ""),
            conda_exe=_b("WAN_CONDA_EXE", ""),
        )
