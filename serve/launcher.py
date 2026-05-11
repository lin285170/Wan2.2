from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List

from .config import Settings


def _torchrun_cmd(settings: Settings, job_json: Path, rdzv_id: str) -> List[str]:
    repo = Path(settings.repo_root).resolve()
    script = repo / "generate_job.py"
    return [
        settings.torchrun_bin,
        f"--nnodes={settings.nnodes}",
        f"--nproc_per_node={settings.nproc_per_node}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={settings.master_addr}:{settings.master_port}",
        f"--rdzv_id={rdzv_id}",
        str(script),
        "--job_json",
        str(job_json.resolve()),
    ]


def _env_for_child(settings: Settings) -> dict:
    env = os.environ.copy()
    repo = str(Path(settings.repo_root).resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo}:{prev}" if prev else repo
    return env


def _build_remote_activate_prefix(settings: Settings) -> str:
    """Build the remote shell prefix for conda activation and environment setup."""
    conda_env = settings.conda_env.strip()
    if not conda_env:
        return ""

    conda_exe = (
        settings.conda_exe.strip()
        or os.environ.get("CONDA_EXE", "")
        or "conda"
    )

    parts = [
        f'eval "$({conda_exe} shell.bash hook)"',
        f"conda activate {shlex.quote(conda_env)}",
    ]

    ld_path = os.environ.get("WAN_REMOTE_LD_LIBRARY_PATH", "").strip()
    if ld_path:
        parts.append(f'export LD_LIBRARY_PATH="{ld_path}:$LD_LIBRARY_PATH"')

    omp_threads = os.environ.get("WAN_REMOTE_OMP_NUM_THREADS", "").strip()
    if omp_threads:
        parts.append(f"export OMP_NUM_THREADS={shlex.quote(omp_threads)}")

    return " && ".join(parts) + " && "


def launch_generate_job(
    settings: Settings,
    job_json: Path,
    rdzv_id: str,
) -> int:
    job_json = job_json.resolve()
    repo = Path(settings.repo_root).resolve()
    cmd = _torchrun_cmd(settings, job_json, rdzv_id)
    env = _env_for_child(settings)

    if settings.nnodes <= 1:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
        )
        return int(proc.returncode)

    if not settings.ssh_second_node.strip():
        raise RuntimeError(
            "WAN_NNODES>1 requires WAN_SSH_SECOND_NODE (e.g. user@gpu-node-1)"
        )

    quoted = " ".join(shlex.quote(c) for c in cmd)

    # ---- 关键修改：组装远程命令 ----
    # 1) conda 激活前缀
    activate_prefix = _build_remote_activate_prefix(settings)
    # 2) 用户自定义前缀（保留兼容）
    user_prefix = settings.ssh_torchrun_prefix.format(repo_root=str(repo))
    # 3) 最终远程命令
    remote_shell = f"{activate_prefix}{user_prefix}{quoted}"

    rem = subprocess.Popen(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            settings.ssh_second_node.strip(),
            "bash", "-lc", remote_shell,  # -l 保证 PATH 等基础环境
        ],
        cwd=str(repo),
        env=env,
    )
    time.sleep(2)
    loc = subprocess.run(cmd, cwd=str(repo), env=env)
    rem.wait()
    if loc.returncode != 0:
        return int(loc.returncode)
    if rem.returncode != 0:
        return int(rem.returncode or 1)
    return 0