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


def launch_generate_job(
    settings: Settings,
    job_json: Path,
    rdzv_id: str,
) -> int:
    """
    Run ``torchrun ... generate_job.py`` locally. For ``nnodes>1``, also starts
    the same command on ``WAN_SSH_SECOND_NODE`` via SSH (both must see the same
    ``job_json`` path, e.g. on NFS).
    """
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
    prefix = settings.ssh_torchrun_prefix.format(repo_root=str(repo))
    remote_shell = f"{prefix}{quoted}"

    rem = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", settings.ssh_second_node.strip(), "bash", "-lc", remote_shell],
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
