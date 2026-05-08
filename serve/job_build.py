from __future__ import annotations

from typing import Any, Dict

from .config import Settings
from .schemas import VideoGenerationRequest


def request_to_job(
    req: VideoGenerationRequest,
    task_id: str,
    settings: Settings,
) -> Dict[str, Any]:
    """Build a flat job dict for ``generate.args_from_job_dict``."""
    job: Dict[str, Any] = {"model": req.model}
    job.update(req.input.model_dump(exclude_none=True))
    job.update(req.parameters.model_dump(exclude_none=True))
    if not job.get("ckpt_dir") and settings.ckpt_dir:
        job["ckpt_dir"] = settings.ckpt_dir
    if not job.get("save_file"):
        job["save_file"] = f"{settings.output_dir.rstrip('/')}/{task_id}.mp4"
    return job
