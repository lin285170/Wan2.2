from __future__ import annotations

from typing import Any, Dict

from .config import Settings
from .schemas import ModelEnum, VideoGenerationRequest

# Default size per model (first supported size from WAN_CONFIGS)
_MODEL_DEFAULT_SIZE = {
    ModelEnum.t2v_a14b.value: "1280*720",
    ModelEnum.i2v_a14b.value: "832*480",
    ModelEnum.ti2v_5b.value: "1280*704",
    ModelEnum.animate_14b.value: "720*1280",
    ModelEnum.s2v_14b.value: "832*480",
}


def request_to_job(
    req: VideoGenerationRequest,
    task_id: str,
    settings: Settings,
) -> Dict[str, Any]:
    """Build a flat job dict for ``generate.args_from_job_dict``."""
    model = req.model.value
    job: Dict[str, Any] = {"model": model}
    job.update(req.input.model_dump(exclude_none=True))
    job.update(req.parameters.model_dump(exclude_none=True))

    if not job.get("ckpt_dir") and settings.ckpt_dir:
        job["ckpt_dir"] = settings.ckpt_dir
    if not job.get("save_file"):
        job["save_file"] = f"{settings.output_dir.rstrip('/')}/{task_id}.mp4"

    # Fill default size if not provided
    if not job.get("size"):
        job["size"] = _MODEL_DEFAULT_SIZE.get(model, "832*480")

    # Ensure video/audio/image fields map to the correct generate.py args
    # generate.py uses --image, --audio, --video directly
    # The job dict already has them from input.model_dump

    return job