from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class VideoInput(BaseModel):
    prompt: Optional[str] = None
    image: Optional[str] = None
    audio: Optional[str] = None


class VideoParameters(BaseModel):
    size: Optional[str] = None
    frame_num: Optional[int] = None
    ckpt_dir: Optional[str] = None
    save_file: Optional[str] = None
    offload_model: Optional[bool] = None
    ulysses_size: Optional[int] = None
    t5_fsdp: Optional[bool] = None
    t5_cpu: Optional[bool] = None
    dit_fsdp: Optional[bool] = None
    use_prompt_extend: Optional[bool] = None
    prompt_extend_method: Optional[Literal["dashscope", "local_qwen"]] = None
    prompt_extend_model: Optional[str] = None
    prompt_extend_target_lang: Optional[Literal["zh", "en"]] = None
    base_seed: Optional[int] = None
    sample_solver: Optional[Literal["unipc", "dpm++"]] = None
    sample_steps: Optional[int] = None
    sample_shift: Optional[float] = None
    sample_guide_scale: Optional[Union[float, List[float]]] = None
    convert_model_dtype: Optional[bool] = None
    task: Optional[str] = None
    # animate / s2v / extras
    src_root_path: Optional[str] = None
    refert_num: Optional[int] = None
    replace_flag: Optional[bool] = None
    use_relighting_lora: Optional[bool] = None
    num_clip: Optional[int] = None
    enable_tts: Optional[bool] = None
    tts_prompt_audio: Optional[str] = None
    tts_prompt_text: Optional[str] = None
    tts_text: Optional[str] = None
    pose_video: Optional[str] = None
    start_from_ref: Optional[bool] = None
    infer_frames: Optional[int] = None


class VideoGenerationRequest(BaseModel):
    model: str = Field(..., description="Model id or task key, e.g. wan2.2-t2v-a14b")
    input: VideoInput = Field(default_factory=VideoInput)
    parameters: VideoParameters = Field(default_factory=VideoParameters)


class OutputTaskId(BaseModel):
    task_id: str


class VideoGenerationResponse(BaseModel):
    request_id: str
    output: OutputTaskId


class TaskStatusBody(BaseModel):
    task_id: str
    task_status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]
    message: str = ""
    output: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
