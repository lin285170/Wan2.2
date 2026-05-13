"""Unit tests for serve module — no running server or Redis required.

Tests config parsing, job_build logic, worker_main routing, and
store signal serialization using mocks.

Usage:
  pytest tests/test_serve.py -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure serve package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@contextmanager
def set_env(**kwargs):
    """Temporarily set environment variables and restore on exit."""
    old = {}
    for k, v in kwargs.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    yield
    for k in kwargs:
        if old[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old[k]


# ============================================================
# serve.config
# ============================================================


class TestSettings:
    def test_defaults(self):
        with set_env(WAN_SERVE_API_KEYS="sk-test"):
            from serve.config import Settings
            s = Settings.from_env()
            assert s.redis_url == "redis://127.0.0.1:6379/0"
            assert s.nnodes == 1
            assert s.nproc_per_node == 1
            assert s.node_rank == 0
            assert s.node_role == "master"
            assert s.signal_key == "wan:signal"
            assert s.master_port == 29500
            assert "sk-test" in s.api_keys

    def test_multi_node_config(self):
        with set_env(
            WAN_SERVE_API_KEYS="sk-test",
            WAN_NNODES="2",
            WAN_NPROC_PER_NODE="4",
            WAN_NODE_RANK="1",
            WAN_NODE_ROLE="worker",
            WAN_MASTER_ADDR="10.0.0.1",
            WAN_MASTER_PORT="29600",
        ):
            from serve.config import Settings
            s = Settings.from_env()
            assert s.nnodes == 2
            assert s.nproc_per_node == 4
            assert s.node_rank == 1
            assert s.node_role == "worker"
            assert s.master_addr == "10.0.0.1"
            assert s.master_port == 29600

    def test_multiple_api_keys(self):
        with set_env(WAN_SERVE_API_KEYS="sk-one,sk-two,sk-three"):
            from serve.config import Settings
            s = Settings.from_env()
            assert s.api_keys == frozenset({"sk-one", "sk-two", "sk-three"})

    def test_empty_api_keys(self):
        # Clear any existing key env vars
        for var in ("WAN_SERVE_API_KEYS", "WAN_SERVE_API_KEY"):
            os.environ.pop(var, None)
        from serve.config import Settings
        s = Settings.from_env()
        assert s.api_keys == frozenset()

    def test_frozen(self):
        with set_env(WAN_SERVE_API_KEYS="sk-test"):
            from serve.config import Settings
            s = Settings.from_env()
            try:
                s.redis_url = "x"
                assert False, "Should be frozen"
            except AttributeError:
                pass


# ============================================================
# serve.job_build
# ============================================================


class TestJobBuild:
    def test_basic_t2v(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest

        env = {
            "WAN_SERVE_API_KEYS": "sk-test",
            "WAN_CKPT_DIR": "/ckpt",
            "WAN_OUTPUT_DIR": "/out",
        }
        for k, v in env.items():
            os.environ[k] = v

        try:
            s = Settings.from_env()
            req = VideoGenerationRequest(
                model="wan2.2-t2v-a14b",
                input={"prompt": "A cat"},
                parameters={"size": "832*480", "frame_num": 81},
            )
            job = request_to_job(req, task_id="wan-abc123", settings=s)
            assert job["model"] == "wan2.2-t2v-a14b"
            assert job["prompt"] == "A cat"
            assert job["size"] == "832*480"
            assert job["frame_num"] == 81
            assert job["ckpt_dir"] == "/ckpt"
            assert job["save_file"] == "/out/wan-abc123.mp4"
        finally:
            for k in env:
                os.environ.pop(k, None)

    def test_parameters_ckpt_dir_overrides_global(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest

        env = {
            "WAN_SERVE_API_KEYS": "sk-test",
            "WAN_CKPT_DIR": "/global-ckpt",
            "WAN_OUTPUT_DIR": "/out",
        }
        for k, v in env.items():
            os.environ[k] = v

        try:
            s = Settings.from_env()
            req = VideoGenerationRequest(
                model="wan2.2-t2v-a14b",
                input={"prompt": "A cat"},
                parameters={"ckpt_dir": "/custom-ckpt"},
            )
            job = request_to_job(req, task_id="wan-xyz", settings=s)
            # per-request ckpt_dir takes precedence
            assert job["ckpt_dir"] == "/custom-ckpt"
        finally:
            for k in env:
                os.environ.pop(k, None)

    def test_none_params_excluded(self):
        from serve.config import Settings
        from serve.job_build import request_to_job
        from serve.schemas import VideoGenerationRequest

        env = {"WAN_SERVE_API_KEYS": "sk-test", "WAN_OUTPUT_DIR": "/out"}
        for k, v in env.items():
            os.environ[k] = v

        try:
            s = Settings.from_env()
            req = VideoGenerationRequest(model="wan2.2-t2v-a14b")
            job = request_to_job(req, task_id="wan-min", settings=s)
            # Parameters that were None should not appear
            assert "size" not in job
            assert "frame_num" not in job
            assert "base_seed" not in job
            # But model, save_file, and ckpt_dir (if global) should be there
            assert "model" in job
            assert "save_file" in job
        finally:
            for k in env:
                os.environ.pop(k, None)


# ============================================================
# serve.worker_main — routing logic
# ============================================================


class TestWorkerRouting:
    def test_master_role_calls_main_master(self):
        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NODE_ROLE"] = "master"

        try:
            with patch("serve.worker_main.main_master") as mock_master, \
                 patch("serve.worker_main.main_worker") as mock_worker:
                from serve.worker_main import main
                with patch("serve.worker_main.Settings") as MockSettings, \
                     patch("serve.worker_main.TaskStore"):
                    MockSettings.from_env.return_value = MagicMock(
                        job_dir="/tmp/jobs",
                        output_dir="/tmp/out",
                        node_role="master",
                    )
                    main()
                    mock_master.assert_called_once()
                    mock_worker.assert_not_called()
        finally:
            os.environ.pop("WAN_SERVE_API_KEYS", None)
            os.environ.pop("WAN_NODE_ROLE", None)

    def test_worker_role_calls_main_worker(self):
        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NODE_ROLE"] = "worker"

        try:
            with patch("serve.worker_main.main_master") as mock_master, \
                 patch("serve.worker_main.main_worker") as mock_worker:
                from serve.worker_main import main
                with patch("serve.worker_main.Settings") as MockSettings, \
                     patch("serve.worker_main.TaskStore"):
                    MockSettings.from_env.return_value = MagicMock(
                        job_dir="/tmp/jobs",
                        output_dir="/tmp/out",
                        node_role="worker",
                    )
                    main()
                    mock_worker.assert_called_once()
                    mock_master.assert_not_called()
        finally:
            os.environ.pop("WAN_SERVE_API_KEYS", None)
            os.environ.pop("WAN_NODE_ROLE", None)


# ============================================================
# serve.store — signal publish
# ============================================================


class TestStoreSignal:
    def test_publish_signal(self):
        from serve.config import Settings
        from serve.store import TaskStore

        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        try:
            s = Settings.from_env()
            store = TaskStore(s)
            mock_redis = MagicMock()
            store._r = mock_redis

            payload = json.dumps({"task_id": "wan-test", "rdzv_id": "wan-abc"})
            store.publish_signal(payload)

            mock_redis.publish.assert_called_once_with(s.signal_key, payload)
        finally:
            os.environ.pop("WAN_SERVE_API_KEYS", None)


# ============================================================
# serve.launcher — torchrun command construction
# ============================================================


class TestLauncher:
    def test_torchrun_cmd_master(self):
        from serve.config import Settings
        from serve.launcher import _torchrun_cmd

        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NNODES"] = "2"
        os.environ["WAN_NPROC_PER_NODE"] = "4"
        os.environ["WAN_NODE_RANK"] = "0"
        os.environ["WAN_MASTER_ADDR"] = "10.0.0.1"
        os.environ["WAN_MASTER_PORT"] = "29500"

        try:
            s = Settings.from_env()
            cmd = _torchrun_cmd(s, Path("/data/jobs/wan-test.json"), "wan-test-rdzv")
            assert cmd[0] == "torchrun"
            assert "--nnodes=2" in cmd
            assert "--nproc_per_node=4" in cmd
            assert "--node_rank=0" in cmd
            assert "--rdzv_backend=c10d" in cmd
            assert "--rdzv_endpoint=10.0.0.1:29500" in cmd
            assert "--rdzv_id=wan-test-rdzv" in cmd
            assert "--job_json" in cmd
        finally:
            for k in ("WAN_SERVE_API_KEYS", "WAN_NNODES", "WAN_NPROC_PER_NODE",
                       "WAN_NODE_RANK", "WAN_MASTER_ADDR", "WAN_MASTER_PORT"):
                os.environ.pop(k, None)

    def test_torchrun_cmd_worker(self):
        from serve.config import Settings
        from serve.launcher import _torchrun_cmd

        os.environ["WAN_SERVE_API_KEYS"] = "sk-test"
        os.environ["WAN_NNODES"] = "2"
        os.environ["WAN_NPROC_PER_NODE"] = "4"
        os.environ["WAN_NODE_RANK"] = "1"
        os.environ["WAN_MASTER_ADDR"] = "10.0.0.1"

        try:
            s = Settings.from_env()
            cmd = _torchrun_cmd(s, Path("/data/jobs/wan-test.json"), "wan-test-rdzv")
            assert "--node_rank=1" in cmd
        finally:
            for k in ("WAN_SERVE_API_KEYS", "WAN_NNODES", "WAN_NPROC_PER_NODE",
                       "WAN_NODE_RANK", "WAN_MASTER_ADDR"):
                os.environ.pop(k, None)


# ============================================================
# serve.schemas — validation
# ============================================================


class TestSchemas:
    def test_video_generation_request_model_required(self):
        from serve.schemas import VideoGenerationRequest
        try:
            VideoGenerationRequest()
            assert False, "model is required"
        except Exception:
            pass

    def test_video_generation_request_with_model(self):
        from serve.schemas import VideoGenerationRequest
        req = VideoGenerationRequest(model="wan2.2-t2v-a14b")
        assert req.model == "wan2.2-t2v-a14b"
        assert req.input.prompt is None
        assert req.parameters.size is None

    def test_video_generation_request_full(self):
        from serve.schemas import VideoGenerationRequest
        req = VideoGenerationRequest(
            model="wan2.2-i2v-a14b",
            input={"prompt": "A cat", "image": "https://example.com/cat.jpg"},
            parameters={"size": "832*480", "frame_num": 81, "base_seed": 42},
        )
        assert req.input.prompt == "A cat"
        assert req.input.image == "https://example.com/cat.jpg"
        assert req.parameters.size == "832*480"
        assert req.parameters.frame_num == 81
        assert req.parameters.base_seed == 42

    def test_task_status_body(self):
        from serve.schemas import TaskStatusBody
        body = TaskStatusBody(task_id="wan-test", task_status="SUCCEEDED")
        assert body.task_id == "wan-test"
        assert body.task_status == "SUCCEEDED"
        assert body.message == ""
        assert body.output == {}

    def test_health_response(self):
        from serve.schemas import HealthResponse
        resp = HealthResponse()
        assert resp.status == "ok"


if __name__ == "__main__":
    unittest.main()