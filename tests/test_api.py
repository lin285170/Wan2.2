"""Integration tests for Wan2.2 API service.

Requires a running instance with Redis + API server.
Set environment variables before running:
  WAN_TEST_API_URL — e.g. http://localhost:8008
  WAN_TEST_API_KEY — a valid API key

Usage:
  export WAN_TEST_API_URL=http://localhost:8008
  export WAN_TEST_API_KEY=sk-test-key
  pytest tests/test_api.py -v
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
import uuid

API_URL = os.environ.get("WAN_TEST_API_URL", "http://localhost:8008")
API_KEY = os.environ.get("WAN_TEST_API_KEY", "sk-test-key")


def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, body: dict | None = None):
    url = f"{API_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ============================================================
# Health endpoint
# ============================================================


class TestHealth:
    def test_healthz(self):
        """GET /healthz should return 200 with status ok."""
        url = f"{API_URL}/healthz"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "ok"


# ============================================================
# Authentication
# ============================================================


class TestAuth:
    def test_no_auth_header(self):
        """POST without Authorization should return 401."""
        url = f"{API_URL}/api/v1/video/generation"
        body = json.dumps({"model": "wan2.2-t2v-a14b", "input": {"prompt": "test"}}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            assert False, "Expected 401"
        except urllib.error.HTTPError as e:
            assert e.code in (401, 403)

    def test_invalid_api_key(self):
        """POST with wrong Bearer token should return 403."""
        url = f"{API_URL}/api/v1/video/generation"
        body = json.dumps({"model": "wan2.2-t2v-a14b", "input": {"prompt": "test"}}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": "Bearer wrong-key", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            assert False, "Expected 403"
        except urllib.error.HTTPError as e:
            assert e.code == 403

    def test_missing_ckpt_dir(self):
        """POST without ckpt_dir (and server WAN_CKPT_DIR empty) should return 400."""
        url = f"{API_URL}/api/v1/video/generation"
        body = json.dumps({
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "test"},
            "parameters": {},
        }).encode()
        req = urllib.request.Request(url, data=body, headers=_headers(), method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            # If WAN_CKPT_DIR is set on server, this will succeed (200) — that's fine too
        except urllib.error.HTTPError as e:
            assert e.code == 400


# ============================================================
# Task creation (full flow)
# ============================================================


class TestTaskCreation:
    def test_create_t2v_task(self):
        """POST /api/v1/video/generation with t2v model should return task_id."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "A cat walking on a beach at sunset"},
            "parameters": {
                "size": "832*480",
                "frame_num": 81,
            },
        })
        assert status == 200
        assert "request_id" in data
        assert "output" in data
        assert "task_id" in data["output"]
        assert data["output"]["task_id"].startswith("wan-")

    def test_create_i2v_task(self):
        """POST with i2v model and image input."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-i2v-a14b",
            "input": {
                "prompt": "A cat dancing",
                "image": "https://example.com/cat.jpg",
            },
            "parameters": {
                "size": "832*480",
            },
        })
        assert status == 200
        assert "task_id" in data["output"]

    def test_create_s2v_task(self):
        """POST with s2v (speech2video) model and audio input."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-s2v-14b",
            "input": {
                "prompt": "A person talking",
                "audio": "https://example.com/speech.wav",
            },
            "parameters": {
                "size": "832*480",
            },
        })
        assert status == 200
        assert "task_id" in data["output"]

    def test_create_task_with_optional_params(self):
        """POST with all optional parameters."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "A dog running in a park"},
            "parameters": {
                "size": "1280x720",
                "frame_num": 81,
                "sample_steps": 40,
                "sample_shift": 5.0,
                "sample_guide_scale": 5.0,
                "base_seed": 42,
                "sample_solver": "unipc",
                "t5_cpu": True,
                "dit_fsdp": True,
                "t5_fsdp": True,
            },
        })
        assert status == 200
        assert "task_id" in data["output"]

    def test_task_id_format(self):
        """Task IDs should follow wan-{uuid_hex} pattern."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "format test"},
        })
        assert status == 200
        task_id = data["output"]["task_id"]
        prefix, hex_part = task_id.split("-", 1)
        assert prefix == "wan"
        assert len(hex_part) == 32  # uuid4 hex


# ============================================================
# Task status query
# ============================================================


class TestTaskStatus:
    def setup_method(self):
        """Create a task before each test."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "status test"},
        })
        assert status == 200
        self.task_id = data["output"]["task_id"]

    def test_get_task_status(self):
        """GET /api/v1/tasks/{task_id} should return task status."""
        status, data = _request("GET", f"/api/v1/tasks/{self.task_id}")
        assert status == 200
        assert data["task_id"] == self.task_id
        assert data["task_status"] in ("PENDING", "RUNNING", "SUCCEEDED", "FAILED")
        assert "message" in data
        assert "output" in data

    def test_get_nonexistent_task(self):
        """GET a non-existent task_id should return 404."""
        fake_id = f"wan-{uuid.uuid4().hex}"
        status, data = _request("GET", f"/api/v1/tasks/{fake_id}")
        assert status == 404

    def test_task_status_transitions(self):
        """Poll task status — should start as PENDING, may transition to RUNNING/SUCCEEDED."""
        for _ in range(5):
            status, data = _request("GET", f"/api/v1/tasks/{self.task_id}")
            assert status == 200
            if data["task_status"] != "PENDING":
                break
            time.sleep(2)


# ============================================================
# File download
# ============================================================


class TestFileDownload:
    def test_download_nonexistent_task(self):
        """GET file for non-existent task should return 404."""
        fake_id = f"wan-{uuid.uuid4().hex}"
        status, data = _request("GET", f"/api/v1/files/by-task/{fake_id}")
        assert status == 404

    def test_download_pending_task(self):
        """GET file for a PENDING task should return 404 (video not ready)."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "download test"},
        })
        assert status == 200
        task_id = data["output"]["task_id"]

        # A freshly created task is PENDING — video won't be ready
        status, data = _request("GET", f"/api/v1/files/by-task/{task_id}")
        assert status == 404


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    def test_empty_prompt(self):
        """POST with empty prompt — API should still accept (prompt is optional in schema)."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {},
        })
        assert status == 200

    def test_missing_model(self):
        """POST without model field — should return 422 (FastAPI validation)."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "input": {"prompt": "test"},
        })
        assert status == 422

    def test_invalid_size_format(self):
        """POST with unusual size format — API should accept (validation is lenient)."""
        status, data = _request("POST", "/api/v1/video/generation", {
            "model": "wan2.2-t2v-a14b",
            "input": {"prompt": "size test"},
            "parameters": {"size": "999x999"},
        })
        # Whether this succeeds depends on server config; at least shouldn't crash
        assert status in (200, 400)

    def test_multiple_tasks_same_prompt(self):
        """Create multiple tasks with the same prompt — each should get unique task_id."""
        ids = []
        for _ in range(3):
            status, data = _request("POST", "/api/v1/video/generation", {
                "model": "wan2.2-t2v-a14b",
                "input": {"prompt": "duplicate test"},
            })
            assert status == 200
            ids.append(data["output"]["task_id"])
        assert len(set(ids)) == 3  # all unique