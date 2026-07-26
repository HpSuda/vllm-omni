# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import math
import os
import shlex
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import FormData, UploadFile
from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import (
    HEADER_SUPER_P95_ESTIMATED_SERVICE_S,
    HEADER_SUPER_P95_SACRIFICIAL,
    estimate_service_time_s,
    normalize_super_p95_hardware_profile,
    parse_super_p95_load_headers,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.trace_logging import write_trace_event

logger = init_logger(__name__)

# Keep dispatcher-side routing aligned with backend batch2 eligibility.
_QWEN_SMALL_IMAGE_MAX_PIXELS = 768 * 768
_CENTRAL_PULL_POLICIES = {
    "central_pull_max_risk",
    "central_pull_cost_damped_risk",
}
_TAIL_DISPATCH_MODES = {
    "immediate",
    "protected_drain",
}


@dataclass
class BackendState:
    name: str
    base_url: str
    hardware_profile: str = "910B2"
    normal_load_s: float = 0.0
    sacrificial_load_s: float = 0.0
    inflight_normal_requests: int = 0
    inflight_sacrificial_requests: int = 0
    latency_ema_s: float = 0.0
    batchable_counts: dict[tuple[Any, ...], int] = field(default_factory=dict)

    def weighted_total_load_s(self, alpha: float) -> float:
        return self.normal_load_s + alpha * self.sacrificial_load_s

    def score_tuple(self, alpha: float) -> tuple[float, float, float, float, str]:
        inflight_score = self.inflight_normal_requests + alpha * self.inflight_sacrificial_requests
        return (
            self.weighted_total_load_s(alpha),
            inflight_score,
            self.latency_ema_s,
            self.normal_load_s,
            self.name,
        )


@dataclass(frozen=True)
class DispatchDecision:
    backend_index: int
    estimated_service_s: float
    is_sacrificial: bool
    arrival_counter: int
    credits_before: int
    credits_after: int
    quota_added: int
    global_max_service_s: float
    central_wait_s: float = 0.0
    central_risk_score: float | None = None
    central_risk_beta: float | None = None
    batch_key: tuple[Any, ...] | None = None


@dataclass
class PendingNormalDispatch:
    arrival_counter: int
    arrival_time_s: float
    estimated_service_s_by_backend: tuple[float, ...]
    credits_before: int
    credits_after: int
    quota_added: int
    global_max_service_s: float
    batch_key: tuple[Any, ...] | None
    future: asyncio.Future[DispatchDecision]


@dataclass
class PendingTailDispatch:
    request_id: str
    arrival_time_s: float
    decision: DispatchDecision
    future: asyncio.Future[DispatchDecision]


@dataclass(frozen=True)
class ManagedBackendSpec:
    device_id: str
    port: int
    base_url: str
    hardware_profile: str


@dataclass
class ManagedBackendProcess:
    spec: ManagedBackendSpec
    process: subprocess.Popen[str]
    log_file: Any


class ManagedBackendLauncher:
    _READY_LOG_PATTERNS = (
        "Application startup complete",
        "Starting vLLM API server",
        "Pure diffusion API server initialized",
    )

    def __init__(
        self,
        *,
        specs: list[ManagedBackendSpec],
        model: str,
        backend_args: list[str],
        backend_env: dict[str, str],
        backend_scheduler: str | None,
        device_env_var: str,
        health_timeout_s: float,
        health_poll_interval_s: float,
        log_dir: str,
        backend_command: str | None = None,
    ) -> None:
        self.specs = specs
        self.model = model
        self.backend_args = backend_args
        self.backend_env = backend_env
        self.backend_scheduler = (backend_scheduler or "").strip()
        self.device_env_var = device_env_var
        self.health_timeout_s = health_timeout_s
        self.health_poll_interval_s = health_poll_interval_s
        self.log_dir = Path(log_dir)
        self.backend_command = backend_command
        self._processes: list[ManagedBackendProcess] = []

    def start_all(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting %d managed super_p95 backends in parallel. Logs: %s",
            len(self.specs),
            self.log_dir,
        )
        try:
            for spec in self.specs:
                self._processes.append(self._start_one(spec))
            self._wait_until_healthy()
            logger.info("All managed super_p95 backends are healthy.")
        except Exception:
            self.stop_all()
            raise

    def stop_all(self) -> None:
        for managed in reversed(self._processes):
            if managed.process.poll() is None:
                managed.process.terminate()
        deadline = time.time() + 20.0
        for managed in reversed(self._processes):
            if managed.process.poll() is None:
                timeout_s = max(deadline - time.time(), 0.0)
                with suppress(subprocess.TimeoutExpired):
                    managed.process.wait(timeout=timeout_s)
            if managed.process.poll() is None:
                managed.process.kill()
                with suppress(subprocess.TimeoutExpired):
                    managed.process.wait(timeout=5.0)
            with suppress(Exception):
                managed.log_file.close()
        self._processes.clear()

    def _start_one(self, spec: ManagedBackendSpec) -> ManagedBackendProcess:
        env = os.environ.copy()
        env[self.device_env_var] = spec.device_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.update(self.backend_env)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        # v0.18 backend scheduler consumes the hardware profile from env.
        # Keep managed launch aligned with v0.16 semantics without depending on
        # a serve CLI flag that does not exist in this branch.
        env["VLLM_OMNI_SUPER_P95_HARDWARE_PROFILE"] = spec.hardware_profile
        env.setdefault("MASTER_PORT", str(22000 + spec.port))
        env.setdefault("VLLM_OMNI_MASTER_PORT", str(22000 + spec.port))
        if self.backend_scheduler:
            env["VLLM_OMNI_DIFFUSION_SCHEDULER"] = self.backend_scheduler
        trace_log_dir = self.backend_env.get("VLLM_OMNI_TRACE_LOG_DIR")
        if trace_log_dir:
            env["VLLM_OMNI_TRACE_LOG_FILE"] = str(Path(trace_log_dir) / f"backend_{spec.port}.jsonl")
            env["VLLM_OMNI_TRACE_NODE"] = f"backend-{spec.port}"
        log_path = self.log_dir / f"backend_{spec.port}.log"
        log_file = open(log_path, "a", encoding="utf-8")
        if self.backend_command is None:
            cmd = [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "serve",
                self.model,
                "--port",
                str(spec.port),
                *self.backend_args,
            ]
        else:
            cmd = [
                self.backend_command,
                "serve",
                self.model,
                "--port",
                str(spec.port),
                *self.backend_args,
            ]
        logger.info(
            "Launching backend port=%s device_id=%s base_url=%s hardware_profile=%s scheduler=%s log=%s cmd=%s",
            spec.port,
            spec.device_id,
            spec.base_url,
            spec.hardware_profile,
            self.backend_scheduler or "<default>",
            log_path,
            shlex.join(cmd),
        )
        shell_cmd = f"{shlex.join(cmd)} >> {shlex.quote(str(log_path))} 2>&1"
        process = subprocess.Popen(
            ["/bin/bash", "-lc", shell_cmd],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return ManagedBackendProcess(spec=spec, process=process, log_file=log_file)

    def _wait_until_healthy(self, target_ports: set[int] | None = None) -> None:
        deadline = time.time() + self.health_timeout_s
        pending = {
            managed.spec.port: managed
            for managed in self._processes
            if target_ports is None or managed.spec.port in target_ports
        }
        while pending:
            failed = [managed for managed in pending.values() if managed.process.poll() is not None]
            if failed:
                details = ", ".join(
                    f"{managed.spec.port}(device={managed.spec.device_id}, log={managed.log_file.name})"
                    for managed in failed
                )
                raise RuntimeError(f"super_p95 backend exited before becoming healthy: {details}")

            ready_ports = [
                port
                for port, managed in pending.items()
                if self._log_indicates_ready(managed) or self._is_healthy(managed.spec.base_url)
            ]
            for port in ready_ports:
                managed = pending.pop(port, None)
                if managed is not None:
                    logger.info(
                        "Managed backend healthy: port=%s device=%s url=%s log=%s",
                        managed.spec.port,
                        managed.spec.device_id,
                        managed.spec.base_url,
                        managed.log_file.name,
                    )

            if not pending:
                return
            if time.time() >= deadline:
                details = ", ".join(
                    f"{managed.spec.port}(device={managed.spec.device_id}, log={managed.log_file.name})"
                    for managed in pending.values()
                )
                raise TimeoutError(f"Timed out waiting for super_p95 backends to become healthy: {details}")
            time.sleep(self.health_poll_interval_s)

    def _log_indicates_ready(self, managed: ManagedBackendProcess) -> bool:
        try:
            managed.log_file.flush()
            with open(managed.log_file.name, encoding="utf-8", errors="ignore") as f:
                tail = f.read()[-32768:]
        except OSError:
            return False
        return any(pattern in tail for pattern in self._READY_LOG_PATTERNS)

    @staticmethod
    def _is_healthy(base_url: str) -> bool:
        try:
            opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
            with opener.open(f"{base_url}/health", timeout=2.0) as response:
                return response.status == 200
        except (urllib_error.URLError, TimeoutError, ValueError):
            return False


class SuperP95Dispatcher:
    def __init__(
        self,
        backend_urls: list[str],
        *,
        backend_hardware_profiles: list[str] | None,
        quota_every: int,
        quota_amount: int,
        threshold_ratio: float,
        sacrificial_load_factor: float,
        request_timeout_s: float,
        long_request_ratio: float = 1.5,
        tail_routing_mode: str = "spread",
        tail_dispatch_mode: str = "immediate",
        normal_routing_policy: str = "assigned_load",
        central_pull_risk_beta: float = 0.5,
        service_time_estimator_name: str = "auto",
        trace_log_file: str | None = None,
        backend_launcher: ManagedBackendLauncher | None = None,
    ) -> None:
        if not 1 <= len(backend_urls) <= 8:
            raise ValueError("super_p95 dispatcher supports 1 to 8 backends")
        hardware_profiles = _parse_backend_hardware_profiles(backend_hardware_profiles, len(backend_urls))
        self.backends = [
            BackendState(
                name=f"backend-{idx}",
                base_url=url.rstrip("/"),
                hardware_profile=hardware_profiles[idx],
            )
            for idx, url in enumerate(backend_urls)
        ]
        self.quota_every = max(quota_every, 1)
        self.quota_amount = max(quota_amount, 0)
        self.threshold_ratio = threshold_ratio
        self.long_request_ratio = max(long_request_ratio, 1.0)
        self.sacrificial_load_factor = sacrificial_load_factor
        self.request_timeout_s = request_timeout_s
        self.trace_log_file = trace_log_file
        if tail_routing_mode not in {"spread", "pack", "sink"}:
            raise ValueError("tail_routing_mode must be one of: pack, sink, spread")
        self.tail_routing_mode = tail_routing_mode
        if tail_dispatch_mode not in _TAIL_DISPATCH_MODES:
            choices = ", ".join(sorted(_TAIL_DISPATCH_MODES))
            raise ValueError(f"tail_dispatch_mode must be one of: {choices}")
        self.tail_dispatch_mode = tail_dispatch_mode
        valid_normal_routing_policies = {"assigned_load", *_CENTRAL_PULL_POLICIES}
        if normal_routing_policy not in valid_normal_routing_policies:
            choices = ", ".join(sorted(valid_normal_routing_policies))
            raise ValueError(f"normal_routing_policy must be one of: {choices}")
        if not math.isfinite(central_pull_risk_beta) or central_pull_risk_beta < 0.0:
            raise ValueError("central_pull_risk_beta must be a finite non-negative number")
        self.normal_routing_policy = normal_routing_policy
        self.central_pull_risk_beta = central_pull_risk_beta
        self.service_time_estimator_name = service_time_estimator_name

        self._lock = asyncio.Lock()
        self.arrival_counter = 0
        self.credits = 0
        self.tail_admitted_count = 0
        self.global_min_service_s: float | None = None
        self.global_max_service_s = 0.0
        self.normal_dispatches = 0
        self.sacrificial_dispatches = 0
        self.quota_refills = 0
        self.quota_credits_added = 0
        self.quota_credits_consumed = 0
        self.central_pull_dispatches = 0
        self.central_queue_max_depth = 0
        self.central_wait_total_s = 0.0
        self.tail_gate_releases = 0
        self.tail_gate_wait_total_s = 0.0
        self.tail_gate_queue_max_depth = 0
        self._pending_normal_dispatches: list[PendingNormalDispatch] = []
        self._pending_tail_dispatches: list[PendingTailDispatch] = []
        self._client: httpx.AsyncClient | None = None
        self._backend_launcher = backend_launcher
        self._video_backend_by_id: dict[str, int] = {}
        self._video_decision_by_id: dict[str, DispatchDecision] = {}
        self._video_request_id_by_id: dict[str, str] = {}
        self._video_received_at_s_by_id: dict[str, float] = {}
        backend_batch2_flag = (
            backend_launcher.backend_env.get("SUPER_P95_QWEN_SMALL_BATCH2") if backend_launcher is not None else None
        )
        self._qwen_small_batch2_enabled = self._env_flag_enabled(
            os.environ.get("SUPER_P95_QWEN_SMALL_BATCH2")
        ) or self._env_flag_enabled(backend_batch2_flag)

    async def startup(self) -> None:
        if self._backend_launcher is not None:
            await asyncio.to_thread(self._backend_launcher.start_all)
        timeout = self.request_timeout_s if self.request_timeout_s > 0 else None
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def shutdown(self) -> None:
        async with self._lock:
            for pending in self._pending_normal_dispatches:
                if not pending.future.done():
                    pending.future.set_exception(RuntimeError("super_p95 dispatcher is shutting down"))
            self._pending_normal_dispatches.clear()
            for pending in self._pending_tail_dispatches:
                self._rollback_pending_tail_locked(pending)
                if not pending.future.done():
                    pending.future.set_exception(RuntimeError("super_p95 dispatcher is shutting down"))
            self._pending_tail_dispatches.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._backend_launcher is not None:
            await asyncio.to_thread(self._backend_launcher.stop_all)

    async def dispatch_json(self, path: str, body: dict[str, Any], incoming_headers: dict[str, str]) -> Response:
        request_id = str(body.get("request_id", ""))
        write_trace_event(
            self.trace_log_file,
            "dispatcher_arrive",
            node="dispatcher",
            request_id=request_id or None,
            path=path,
            **_trace_request_fields(path, body),
        )
        decision = await self._choose_backend(path, body)
        backend = self.backends[decision.backend_index]
        write_trace_event(
            self.trace_log_file,
            "dispatch",
            node="dispatcher",
            request_id=request_id or None,
            path=path,
            backend=backend.name,
            backend_url=backend.base_url,
            sacrificial=decision.is_sacrificial,
            estimated_service_s=decision.estimated_service_s,
            arrival_counter=decision.arrival_counter,
            credits_before=decision.credits_before,
            credits_after=decision.credits_after,
            quota_added=decision.quota_added,
            global_max_service_s=decision.global_max_service_s,
            central_wait_s=decision.central_wait_s,
            central_risk_score=decision.central_risk_score,
            central_risk_beta=decision.central_risk_beta,
            normal_routing_policy=self.normal_routing_policy,
            tail_routing_mode=self.tail_routing_mode,
            tail_dispatch_mode=self.tail_dispatch_mode,
            tail_gate_wait_s=(
                decision.central_wait_s
                if decision.is_sacrificial and self.tail_dispatch_mode == "protected_drain"
                else None
            ),
            queue_class="tail" if decision.is_sacrificial else "normal",
            service_time_estimator=self.service_time_estimator_name,
            **_trace_request_fields(path, body),
        )
        headers = self._build_forward_headers(incoming_headers, decision)
        assert self._client is not None

        start_time = time.perf_counter()
        try:
            response = await self._client.post(f"{backend.base_url}{path}", json=body, headers=headers)
        except Exception:
            elapsed_s = time.perf_counter() - start_time
            await self._mark_failed_response(decision, elapsed_s)
            raise

        elapsed_s = time.perf_counter() - start_time
        await self._apply_response_feedback(decision, response.headers, elapsed_s)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def dispatch_form(self, path: str, form: FormData, incoming_headers: dict[str, str]) -> Response:
        body = _form_to_estimation_dict(form)
        request_received_at_s = time.perf_counter()
        request_id = str(body.get("request_id", ""))
        write_trace_event(
            self.trace_log_file,
            "dispatcher_arrive",
            node="dispatcher",
            request_id=request_id or None,
            path=path,
            **_trace_request_fields(path, body),
        )
        decision = await self._choose_backend(path, body)
        backend = self.backends[decision.backend_index]
        write_trace_event(
            self.trace_log_file,
            "dispatch",
            node="dispatcher",
            request_id=request_id or None,
            path=path,
            backend=backend.name,
            backend_url=backend.base_url,
            sacrificial=decision.is_sacrificial,
            estimated_service_s=decision.estimated_service_s,
            arrival_counter=decision.arrival_counter,
            credits_before=decision.credits_before,
            credits_after=decision.credits_after,
            quota_added=decision.quota_added,
            global_max_service_s=decision.global_max_service_s,
            central_wait_s=decision.central_wait_s,
            central_risk_score=decision.central_risk_score,
            central_risk_beta=decision.central_risk_beta,
            normal_routing_policy=self.normal_routing_policy,
            tail_routing_mode=self.tail_routing_mode,
            tail_dispatch_mode=self.tail_dispatch_mode,
            tail_gate_wait_s=(
                decision.central_wait_s
                if decision.is_sacrificial and self.tail_dispatch_mode == "protected_drain"
                else None
            ),
            queue_class="tail" if decision.is_sacrificial else "normal",
            service_time_estimator=self.service_time_estimator_name,
            **_trace_request_fields(path, body),
        )
        headers = self._build_forward_headers(incoming_headers, decision)
        headers.pop("content-type", None)

        data: dict[str, str] = {}
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                payload = await value.read()
                files.append(
                    (
                        key,
                        (
                            value.filename or "upload.bin",
                            payload,
                            value.content_type or "application/octet-stream",
                        ),
                    )
                )
            else:
                data[key] = str(value)

        assert self._client is not None
        start_time = time.perf_counter()
        try:
            response = await self._client.post(
                f"{backend.base_url}{path}", data=data, files=files or None, headers=headers
            )
        except Exception as exc:
            elapsed_s = time.perf_counter() - start_time
            await self._mark_failed_response(decision, elapsed_s)
            write_trace_event(
                self.trace_log_file,
                "dispatcher_failed",
                node="dispatcher",
                request_id=request_id or None,
                backend=backend.name,
                queue_class="tail" if decision.is_sacrificial else "normal",
                estimated_service_s=decision.estimated_service_s,
                central_wait_s=decision.central_wait_s,
                dispatcher_e2e_s=time.perf_counter() - request_received_at_s,
                error=repr(exc),
            )
            raise

        elapsed_s = time.perf_counter() - start_time
        if path == "/v1/videos" and response.status_code < 400:
            remembered = self._remember_video_backend(
                path,
                response,
                decision,
                request_id=request_id,
                request_received_at_s=request_received_at_s,
            )
            if not remembered:
                await self._apply_response_feedback(decision, response.headers, elapsed_s)
                write_trace_event(
                    self.trace_log_file,
                    "dispatcher_failed",
                    node="dispatcher",
                    request_id=request_id or None,
                    backend=backend.name,
                    queue_class="tail" if decision.is_sacrificial else "normal",
                    estimated_service_s=decision.estimated_service_s,
                    central_wait_s=decision.central_wait_s,
                    dispatcher_e2e_s=time.perf_counter() - request_received_at_s,
                    error="Backend response did not contain a video job id.",
                )
        else:
            await self._apply_response_feedback(decision, response.headers, elapsed_s)
            if response.status_code >= 400:
                write_trace_event(
                    self.trace_log_file,
                    "dispatcher_failed",
                    node="dispatcher",
                    request_id=request_id or None,
                    backend=backend.name,
                    queue_class="tail" if decision.is_sacrificial else "normal",
                    estimated_service_s=decision.estimated_service_s,
                    central_wait_s=decision.central_wait_s,
                    dispatcher_e2e_s=time.perf_counter() - request_received_at_s,
                    status_code=response.status_code,
                    error=response.text,
                )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def proxy_get(self, path: str) -> Response:
        backend = self.backends[self._backend_index_for_proxy_path(path)]
        assert self._client is not None
        response = await self._client.get(f"{backend.base_url}{path}")
        await self._maybe_release_video_load_from_get(path, response)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def proxy_delete(self, path: str) -> Response:
        backend_index = self._backend_index_for_proxy_path(path)
        backend = self.backends[backend_index]
        assert self._client is not None
        response = await self._client.delete(f"{backend.base_url}{path}")
        if response.status_code < 400:
            video_id = self._video_id_from_proxy_path(path)
            if video_id is not None:
                await self._release_video_load(video_id)
                self._video_backend_by_id.pop(video_id, None)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    def _remember_video_backend(
        self,
        path: str,
        response: httpx.Response,
        decision: DispatchDecision,
        *,
        request_id: str,
        request_received_at_s: float,
    ) -> bool:
        if path != "/v1/videos" or response.status_code >= 400:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False
        video_id = payload.get("id") if isinstance(payload, dict) else None
        if isinstance(video_id, str) and video_id:
            self._video_backend_by_id[video_id] = decision.backend_index
            self._video_decision_by_id[video_id] = decision
            self._video_request_id_by_id[video_id] = request_id
            self._video_received_at_s_by_id[video_id] = request_received_at_s
            write_trace_event(
                self.trace_log_file,
                "dispatcher_job_accepted",
                node="dispatcher",
                request_id=request_id or None,
                video_id=video_id,
                backend=self.backends[decision.backend_index].name,
                queue_class="tail" if decision.is_sacrificial else "normal",
                estimated_service_s=decision.estimated_service_s,
                central_wait_s=decision.central_wait_s,
            )
            return True
        return False

    async def _maybe_release_video_load_from_get(self, path: str, response: httpx.Response) -> None:
        video_id = self._video_id_from_proxy_path(path)
        if video_id is None or "/" in path.removeprefix(f"/v1/videos/{video_id}"):
            return
        if response.status_code >= 400:
            return
        try:
            payload = response.json()
        except ValueError:
            return
        status = payload.get("status") if isinstance(payload, dict) else None
        if status in {"completed", "failed", "cancelled", "canceled"}:
            await self._release_video_load(
                video_id,
                status=str(status),
                inference_time_s=_parse_float(payload.get("inference_time_s")),
                error=payload.get("error"),
            )

    async def _release_video_load(
        self,
        video_id: str,
        *,
        status: str = "cancelled",
        inference_time_s: float | None = None,
        error: Any = None,
    ) -> None:
        decision = self._video_decision_by_id.pop(video_id, None)
        if decision is None:
            return
        request_id = self._video_request_id_by_id.pop(video_id, "")
        received_at_s = self._video_received_at_s_by_id.pop(video_id, None)
        async with self._lock:
            backend = self.backends[decision.backend_index]
            self._dec_inflight(backend, decision.is_sacrificial)
            self._fallback_remove_estimated_load(backend, decision)
            self._assign_waiting_normals_locked()
        terminal_event = "dispatcher_complete" if status == "completed" else "dispatcher_failed"
        write_trace_event(
            self.trace_log_file,
            terminal_event,
            node="dispatcher",
            request_id=request_id or None,
            video_id=video_id,
            backend=backend.name,
            status=status,
            queue_class="tail" if decision.is_sacrificial else "normal",
            estimated_service_s=decision.estimated_service_s,
            central_wait_s=decision.central_wait_s,
            backend_inference_time_s=inference_time_s,
            dispatcher_e2e_s=(time.perf_counter() - received_at_s if received_at_s is not None else None),
            error=error,
        )

    def _backend_index_for_proxy_path(self, path: str) -> int:
        video_id = self._video_id_from_proxy_path(path)
        if video_id is None:
            return 0
        return self._video_backend_by_id.get(video_id, 0)

    @staticmethod
    def _video_id_from_proxy_path(path: str) -> str | None:
        prefix = "/v1/videos/"
        if not path.startswith(prefix):
            return None
        suffix = path[len(prefix) :]
        video_id = suffix.split("/", 1)[0]
        return video_id or None

    async def health(self) -> JSONResponse:
        assert self._client is not None
        statuses: list[dict[str, Any]] = []
        overall_healthy = True
        for backend in self.backends:
            try:
                response = await self._client.get(f"{backend.base_url}/health")
                healthy = response.status_code == 200
                detail = response.text
            except Exception as exc:
                healthy = False
                detail = str(exc)
            overall_healthy = overall_healthy and healthy
            statuses.append({"backend": backend.name, "url": backend.base_url, "healthy": healthy, "detail": detail})
        async with self._lock:
            central_queue_depth = len(self._pending_normal_dispatches)
            central_queue_max_depth = self.central_queue_max_depth
            central_pull_dispatches = self.central_pull_dispatches
            tail_gate_queue_depth = len(self._pending_tail_dispatches)
            tail_gate_queue_max_depth = self.tail_gate_queue_max_depth
            tail_gate_releases = self.tail_gate_releases
            tail_gate_wait_total_s = self.tail_gate_wait_total_s
        return JSONResponse(
            status_code=200 if overall_healthy else 503,
            content={
                "status": "healthy" if overall_healthy else "degraded",
                "backends": statuses,
                "normal_routing_policy": self.normal_routing_policy,
                "central_pull_risk_beta": self.central_pull_risk_beta,
                "tail_routing_mode": self.tail_routing_mode,
                "tail_dispatch_mode": self.tail_dispatch_mode,
                "service_time_estimator": self.service_time_estimator_name,
                "request_trace_enabled": self.trace_log_file is not None,
                "trace_log_file": self.trace_log_file,
                "central_queue_depth": central_queue_depth,
                "central_queue_max_depth": central_queue_max_depth,
                "central_pull_dispatches": central_pull_dispatches,
                "tail_gate_queue_depth": tail_gate_queue_depth,
                "tail_gate_queue_max_depth": tail_gate_queue_max_depth,
                "tail_gate_releases": tail_gate_releases,
                "tail_gate_wait_total_s": tail_gate_wait_total_s,
            },
        )

    async def _choose_backend(self, path: str, body: dict[str, Any]) -> DispatchDecision:
        arrival_time_s = time.perf_counter()
        estimated_service_s_by_backend = [
            self._estimate_service_s(path, body, backend.hardware_profile) for backend in self.backends
        ]
        estimated_service_s = min(estimated_service_s_by_backend)
        pending_normal: PendingNormalDispatch | None = None
        pending_tail: PendingTailDispatch | None = None
        async with self._lock:
            self.arrival_counter += 1
            arrival_counter = self.arrival_counter
            credits_before = self.credits
            quota_added = 0
            if self.arrival_counter % self.quota_every == 0:
                self.credits += self.quota_amount
                quota_added = self.quota_amount
                self.quota_refills += 1
                self.quota_credits_added += self.quota_amount

            if self.global_min_service_s is None:
                self.global_min_service_s = estimated_service_s
            else:
                self.global_min_service_s = min(self.global_min_service_s, estimated_service_s)
            self.global_max_service_s = max(self.global_max_service_s, estimated_service_s)
            is_sacrificial = self.credits > 0 and self._is_sacrificial_candidate(estimated_service_s)
            if is_sacrificial:
                self.credits -= 1
                self.tail_admitted_count += 1
                self.sacrificial_dispatches += 1
                self.quota_credits_consumed += 1

            batch_key = None if is_sacrificial else self._batch_routing_key(path, body)
            if not is_sacrificial and self.normal_routing_policy in _CENTRAL_PULL_POLICIES:
                future = asyncio.get_running_loop().create_future()
                pending_normal = PendingNormalDispatch(
                    arrival_counter=arrival_counter,
                    arrival_time_s=arrival_time_s,
                    estimated_service_s_by_backend=tuple(estimated_service_s_by_backend),
                    credits_before=credits_before,
                    credits_after=self.credits,
                    quota_added=quota_added,
                    global_max_service_s=self.global_max_service_s,
                    batch_key=batch_key,
                    future=future,
                )
                self._pending_normal_dispatches.append(pending_normal)
                self.central_queue_max_depth = max(
                    self.central_queue_max_depth,
                    len(self._pending_normal_dispatches),
                )
                write_trace_event(
                    self.trace_log_file,
                    "central_enqueue",
                    node="dispatcher",
                    request_id=str(body.get("request_id", "")) or None,
                    arrival_counter=arrival_counter,
                    estimated_service_s=estimated_service_s,
                    estimated_service_s_by_backend=estimated_service_s_by_backend,
                    queue_depth=len(self._pending_normal_dispatches),
                    queue_class="normal",
                    normal_routing_policy=self.normal_routing_policy,
                    central_risk_beta=self._central_pull_beta(),
                    service_time_estimator=self.service_time_estimator_name,
                    **_trace_request_fields(path, body),
                )
                self._assign_waiting_normals_locked()
            elif is_sacrificial and self.tail_dispatch_mode == "protected_drain":
                decision = self._bind_request_locked(
                    estimated_service_s_by_backend=estimated_service_s_by_backend,
                    is_sacrificial=True,
                    arrival_counter=arrival_counter,
                    credits_before=credits_before,
                    credits_after=self.credits,
                    quota_added=quota_added,
                    global_max_service_s=self.global_max_service_s,
                    batch_key=None,
                    central_wait_s=0.0,
                )
                future = asyncio.get_running_loop().create_future()
                pending_tail = PendingTailDispatch(
                    request_id=str(body.get("request_id", "")),
                    arrival_time_s=arrival_time_s,
                    decision=decision,
                    future=future,
                )
                self._pending_tail_dispatches.append(pending_tail)
                self.tail_gate_queue_max_depth = max(
                    self.tail_gate_queue_max_depth,
                    len(self._pending_tail_dispatches),
                )
                write_trace_event(
                    self.trace_log_file,
                    "tail_gate_enqueue",
                    node="dispatcher",
                    request_id=pending_tail.request_id or None,
                    arrival_counter=arrival_counter,
                    backend=self.backends[decision.backend_index].name,
                    estimated_service_s=decision.estimated_service_s,
                    central_queue_depth=len(self._pending_normal_dispatches),
                    tail_gate_queue_depth=len(self._pending_tail_dispatches),
                    tail_dispatch_mode=self.tail_dispatch_mode,
                    **_trace_request_fields(path, body),
                )
                self._assign_waiting_normals_locked()
            else:
                if not is_sacrificial:
                    self.normal_dispatches += 1
                return self._bind_request_locked(
                    estimated_service_s_by_backend=estimated_service_s_by_backend,
                    is_sacrificial=is_sacrificial,
                    arrival_counter=arrival_counter,
                    credits_before=credits_before,
                    credits_after=self.credits,
                    quota_added=quota_added,
                    global_max_service_s=self.global_max_service_s,
                    batch_key=batch_key,
                    central_wait_s=0.0,
                )

        assert pending_normal is not None or pending_tail is not None
        try:
            if pending_normal is not None:
                return await pending_normal.future
            assert pending_tail is not None
            return await pending_tail.future
        except BaseException:
            async with self._lock:
                if pending_normal is not None:
                    with suppress(ValueError):
                        self._pending_normal_dispatches.remove(pending_normal)
                    if not pending_normal.future.done():
                        pending_normal.future.cancel()
                    self._assign_waiting_normals_locked()
                else:
                    assert pending_tail is not None
                    removed = False
                    with suppress(ValueError):
                        self._pending_tail_dispatches.remove(pending_tail)
                        removed = True
                    if removed:
                        self._rollback_pending_tail_locked(pending_tail)
                    if not pending_tail.future.done():
                        pending_tail.future.cancel()
                    self._assign_waiting_normals_locked()
            raise

    def _bind_request_locked(
        self,
        *,
        estimated_service_s_by_backend: list[float] | tuple[float, ...],
        is_sacrificial: bool,
        arrival_counter: int,
        credits_before: int,
        credits_after: int,
        quota_added: int,
        global_max_service_s: float,
        batch_key: tuple[Any, ...] | None,
        central_wait_s: float,
        central_risk_score: float | None = None,
        central_risk_beta: float | None = None,
        backend_index: int | None = None,
    ) -> DispatchDecision:
        if backend_index is None:
            backend_index = (
                self._select_tail_backend_index()
                if is_sacrificial
                else self._select_backend_index(batch_key)
            )
        backend = self.backends[backend_index]
        selected_estimated_service_s = estimated_service_s_by_backend[backend_index]
        if is_sacrificial:
            backend.sacrificial_load_s += selected_estimated_service_s
            backend.inflight_sacrificial_requests += 1
        else:
            backend.normal_load_s += selected_estimated_service_s
            backend.inflight_normal_requests += 1
            if batch_key is not None:
                backend.batchable_counts[batch_key] = backend.batchable_counts.get(batch_key, 0) + 1
        logger.info(
            "super_p95 dispatch arrival=%d backend=%s sacrificial=%s "
            "quota_added=%d credits_before=%d credits_after=%d "
            "estimated_service_s=%.4f global_max_service_s=%.4f "
            "central_wait_s=%.4f central_queue_depth=%d "
            "batch_key=%s backend_batchable_count=%d "
            "normal_dispatches=%d sacrificial_dispatches=%d",
            arrival_counter,
            backend.name,
            is_sacrificial,
            quota_added,
            credits_before,
            credits_after,
            selected_estimated_service_s,
            global_max_service_s,
            central_wait_s,
            len(self._pending_normal_dispatches),
            batch_key,
            backend.batchable_counts.get(batch_key, 0) if batch_key is not None else 0,
            self.normal_dispatches,
            self.sacrificial_dispatches,
        )
        return DispatchDecision(
            backend_index=backend_index,
            estimated_service_s=selected_estimated_service_s,
            is_sacrificial=is_sacrificial,
            arrival_counter=arrival_counter,
            credits_before=credits_before,
            credits_after=credits_after,
            quota_added=quota_added,
            global_max_service_s=global_max_service_s,
            central_wait_s=central_wait_s,
            central_risk_score=central_risk_score,
            central_risk_beta=central_risk_beta,
            batch_key=batch_key,
        )

    def _assign_waiting_normals_locked(self) -> None:
        while self._pending_normal_dispatches:
            available_backend_indices = [
                idx for idx, backend in enumerate(self.backends) if backend.inflight_normal_requests == 0
            ]
            if not available_backend_indices:
                break

            backend_index = min(
                available_backend_indices,
                key=lambda idx: (
                    self.backends[idx].inflight_sacrificial_requests > 0,
                    self.backends[idx].sacrificial_load_s,
                    self.backends[idx].latency_ema_s,
                    self.backends[idx].name,
                ),
            )
            now_s = time.perf_counter()
            risk_beta = self._central_pull_beta()

            def risk_score(item: PendingNormalDispatch) -> float:
                return now_s - item.arrival_time_s + risk_beta * item.estimated_service_s_by_backend[backend_index]

            selected = max(
                self._pending_normal_dispatches,
                key=lambda item: (
                    risk_score(item),
                    -item.arrival_counter,
                ),
            )
            self._pending_normal_dispatches.remove(selected)
            central_wait_s = max(now_s - selected.arrival_time_s, 0.0)
            self.central_pull_dispatches += 1
            self.central_wait_total_s += central_wait_s
            self.normal_dispatches += 1
            decision = self._bind_request_locked(
                estimated_service_s_by_backend=selected.estimated_service_s_by_backend,
                is_sacrificial=False,
                arrival_counter=selected.arrival_counter,
                credits_before=selected.credits_before,
                credits_after=selected.credits_after,
                quota_added=selected.quota_added,
                global_max_service_s=selected.global_max_service_s,
                batch_key=selected.batch_key,
                central_wait_s=central_wait_s,
                central_risk_score=risk_score(selected),
                central_risk_beta=risk_beta,
                backend_index=backend_index,
            )
            if not selected.future.done():
                selected.future.set_result(decision)
        self._release_waiting_tails_locked()

    def _release_waiting_tails_locked(self) -> None:
        if self.tail_dispatch_mode != "protected_drain":
            return
        if self._pending_normal_dispatches:
            return

        now_s = time.perf_counter()
        for pending in tuple(self._pending_tail_dispatches):
            backend = self.backends[pending.decision.backend_index]
            if backend.inflight_normal_requests > 0:
                continue
            self._pending_tail_dispatches.remove(pending)
            central_wait_s = max(now_s - pending.arrival_time_s, 0.0)
            decision = replace(
                pending.decision,
                central_wait_s=central_wait_s,
            )
            self.tail_gate_releases += 1
            self.tail_gate_wait_total_s += central_wait_s
            write_trace_event(
                self.trace_log_file,
                "tail_gate_release",
                node="dispatcher",
                request_id=pending.request_id or None,
                arrival_counter=decision.arrival_counter,
                backend=backend.name,
                estimated_service_s=decision.estimated_service_s,
                central_wait_s=central_wait_s,
                central_queue_depth=len(self._pending_normal_dispatches),
                tail_gate_queue_depth=len(self._pending_tail_dispatches),
                tail_dispatch_mode=self.tail_dispatch_mode,
            )
            if not pending.future.done():
                pending.future.set_result(decision)

    def _rollback_pending_tail_locked(self, pending: PendingTailDispatch) -> None:
        backend = self.backends[pending.decision.backend_index]
        self._dec_inflight(backend, is_sacrificial=True)
        self._fallback_remove_estimated_load(backend, pending.decision)

    def _central_pull_beta(self) -> float:
        if self.normal_routing_policy == "central_pull_max_risk":
            return 1.0
        return self.central_pull_risk_beta

    def _select_backend_index(self, batch_key: tuple[Any, ...] | None) -> int:
        if batch_key is not None:
            odd_candidates = [
                idx for idx, backend in enumerate(self.backends) if backend.batchable_counts.get(batch_key, 0) % 2 == 1
            ]
            if odd_candidates:
                return min(
                    odd_candidates,
                    key=lambda idx: self.backends[idx].score_tuple(self.sacrificial_load_factor),
                )
        return min(
            range(len(self.backends)),
            key=lambda idx: self.backends[idx].score_tuple(self.sacrificial_load_factor),
        )

    def _select_tail_backend_index(self) -> int:
        if self.tail_routing_mode == "pack":
            tail_loaded = [
                idx
                for idx, backend in enumerate(self.backends)
                if backend.inflight_sacrificial_requests > 0
                or backend.sacrificial_load_s > 0.0
            ]
            if tail_loaded:
                return min(
                    tail_loaded,
                    key=lambda idx: (
                        -self.backends[idx].sacrificial_load_s,
                        self.backends[idx].normal_load_s,
                        self.backends[idx].latency_ema_s,
                        self.backends[idx].name,
                    ),
                )
            return min(
                range(len(self.backends)),
                key=lambda idx: (
                    self.backends[idx].normal_load_s
                    + self.backends[idx].sacrificial_load_s,
                    self.backends[idx].latency_ema_s,
                    self.backends[idx].name,
                ),
            )
        if self.tail_routing_mode == "sink":
            return min(
                range(len(self.backends)),
                key=lambda idx: (
                    -self.backends[idx].normal_load_s,
                    -self.backends[idx].sacrificial_load_s,
                    self.backends[idx].latency_ema_s,
                    self.backends[idx].name,
                ),
            )
        return self._select_backend_index(batch_key=None)

    async def _apply_response_feedback(
        self,
        decision: DispatchDecision,
        headers: httpx.Headers,
        elapsed_s: float,
    ) -> None:
        async with self._lock:
            backend = self.backends[decision.backend_index]
            self._update_latency_ema(backend, elapsed_s)
            self._dec_inflight(backend, decision.is_sacrificial)
            self._dec_batchable_count(backend, decision.batch_key)
            authoritative = parse_super_p95_load_headers(headers)
            if authoritative is not None:
                backend.normal_load_s = authoritative.normal_load_s
                backend.sacrificial_load_s = authoritative.sacrificial_load_s
            else:
                self._fallback_remove_estimated_load(backend, decision)
            self._assign_waiting_normals_locked()

    async def _mark_failed_response(self, decision: DispatchDecision, elapsed_s: float) -> None:
        async with self._lock:
            backend = self.backends[decision.backend_index]
            self._update_latency_ema(backend, elapsed_s)
            self._dec_inflight(backend, decision.is_sacrificial)
            self._dec_batchable_count(backend, decision.batch_key)
            self._fallback_remove_estimated_load(backend, decision)
            self._assign_waiting_normals_locked()

    def _is_sacrificial_candidate(self, estimated_service_s: float) -> bool:
        if self.global_max_service_s <= 0.0:
            return False
        if estimated_service_s < self.threshold_ratio * self.global_max_service_s:
            return False
        if self.global_min_service_s is None or self.global_min_service_s <= 0.0:
            return False
        return estimated_service_s >= self.long_request_ratio * self.global_min_service_s

    @staticmethod
    def _update_latency_ema(backend: BackendState, elapsed_s: float) -> None:
        if elapsed_s <= 0.0:
            return
        if backend.latency_ema_s <= 0.0:
            backend.latency_ema_s = elapsed_s
        else:
            backend.latency_ema_s = 0.9 * backend.latency_ema_s + 0.1 * elapsed_s

    @staticmethod
    def _dec_inflight(backend: BackendState, is_sacrificial: bool) -> None:
        if is_sacrificial:
            backend.inflight_sacrificial_requests = max(backend.inflight_sacrificial_requests - 1, 0)
        else:
            backend.inflight_normal_requests = max(backend.inflight_normal_requests - 1, 0)

    @staticmethod
    def _dec_batchable_count(backend: BackendState, batch_key: tuple[Any, ...] | None) -> None:
        if batch_key is None:
            return
        count = backend.batchable_counts.get(batch_key, 0)
        if count <= 1:
            backend.batchable_counts.pop(batch_key, None)
        else:
            backend.batchable_counts[batch_key] = count - 1

    @staticmethod
    def _fallback_remove_estimated_load(backend: BackendState, decision: DispatchDecision) -> None:
        if decision.is_sacrificial:
            backend.sacrificial_load_s = max(backend.sacrificial_load_s - decision.estimated_service_s, 0.0)
        else:
            backend.normal_load_s = max(backend.normal_load_s - decision.estimated_service_s, 0.0)

    @staticmethod
    def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
        blocked = {"content-length", "transfer-encoding", "connection", "content-encoding"}
        return {key: value for key, value in headers.items() if key.lower() not in blocked}

    @staticmethod
    def _build_forward_headers(incoming_headers: dict[str, str], decision: DispatchDecision) -> dict[str, str]:
        blocked = {"host", "content-length"}
        headers = {key: value for key, value in incoming_headers.items() if key.lower() not in blocked}
        headers[HEADER_SUPER_P95_ESTIMATED_SERVICE_S] = f"{decision.estimated_service_s:.6f}"
        if decision.is_sacrificial:
            headers[HEADER_SUPER_P95_SACRIFICIAL] = "1"
        else:
            headers.pop(HEADER_SUPER_P95_SACRIFICIAL, None)
        return headers

    @staticmethod
    def _env_flag_enabled(value: str | None) -> bool:
        return str(value or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _batch_routing_key(self, path: str, body: dict[str, Any]) -> tuple[Any, ...] | None:
        if not self._qwen_small_batch2_enabled:
            return None
        model = str(body.get("model") or "").lower()
        if model and "qwen" not in model:
            return None
        if path == "/v1/chat/completions":
            params = body.get("extra_body") or {}
            width = params.get("width")
            height = params.get("height")
            steps = params.get("num_inference_steps")
            frames = params.get("num_frames")
            negative_prompt = params.get("negative_prompt")
            guidance_scale = params.get("guidance_scale")
            true_cfg_scale = params.get("true_cfg_scale")
            cfg_text_scale = params.get("cfg_text_scale")
            cfg_img_scale = params.get("cfg_img_scale")
            seed = params.get("seed")
            generator_device = params.get("generator_device")
            num_outputs = params.get("num_outputs_per_prompt", 1)
        elif path == "/v1/images/generations":
            width, height = _parse_size(body.get("size"))
            steps = body.get("num_inference_steps")
            frames = body.get("num_frames")
            negative_prompt = body.get("negative_prompt")
            guidance_scale = body.get("guidance_scale")
            true_cfg_scale = body.get("true_cfg_scale")
            cfg_text_scale = body.get("cfg_text_scale")
            cfg_img_scale = body.get("cfg_img_scale")
            seed = body.get("seed")
            generator_device = body.get("generator_device")
            num_outputs = body.get("n", body.get("num_outputs_per_prompt", 1))
        else:
            return None

        try:
            width_i = int(width)
            height_i = int(height)
        except (TypeError, ValueError):
            return None
        try:
            frames_i = int(frames) if frames is not None else 1
        except (TypeError, ValueError):
            return None
        try:
            num_outputs_i = int(num_outputs or 1)
        except (TypeError, ValueError):
            return None
        if frames_i > 1 or num_outputs_i != 1:
            return None
        if width_i * height_i > _QWEN_SMALL_IMAGE_MAX_PIXELS:
            return None
        return (
            width_i,
            height_i,
            steps,
            guidance_scale,
            true_cfg_scale,
            cfg_text_scale,
            cfg_img_scale,
            negative_prompt,
            seed,
            generator_device,
        )

    @staticmethod
    def _estimate_service_s(path: str, body: dict[str, Any], hardware_profile: str) -> float:
        if path == "/v1/chat/completions":
            extra_body = body.get("extra_body") or {}
            return _estimate_service_s_from_values(
                width=extra_body.get("width"),
                height=extra_body.get("height"),
                num_inference_steps=extra_body.get("num_inference_steps"),
                num_frames=extra_body.get("num_frames"),
                hardware_profile=hardware_profile,
            )
        if path == "/v1/images/generations":
            width, height = _parse_size(body.get("size"))
            return _estimate_service_s_from_values(
                width=width,
                height=height,
                num_inference_steps=body.get("num_inference_steps"),
                num_frames=body.get("num_frames"),
                hardware_profile=hardware_profile,
            )
        if path == "/v1/videos":
            width, height = _parse_size(body.get("size"))
            num_frames = body.get("num_frames")
            if num_frames is None:
                seconds = _parse_int(body.get("seconds"))
                fps = _parse_int(body.get("fps"))
                if seconds is not None:
                    num_frames = seconds * (fps if fps is not None else 24)
            return _estimate_service_s_from_values(
                width=body.get("width", width),
                height=body.get("height", height),
                num_inference_steps=body.get("num_inference_steps"),
                num_frames=num_frames,
                hardware_profile=hardware_profile,
            )
        raise HTTPException(status_code=400, detail=f"Unsupported super_p95 path: {path}")


def _parse_size(size: Any) -> tuple[int | None, int | None]:
    if not isinstance(size, str) or "x" not in size.lower():
        return None, None
    width_str, height_str = size.lower().split("x", 1)
    try:
        return int(width_str), int(height_str)
    except ValueError:
        return None, None


def _estimate_service_s_from_values(
    *,
    width: Any,
    height: Any,
    num_inference_steps: Any,
    num_frames: Any,
    hardware_profile: str,
) -> float:
    sampling_params = OmniDiffusionSamplingParams(
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        num_inference_steps=int(num_inference_steps) if num_inference_steps is not None else 25,
        num_frames=int(num_frames) if num_frames is not None else 1,
    )
    request = OmniDiffusionRequest(prompts=["super_p95"], sampling_params=sampling_params, request_ids=["super_p95"])
    return estimate_service_time_s(request, hardware_profile=hardware_profile)


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_request_fields(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Return scheduling-visible input fields for a trace event."""

    width = _parse_int(body.get("width"))
    height = _parse_int(body.get("height"))
    size_width, size_height = _parse_size(body.get("size"))
    width = width or size_width
    height = height or size_height
    steps = _parse_int(body.get("num_inference_steps"))
    fps = _parse_int(body.get("fps"))
    frames = _parse_int(body.get("num_frames"))
    if frames is None:
        seconds = _parse_int(body.get("seconds"))
        if seconds is not None:
            frames = seconds * (fps or 24)

    fields: dict[str, Any] = {
        "width": width,
        "height": height,
        "num_frames": frames,
        "num_inference_steps": steps,
        "fps": fps,
    }
    if path == "/v1/videos":
        workload_key = (width, height, steps, frames)
        workload_class = {
            (854, 480, 3, 80): "short",
            (854, 480, 4, 120): "medium",
            (1280, 720, 6, 80): "long",
        }.get(workload_key, "custom")
        fields["workload_class"] = workload_class
    return fields


def _form_to_estimation_dict(form: FormData) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            continue
        body[key] = value
    return body


def _parse_backend_hardware_profiles(raw: list[str] | None | str, num_backends: int) -> list[str]:
    if raw is None:
        return [normalize_super_p95_hardware_profile(None)] * num_backends
    if isinstance(raw, str):
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        parsed = [item.strip() for item in raw if item.strip()]
    if len(parsed) == 1 and num_backends > 1:
        parsed = parsed * num_backends
    if len(parsed) != num_backends:
        raise ValueError(f"--backend-hardware-profiles must provide exactly {num_backends} entries")
    return [normalize_super_p95_hardware_profile(item) for item in parsed]


def build_app(dispatcher: SuperP95Dispatcher) -> FastAPI:
    app = FastAPI(title="super_p95 dispatcher")

    @app.on_event("startup")
    async def _startup() -> None:
        await dispatcher.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await dispatcher.shutdown()

    @app.get("/health")
    async def health() -> JSONResponse:
        return await dispatcher.health()

    @app.get("/v1/models")
    async def models() -> Response:
        return await dispatcher.proxy_get("/v1/models")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        return await dispatcher.dispatch_json("/v1/chat/completions", body, dict(request.headers))

    @app.post("/v1/images/generations")
    async def image_generations(request: Request) -> Response:
        body = await request.json()
        return await dispatcher.dispatch_json("/v1/images/generations", body, dict(request.headers))

    @app.post("/v1/videos")
    async def videos(request: Request) -> Response:
        form = await request.form()
        return await dispatcher.dispatch_form("/v1/videos", form, dict(request.headers))

    @app.get("/v1/videos/{video_id}")
    async def retrieve_video(video_id: str) -> Response:
        return await dispatcher.proxy_get(f"/v1/videos/{video_id}")

    @app.delete("/v1/videos/{video_id}")
    async def delete_video(video_id: str) -> Response:
        return await dispatcher.proxy_delete(f"/v1/videos/{video_id}")

    @app.get("/v1/videos/{video_id}/content")
    async def retrieve_video_content(video_id: str) -> Response:
        return await dispatcher.proxy_get(f"/v1/videos/{video_id}/content")

    return app


def build_arg_parser(
    *,
    description: str = "super_p95 dispatcher for diffusion servers",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--backend-url",
        dest="backend_urls",
        action="append",
        help="Backend base URL. Repeat 1 to 8 times, e.g. http://127.0.0.1:8091",
    )
    parser.add_argument("--num-servers", type=int, help="Number of managed backend servers to launch (1-8).")
    parser.add_argument(
        "--device-ids",
        help="Comma-separated device ids for managed launch, e.g. 0,1,2,3. Defaults to 0..num_servers-1.",
    )
    parser.add_argument("--model", help="Model name for managed backend launch, e.g. Qwen/Qwen-Image.")
    parser.add_argument("--backend-host", default="127.0.0.1", help="Host used for managed backend URLs.")
    parser.add_argument("--backend-start-port", type=int, default=8091, help="Starting port for managed backends.")
    parser.add_argument(
        "--backend-hardware-profiles",
        help="Comma-separated hardware profiles for backends, e.g. 910B2,910B3.",
    )
    parser.add_argument(
        "--backend-args",
        action="append",
        default=["--omni"],
        help="Extra CLI args appended to each managed backend command.",
    )
    parser.add_argument(
        "--backend-env",
        action="append",
        default=[],
        help="Extra environment for managed backends in KEY=VALUE form. Repeat as needed.",
    )
    parser.add_argument(
        "--backend-scheduler",
        default="",
        help=(
            "Optional managed-backend diffusion scheduler. "
            "Examples: step_baseline, super_p95_step. "
            "Sets VLLM_OMNI_DIFFUSION_SCHEDULER for each managed backend."
        ),
    )
    parser.add_argument(
        "--device-env-var",
        default="ASCEND_RT_VISIBLE_DEVICES",
        help="Environment variable used to pin a managed backend to a single device.",
    )
    parser.add_argument(
        "--backend-health-timeout-s",
        type=float,
        default=900.0,
        help="How long dispatcher startup waits for managed backends to pass /health.",
    )
    parser.add_argument(
        "--backend-health-poll-interval-s",
        type=float,
        default=5.0,
        help="Polling interval while waiting for managed backends to pass /health.",
    )
    parser.add_argument(
        "--backend-log-dir",
        default="/tmp/super_p95_backends",
        help="Directory for managed backend stdout/stderr logs.",
    )
    parser.add_argument("--quota-every", type=int, default=20)
    parser.add_argument("--quota-amount", type=int, default=1)
    parser.add_argument("--threshold-ratio", type=float, default=0.8)
    parser.add_argument(
        "--long-request-ratio",
        type=float,
        default=1.5,
        help=(
            "Minimum estimated-service ratio against the shortest observed request before a tail budget "
            "can be spent. This keeps budget unused when the observed workload has no clearly long request."
        ),
    )
    parser.add_argument("--sacrificial-load-factor", type=float, default=0.1)
    parser.add_argument(
        "--tail-routing-mode",
        choices=("spread", "pack", "sink"),
        default="spread",
        help=(
            "Tail backend placement. spread preserves the current least-load "
            "behavior; pack reuses an existing Tail backend; sink chooses the "
            "largest Normal backlog."
        ),
    )
    parser.add_argument(
        "--tail-dispatch-mode",
        choices=tuple(sorted(_TAIL_DISPATCH_MODES)),
        default="immediate",
        help=(
            "Tail start gate. immediate forwards Tail requests as soon as they "
            "are classified; protected_drain reserves the selected backend but "
            "waits until the central Normal queue and that backend's Normal "
            "work have drained."
        ),
    )
    parser.add_argument(
        "--normal-routing-policy",
        choices=("assigned_load", "central_pull_max_risk", "central_pull_cost_damped_risk"),
        default="assigned_load",
        help=(
            "Normal request binding policy. Central-pull policies keep unstarted "
            "Normal requests in an online global queue. max_risk selects the "
            "maximum (wait + estimate); cost_damped_risk uses "
            "(wait + beta * estimate)."
        ),
    )
    parser.add_argument(
        "--central-pull-risk-beta",
        type=float,
        default=0.5,
        help=(
            "Estimated-service weight for central_pull_cost_damped_risk. "
            "The Wan2.2 P95 candidate uses 0.5; central_pull_max_risk always uses 1.0."
        ),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=0.0,
        help="Dispatcher-to-backend request timeout in seconds. Set to 0 to disable timeout entirely.",
    )
    parser.add_argument(
        "--trace-log-dir",
        type=str,
        default=None,
        help="Optional directory for dispatcher/backend JSONL trace logs.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _parse_backend_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --backend-env value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --backend-env value {item!r}; key cannot be empty")
        env[key] = value
    return env


def _parse_usp_degree(backend_args: list[str]) -> int:
    for idx, token in enumerate(backend_args):
        if token == "--usp" and idx + 1 < len(backend_args):
            with suppress(ValueError):
                return max(int(backend_args[idx + 1]), 1)
    return 1


def _parse_device_ids(device_ids: str | None, num_servers: int, usp_degree: int) -> list[str]:
    if device_ids is None:
        if num_servers == 1 and usp_degree > 1:
            return [",".join(str(idx) for idx in range(usp_degree))]
        return [str(idx) for idx in range(num_servers)]

    raw = device_ids.strip()
    if num_servers == 1:
        return [raw]

    if ";" in raw:
        parsed = [item.strip() for item in raw.split(";") if item.strip()]
        if len(parsed) != num_servers:
            raise ValueError(f"--device-ids must provide exactly {num_servers} ';'-separated groups")
        return parsed

    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parsed) != num_servers:
        raise ValueError(
            f"--device-ids must provide exactly {num_servers} entries. "
            "For multi-card backends, use ';'-separated groups, e.g. '0,1,2,3;4,5,6,7'."
        )
    return parsed


def _build_managed_specs(
    host: str,
    start_port: int,
    device_ids: list[str],
    hardware_profiles: list[str],
) -> list[ManagedBackendSpec]:
    return [
        ManagedBackendSpec(
            device_id=device_id,
            port=start_port + idx,
            base_url=f"http://{host}:{start_port + idx}",
            hardware_profile=hardware_profiles[idx],
        )
        for idx, device_id in enumerate(device_ids)
    ]


def build_dispatcher_from_args(
    args: argparse.Namespace,
    *,
    dispatcher_cls: type[SuperP95Dispatcher] = SuperP95Dispatcher,
    dispatcher_kwargs: dict[str, Any] | None = None,
) -> SuperP95Dispatcher:
    manual_urls = [url.rstrip("/") for url in (args.backend_urls or [])]
    use_managed = args.num_servers is not None or args.model is not None or args.device_ids is not None
    backend_hardware_profiles_arg = getattr(args, "backend_hardware_profiles", None)

    if manual_urls and use_managed:
        raise ValueError("Choose either --backend-url or managed launch args, not both")

    backend_launcher = None
    if manual_urls:
        backend_urls = manual_urls
    else:
        if args.num_servers is None or args.model is None:
            raise ValueError("Managed launch requires both --num-servers and --model")
        if not 1 <= args.num_servers <= 8:
            raise ValueError("--num-servers must be between 1 and 8")
        backend_args: list[str] = []
        for raw in args.backend_args:
            backend_args.extend(shlex.split(raw))
        # Preserve the first occurrence of each arg while avoiding duplicate
        # flags such as "--omni --omni" from parser defaults plus explicit CLI.
        deduped_backend_args: list[str] = []
        seen_backend_args: set[str] = set()
        for token in backend_args:
            if token in seen_backend_args:
                continue
            seen_backend_args.add(token)
            deduped_backend_args.append(token)
        backend_args = deduped_backend_args
        usp_degree = _parse_usp_degree(backend_args)
        device_ids = _parse_device_ids(args.device_ids, args.num_servers, usp_degree)
        hardware_profiles = _parse_backend_hardware_profiles(backend_hardware_profiles_arg, args.num_servers)
        specs = _build_managed_specs(args.backend_host, args.backend_start_port, device_ids, hardware_profiles)
        backend_urls = [spec.base_url for spec in specs]
        backend_env = _parse_backend_env(args.backend_env)
        if args.trace_log_dir:
            backend_env["VLLM_OMNI_TRACE_LOG_DIR"] = args.trace_log_dir
        backend_launcher = ManagedBackendLauncher(
            specs=specs,
            model=args.model,
            backend_args=backend_args,
            backend_env=backend_env,
            backend_scheduler=args.backend_scheduler,
            device_env_var=args.device_env_var,
            health_timeout_s=args.backend_health_timeout_s,
            health_poll_interval_s=args.backend_health_poll_interval_s,
            log_dir=args.backend_log_dir,
        )

    return dispatcher_cls(
        backend_urls=backend_urls,
        backend_hardware_profiles=_parse_backend_hardware_profiles(backend_hardware_profiles_arg, len(backend_urls)),
        quota_every=args.quota_every,
        quota_amount=args.quota_amount,
        threshold_ratio=args.threshold_ratio,
        long_request_ratio=getattr(args, "long_request_ratio", 1.5),
        sacrificial_load_factor=args.sacrificial_load_factor,
        request_timeout_s=args.request_timeout_s,
        tail_routing_mode=getattr(args, "tail_routing_mode", "spread"),
        tail_dispatch_mode=getattr(args, "tail_dispatch_mode", "immediate"),
        normal_routing_policy=getattr(
            args,
            "normal_routing_policy",
            "assigned_load",
        ),
        central_pull_risk_beta=getattr(args, "central_pull_risk_beta", 0.5),
        trace_log_file=(str(Path(args.trace_log_dir) / "dispatcher.jsonl") if args.trace_log_dir else None),
        backend_launcher=backend_launcher,
        **(dispatcher_kwargs or {}),
    )


def main() -> None:
    args = parse_args()
    dispatcher = build_dispatcher_from_args(args)
    uvicorn.run(build_app(dispatcher), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
