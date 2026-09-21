# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-level scheduling policy on the native diffusion step lifecycle.

The dispatcher assigns Tail membership and supplies calibrated service estimates.
Normal requests retain FIFO order; pending Tail requests use newest-first order.
Only Normal-over-Tail preemption is allowed, at a completed denoise-step boundary.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    SchedulerRequestState,
)
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_TRACE_REQUEST_ID,
    SuperP95LoadSnapshot,
    estimate_service_time_s,
    get_super_p95_request_metadata,
    normalize_super_p95_hardware_profile,
)
from vllm_omni.trace_logging import write_trace_event


def small_image_batch2_enabled() -> bool:
    return os.environ.get("SUPER_P95_QWEN_SMALL_BATCH2", "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class _RequestMetadata:
    arrival_seq: int
    is_tail: bool
    estimated_service_s: float


class SuperP95StepScheduler(StepScheduler):
    """Dense-mode policy retaining native request identity, progress and cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self._metadata: dict[str, _RequestMetadata] = {}
        self._arrival_seq = 0
        self._wave_batch_ids: set[str] = set()
        self._trace_log_file = os.environ.get("VLLM_OMNI_TRACE_LOG_FILE")
        self._trace_node = os.environ.get("VLLM_OMNI_TRACE_NODE", "backend")
        self._hardware_profile = normalize_super_p95_hardware_profile(
            os.environ.get("VLLM_OMNI_SUPER_P95_HARDWARE_PROFILE")
        )

    @staticmethod
    def validate_kv_mode(od_config) -> None:
        # FIFO admission is intentional. Native paged-KV admission may defer a
        # head request until remote blocks become available; supporting that
        # needs a separate policy for bypass/fairness and paused row capacity.
        mode = getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        if mode != DiffusionKVCacheMode.DENSE_LEGACY or getattr(od_config, "kv_transfer_config", None) is not None:
            raise ValueError(
                "super_p95_step supports diffusion_kv_mode='dense_legacy' only, without native KV transfer; "
                "paged-KV admission/preemption is not supported by this policy."
            )

    def initialize(self, od_config, **kwargs: Any) -> None:
        self.validate_kv_mode(od_config)
        super().initialize(od_config, **kwargs)

    def _reset_scheduler_state(self) -> None:
        super()._reset_scheduler_state()
        self._metadata.clear()
        self._arrival_seq = 0
        self._wave_batch_ids.clear()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        is_tail, estimate = get_super_p95_request_metadata(request.sampling_params.extra_args)
        if estimate is None:
            estimate = estimate_service_time_s(request, hardware_profile=self._hardware_profile)
        request_id = super().add_request(request)
        self._metadata[request_id] = _RequestMetadata(self._arrival_seq, is_tail, estimate)
        self._arrival_seq += 1
        self._trace("scheduler_enqueue", request_id)
        return request_id

    def _sort_waiting(self) -> None:
        def key(request_id: str) -> tuple[bool, int]:
            meta = self._metadata[request_id]
            return meta.is_tail, -meta.arrival_seq if meta.is_tail else meta.arrival_seq

        self._waiting = deque(sorted(self._waiting, key=key))

    def schedule(self) -> DiffusionSchedulerOutput:
        normal_waiting = any(not self._metadata[request_id].is_tail for request_id in self._waiting)
        if normal_waiting:
            for request_id in tuple(self._running):
                if self._metadata[request_id].is_tail and self.preempt_request(request_id):
                    self._trace("scheduler_preempt", request_id, reason="normal_outranks_tail")

        self._sort_waiting()
        self._wave_batch_ids.clear()
        if not self._running and self._waiting:
            first_id = self._waiting[0]
            self._wave_batch_ids.add(first_id)
            partner_id = self._find_batch_partner(first_id)
            if partner_id is not None:
                self._wave_batch_ids.add(partner_id)
                self._waiting.remove(partner_id)
                self._waiting.insert(1, partner_id)

        previous_running = set(self._running)
        output = super().schedule()
        new_ids = {request.request_id for request in output.scheduled_new_reqs}
        for request_id in output.scheduled_request_ids:
            if request_id not in previous_running:
                self._trace(
                    "scheduler_select",
                    request_id,
                    selection_kind="first" if request_id in new_ids else "resume",
                    was_new_request=request_id in new_ids,
                )
        return output

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        # Never inject work into an already-running wave. The optional pair is
        # chosen together before admission; native key/KV checks still apply.
        return state.request_id in self._wave_batch_ids and super()._can_schedule_waiting(state)

    def _batch_key(self, request_id: str) -> tuple[Any, ...] | None:
        if not small_image_batch2_enabled() or self.max_num_running_reqs < 2:
            return None
        state = self._request_states[request_id]
        if self._metadata[request_id].is_tail or state.status != DiffusionRequestStatus.WAITING:
            return None
        model_name = " ".join(
            str(getattr(self.od_config, attr, "") or "")
            for attr in ("model", "model_name", "model_path", "model_class_name")
        ).lower()
        if "qwen" not in model_name:
            return None
        req, sampling = state.req, state.req.sampling_params
        if isinstance(req.prompt, dict) and req.prompt.get("multi_modal_data"):
            return None
        width, height = sampling.width, sampling.height
        if not width or not height or int(width) * int(height) > 768 * 768:
            return None
        if sampling.num_outputs_per_prompt != 1 or self._request_progress[request_id].current_step != 0:
            return None
        extra = sampling.extra_args or {}
        return (
            state.sampling_params_key,
            self._request_progress[request_id].total_steps,
            extra.get("cfg_text_scale"),
            extra.get("cfg_img_scale"),
            req.prompt.get("negative_prompt") if isinstance(req.prompt, dict) else None,
            sampling.seed,
            sampling.generator_device,
        )

    def _find_batch_partner(self, first_id: str) -> str | None:
        first_key = self._batch_key(first_id)
        if first_key is None:
            return None
        normal_ids = [request_id for request_id in self._waiting if not self._metadata[request_id].is_tail]
        if len(normal_ids) - 1 < _positive_env("SUPER_P95_QWEN_SMALL_BATCH_MIN_PENDING", 1):
            return None
        window = _positive_env("SUPER_P95_IMAGE_BATCH_SEARCH_WINDOW", 32)
        for request_id in normal_ids[1 : window + 1]:
            if self._batch_key(request_id) == first_key:
                return request_id
        return None

    def get_load_snapshot(self) -> SuperP95LoadSnapshot:
        normal, tail = 0.0, 0.0
        for request_id, meta in self._metadata.items():
            state = self._request_states.get(request_id)
            progress = self._request_progress.get(request_id)
            if state is None or state.is_finished() or progress is None:
                continue
            remaining = max(progress.total_steps - progress.current_step, 0)
            estimate = meta.estimated_service_s * remaining / max(progress.total_steps, 1)
            if meta.is_tail:
                tail += estimate
            else:
                normal += estimate
        return SuperP95LoadSnapshot(normal_load_s=normal, sacrificial_load_s=tail)

    def _finish_requests(self, statuses, errors=None) -> set[str]:
        finished = super()._finish_requests(statuses, errors)
        for request_id in finished:
            status = statuses[request_id]
            self._trace(
                "scheduler_complete" if status == DiffusionRequestStatus.FINISHED_COMPLETED else "scheduler_failed",
                request_id,
                status=status.name,
                error=None if errors is None else errors.get(request_id),
            )
        return finished

    def _pop_extra_request_state(self, request_id: str) -> None:
        super()._pop_extra_request_state(request_id)
        self._metadata.pop(request_id, None)
        self._wave_batch_ids.discard(request_id)

    def _trace(self, event: str, request_id: str, **fields: Any) -> None:
        meta = self._metadata.get(request_id)
        if meta is None:
            return
        progress = self._request_progress.get(request_id)
        state = self._request_states.get(request_id)
        extra = (state.req.sampling_params.extra_args or {}) if state is not None else {}
        write_trace_event(
            self._trace_log_file,
            event,
            node=self._trace_node,
            request_id=extra.get(EXTRA_ARG_SUPER_P95_TRACE_REQUEST_ID) or request_id,
            scheduler_request_id=request_id,
            queue_class="tail" if meta.is_tail else "normal",
            sacrificial=meta.is_tail,
            estimated_service_s=meta.estimated_service_s,
            completed_steps=progress.current_step if progress is not None else None,
            total_steps=progress.total_steps if progress is not None else None,
            normal_pending_depth=sum(not self._metadata[rid].is_tail for rid in self._waiting),
            tail_pending_depth=sum(self._metadata[rid].is_tail for rid in self._waiting),
            running_depth=len(self._running),
            **fields,
        )
