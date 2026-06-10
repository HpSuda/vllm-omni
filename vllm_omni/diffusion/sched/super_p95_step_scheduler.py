# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.super_p95 import (
    SuperP95LoadSnapshot,
    estimate_service_time_s,
    get_super_p95_request_metadata,
    normalize_super_p95_hardware_profile,
)

logger = init_logger(__name__)

# Requests up to this many pixels qualify for batch2 pairing when
# SUPER_P95_QWEN_SMALL_BATCH2 is enabled (any shape, e.g. 512x768).
_SMALL_IMAGE_MAX_PIXELS = 768 * 768


@dataclass(order=True)
class _QueuedRequest:
    sort_key: tuple[int, ...] = field(init=False, repr=False)
    arrival_seq: int
    arrival_time_s: float
    sched_req_id: str = field(compare=False)
    estimated_service_s: float = field(compare=False)
    total_steps: int = field(compare=False, default=1)
    completed_steps: int = field(compare=False, default=0)
    is_sacrificial: bool = field(compare=False, default=False)

    def __post_init__(self) -> None:
        self.refresh_sort_key()

    @property
    def remaining_steps(self) -> int:
        return max(self.total_steps - self.completed_steps, 0)

    @property
    def remaining_service_s(self) -> float:
        return self.estimated_service_s * self.remaining_steps / max(self.total_steps, 1)

    def refresh_sort_key(self) -> None:
        if self.is_sacrificial:
            # Sacrificial requests are the explicit cost sink: newer sacrificial
            # arrivals are allowed to jump ahead of older sacrificial requests.
            self.sort_key = (-self.arrival_seq,)
        else:
            # Normal requests preserve FIFO to avoid inflating worst-case delay
            # by reordering within the main queue.
            self.sort_key = (self.arrival_seq,)


class SuperP95StepScheduler(_BaseScheduler):
    """Stepwise-native super_p95 scheduler for v0.18 diffusion execution.

    This scheduler assumes engine drives one denoise step per schedule cycle.
    Preemption is therefore modeled by choosing a different request on the
    next scheduling cycle rather than by async interruption within a request.
    """

    def __init__(self) -> None:
        super().__init__()
        self._normal_pending: list[_QueuedRequest] = []
        self._sacrificial_pending: list[_QueuedRequest] = []
        self._queue_metadata: dict[str, _QueuedRequest] = {}
        self._arrival_seq: int = 0
        self._current_time_s: float = 0.0
        self._hardware_profile = normalize_super_p95_hardware_profile(
            os.environ.get("VLLM_OMNI_SUPER_P95_HARDWARE_PROFILE")
        )
        self._num_added_normal = 0
        self._num_added_sacrificial = 0
        self._num_scheduled_normal = 0
        self._num_scheduled_sacrificial = 0
        self._num_preemptions = 0

    def has_requests(self) -> bool:
        return bool(self._running or self._normal_pending or self._sacrificial_pending)

    def get_load_snapshot(self) -> SuperP95LoadSnapshot:
        normal = 0.0
        sacrificial = 0.0
        for queued in self._queue_metadata.values():
            state = self._request_states.get(queued.sched_req_id)
            if state is None or state.is_finished():
                continue
            if queued.is_sacrificial:
                sacrificial += queued.remaining_service_s
            else:
                normal += queued.remaining_service_s
        return SuperP95LoadSnapshot(normal_load_s=normal, sacrificial_load_s=sacrificial)

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        state = DiffusionRequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._register_request_ids(request.request_ids, sched_req_id)

        sacrificial, estimated_service_s = get_super_p95_request_metadata(request.sampling_params.extra_args)
        if estimated_service_s is None:
            estimated_service_s = estimate_service_time_s(request, hardware_profile=self._hardware_profile)

        queued = _QueuedRequest(
            arrival_seq=self._arrival_seq,
            arrival_time_s=self._current_time_s,
            sched_req_id=sched_req_id,
            estimated_service_s=estimated_service_s,
            total_steps=max(int(request.sampling_params.num_inference_steps or 1), 1),
            is_sacrificial=sacrificial,
        )
        self._arrival_seq += 1
        self._queue_metadata[sched_req_id] = queued
        self._push_pending(queued)
        if sacrificial:
            self._num_added_sacrificial += 1
        else:
            self._num_added_normal += 1
        logger.info(
            "super_p95 scheduler add req=%s sacrificial=%s estimated_service_s=%.4f "
            "total_steps=%d added_normal=%d added_sacrificial=%d",
            sched_req_id,
            sacrificial,
            estimated_service_s,
            queued.total_steps,
            self._num_added_normal,
            self._num_added_sacrificial,
        )
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        active_id = self._running[0] if self._running else None
        active_queued = self._queue_metadata.get(active_id) if active_id is not None else None
        candidate = self.peek_next_request()

        if active_queued is not None and candidate is not None and self.request_outranks(candidate, active_queued):
            self._running.clear()
            active_state = self._request_states.get(active_queued.sched_req_id)
            if active_state is not None and not active_state.is_finished():
                active_state.status = DiffusionRequestStatus.PREEMPTED
                self._push_pending(active_queued)
                self._num_preemptions += 1
                logger.info(
                    "super_p95 scheduler preempt running=%s sacrificial=%s candidate=%s candidate_sacrificial=%s "
                    "preemptions=%d normal_pending=%d sacrificial_pending=%d",
                    active_queued.sched_req_id,
                    active_queued.is_sacrificial,
                    candidate.sched_req_id,
                    candidate.is_sacrificial,
                    self._num_preemptions,
                    len(self._normal_pending),
                    len(self._sacrificial_pending),
                )
            active_queued = None

        if active_queued is not None:
            scheduled_cached_req_ids.extend(self._running)
        else:
            queued = self._pop_next_queued()
            if queued is not None:
                selected = [queued]
                partner = self._pop_batch_partner(queued)
                if partner is not None:
                    selected.append(partner)
                    logger.info(
                        "super_p95 scheduler batch2 select reqs=%s,%s width_height_steps=%s window=%d",
                        queued.sched_req_id,
                        partner.sched_req_id,
                        self._batch_key(queued),
                        self._batch_search_window(),
                    )
                self._running = [item.sched_req_id for item in selected]
                for item in selected:
                    state = self._request_states.get(item.sched_req_id)
                    if state is None:
                        continue
                    was_new_request = state.status == DiffusionRequestStatus.WAITING and item.completed_steps == 0
                    state.status = DiffusionRequestStatus.RUNNING
                    if item.is_sacrificial:
                        self._num_scheduled_sacrificial += 1
                    else:
                        self._num_scheduled_normal += 1
                    logger.info(
                        "super_p95 scheduler select req=%s sacrificial=%s completed_steps=%d total_steps=%d "
                        "scheduled_normal=%d scheduled_sacrificial=%d normal_pending=%d sacrificial_pending=%d",
                        item.sched_req_id,
                        item.is_sacrificial,
                        item.completed_steps,
                        item.total_steps,
                        self._num_scheduled_normal,
                        self._num_scheduled_sacrificial,
                        len(self._normal_pending),
                        len(self._sacrificial_pending),
                    )
                    if was_new_request:
                        scheduled_new_reqs.append(NewRequestData.from_state(state))
                    else:
                        scheduled_cached_req_ids.append(item.sched_req_id)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._normal_pending) + len(self._sacrificial_pending),
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: DiffusionOutput) -> set[str]:
        if not sched_output.scheduled_req_ids:
            return set()
        sched_req_id = sched_output.scheduled_req_ids[0]
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
        statuses = {
            sched_req_id: (
                DiffusionRequestStatus.FINISHED_ERROR if output.error else DiffusionRequestStatus.FINISHED_COMPLETED
            )
        }
        errors = {sched_req_id: output.error}
        queued = self._queue_metadata.get(sched_req_id)
        if queued is not None:
            self._current_time_s += queued.remaining_service_s
            queued.completed_steps = queued.total_steps
        return self._finish_requests(statuses, errors)

    def update_from_runner_output(self, sched_output: DiffusionSchedulerOutput, runner_output) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()
        runner_req_ids = getattr(runner_output, "req_ids", None) or [runner_output.req_id]
        if list(runner_req_ids) != scheduled_req_ids:
            raise ValueError(f"runner output {runner_req_ids!r} does not match scheduled {scheduled_req_ids!r}")

        outputs = getattr(runner_output, "outputs", None) or {}
        statuses: dict[str, DiffusionRequestStatus] = {}
        errors: dict[str, str | None] = {}

        for sched_req_id in scheduled_req_ids:
            queued = self._queue_metadata.get(sched_req_id)
            state = self._request_states.get(sched_req_id)
            if queued is None or state is None or state.is_finished():
                continue

            prev_completed_steps = queued.completed_steps
            completed_steps = prev_completed_steps
            if runner_output.step_index is not None:
                completed_steps = min(max(int(runner_output.step_index), prev_completed_steps), queued.total_steps)

            delta_steps = max(completed_steps - prev_completed_steps, 0)
            self._current_time_s += queued.estimated_service_s * delta_steps / max(queued.total_steps, 1)
            queued.completed_steps = completed_steps
            queued.refresh_sort_key()

            if runner_output.finished:
                if sched_req_id in self._running:
                    self._running.remove(sched_req_id)
                result = outputs.get(sched_req_id, runner_output.result if len(scheduled_req_ids) == 1 else None)
                statuses[sched_req_id] = (
                    DiffusionRequestStatus.FINISHED_ERROR
                    if result is not None and result.error
                    else DiffusionRequestStatus.FINISHED_COMPLETED
                )
                errors[sched_req_id] = None if result is None else result.error
                queued.completed_steps = queued.total_steps
            else:
                state.status = DiffusionRequestStatus.RUNNING

        if runner_output.finished:
            return self._finish_requests(statuses, errors)
        return set()

    def abort_request(self, sched_req_id: str) -> bool:
        if self.get_request_state(sched_req_id) is None:
            return False
        self.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        return True

    def preempt_request(self, sched_req_id: str) -> bool:
        if sched_req_id not in self._request_states:
            return False
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
            state = self._request_states[sched_req_id]
            state.status = DiffusionRequestStatus.PREEMPTED
            queued = self._queue_metadata.get(sched_req_id)
            if queued is not None:
                self._push_pending(queued)
            return True
        return False

    def _reset_scheduler_state(self) -> None:
        self._normal_pending.clear()
        self._sacrificial_pending.clear()
        self._queue_metadata.clear()
        self._arrival_seq = 0
        self._current_time_s = 0.0
        self._num_added_normal = 0
        self._num_added_sacrificial = 0
        self._num_scheduled_normal = 0
        self._num_scheduled_sacrificial = 0
        self._num_preemptions = 0

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._queue_metadata.pop(sched_req_id, None)

    def _push_pending(self, queued: _QueuedRequest) -> None:
        queued.refresh_sort_key()
        heap = self._sacrificial_pending if queued.is_sacrificial else self._normal_pending
        heapq.heappush(heap, queued)

    def _pop_next_queued(self) -> _QueuedRequest | None:
        for heap in (self._normal_pending, self._sacrificial_pending):
            while heap:
                queued = heapq.heappop(heap)
                state = self._request_states.get(queued.sched_req_id)
                if state is None or state.is_finished():
                    continue
                return queued
        return None

    def peek_next_request(self) -> _QueuedRequest | None:
        for heap in (self._normal_pending, self._sacrificial_pending):
            while heap:
                queued = heap[0]
                state = self._request_states.get(queued.sched_req_id)
                if state is None or state.is_finished():
                    heapq.heappop(heap)
                    continue
                return queued
        return None

    def _batch2_enabled(self) -> bool:
        return os.environ.get("SUPER_P95_QWEN_SMALL_BATCH2", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _batch_search_window(self) -> int:
        raw = os.environ.get("SUPER_P95_IMAGE_BATCH_SEARCH_WINDOW", "32")
        try:
            return max(int(raw), 2)
        except ValueError:
            return 32

    def _batch_min_pending(self) -> int:
        raw = os.environ.get("SUPER_P95_QWEN_SMALL_BATCH_MIN_PENDING", "1")
        try:
            return max(int(raw), 1)
        except ValueError:
            return 1

    def _batch_key(self, queued: _QueuedRequest) -> tuple | None:
        if not self._batch2_enabled():
            return None
        if queued.is_sacrificial or queued.completed_steps != 0:
            return None
        model_name = " ".join(
            str(getattr(self.od_config, attr, "") or "")
            for attr in ("model", "model_name", "model_path", "model_class_name")
        ).lower()
        if model_name and "qwen" not in model_name:
            return None
        state = self._request_states.get(queued.sched_req_id)
        if state is None or state.status != DiffusionRequestStatus.WAITING or state.is_finished():
            return None
        req = state.req
        if len(req.prompts) != 1 or len(req.request_ids) > 1:
            return None
        prompt = req.prompts[0]
        if isinstance(prompt, dict) and prompt.get("multi_modal_data"):
            return None
        sampling = req.sampling_params
        width = getattr(sampling, "width", None)
        height = getattr(sampling, "height", None)
        if width is None or height is None:
            return None
        # Batch partners must still match exactly on (width, height) via the
        # returned key; this only gates eligibility by total pixel count.
        if int(width) * int(height) > _SMALL_IMAGE_MAX_PIXELS:
            return None
        if int(getattr(sampling, "num_outputs_per_prompt", 1) or 1) != 1:
            return None
        extra_args = getattr(sampling, "extra_args", {}) or {}
        return (
            width,
            height,
            getattr(sampling, "num_inference_steps", None),
            getattr(sampling, "guidance_scale", None),
            getattr(sampling, "true_cfg_scale", None),
            extra_args.get("cfg_text_scale"),
            extra_args.get("cfg_img_scale"),
            prompt.get("negative_prompt") if isinstance(prompt, dict) else None,
            getattr(sampling, "seed", None),
            getattr(sampling, "generator_device", None),
        )

    def _pop_batch_partner(self, first: _QueuedRequest) -> _QueuedRequest | None:
        first_key = self._batch_key(first)
        if first_key is None:
            return None
        heap = self._normal_pending
        if len(heap) < self._batch_min_pending():
            return None
        window = self._batch_search_window()
        candidates = heapq.nsmallest(min(len(heap), window), heap)
        for candidate in candidates:
            if candidate.sched_req_id == first.sched_req_id:
                continue
            if self._batch_key(candidate) != first_key:
                continue
            heap.remove(candidate)
            heapq.heapify(heap)
            return candidate
        return None

    @staticmethod
    def request_outranks(candidate: _QueuedRequest, incumbent: _QueuedRequest) -> bool:
        return not candidate.is_sacrificial and incumbent.is_sacrificial
