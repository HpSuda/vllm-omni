# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.diffusion.simulator.models import (
    BackendState,
    BackendView,
    ExperimentConfig,
    Phase,
    Priority,
    RequestStatus,
    SimRequest,
)
from benchmarks.diffusion.simulator.policies import (
    Classifier,
    LocalScheduler,
    Router,
    build_classifier,
    build_router,
    build_scheduler,
)
from benchmarks.diffusion.simulator.service_models import build_service_timing_model


class _EventKind(str, Enum):
    ARRIVAL = "arrival"
    PHASE_COMPLETE = "phase_complete"
    SWITCH_COMPLETE = "switch_complete"


@dataclass(order=True, frozen=True)
class _Event:
    time_s: float
    sequence: int
    kind: _EventKind = field(compare=False)
    request_id: str = field(compare=False)
    backend_name: str | None = field(compare=False, default=None)
    phase: Phase | None = field(compare=False, default=None)


@dataclass(frozen=True)
class SimulationResult:
    seed: int
    request_rate: float
    metrics: dict[str, Any]
    requests: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]

    def to_dict(self, *, include_requests: bool = True, include_events: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seed": self.seed,
            "request_rate": self.request_rate,
            "metrics": self.metrics,
        }
        if include_requests:
            result["requests"] = list(self.requests)
        if include_events:
            result["events"] = list(self.events)
        return result


def percentile(values: list[float], percentile_value: float) -> float:
    """NumPy-compatible default linear percentile (Hyndman-Fan type 7)."""

    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _mean_one_lognormal(rng: random.Random, sigma: float) -> float:
    if sigma == 0.0:
        return 1.0
    return math.exp(rng.gauss(-0.5 * sigma * sigma, sigma))


class Simulator:
    """Single-run, online, CPU-only discrete-event simulation engine."""

    def __init__(self, config: ExperimentConfig, *, seed: int | None = None, collect_events: bool = False) -> None:
        self.config = config
        self.seed = config.simulation.seed if seed is None else seed
        self.collect_events = collect_events
        self.classifier: Classifier = build_classifier(config.policy.classifier)
        self.router: Router = build_router(config.policy.router)
        self.scheduler: LocalScheduler = build_scheduler(config.policy.scheduler)
        self.backends = {backend.name: BackendState(config=backend) for backend in config.topology.backends}
        self.service_timing_model = build_service_timing_model(config)
        self.requests, self.request_rate = self._generate_requests()
        self._event_queue: list[_Event] = []
        self._event_sequence = 0
        self._trace_events: list[dict[str, Any]] = []
        self._now_s = 0.0
        self._completed = 0
        self._preemptions = 0
        self._steals = 0
        self._global_normal_waiting_ids: list[str] = []
        self._central_pull_time_s: dict[str, float] = {}
        self._protected_pulls = 0
        self._central_normal_queue_max_depth = 0
        self._switch_cost_total_s = 0.0
        self._steal_cost_total_s = 0.0
        self._protected_pull_cost_total_s = 0.0

    def _generate_requests(self) -> tuple[dict[str, SimRequest], float]:
        workload = self.config.workload
        profile_rng = random.Random(self.seed)
        arrival_rng = random.Random(self.seed)
        actual_rng = random.Random(self.seed + 1_000_003)
        estimate_rng = random.Random(self.seed + 2_000_003)

        request_types = list(workload.request_types)
        sampled_types = profile_rng.choices(
            request_types,
            weights=[request_type.weight for request_type in request_types],
            k=workload.num_requests,
        )
        service = self.config.service
        base_profiles = {
            id(request_type): self.service_timing_model.profile(request_type) for request_type in request_types
        }
        weighted_mean_service_s = sum(
            request_type.weight * base_profiles[id(request_type)].total_s * service.actual_service_scale
            for request_type in request_types
        ) / sum(request_type.weight for request_type in request_types)
        if workload.request_rate is not None:
            request_rate = workload.request_rate
        else:
            assert workload.utilization is not None
            aggregate_speed = sum(backend.speed for backend in self.config.topology.backends)
            request_rate = workload.utilization * aggregate_speed / weighted_mean_service_s

        arrivals = [0.0]
        for _ in range(1, workload.num_requests):
            arrivals.append(arrivals[-1] + arrival_rng.expovariate(request_rate))

        requests: dict[str, SimRequest] = {}
        for index, (request_type, arrival_time_s) in enumerate(zip(sampled_types, arrivals)):
            base_profile = base_profiles[id(request_type)]
            jitter_multiplier = _mean_one_lognormal(
                actual_rng,
                service.actual_jitter_sigma,
            )
            actual_multiplier = service.actual_service_scale * jitter_multiplier
            actual_profile = base_profile.scaled(actual_multiplier)
            nominal_estimate_s = (
                request_type.estimated_service_s
                if request_type.estimated_service_s is not None
                else request_type.nominal_service_s
            )
            if nominal_estimate_s is None:
                nominal_estimate_s = base_profile.total_s
            estimated_total_s = nominal_estimate_s * _mean_one_lognormal(estimate_rng, service.estimate_error_sigma)
            diagnostics = dict(base_profile.diagnostics)
            diagnostics["actual_service_multiplier"] = actual_multiplier
            request_id = f"request-{index:05d}"
            requests[request_id] = SimRequest(
                request_id=request_id,
                arrival_seq=index,
                arrival_time_s=arrival_time_s,
                request_type=request_type,
                actual_encode_s=actual_profile.encode_s,
                actual_step_s=actual_profile.denoise_s / request_type.num_inference_steps,
                actual_decode_s=actual_profile.decode_s,
                actual_text_encode_s=actual_profile.text_encode_s,
                actual_latent_prepare_s=actual_profile.latent_prepare_s,
                actual_denoise_compute_s=actual_profile.denoise_compute_s,
                actual_denoise_overhead_s=actual_profile.denoise_overhead_s,
                actual_usp_communication_s=actual_profile.usp_communication_s,
                actual_hsdp_communication_s=actual_profile.hsdp_communication_s,
                actual_vae_decode_s=actual_profile.vae_decode_s,
                actual_postprocess_s=actual_profile.postprocess_s,
                estimated_encode_s=(estimated_total_s * base_profile.estimate_encode_fraction),
                estimated_step_s=(
                    estimated_total_s * base_profile.estimate_denoise_fraction / request_type.num_inference_steps
                ),
                estimated_decode_s=(estimated_total_s * base_profile.estimate_decode_fraction),
                service_diagnostics=diagnostics,
            )
        return requests, request_rate

    def _push_event(
        self,
        *,
        time_s: float,
        kind: _EventKind,
        request_id: str,
        backend_name: str | None = None,
        phase: Phase | None = None,
    ) -> None:
        if time_s < self._now_s:
            raise RuntimeError("cannot schedule an event in the past")
        heapq.heappush(
            self._event_queue,
            _Event(
                time_s=time_s,
                sequence=self._event_sequence,
                kind=kind,
                request_id=request_id,
                backend_name=backend_name,
                phase=phase,
            ),
        )
        self._event_sequence += 1

    def _trace(
        self, event: str, *, request: SimRequest | None = None, backend: BackendState | None = None, **details: Any
    ) -> None:
        if not self.collect_events:
            return
        record: dict[str, Any] = {"time_s": self._now_s, "event": event}
        if request is not None:
            record["request_id"] = request.request_id
        if backend is not None:
            record["backend"] = backend.name
        record.update(details)
        self._trace_events.append(record)

    def _load_for_backend(self, backend: BackendState) -> tuple[float, float]:
        if self.router.load_view == "assigned":
            return backend.assigned_normal_load_s, backend.assigned_sacrificial_load_s

        normal_load_s = 0.0
        sacrificial_load_s = 0.0
        for request in self.requests.values():
            if request.backend_name != backend.name or request.status == RequestStatus.COMPLETED:
                continue
            remaining = request.remaining_estimated_s(backend.config.speed)
            if request.priority == Priority.SACRIFICIAL:
                sacrificial_load_s += remaining
            else:
                normal_load_s += remaining
        return normal_load_s, sacrificial_load_s

    def _backend_view(self, backend: BackendState) -> BackendView:
        normal_load_s, sacrificial_load_s = self._load_for_backend(backend)
        return BackendView(
            name=backend.name,
            speed=backend.config.speed,
            normal_load_s=normal_load_s,
            sacrificial_load_s=sacrificial_load_s,
            inflight_normal=backend.inflight_normal,
            inflight_sacrificial=backend.inflight_sacrificial,
            latency_ema_s=backend.latency_ema_s,
            running_request_id=backend.running_id,
            pending_requests=len(backend.pending_ids),
        )

    def _maybe_pull_protected(
        self,
        target: BackendState,
        incumbent: SimRequest | None,
    ) -> tuple[str | None, float]:
        """Bind one never-started Normal from the online global queue."""

        if not getattr(self.scheduler, "global_protected_pull", False):
            return None, 0.0
        if incumbent is not None:
            if incumbent.priority == Priority.NORMAL:
                return None, 0.0
            if not getattr(self.scheduler, "preempt_normal_over_sacrificial", False):
                return None, 0.0

        candidate_views = tuple(
            self.requests[request_id].to_view(target.config.speed) for request_id in self._global_normal_waiting_ids
        )
        choose_protected_pull = getattr(self.scheduler, "choose_protected_pull", None)
        if not candidate_views or choose_protected_pull is None:
            return None, 0.0
        selected_id = choose_protected_pull(now_s=self._now_s, pending=candidate_views)
        if selected_id is None:
            return None, 0.0
        if selected_id not in self._global_normal_waiting_ids:
            raise RuntimeError(f"global protected pull selected ineligible request {selected_id!r}")
        request = self.requests[selected_id]
        if (
            request.priority != Priority.NORMAL
            or request.status != RequestStatus.WAITING
            or request.first_start_time_s is not None
            or request.backend_name is not None
        ):
            raise RuntimeError(f"global protected pull found invalid request state for {selected_id!r}")

        queue_depth_before = len(self._global_normal_waiting_ids)
        self._global_normal_waiting_ids.remove(selected_id)
        request.backend_name = target.name
        target.pending_ids.append(selected_id)
        target_estimate_s = request.estimated_total_on_backend_s(target.config.speed)
        target.assigned_normal_load_s += target_estimate_s
        target.inflight_normal += 1
        self._central_pull_time_s[selected_id] = self._now_s
        self._protected_pulls += 1
        cost_s = float(getattr(self.scheduler, "protected_pull_cost_s", 0.0))
        self._trace(
            "protected_pull",
            request=request,
            backend=target,
            cost_s=cost_s,
            protected_pull_order=str(getattr(self.scheduler, "protected_pull_order", "fifo")),
            online_risk_s=(self._now_s - request.arrival_time_s + request.remaining_estimated_s(target.config.speed)),
            target_estimate_s=target_estimate_s,
            queue_depth_before=queue_depth_before,
            queue_depth_after=len(self._global_normal_waiting_ids),
        )
        return selected_id, cost_s

    def _schedule_idle_backends_for_protected_pull(self) -> None:
        """Offer global work to idle backends in a stable fastest-first order."""

        for backend in sorted(
            self.backends.values(),
            key=lambda item: (-item.config.speed, item.name),
        ):
            if backend.running_id is None:
                self._schedule_backend(backend, incumbent=None)

    def _maybe_steal_protected(
        self,
        target: BackendState,
        incumbent: SimRequest | None,
    ) -> str | None:
        if not getattr(self.scheduler, "global_work_stealing", False):
            return None
        if any(self.requests[request_id].priority == Priority.NORMAL for request_id in target.pending_ids):
            return None
        if incumbent is not None:
            if incumbent.priority != Priority.SACRIFICIAL:
                return None
            if not getattr(self.scheduler, "preempt_normal_over_sacrificial", False):
                return None

        steal_hysteresis_s = float(getattr(self.scheduler, "steal_hysteresis_s", 0.0))
        steal_cost_s = float(getattr(self.scheduler, "steal_cost_s", 0.0))
        threshold_s = steal_hysteresis_s + steal_cost_s
        target_normal_load_s, _ = self._load_for_backend(target)
        candidates: list[tuple[SimRequest, BackendState, float, float, float, float]] = []
        for donor in self.backends.values():
            if donor.name == target.name:
                continue
            donor_normal_load_s, _ = self._load_for_backend(donor)
            for request_id in donor.pending_ids:
                request = self.requests[request_id]
                if (
                    request.priority != Priority.NORMAL
                    or request.status != RequestStatus.WAITING
                    or request.first_start_time_s is not None
                ):
                    continue
                target_completion_s = target_normal_load_s + request.estimated_total_on_backend_s(target.config.speed)
                benefit_s = donor_normal_load_s - target_completion_s
                if benefit_s <= threshold_s:
                    continue
                risk_s = self._now_s - request.arrival_time_s + request.remaining_estimated_s(donor.config.speed)
                candidates.append(
                    (
                        request,
                        donor,
                        donor_normal_load_s,
                        target_completion_s,
                        benefit_s,
                        risk_s,
                    )
                )

        if not candidates:
            return None
        steal_order = str(getattr(self.scheduler, "steal_order", "max_risk"))
        if steal_order == "oldest":
            selected = min(
                candidates,
                key=lambda candidate: (
                    candidate[0].arrival_seq,
                    candidate[0].request_id,
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda candidate: (
                    -candidate[5],
                    candidate[0].arrival_seq,
                    candidate[0].request_id,
                ),
            )
        (
            request,
            donor,
            donor_completion_s,
            target_completion_s,
            benefit_s,
            risk_s,
        ) = selected

        donor.pending_ids.remove(request.request_id)
        target.pending_ids.append(request.request_id)
        donor_estimate_s = request.estimated_total_on_backend_s(donor.config.speed)
        target_estimate_s = request.estimated_total_on_backend_s(target.config.speed)
        donor.assigned_normal_load_s = max(
            donor.assigned_normal_load_s - donor_estimate_s,
            0.0,
        )
        target.assigned_normal_load_s += target_estimate_s
        if donor.inflight_normal <= 0:
            raise RuntimeError(f"backend {donor.name} has no normal inflight work to steal")
        donor.inflight_normal -= 1
        target.inflight_normal += 1
        request.backend_name = target.name
        self._steals += 1
        self._trace(
            "steal",
            request=request,
            backend=target,
            source_backend=donor.name,
            donor_completion_s=donor_completion_s,
            target_completion_s=target_completion_s,
            benefit_s=benefit_s,
            risk_s=risk_s,
            hysteresis_s=steal_hysteresis_s,
            cost_s=steal_cost_s,
            steal_order=steal_order,
        )
        return request.request_id

    def _handle_arrival(self, event: _Event) -> None:
        request = self.requests[event.request_id]
        if request.status != RequestStatus.NOT_ARRIVED:
            raise RuntimeError(f"duplicate arrival for {request.request_id}")
        request.status = RequestStatus.WAITING
        minimum_estimate_s = min(
            request.estimated_total_on_backend_s(backend.config.speed) for backend in self.backends.values()
        )
        backend_views = tuple(self._backend_view(backend) for backend in self.backends.values())
        classification = self.classifier.classify(
            request.to_view(),
            minimum_estimate_s,
            now_s=self._now_s,
            backends=backend_views,
        )
        request.priority = classification.priority

        if getattr(self.scheduler, "global_protected_pull", False) and request.priority == Priority.NORMAL:
            self._global_normal_waiting_ids.append(request.request_id)
            self._central_normal_queue_max_depth = max(
                self._central_normal_queue_max_depth,
                len(self._global_normal_waiting_ids),
            )
            self._trace(
                "arrival",
                request=request,
                request_type=request.request_type.name,
                priority=request.priority.value,
                estimated_service_s=minimum_estimate_s,
                classifier=classification.details,
                router={"mode": "deferred_global_normal"},
                binding="deferred",
                central_queue_depth=len(self._global_normal_waiting_ids),
            )
            self._schedule_idle_backends_for_protected_pull()
            return

        routing = self.router.choose(request.to_view(), backend_views)
        backend = self.backends.get(routing.backend_name)
        if backend is None:
            raise RuntimeError(f"router selected unknown backend {routing.backend_name!r}")

        request.backend_name = backend.name
        backend.pending_ids.append(request.request_id)
        selected_estimate_s = request.estimated_total_on_backend_s(backend.config.speed)
        if request.priority == Priority.SACRIFICIAL:
            backend.assigned_sacrificial_load_s += selected_estimate_s
            backend.inflight_sacrificial += 1
        else:
            backend.assigned_normal_load_s += selected_estimate_s
            backend.inflight_normal += 1

        self._trace(
            "arrival",
            request=request,
            backend=backend,
            request_type=request.request_type.name,
            priority=request.priority.value,
            estimated_service_s=selected_estimate_s,
            classifier=classification.details,
            router=routing.details,
        )
        if getattr(self.scheduler, "global_protected_pull", False):
            self._schedule_idle_backends_for_protected_pull()
        else:
            if backend.running_id is None:
                self._schedule_backend(backend, incumbent=None)
        if getattr(self.scheduler, "global_work_stealing", False):
            for idle_backend in self.backends.values():
                if idle_backend.name != backend.name and idle_backend.running_id is None:
                    self._schedule_backend(idle_backend, incumbent=None)

    def _schedule_backend(self, backend: BackendState, incumbent: SimRequest | None) -> None:
        if incumbent is None and backend.running_id is not None:
            return
        pulled_id, protected_pull_cost_s = self._maybe_pull_protected(backend, incumbent)
        stolen_id = None if pulled_id is not None else self._maybe_steal_protected(backend, incumbent)
        pending = tuple(self.requests[request_id].to_view(backend.config.speed) for request_id in backend.pending_ids)
        if pulled_id is not None:
            selected_id = pulled_id
        elif (
            getattr(self.scheduler, "global_protected_pull", False)
            and incumbent is not None
            and incumbent.priority == Priority.NORMAL
        ):
            # Central Pull never makes a running protected request yield to
            # another protected request. It only dispatches at a free or Tail
            # boundary.
            selected_id = incumbent.request_id
        else:
            selected_id = self.scheduler.choose(
                now_s=self._now_s,
                backend=self._backend_view(backend),
                incumbent=None if incumbent is None else incumbent.to_view(backend.config.speed),
                pending=pending,
            )
        candidate_ids = set(backend.pending_ids)
        if incumbent is not None:
            candidate_ids.add(incumbent.request_id)
        if selected_id is None:
            if candidate_ids:
                raise RuntimeError(f"scheduler returned no request for non-empty backend {backend.name}")
            backend.running_id = None
            return
        if selected_id not in candidate_ids:
            raise RuntimeError(f"scheduler selected ineligible request {selected_id!r}")

        backend.running_id = None
        selected = self.requests[selected_id]
        preemption_cost_s = 0.0
        steal_cost_s = float(getattr(self.scheduler, "steal_cost_s", 0.0)) if selected_id == stolen_id else 0.0
        if selected_id != pulled_id:
            protected_pull_cost_s = 0.0
        if incumbent is not None and selected_id != incumbent.request_id:
            incumbent.status = RequestStatus.WAITING
            incumbent.preemptions += 1
            self._preemptions += 1
            backend.pending_ids.append(incumbent.request_id)
            preemption_cost_s = self.config.service.preemption_cost_s
            self._trace(
                "preempt",
                request=incumbent,
                backend=backend,
                selected_request_id=selected_id,
                completed_steps=incumbent.completed_steps,
                cost_s=preemption_cost_s,
            )

        if selected_id in backend.pending_ids:
            backend.pending_ids.remove(selected_id)
        selected.status = RequestStatus.RUNNING
        backend.running_id = selected_id
        selected.dispatch_count += 1
        is_resume = False
        dispatch_cost_s = preemption_cost_s + steal_cost_s + protected_pull_cost_s
        if selected.first_start_time_s is None:
            selected.first_start_time_s = self._now_s + dispatch_cost_s
        elif incumbent is None or selected_id != incumbent.request_id:
            selected.resumes += 1
            is_resume = True

        if dispatch_cost_s > 0.0:
            backend.busy_time_s += dispatch_cost_s
            self._switch_cost_total_s += preemption_cost_s
            self._steal_cost_total_s += steal_cost_s
            self._protected_pull_cost_total_s += protected_pull_cost_s
            self._trace(
                "switch_start",
                request=selected,
                backend=backend,
                duration_s=dispatch_cost_s,
                preemption_cost_s=preemption_cost_s,
                steal_cost_s=steal_cost_s,
                protected_pull_cost_s=protected_pull_cost_s,
            )
            self._push_event(
                time_s=self._now_s + dispatch_cost_s,
                kind=_EventKind.SWITCH_COMPLETE,
                request_id=selected_id,
                backend_name=backend.name,
            )
        else:
            if is_resume:
                self._trace("resume", request=selected, backend=backend, completed_steps=selected.completed_steps)
            self._start_next_phase(backend, selected)

    def _start_next_phase(self, backend: BackendState, request: SimRequest) -> None:
        if backend.running_id != request.request_id or request.status != RequestStatus.RUNNING:
            raise RuntimeError(f"cannot start phase for non-running request {request.request_id}")
        phase = request.next_phase
        duration_s = request.actual_phase_duration_s(phase, backend.config.speed)
        denoise_compute_s = (
            request.actual_denoise_compute_s / request.total_steps / backend.config.speed
            if phase == Phase.DENOISE
            else 0.0
        )
        denoise_overhead_s = (
            request.actual_denoise_overhead_s / request.total_steps / backend.config.speed
            if phase == Phase.DENOISE
            else 0.0
        )
        usp_communication_s = (
            request.actual_usp_communication_s / request.total_steps / backend.config.speed
            if phase == Phase.DENOISE
            else 0.0
        )
        hsdp_communication_s = (
            request.actual_hsdp_communication_s / request.total_steps / backend.config.speed
            if phase == Phase.DENOISE
            else 0.0
        )
        backend.busy_time_s += duration_s
        self._trace(
            "phase_start",
            request=request,
            backend=backend,
            phase=phase.value,
            step_index=request.completed_steps if phase == Phase.DENOISE else None,
            duration_s=duration_s,
            text_encode_s=(request.actual_text_encode_s / backend.config.speed if phase == Phase.ENCODE else 0.0),
            latent_prepare_s=(request.actual_latent_prepare_s / backend.config.speed if phase == Phase.ENCODE else 0.0),
            denoise_compute_s=denoise_compute_s,
            denoise_overhead_s=denoise_overhead_s,
            usp_communication_s=usp_communication_s,
            hsdp_communication_s=hsdp_communication_s,
            vae_decode_s=(request.actual_vae_decode_s / backend.config.speed if phase == Phase.DECODE else 0.0),
            postprocess_s=(request.actual_postprocess_s / backend.config.speed if phase == Phase.DECODE else 0.0),
        )
        self._push_event(
            time_s=self._now_s + duration_s,
            kind=_EventKind.PHASE_COMPLETE,
            request_id=request.request_id,
            backend_name=backend.name,
            phase=phase,
        )

    def _handle_switch_complete(self, event: _Event) -> None:
        assert event.backend_name is not None
        backend = self.backends[event.backend_name]
        request = self.requests[event.request_id]
        if backend.running_id != request.request_id:
            raise RuntimeError("context switch completed for a request that is no longer selected")
        self._trace("switch_complete", request=request, backend=backend)
        if request.resumes > 0:
            self._trace("resume", request=request, backend=backend, completed_steps=request.completed_steps)
        self._start_next_phase(backend, request)

    def _handle_phase_complete(self, event: _Event) -> None:
        assert event.backend_name is not None and event.phase is not None
        backend = self.backends[event.backend_name]
        request = self.requests[event.request_id]
        if backend.running_id != request.request_id or request.status != RequestStatus.RUNNING:
            raise RuntimeError(f"stale phase completion for {request.request_id}")

        if event.phase == Phase.ENCODE:
            if request.encode_completed:
                raise RuntimeError(f"request {request.request_id} encoded more than once")
            request.encode_completed = True
        elif event.phase == Phase.DENOISE:
            request.completed_steps += 1
            if request.completed_steps > request.total_steps:
                raise RuntimeError(f"request {request.request_id} ran too many denoise steps")
        else:
            if request.decode_completed:
                raise RuntimeError(f"request {request.request_id} decoded more than once")
            request.decode_completed = True

        self._trace(
            "phase_complete",
            request=request,
            backend=backend,
            phase=event.phase.value,
            completed_steps=request.completed_steps,
        )

        # Match the production runner RPC boundaries: encode immediately flows
        # into step 1, and the final step immediately flows into decode.
        if event.phase == Phase.ENCODE:
            self._start_next_phase(backend, request)
            return
        if event.phase == Phase.DENOISE and request.completed_steps == request.total_steps:
            self._start_next_phase(backend, request)
            return
        if event.phase == Phase.DENOISE:
            self._schedule_backend(backend, incumbent=request)
            return

        request.status = RequestStatus.COMPLETED
        request.completion_time_s = self._now_s
        backend.running_id = None
        backend.completed_requests += 1
        selected_estimate_s = request.estimated_total_on_backend_s(backend.config.speed)
        if request.priority == Priority.SACRIFICIAL:
            backend.inflight_sacrificial = max(backend.inflight_sacrificial - 1, 0)
            backend.assigned_sacrificial_load_s = max(backend.assigned_sacrificial_load_s - selected_estimate_s, 0.0)
        else:
            backend.inflight_normal = max(backend.inflight_normal - 1, 0)
            backend.assigned_normal_load_s = max(backend.assigned_normal_load_s - selected_estimate_s, 0.0)
        if self.router.update_latency_ema:
            latency_s = request.latency_s
            if backend.latency_ema_s <= 0.0:
                backend.latency_ema_s = latency_s
            else:
                backend.latency_ema_s = 0.9 * backend.latency_ema_s + 0.1 * latency_s
        self._completed += 1
        self._trace(
            "complete",
            request=request,
            backend=backend,
            latency_s=request.latency_s,
            priority=request.priority.value,
        )
        self._schedule_backend(backend, incumbent=None)

    def run(self) -> SimulationResult:
        for request in self.requests.values():
            self._push_event(
                time_s=request.arrival_time_s,
                kind=_EventKind.ARRIVAL,
                request_id=request.request_id,
            )

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.time_s < self._now_s:
                raise RuntimeError("simulation clock moved backwards")
            self._now_s = event.time_s
            if event.kind == _EventKind.ARRIVAL:
                self._handle_arrival(event)
            elif event.kind == _EventKind.PHASE_COMPLETE:
                self._handle_phase_complete(event)
            else:
                self._handle_switch_complete(event)

        self._verify_complete()
        return SimulationResult(
            seed=self.seed,
            request_rate=self.request_rate,
            metrics=self._build_metrics(),
            requests=tuple(self._request_record(request) for request in self.requests.values()),
            events=tuple(self._trace_events),
        )

    def _verify_complete(self) -> None:
        expected = len(self.requests)
        if self._completed != expected:
            raise RuntimeError(f"simulation drained with {self._completed}/{expected} requests completed")
        for request in self.requests.values():
            if request.status != RequestStatus.COMPLETED:
                raise RuntimeError(f"request {request.request_id} did not complete")
            if (
                not request.encode_completed
                or request.completed_steps != request.total_steps
                or not request.decode_completed
            ):
                raise RuntimeError(f"request {request.request_id} has incomplete work despite completion")
        for backend in self.backends.values():
            if backend.running_id is not None or backend.pending_ids:
                raise RuntimeError(f"backend {backend.name} retained work after simulation")
            if backend.inflight_normal or backend.inflight_sacrificial:
                raise RuntimeError(f"backend {backend.name} retained inflight accounting")
        if self._global_normal_waiting_ids:
            raise RuntimeError("simulation retained work in the global Normal queue")

        expected_busy_s = self._switch_cost_total_s + self._steal_cost_total_s + self._protected_pull_cost_total_s
        for request in self.requests.values():
            assert request.backend_name is not None
            expected_busy_s += request.actual_total_s / self.backends[request.backend_name].config.speed
        observed_busy_s = sum(backend.busy_time_s for backend in self.backends.values())
        if not math.isclose(observed_busy_s, expected_busy_s, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(
                f"work conservation failed: observed busy={observed_busy_s}, expected busy={expected_busy_s}"
            )

    def _request_record(self, request: SimRequest) -> dict[str, Any]:
        assert request.backend_name is not None and request.first_start_time_s is not None
        backend = self.backends[request.backend_name]
        service_s = request.actual_total_s / backend.config.speed
        text_encode_s = request.actual_text_encode_s / backend.config.speed
        latent_prepare_s = request.actual_latent_prepare_s / backend.config.speed
        denoise_compute_s = request.actual_denoise_compute_s / backend.config.speed
        denoise_overhead_s = request.actual_denoise_overhead_s / backend.config.speed
        usp_communication_s = request.actual_usp_communication_s / backend.config.speed
        hsdp_communication_s = request.actual_hsdp_communication_s / backend.config.speed
        vae_decode_s = request.actual_vae_decode_s / backend.config.speed
        postprocess_s = request.actual_postprocess_s / backend.config.speed
        record = {
            "request_id": request.request_id,
            "arrival_seq": request.arrival_seq,
            "request_type": request.request_type.name,
            "arrival_time_s": request.arrival_time_s,
            "first_start_time_s": request.first_start_time_s,
            "completion_time_s": request.completion_time_s,
            "latency_s": request.latency_s,
            "queue_before_first_start_s": request.first_start_time_s - request.arrival_time_s,
            "central_wait_s": (
                self._central_pull_time_s[request.request_id] - request.arrival_time_s
                if request.request_id in self._central_pull_time_s
                else 0.0
            ),
            "service_s": service_s,
            "text_encode_s": text_encode_s,
            "latent_prepare_s": latent_prepare_s,
            "denoise_compute_s": denoise_compute_s,
            "denoise_overhead_s": denoise_overhead_s,
            "usp_communication_s": usp_communication_s,
            "hsdp_communication_s": hsdp_communication_s,
            "vae_decode_s": vae_decode_s,
            "postprocess_s": postprocess_s,
            "slowdown": request.latency_s / service_s,
            "estimated_service_s": request.estimated_total_on_backend_s(backend.config.speed),
            "priority": request.priority.value,
            "backend": request.backend_name,
            "preemptions": request.preemptions,
            "resumes": request.resumes,
            "dispatch_count": request.dispatch_count,
            "completed_steps": request.completed_steps,
        }
        record.update(request.service_diagnostics)
        return record

    def _build_metrics(self) -> dict[str, Any]:
        latencies = [request.latency_s for request in self.requests.values()]
        first_arrival_s = min(request.arrival_time_s for request in self.requests.values())
        last_completion_s = max(request.completion_time_s or 0.0 for request in self.requests.values())
        makespan_s = last_completion_s - first_arrival_s
        sacrificial = [request for request in self.requests.values() if request.priority == Priority.SACRIFICIAL]
        normal = [request for request in self.requests.values() if request.priority == Priority.NORMAL]
        central_waits_s = [
            self._central_pull_time_s[request.request_id] - request.arrival_time_s
            for request in normal
            if request.request_id in self._central_pull_time_s
        ]
        service_total_s = 0.0
        text_encode_total_s = 0.0
        latent_prepare_total_s = 0.0
        denoise_compute_total_s = 0.0
        denoise_overhead_total_s = 0.0
        usp_communication_total_s = 0.0
        hsdp_communication_total_s = 0.0
        vae_decode_total_s = 0.0
        postprocess_total_s = 0.0
        denoise_flops_total = 0.0
        executed_denoise_flops_across_ranks_total = 0.0
        usp_communication_bytes_per_rank_total = 0.0
        hsdp_communication_bytes_per_rank_total = 0.0
        for request in self.requests.values():
            assert request.backend_name is not None
            backend_speed = self.backends[request.backend_name].config.speed
            service_total_s += request.actual_total_s / backend_speed
            text_encode_total_s += request.actual_text_encode_s / backend_speed
            latent_prepare_total_s += request.actual_latent_prepare_s / backend_speed
            denoise_compute_total_s += request.actual_denoise_compute_s / backend_speed
            denoise_overhead_total_s += request.actual_denoise_overhead_s / backend_speed
            usp_communication_total_s += request.actual_usp_communication_s / backend_speed
            hsdp_communication_total_s += request.actual_hsdp_communication_s / backend_speed
            vae_decode_total_s += request.actual_vae_decode_s / backend_speed
            postprocess_total_s += request.actual_postprocess_s / backend_speed
            denoise_flops_total += float(request.service_diagnostics.get("denoise_flops", 0.0))
            executed_denoise_flops_across_ranks_total += float(
                request.service_diagnostics.get(
                    "executed_denoise_flops_across_ranks",
                    0.0,
                )
            )
            usp_communication_bytes_per_rank_total += float(
                request.service_diagnostics.get(
                    "usp_communication_bytes_per_rank",
                    0.0,
                )
            )
            hsdp_communication_bytes_per_rank_total += float(
                request.service_diagnostics.get(
                    "hsdp_communication_bytes_per_rank",
                    0.0,
                )
            )
        communication_total_s = usp_communication_total_s + hsdp_communication_total_s

        def latency_summary(requests: list[SimRequest]) -> dict[str, float] | None:
            if not requests:
                return None
            values = [request.latency_s for request in requests]
            return {
                "mean_s": sum(values) / len(values),
                "p50_s": percentile(values, 50),
                "p95_s": percentile(values, 95),
                "p99_s": percentile(values, 99),
                "max_s": max(values),
            }

        return {
            "completed_requests": len(self.requests),
            "failed_requests": 0,
            "makespan_s": makespan_s,
            "throughput_rps": len(self.requests) / makespan_s if makespan_s > 0.0 else math.inf,
            "latency_mean_s": sum(latencies) / len(latencies),
            "latency_p50_s": percentile(latencies, 50),
            "latency_p95_s": percentile(latencies, 95),
            "latency_p99_s": percentile(latencies, 99),
            "latency_max_s": max(latencies),
            "normal_requests": len(normal),
            "sacrificial_requests": len(sacrificial),
            "preemptions": self._preemptions,
            "steals": self._steals,
            "protected_pulls": self._protected_pulls,
            "central_normal_queue_max_depth": self._central_normal_queue_max_depth,
            "central_wait_mean_s": (sum(central_waits_s) / len(central_waits_s) if central_waits_s else 0.0),
            "central_wait_p50_s": (percentile(central_waits_s, 50) if central_waits_s else 0.0),
            "central_wait_p95_s": (percentile(central_waits_s, 95) if central_waits_s else 0.0),
            "switch_cost_total_s": self._switch_cost_total_s,
            "steal_cost_total_s": self._steal_cost_total_s,
            "protected_pull_cost_total_s": self._protected_pull_cost_total_s,
            "service_total_s": service_total_s,
            "text_encode_total_s": text_encode_total_s,
            "latent_prepare_total_s": latent_prepare_total_s,
            "denoise_compute_total_s": denoise_compute_total_s,
            "denoise_overhead_total_s": denoise_overhead_total_s,
            "usp_communication_total_s": usp_communication_total_s,
            "hsdp_communication_total_s": hsdp_communication_total_s,
            "vae_decode_total_s": vae_decode_total_s,
            "postprocess_total_s": postprocess_total_s,
            "communication_total_s": communication_total_s,
            "communication_fraction_of_service": (
                communication_total_s / service_total_s if service_total_s > 0.0 else 0.0
            ),
            "usp_communication_fraction_of_service": (
                usp_communication_total_s / service_total_s if service_total_s > 0.0 else 0.0
            ),
            "hsdp_communication_fraction_of_service": (
                hsdp_communication_total_s / service_total_s if service_total_s > 0.0 else 0.0
            ),
            "denoise_flops_total": denoise_flops_total,
            "executed_denoise_flops_across_ranks_total": (executed_denoise_flops_across_ranks_total),
            "usp_communication_bytes_per_rank_total": (usp_communication_bytes_per_rank_total),
            "hsdp_communication_bytes_per_rank_total": (hsdp_communication_bytes_per_rank_total),
            "normal_latency": latency_summary(normal),
            "sacrificial_latency": latency_summary(sacrificial),
            "backend_utilization": {
                backend.name: (backend.busy_time_s / makespan_s if makespan_s > 0.0 else 0.0)
                for backend in self.backends.values()
            },
            "backend_completed_requests": {
                backend.name: backend.completed_requests for backend in self.backends.values()
            },
        }


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return a JSON-safe, parser-roundtrippable resolved configuration."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("resolved configuration keys must be strings")
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("resolved configuration numbers must be finite")
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ValueError(f"resolved configuration contains unsupported type {type(value).__name__!r}")

    def component(component_config: Any) -> dict[str, Any]:
        return {
            "type": component_config.kind,
            **normalize(component_config.options),
        }

    return {
        "version": 1,
        "name": config.name,
        "simulation": {
            "seed": config.simulation.seed,
            "runs": config.simulation.runs,
        },
        "workload": {
            "num_requests": config.workload.num_requests,
            "utilization": config.workload.utilization,
            "request_rate": config.workload.request_rate,
            "request_types": [
                {
                    "name": request_type.name,
                    "weight": request_type.weight,
                    "width": request_type.width,
                    "height": request_type.height,
                    "num_inference_steps": request_type.num_inference_steps,
                    "num_frames": request_type.num_frames,
                    "nominal_service_s": request_type.nominal_service_s,
                    "estimated_service_s": request_type.estimated_service_s,
                    "metadata": normalize(request_type.metadata),
                }
                for request_type in config.workload.request_types
            ],
        },
        "service": {
            "encode_fraction": config.service.encode_fraction,
            "decode_fraction": config.service.decode_fraction,
            "actual_jitter_sigma": config.service.actual_jitter_sigma,
            "estimate_error_sigma": config.service.estimate_error_sigma,
            "preemption_cost_s": config.service.preemption_cost_s,
            "actual_service_scale": config.service.actual_service_scale,
            "hsdp": {
                "enabled": config.service.hsdp.enabled,
                "shard_size": config.service.hsdp.shard_size,
                "communication_overhead_weight": (config.service.hsdp.communication_overhead_weight),
            },
            "timing_model": {
                "type": config.service.timing_model.kind,
                **normalize(config.service.timing_model.options),
            },
        },
        "topology": {
            "name": config.topology.name,
            "backends": [
                {
                    "name": backend.name,
                    "speed": backend.speed,
                    "devices": backend.devices,
                }
                for backend in config.topology.backends
            ],
        },
        "policy": {
            "classifier": component(config.policy.classifier),
            "router": component(config.policy.router),
            "scheduler": component(config.policy.scheduler),
        },
    }
