# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online visible-backlog leveling for pure-P95 diffusion scheduling.

The planner consumes only scheduler-visible state: arrived Normal requests,
estimated backend release times, completed latencies, active Normal
projections, and the number of outstanding Tail requests. It plans the whole
visible backlog, but returns only one bounded release-wave prefix. A dispatcher
can commit that prefix and replan after the wave drains.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.diffusion.tail_aware_release_calendar import percentile_type7

_TAIL_LATENCY_PLACEHOLDER_S = 1.0e12


@dataclass(frozen=True)
class BacklogLevelingRequest:
    request_id: str
    sequence: int
    arrival_time_s: float
    estimated_service_s_by_backend: tuple[float, ...]

    def estimate_on(self, backend_index: int) -> float:
        return self.estimated_service_s_by_backend[backend_index]


@dataclass(frozen=True)
class BacklogLevelingConfig:
    trigger_pending: int = 16
    commit_size: int = 8
    max_descent_rounds: int = 6
    candidate_cap: int = 20_000
    risk_beta: float = 0.85
    band_risk_beta: float = 0.625
    band_min_pending: int = 10
    band_max_pending: int = 27

    def __post_init__(self) -> None:
        for name in (
            "trigger_pending",
            "commit_size",
            "max_descent_rounds",
            "candidate_cap",
            "band_min_pending",
            "band_max_pending",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.trigger_pending <= 0:
            raise ValueError("trigger_pending must be positive")
        if self.commit_size <= 0:
            raise ValueError("commit_size must be positive")
        if self.max_descent_rounds < 0:
            raise ValueError("max_descent_rounds cannot be negative")
        if self.candidate_cap <= 0:
            raise ValueError("candidate_cap must be positive")
        if self.band_min_pending <= 0:
            raise ValueError("band_min_pending must be positive")
        if self.band_max_pending < self.band_min_pending:
            raise ValueError("band_max_pending cannot be less than band_min_pending")
        for name in ("risk_beta", "band_risk_beta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class BacklogLevelingJob:
    request_id: str
    backend_index: int
    start_offset_s: float
    finish_offset_s: float
    projected_latency_s: float


@dataclass(frozen=True)
class BacklogLevelingPlan:
    committed_request_ids: tuple[str, ...]
    jobs: tuple[BacklogLevelingJob, ...]
    release_calendar_s: tuple[float, ...]
    predicted_before_p95_s: float
    predicted_after_p95_s: float
    predicted_before_max_normal_s: float
    predicted_after_max_normal_s: float
    predicted_before_mean_normal_s: float
    predicted_after_mean_normal_s: float
    evaluated_candidates: int
    descent_rounds: int
    elapsed_ms: float
    projected_cohort_size: int


def _risk_beta(depth: int, config: BacklogLevelingConfig) -> float:
    if config.band_min_pending <= depth <= config.band_max_pending:
        return config.band_risk_beta
    return config.risk_beta


def _queue_band_select(
    requests: Sequence[BacklogLevelingRequest],
    *,
    dispatch_s: float,
    backend_index: int,
    config: BacklogLevelingConfig,
) -> BacklogLevelingRequest:
    beta = _risk_beta(len(requests), config)
    return min(
        requests,
        key=lambda request: (
            -(dispatch_s - request.arrival_time_s + beta * request.estimate_on(backend_index)),
            request.sequence,
            request.request_id,
        ),
    )


def _objective(
    latencies_s: Sequence[float],
) -> tuple[float, float, float]:
    finite = [latency_s for latency_s in latencies_s if latency_s < _TAIL_LATENCY_PLACEHOLDER_S]
    if not finite:
        return math.inf, math.inf, math.inf
    # P95 is the only optimized metric. Maximum and mean are deterministic
    # tie-breakers and never trade away a lower P95.
    return (
        percentile_type7(latencies_s, 0.95),
        max(finite),
        statistics.fmean(finite),
    )


def _schedule_order(
    order: Sequence[BacklogLevelingRequest],
    *,
    now_s: float,
    release_calendar_s: Sequence[float],
    fixed_latencies_s: Sequence[float],
    outstanding_tail_count: int,
) -> tuple[tuple[float, float, float], tuple[BacklogLevelingJob, ...]]:
    slots_s = list(release_calendar_s)
    latencies_s = list(fixed_latencies_s)
    latencies_s.extend([_TAIL_LATENCY_PLACEHOLDER_S] * outstanding_tail_count)
    jobs: list[BacklogLevelingJob] = []
    for request in order:
        backend_index = min(
            range(len(slots_s)),
            key=lambda index: (slots_s[index], index),
        )
        start_offset_s = slots_s[backend_index]
        finish_offset_s = start_offset_s + request.estimate_on(backend_index)
        slots_s[backend_index] = finish_offset_s
        projected_latency_s = now_s + finish_offset_s - request.arrival_time_s
        latencies_s.append(projected_latency_s)
        jobs.append(
            BacklogLevelingJob(
                request_id=request.request_id,
                backend_index=backend_index,
                start_offset_s=start_offset_s,
                finish_offset_s=finish_offset_s,
                projected_latency_s=projected_latency_s,
            )
        )
    return _objective(latencies_s), tuple(jobs)


def _evaluate_order(
    order: Sequence[BacklogLevelingRequest],
    *,
    now_s: float,
    release_calendar_s: Sequence[float],
    fixed_latencies_s: Sequence[float],
    outstanding_tail_count: int,
) -> tuple[float, float, float]:
    """Evaluate one order without allocating per-request trace objects."""

    slots_s = list(release_calendar_s)
    latencies_s = list(fixed_latencies_s)
    latencies_s.extend([_TAIL_LATENCY_PLACEHOLDER_S] * outstanding_tail_count)
    for request in order:
        backend_index = min(
            range(len(slots_s)),
            key=lambda index: (slots_s[index], index),
        )
        finish_offset_s = slots_s[backend_index] + request.estimate_on(backend_index)
        slots_s[backend_index] = finish_offset_s
        latencies_s.append(now_s + finish_offset_s - request.arrival_time_s)
    return _objective(latencies_s)


def _greedy_order(
    pending: Sequence[BacklogLevelingRequest],
    *,
    now_s: float,
    release_calendar_s: Sequence[float],
    config: BacklogLevelingConfig,
) -> tuple[BacklogLevelingRequest, ...]:
    remaining = list(pending)
    slots_s = list(release_calendar_s)
    order: list[BacklogLevelingRequest] = []
    while remaining:
        backend_index = min(
            range(len(slots_s)),
            key=lambda index: (slots_s[index], index),
        )
        dispatch_s = now_s + slots_s[backend_index]
        selected = _queue_band_select(
            remaining,
            dispatch_s=dispatch_s,
            backend_index=backend_index,
            config=config,
        )
        slots_s[backend_index] += selected.estimate_on(backend_index)
        order.append(selected)
        remaining.remove(selected)
    return tuple(order)


def plan_backlog_leveling(
    *,
    pending: Sequence[BacklogLevelingRequest],
    now_s: float,
    release_calendar_s: Sequence[float],
    completed_latencies_s: Sequence[float] = (),
    active_normal_projected_latencies_s: Sequence[float] = (),
    outstanding_tail_count: int = 0,
    config: BacklogLevelingConfig | None = None,
) -> BacklogLevelingPlan:
    """Level the Type-7 P95 boundary across all currently visible chains."""

    if not pending:
        raise ValueError("pending cannot be empty")
    if not release_calendar_s:
        raise ValueError("release_calendar_s cannot be empty")
    if outstanding_tail_count < 0:
        raise ValueError("outstanding_tail_count cannot be negative")
    backend_count = len(release_calendar_s)
    for request in pending:
        if len(request.estimated_service_s_by_backend) != backend_count:
            raise ValueError("every request must provide one estimate per release slot")
        if not math.isfinite(request.arrival_time_s) or request.arrival_time_s < 0.0:
            raise ValueError("request arrival times must be finite and non-negative")
        if any(
            not math.isfinite(estimate_s) or estimate_s <= 0.0 for estimate_s in request.estimated_service_s_by_backend
        ):
            raise ValueError("request estimates must be finite and positive")
    values = [
        now_s,
        *release_calendar_s,
        *completed_latencies_s,
        *active_normal_projected_latencies_s,
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("times and latencies must be finite and non-negative")

    effective_config = config or BacklogLevelingConfig(
        commit_size=backend_count,
    )
    started = time.perf_counter()
    fixed_latencies_s = (
        *completed_latencies_s,
        *active_normal_projected_latencies_s,
    )
    order = _greedy_order(
        pending,
        now_s=now_s,
        release_calendar_s=release_calendar_s,
        config=effective_config,
    )
    initial_objective, jobs = _schedule_order(
        order,
        now_s=now_s,
        release_calendar_s=release_calendar_s,
        fixed_latencies_s=fixed_latencies_s,
        outstanding_tail_count=outstanding_tail_count,
    )
    objective = initial_objective
    evaluated = 1
    descent_rounds = 0

    for _ in range(effective_config.max_descent_rounds):
        best: (
            tuple[
                tuple[float, float, float],
                tuple[BacklogLevelingRequest, ...],
            ]
            | None
        ) = None
        stop = False
        for left in range(len(order) - 1):
            for right in range(left + 1, len(order)):
                if evaluated >= effective_config.candidate_cap:
                    stop = True
                    break
                candidate = list(order)
                candidate[left], candidate[right] = (
                    candidate[right],
                    candidate[left],
                )
                candidate_order = tuple(candidate)
                candidate_objective = _evaluate_order(
                    candidate_order,
                    now_s=now_s,
                    release_calendar_s=release_calendar_s,
                    fixed_latencies_s=fixed_latencies_s,
                    outstanding_tail_count=outstanding_tail_count,
                )
                evaluated += 1
                if candidate_objective < objective and (best is None or candidate_objective < best[0]):
                    best = (
                        candidate_objective,
                        candidate_order,
                    )
            if stop:
                break
        if best is None:
            break
        objective, order = best
        _, jobs = _schedule_order(
            order,
            now_s=now_s,
            release_calendar_s=release_calendar_s,
            fixed_latencies_s=fixed_latencies_s,
            outstanding_tail_count=outstanding_tail_count,
        )
        descent_rounds += 1
        if stop:
            break

    commit_count = min(
        effective_config.commit_size,
        len(jobs),
    )
    return BacklogLevelingPlan(
        committed_request_ids=tuple(job.request_id for job in jobs[:commit_count]),
        jobs=jobs,
        release_calendar_s=tuple(release_calendar_s),
        predicted_before_p95_s=initial_objective[0],
        predicted_after_p95_s=objective[0],
        predicted_before_max_normal_s=initial_objective[1],
        predicted_after_max_normal_s=objective[1],
        predicted_before_mean_normal_s=initial_objective[2],
        predicted_after_mean_normal_s=objective[2],
        evaluated_candidates=evaluated,
        descent_rounds=descent_rounds,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        projected_cohort_size=(len(fixed_latencies_s) + len(pending) + outstanding_tail_count),
    )
