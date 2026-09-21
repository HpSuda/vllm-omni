# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online Tail-aware release-calendar beam planning.

The planner deliberately consumes only scheduler-visible state. It does not
know future arrivals or the final benchmark size, and it never changes a
request that has already started.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_TAIL_LATENCY_PLACEHOLDER_S = 1.0e12


@dataclass(frozen=True)
class ReleaseCalendarRequest:
    request_id: str
    sequence: int
    arrival_time_s: float
    estimated_service_s_by_backend: tuple[float, ...]

    def estimate_on(self, backend_index: int) -> float:
        return self.estimated_service_s_by_backend[backend_index]


@dataclass(frozen=True)
class ReleaseCalendarBeamConfig:
    horizon: int = 4
    beam_width: int = 16
    branch_width: int = 6
    risk_slack_s: float = 100.0
    min_pending: int = 10
    max_pending: int = 27
    history_size: int = 128
    candidate_cap: int = 4096
    risk_beta: float = 0.85
    band_risk_beta: float = 0.625
    band_min_pending: int = 10
    band_max_pending: int = 27

    def __post_init__(self) -> None:
        for name in (
            "horizon",
            "beam_width",
            "branch_width",
            "min_pending",
            "max_pending",
            "history_size",
            "candidate_cap",
            "band_min_pending",
            "band_max_pending",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.branch_width <= 0:
            raise ValueError("branch_width must be positive")
        if self.min_pending <= 0:
            raise ValueError("min_pending must be positive")
        if self.max_pending < self.min_pending:
            raise ValueError("max_pending cannot be less than min_pending")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.candidate_cap <= 0:
            raise ValueError("candidate_cap must be positive")
        if self.band_min_pending <= 0:
            raise ValueError("band_min_pending must be positive")
        if self.band_max_pending < self.band_min_pending:
            raise ValueError("band_max_pending cannot be less than band_min_pending")
        for name in ("risk_slack_s", "risk_beta", "band_risk_beta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class ReleaseCalendarPlan:
    selected_request_id: str
    used_beam: bool
    fallback_reason: str | None
    elapsed_ms: float
    candidate_count: int
    predicted_before_p95_s: float | None
    predicted_after_p95_s: float | None
    predicted_before_normal_boundary_s: float | None
    predicted_after_normal_boundary_s: float | None
    predicted_before_mean_s: float | None
    predicted_after_mean_s: float | None
    prefix: tuple[str, ...]
    release_calendar_s: tuple[float, ...]
    completed_history_count: int
    active_normal_count: int
    outstanding_tail_count: int
    projected_cohort_size: int


@dataclass(frozen=True)
class _BeamState:
    slots_s: tuple[float, ...]
    projected_latencies_s: tuple[float, ...]
    remaining: tuple[ReleaseCalendarRequest, ...]
    first_request_id: str
    prefix: tuple[str, ...]
    rollout_objective: tuple[float, float]


def percentile_type7(values: Sequence[float], quantile: float) -> float:
    """Hyndman-Fan type-7 percentile, matching NumPy's default."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _objective(values: Sequence[float]) -> tuple[float, float]:
    finite = [value for value in values if value < _TAIL_LATENCY_PLACEHOLDER_S]
    mean_s = statistics.fmean(finite) if finite else math.inf
    return percentile_type7(values, 0.95), mean_s


def _risk_beta(depth: int, config: ReleaseCalendarBeamConfig) -> float:
    if config.band_min_pending <= depth <= config.band_max_pending:
        return config.band_risk_beta
    return config.risk_beta


def _request_risk(
    request: ReleaseCalendarRequest,
    *,
    dispatch_s: float,
    backend_index: int,
    depth: int,
    config: ReleaseCalendarBeamConfig,
) -> float:
    return dispatch_s - request.arrival_time_s + _risk_beta(depth, config) * request.estimate_on(backend_index)


def _queue_band_select(
    requests: Sequence[ReleaseCalendarRequest],
    *,
    dispatch_s: float,
    backend_index: int,
    config: ReleaseCalendarBeamConfig,
) -> ReleaseCalendarRequest:
    depth = len(requests)
    return min(
        requests,
        key=lambda request: (
            -_request_risk(
                request,
                dispatch_s=dispatch_s,
                backend_index=backend_index,
                depth=depth,
                config=config,
            ),
            request.sequence,
            request.request_id,
        ),
    )


def _critical_candidates(
    remaining: tuple[ReleaseCalendarRequest, ...],
    *,
    dispatch_s: float,
    backend_index: int,
    config: ReleaseCalendarBeamConfig,
) -> tuple[ReleaseCalendarRequest, ...]:
    depth = len(remaining)
    by_risk = sorted(
        remaining,
        key=lambda request: (
            -_request_risk(
                request,
                dispatch_s=dispatch_s,
                backend_index=backend_index,
                depth=depth,
                config=config,
            ),
            request.sequence,
            request.request_id,
        ),
    )
    maximum = _request_risk(
        by_risk[0],
        dispatch_s=dispatch_s,
        backend_index=backend_index,
        depth=depth,
        config=config,
    )
    eligible = [
        request
        for request in by_risk
        if _request_risk(
            request,
            dispatch_s=dispatch_s,
            backend_index=backend_index,
            depth=depth,
            config=config,
        )
        >= maximum - config.risk_slack_s
    ][: config.branch_width]

    # Preserve the prototype's structural shortest alternative. This can add
    # one request beyond branch_width, while candidate_cap remains the hard
    # bound on total planner work.
    shortest = min(
        remaining,
        key=lambda request: (
            request.estimate_on(backend_index),
            -_request_risk(
                request,
                dispatch_s=dispatch_s,
                backend_index=backend_index,
                depth=depth,
                config=config,
            ),
            request.sequence,
            request.request_id,
        ),
    )
    if _request_risk(
        shortest,
        dispatch_s=dispatch_s,
        backend_index=backend_index,
        depth=depth,
        config=config,
    ) >= maximum - config.risk_slack_s and all(request.request_id != shortest.request_id for request in eligible):
        eligible.append(shortest)
    return tuple(eligible)


def _next_backend_index(
    slots_s: Sequence[float],
    *,
    first_backend_index: int | None = None,
) -> int:
    if first_backend_index is not None:
        return first_backend_index
    return min(
        range(len(slots_s)),
        key=lambda index: (slots_s[index], index),
    )


def _greedy_complete(
    *,
    now_s: float,
    slots_s: tuple[float, ...],
    projected_latencies_s: tuple[float, ...],
    remaining: tuple[ReleaseCalendarRequest, ...],
    config: ReleaseCalendarBeamConfig,
    first_backend_index: int | None = None,
) -> tuple[tuple[float, float], tuple[str, ...]]:
    mutable_slots = list(slots_s)
    mutable_projected = list(projected_latencies_s)
    mutable_remaining = list(remaining)
    order: list[str] = []
    force_backend_index = first_backend_index
    while mutable_remaining:
        backend_index = _next_backend_index(
            mutable_slots,
            first_backend_index=force_backend_index,
        )
        force_backend_index = None
        dispatch_s = now_s + mutable_slots[backend_index]
        selected = _queue_band_select(
            mutable_remaining,
            dispatch_s=dispatch_s,
            backend_index=backend_index,
            config=config,
        )
        finish_s = dispatch_s + selected.estimate_on(backend_index)
        mutable_slots[backend_index] = finish_s - now_s
        mutable_projected.append(finish_s - selected.arrival_time_s)
        mutable_remaining.remove(selected)
        order.append(selected.request_id)
    return _objective(mutable_projected), tuple(order)


def _normal_boundary(
    *,
    now_s: float,
    slots_s: tuple[float, ...],
    projected_latencies_s: tuple[float, ...],
    remaining: tuple[ReleaseCalendarRequest, ...],
    config: ReleaseCalendarBeamConfig,
    first_backend_index: int | None = None,
) -> float | None:
    """Return the largest finite latency from the same greedy rollout."""

    mutable_slots = list(slots_s)
    mutable_projected = list(projected_latencies_s)
    mutable_remaining = list(remaining)
    force_backend_index = first_backend_index
    while mutable_remaining:
        backend_index = _next_backend_index(
            mutable_slots,
            first_backend_index=force_backend_index,
        )
        force_backend_index = None
        dispatch_s = now_s + mutable_slots[backend_index]
        selected = _queue_band_select(
            mutable_remaining,
            dispatch_s=dispatch_s,
            backend_index=backend_index,
            config=config,
        )
        finish_s = dispatch_s + selected.estimate_on(backend_index)
        mutable_slots[backend_index] = finish_s - now_s
        mutable_projected.append(finish_s - selected.arrival_time_s)
        mutable_remaining.remove(selected)
    return max(
        (value for value in mutable_projected if value < _TAIL_LATENCY_PLACEHOLDER_S),
        default=None,
    )


def _validated_release_calendar(
    release_calendar_s: Sequence[float] | None,
    pending: Sequence[ReleaseCalendarRequest],
) -> tuple[float, ...] | None:
    if release_calendar_s is None or not release_calendar_s:
        return None
    slots: list[float] = []
    for raw_slot in release_calendar_s:
        if (
            isinstance(raw_slot, bool)
            or not isinstance(raw_slot, (int, float))
            or not math.isfinite(raw_slot)
            or raw_slot < 0.0
        ):
            return None
        slots.append(float(raw_slot))
    for request in pending:
        estimates = request.estimated_service_s_by_backend
        if len(estimates) != len(slots):
            return None
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0
            for value in estimates
        ):
            return None
    return tuple(slots)


def _fallback_plan(
    *,
    pending: Sequence[ReleaseCalendarRequest],
    now_s: float,
    backend_index: int,
    config: ReleaseCalendarBeamConfig,
    release_calendar_s: tuple[float, ...],
    reason: str,
    started_at_s: float,
    completed_history_count: int = 0,
    active_normal_count: int = 0,
    outstanding_tail_count: int = 0,
) -> ReleaseCalendarPlan:
    selected = _queue_band_select(
        pending,
        dispatch_s=now_s,
        backend_index=backend_index,
        config=config,
    )
    return ReleaseCalendarPlan(
        selected_request_id=selected.request_id,
        used_beam=False,
        fallback_reason=reason,
        elapsed_ms=(time.perf_counter() - started_at_s) * 1000.0,
        candidate_count=1,
        predicted_before_p95_s=None,
        predicted_after_p95_s=None,
        predicted_before_normal_boundary_s=None,
        predicted_after_normal_boundary_s=None,
        predicted_before_mean_s=None,
        predicted_after_mean_s=None,
        prefix=(selected.request_id,),
        release_calendar_s=release_calendar_s,
        completed_history_count=completed_history_count,
        active_normal_count=active_normal_count,
        outstanding_tail_count=outstanding_tail_count,
        projected_cohort_size=(completed_history_count + active_normal_count + outstanding_tail_count + len(pending)),
    )


def plan_release_calendar(
    *,
    pending: Sequence[ReleaseCalendarRequest],
    now_s: float,
    release_calendar_s: Sequence[float] | None,
    first_backend_index: int,
    completed_latencies_s: Iterable[float],
    active_normal_projected_latencies_s: Iterable[float],
    outstanding_tail_count: int,
    config: ReleaseCalendarBeamConfig,
    unavailable_release_reason: str = "missing_release_calendar",
) -> ReleaseCalendarPlan:
    """Plan several visible release decisions, then execute only the first."""

    started_at_s = time.perf_counter()
    if not pending:
        raise ValueError("pending cannot be empty")
    completed = [
        float(value)
        for value in completed_latencies_s
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0
    ][-config.history_size :]
    active = [
        float(value)
        for value in active_normal_projected_latencies_s
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0
    ]
    tail_count = max(int(outstanding_tail_count), 0)
    if not 0 <= first_backend_index < (len(release_calendar_s) if release_calendar_s is not None else 0):
        # The request estimates are still sufficient for a stable Queue-Band
        # fallback when callers pass a missing release calendar.
        fallback_backend = max(first_backend_index, 0)
        if pending and pending[0].estimated_service_s_by_backend:
            fallback_backend = min(
                fallback_backend,
                len(pending[0].estimated_service_s_by_backend) - 1,
            )
        return _fallback_plan(
            pending=pending,
            now_s=now_s,
            backend_index=fallback_backend,
            config=config,
            release_calendar_s=(),
            reason=unavailable_release_reason,
            started_at_s=started_at_s,
            completed_history_count=len(completed),
            active_normal_count=len(active),
            outstanding_tail_count=tail_count,
        )
    slots = _validated_release_calendar(release_calendar_s, pending)
    if slots is None:
        return _fallback_plan(
            pending=pending,
            now_s=now_s,
            backend_index=first_backend_index,
            config=config,
            release_calendar_s=(),
            reason=unavailable_release_reason,
            started_at_s=started_at_s,
            completed_history_count=len(completed),
            active_normal_count=len(active),
            outstanding_tail_count=tail_count,
        )

    depth = len(pending)
    if not config.min_pending <= depth <= config.max_pending:
        return _fallback_plan(
            pending=pending,
            now_s=now_s,
            backend_index=first_backend_index,
            config=config,
            release_calendar_s=slots,
            reason="outside_pending_window",
            started_at_s=started_at_s,
            completed_history_count=len(completed),
            active_normal_count=len(active),
            outstanding_tail_count=tail_count,
        )

    initial_projected = tuple(
        [
            *completed,
            *active,
            *([_TAIL_LATENCY_PLACEHOLDER_S] * tail_count),
        ]
    )
    before_objective, _ = _greedy_complete(
        now_s=now_s,
        slots_s=slots,
        projected_latencies_s=initial_projected,
        remaining=tuple(pending),
        config=config,
        first_backend_index=first_backend_index,
    )
    base = _queue_band_select(
        pending,
        dispatch_s=now_s + slots[first_backend_index],
        backend_index=first_backend_index,
        config=config,
    )
    initial = _BeamState(
        slots_s=slots,
        projected_latencies_s=initial_projected,
        remaining=tuple(pending),
        first_request_id=base.request_id,
        prefix=(),
        rollout_objective=before_objective,
    )
    beam = [initial]
    candidate_count = 0
    cap_reached = False
    for depth_index in range(min(config.horizon, len(pending))):
        children: list[_BeamState] = []
        for state in beam:
            if not state.remaining:
                children.append(state)
                continue
            backend_index = _next_backend_index(
                state.slots_s,
                first_backend_index=(first_backend_index if depth_index == 0 else None),
            )
            dispatch_s = now_s + state.slots_s[backend_index]
            for selected in _critical_candidates(
                state.remaining,
                dispatch_s=dispatch_s,
                backend_index=backend_index,
                config=config,
            ):
                if candidate_count >= config.candidate_cap:
                    cap_reached = True
                    break
                child_slots = list(state.slots_s)
                finish_s = dispatch_s + selected.estimate_on(backend_index)
                child_slots[backend_index] = finish_s - now_s
                child_projected = (
                    *state.projected_latencies_s,
                    finish_s - selected.arrival_time_s,
                )
                child_remaining = tuple(
                    request for request in state.remaining if request.request_id != selected.request_id
                )
                child_objective, _ = _greedy_complete(
                    now_s=now_s,
                    slots_s=tuple(child_slots),
                    projected_latencies_s=child_projected,
                    remaining=child_remaining,
                    config=config,
                )
                children.append(
                    _BeamState(
                        slots_s=tuple(child_slots),
                        projected_latencies_s=child_projected,
                        remaining=child_remaining,
                        first_request_id=(selected.request_id if not state.prefix else state.first_request_id),
                        prefix=(*state.prefix, selected.request_id),
                        rollout_objective=child_objective,
                    )
                )
                candidate_count += 1
            if cap_reached:
                break
        if not children:
            break
        children.sort(
            key=lambda state: (
                state.rollout_objective,
                state.first_request_id != base.request_id,
                state.prefix,
            )
        )
        beam = children[: config.beam_width]
        if cap_reached or not beam[0].remaining:
            break

    if not beam or not beam[0].prefix:
        return _fallback_plan(
            pending=pending,
            now_s=now_s,
            backend_index=first_backend_index,
            config=config,
            release_calendar_s=slots,
            reason="candidate_cap",
            started_at_s=started_at_s,
            completed_history_count=len(completed),
            active_normal_count=len(active),
            outstanding_tail_count=tail_count,
        )
    best = min(
        beam,
        key=lambda state: (
            state.rollout_objective,
            state.first_request_id != base.request_id,
            state.prefix,
        ),
    )
    projected_cohort_size = len(completed) + len(active) + tail_count + len(pending)
    finite_cohort_size = projected_cohort_size - tail_count
    p95_upper_rank = math.ceil((projected_cohort_size - 1) * 0.95)
    p95_uses_placeholder = p95_upper_rank >= finite_cohort_size
    before_p95_s = None if p95_uses_placeholder else before_objective[0]
    after_p95_s = None if p95_uses_placeholder else best.rollout_objective[0]
    before_boundary_s = _normal_boundary(
        now_s=now_s,
        slots_s=slots,
        projected_latencies_s=initial_projected,
        remaining=tuple(pending),
        config=config,
        first_backend_index=first_backend_index,
    )
    after_boundary_s = _normal_boundary(
        now_s=now_s,
        slots_s=best.slots_s,
        projected_latencies_s=best.projected_latencies_s,
        remaining=best.remaining,
        config=config,
    )
    return ReleaseCalendarPlan(
        selected_request_id=best.first_request_id,
        used_beam=True,
        fallback_reason=("candidate_cap" if cap_reached else None),
        elapsed_ms=(time.perf_counter() - started_at_s) * 1000.0,
        candidate_count=candidate_count,
        predicted_before_p95_s=before_p95_s,
        predicted_after_p95_s=after_p95_s,
        predicted_before_normal_boundary_s=before_boundary_s,
        predicted_after_normal_boundary_s=after_boundary_s,
        predicted_before_mean_s=before_objective[1],
        predicted_after_mean_s=best.rollout_objective[1],
        prefix=best.prefix,
        release_calendar_s=slots,
        completed_history_count=len(completed),
        active_normal_count=len(active),
        outstanding_tail_count=tail_count,
        projected_cohort_size=projected_cohort_size,
    )
