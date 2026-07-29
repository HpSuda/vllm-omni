# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.diffusion.tail_aware_release_calendar import (
    ReleaseCalendarBeamConfig,
    ReleaseCalendarRequest,
    percentile_type7,
    plan_release_calendar,
)


def _request(
    index: int,
    *,
    arrival_s: float,
    estimate_s: float,
    backend_count: int = 2,
) -> ReleaseCalendarRequest:
    return ReleaseCalendarRequest(
        request_id=f"request-{index}",
        sequence=index,
        arrival_time_s=arrival_s,
        estimated_service_s_by_backend=((estimate_s,) * backend_count),
    )


def test_percentile_type7_matches_linear_interpolation() -> None:
    assert percentile_type7(
        [1.0, 2.0, 3.0, 4.0],
        0.95,
    ) == pytest.approx(3.85)


def test_beam_is_bounded_and_executes_only_first_action() -> None:
    pending = [
        _request(0, arrival_s=0.0, estimate_s=80.0),
        _request(1, arrival_s=20.0, estimate_s=15.0),
        _request(2, arrival_s=25.0, estimate_s=30.0),
        _request(3, arrival_s=30.0, estimate_s=10.0),
    ]
    plan = plan_release_calendar(
        pending=pending,
        now_s=50.0,
        release_calendar_s=(0.0, 25.0),
        first_backend_index=0,
        completed_latencies_s=(30.0, 40.0),
        active_normal_projected_latencies_s=(65.0,),
        outstanding_tail_count=0,
        config=ReleaseCalendarBeamConfig(
            horizon=2,
            beam_width=3,
            branch_width=2,
            risk_slack_s=1000.0,
            min_pending=1,
            max_pending=10,
            history_size=8,
            candidate_cap=10,
        ),
    )

    assert plan.used_beam is True
    assert plan.selected_request_id == plan.prefix[0]
    assert len(plan.prefix) <= 2
    assert plan.candidate_count <= 10
    assert plan.release_calendar_s == (0.0, 25.0)


def test_branch_width_allows_one_structural_shortest_candidate() -> None:
    pending = [
        _request(
            index,
            arrival_s=float(-index),
            estimate_s=(1.0 if index == 6 else 100.0 + index),
            backend_count=1,
        )
        for index in range(7)
    ]
    plan = plan_release_calendar(
        pending=pending,
        now_s=0.0,
        release_calendar_s=(0.0,),
        first_backend_index=0,
        completed_latencies_s=(),
        active_normal_projected_latencies_s=(),
        outstanding_tail_count=0,
        config=ReleaseCalendarBeamConfig(
            horizon=1,
            beam_width=16,
            branch_width=6,
            risk_slack_s=1000.0,
            min_pending=1,
            max_pending=10,
            candidate_cap=20,
        ),
    )

    assert plan.candidate_count == 7


def test_outstanding_tail_uses_readable_normal_boundary() -> None:
    pending = [_request(index, arrival_s=0.0, estimate_s=10.0) for index in range(19)]
    plan = plan_release_calendar(
        pending=pending,
        now_s=0.0,
        release_calendar_s=(0.0, 0.0),
        first_backend_index=0,
        completed_latencies_s=(),
        active_normal_projected_latencies_s=(),
        outstanding_tail_count=1,
        config=ReleaseCalendarBeamConfig(
            horizon=1,
            beam_width=2,
            branch_width=1,
            risk_slack_s=0.0,
            min_pending=1,
            max_pending=30,
        ),
    )

    assert plan.predicted_before_p95_s is None
    assert plan.predicted_after_p95_s is None
    assert plan.predicted_before_normal_boundary_s is not None
    assert plan.predicted_after_normal_boundary_s is not None
    assert plan.outstanding_tail_count == 1
    assert plan.projected_cohort_size == 20


def test_missing_running_tail_eta_falls_back_to_queue_band() -> None:
    pending = [
        _request(0, arrival_s=0.0, estimate_s=20.0),
        _request(1, arrival_s=1.0, estimate_s=10.0),
    ]
    plan = plan_release_calendar(
        pending=pending,
        now_s=10.0,
        release_calendar_s=None,
        first_backend_index=0,
        completed_latencies_s=(),
        active_normal_projected_latencies_s=(),
        outstanding_tail_count=1,
        unavailable_release_reason="running_tail_eta_unavailable",
        config=ReleaseCalendarBeamConfig(
            min_pending=1,
            max_pending=10,
        ),
    )

    assert plan.used_beam is False
    assert plan.fallback_reason == "running_tail_eta_unavailable"
    assert plan.outstanding_tail_count == 1
