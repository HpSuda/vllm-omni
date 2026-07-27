# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.diffusion.backlog_leveling import (
    BacklogLevelingConfig,
    BacklogLevelingRequest,
    plan_backlog_leveling,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(
    request_id: str,
    sequence: int,
    arrival_time_s: float,
    service_s: float,
) -> BacklogLevelingRequest:
    return BacklogLevelingRequest(
        request_id=request_id,
        sequence=sequence,
        arrival_time_s=arrival_time_s,
        estimated_service_s_by_backend=(service_s, service_s),
    )


def test_visible_backlog_leveling_commits_improved_prefix() -> None:
    pending = [
        _request("r0", 0, 13.436, 10.0),
        _request("r1", 1, 25.507, 30.0),
        _request("r2", 2, 76.096, 30.0),
        _request("r3", 3, 65.159, 10.0),
        _request("r4", 4, 9.386, 10.0),
        _request("r5", 5, 89.332, 30.0),
        _request("r6", 6, 43.277, 10.0),
        _request("r7", 7, 69.583, 30.0),
    ]

    plan = plan_backlog_leveling(
        pending=pending,
        now_s=100.0,
        release_calendar_s=(0.0, 20.0),
        completed_latencies_s=(20.0, 50.0),
        config=BacklogLevelingConfig(
            trigger_pending=4,
            commit_size=2,
            max_descent_rounds=3,
            candidate_cap=1000,
            band_min_pending=2,
            band_max_pending=8,
        ),
    )

    assert plan.predicted_after_p95_s < plan.predicted_before_p95_s
    assert plan.committed_request_ids == tuple(job.request_id for job in plan.jobs[:2])
    assert len(plan.committed_request_ids) == 2
    assert set(plan.committed_request_ids) <= {request.request_id for request in pending}
    assert plan.evaluated_candidates <= 1000
    assert plan.projected_cohort_size == 10


def test_visible_backlog_leveling_rejects_backend_shape_mismatch() -> None:
    request = BacklogLevelingRequest(
        request_id="request",
        sequence=0,
        arrival_time_s=0.0,
        estimated_service_s_by_backend=(10.0,),
    )

    with pytest.raises(ValueError, match="one estimate per release slot"):
        plan_backlog_leveling(
            pending=[request],
            now_s=1.0,
            release_calendar_s=(0.0, 1.0),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trigger_pending", 0),
        ("commit_size", 0),
        ("max_descent_rounds", -1),
        ("candidate_cap", 0),
    ],
)
def test_backlog_leveling_config_rejects_invalid_integer(
    field: str,
    value: int,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        BacklogLevelingConfig(**kwargs)
