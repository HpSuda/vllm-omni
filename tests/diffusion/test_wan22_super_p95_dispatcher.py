# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from fastapi import HTTPException

from benchmarks.diffusion.wan22_super_p95_dispatcher import (
    WAN22_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
    Wan22ServiceTimeEstimator,
    Wan22SuperP95Dispatcher,
    parse_args,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_release_calendar_tail_pack_backfill_mode_applies_internal_defaults() -> None:
    args = parse_args(
        [
            "--wan22-scheduling-mode",
            WAN22_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
        ]
    )

    assert args.quota_every == 20
    assert args.quota_amount == 1
    assert args.threshold_ratio == 0.8
    assert args.long_request_ratio == 1.5
    assert args.sacrificial_load_factor == 0.1
    assert args.normal_routing_policy == "central_pull_tail_aware_release_calendar_beam"
    assert args.central_pull_risk_beta == 0.85
    assert args.central_pull_band_risk_beta == 0.625
    assert args.central_pull_band_min_pending == 10
    assert args.central_pull_band_max_pending == 27
    assert args.central_pull_mix_risk_beta == 0.4
    assert args.central_pull_mix_min_pending == 16
    assert args.central_pull_mix_max_pending == 26
    assert args.central_pull_mix_max_long_fraction == 0.32
    assert args.central_pull_beam_horizon == 4
    assert args.central_pull_beam_width == 16
    assert args.central_pull_beam_branch_width == 6
    assert args.central_pull_beam_risk_slack_s == 100.0
    assert args.central_pull_beam_min_pending == 10
    assert args.central_pull_beam_max_pending == 27
    assert args.central_pull_beam_history_size == 128
    assert args.central_pull_beam_candidate_cap == 4096
    assert args.tail_routing_mode == "pack"
    assert args.tail_dispatch_mode == "protected_drain"
    assert args.tail_idle_backfill is True


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("production", 119.71),
        ("2xusp4", 188.720),
        ("4xusp2", 357.126),
        ("8xusp1_inferred", 612.299),
    ],
)
def test_wan22_estimator_uses_topology_profile(profile: str, expected: float) -> None:
    estimator = Wan22ServiceTimeEstimator(profile)

    actual = estimator.estimate(
        "/v1/videos",
        {
            "size": "1280x720",
            "num_inference_steps": "6",
            "num_frames": "80",
        },
        "910B3",
    )

    assert actual == pytest.approx(expected)


def test_wan22_estimator_derives_frames_from_seconds_and_fps() -> None:
    estimator = Wan22ServiceTimeEstimator("4xusp2")

    actual = estimator.estimate(
        "/v1/videos",
        {
            "size": "854x480",
            "num_inference_steps": "4",
            "seconds": "5",
            "fps": "24",
        },
        "910B3",
    )

    assert actual == pytest.approx(132.733)


def test_wan22_estimator_scales_unknown_input_from_long_anchor() -> None:
    estimator = Wan22ServiceTimeEstimator("4xusp2")

    actual = estimator.estimate(
        "/v1/videos",
        {
            "size": "1280x720",
            "num_inference_steps": "3",
            "num_frames": "80",
        },
        "910B3",
    )

    # Unknown frame counts are normalized from 80 to 81 for Wan2.2 VAE
    # compatibility before applying the coarse compute-volume scaling.
    assert actual == pytest.approx(357.126 * 3 * 81 / (6 * 80))


def test_wan22_estimator_rejects_non_video_endpoint() -> None:
    estimator = Wan22ServiceTimeEstimator("4xusp2")

    with pytest.raises(HTTPException, match="only supports /v1/videos"):
        estimator.estimate(
            "/v1/images/generations",
            {"size": "1024x1024", "num_inference_steps": 25},
            "910B3",
        )


def test_wan22_estimator_rejects_uncalibrated_hardware_profile() -> None:
    estimator = Wan22ServiceTimeEstimator("4xusp2")

    with pytest.raises(HTTPException, match="not calibrated"):
        estimator.estimate(
            "/v1/videos",
            {
                "size": "854x480",
                "num_inference_steps": "3",
                "num_frames": "80",
            },
            "910B2",
        )


def test_wan22_dispatcher_exposes_dedicated_estimator() -> None:
    dispatcher = Wan22SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=["910B3"],
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        wan22_estimator=Wan22ServiceTimeEstimator("4xusp2"),
    )

    actual = dispatcher._estimate_service_s(
        "/v1/videos",
        {
            "size": "854x480",
            "num_inference_steps": "3",
            "num_frames": "80",
        },
        "910B3",
    )

    assert dispatcher.service_time_estimator_name == "wan22:4xusp2"
    assert actual == pytest.approx(68.602)
