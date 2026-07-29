# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.diffusion.qwen_image_super_p95_dispatcher import (
    QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
    build_qwen_image_dispatcher,
    parse_args,
)
from benchmarks.diffusion.super_p95_dispatcher import SuperP95Dispatcher

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_release_calendar_tail_pack_backfill_mode_applies_qwen_defaults() -> None:
    args = parse_args(
        [
            "--qwen-image-scheduling-mode",
            QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
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
    assert args.central_pull_mix_risk_beta == 0.4
    assert args.central_pull_beam_horizon == 4
    assert args.central_pull_beam_width == 16
    assert args.central_pull_beam_branch_width == 6
    assert args.tail_routing_mode == "pack"
    assert args.tail_dispatch_mode == "protected_drain"
    assert args.tail_idle_backfill is True
    assert args.backend_scheduler == "super_p95_step"

    backend_env = dict(item.split("=", 1) for item in args.backend_env)
    assert backend_env["SUPER_P95_QWEN_SMALL_BATCH2"] == "1"
    assert backend_env["SUPER_P95_IMAGE_BATCH_SEARCH_WINDOW"] == "64"
    assert backend_env["SUPER_P95_QWEN_SMALL_BATCH_MIN_PENDING"] == "1"


def test_qwen_mode_preserves_explicit_backend_env_override() -> None:
    args = parse_args(
        [
            "--qwen-image-scheduling-mode",
            QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
            "--backend-env",
            "SUPER_P95_IMAGE_BATCH_SEARCH_WINDOW=32",
        ]
    )

    backend_env = dict(item.split("=", 1) for item in args.backend_env)
    assert backend_env["SUPER_P95_IMAGE_BATCH_SEARCH_WINDOW"] == "32"


def test_qwen_mode_models_running_tail_as_preemptible_for_normal() -> None:
    args = parse_args(
        [
            "--backend-url",
            "http://backend-0",
            "--qwen-image-scheduling-mode",
            QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
        ]
    )

    dispatcher = build_qwen_image_dispatcher(args)

    assert dispatcher.running_tail_preemptible_for_normal is True


@pytest.mark.parametrize(
    ("size", "steps", "expected"),
    [
        ("512x512", 20, 8.64),
        ("768x768", 20, 8.64),
        ("1024x1024", 25, 14.22),
        ("1536x1536", 35, 49.34),
    ],
)
def test_qwen_estimator_uses_910b3_image_anchors(
    size: str,
    steps: int,
    expected: float,
) -> None:
    actual = SuperP95Dispatcher._estimate_service_s(
        "/v1/images/generations",
        {
            "size": size,
            "num_inference_steps": steps,
        },
        "910B3",
    )

    assert actual == pytest.approx(expected)
