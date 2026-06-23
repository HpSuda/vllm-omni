# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from benchmarks.diffusion.super_p95_dispatcher import SuperP95Dispatcher
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import estimate_service_time_s
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_estimate_uses_seconds_and_fps_when_num_frames_missing() -> None:
    body = {
        "size": "854x480",
        "seconds": "4",
        "fps": "24",
        "num_inference_steps": "3",
    }

    estimated = SuperP95Dispatcher._estimate_service_s("/v1/videos", body, "910B3")
    expected = estimate_service_time_s(
        OmniDiffusionRequest(
            prompts=["video"],
            sampling_params=OmniDiffusionSamplingParams(
                width=854,
                height=480,
                num_inference_steps=3,
                num_frames=96,
            ),
            request_ids=["video"],
        ),
        hardware_profile="910B3",
    )

    assert estimated == pytest.approx(expected)


def test_video_estimate_defaults_to_24_fps_for_seconds_only_requests() -> None:
    body = {
        "size": "854x480",
        "seconds": "4",
        "num_inference_steps": "3",
    }

    estimated = SuperP95Dispatcher._estimate_service_s("/v1/videos", body, "910B3")
    expected = estimate_service_time_s(
        OmniDiffusionRequest(
            prompts=["video"],
            sampling_params=OmniDiffusionSamplingParams(
                width=854,
                height=480,
                num_inference_steps=3,
                num_frames=96,
            ),
            request_ids=["video"],
        ),
        hardware_profile="910B3",
    )

    assert estimated == pytest.approx(expected)


def test_dispatcher_preserves_tail_budget_without_obvious_long_request() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )

    async def _run():
        decisions = []
        for _ in range(20):
            decisions.append(
                await dispatcher._choose_backend(
                    "/v1/videos",
                    {
                        "width": "854",
                        "height": "480",
                        "num_inference_steps": "3",
                        "num_frames": "80",
                    },
                )
            )
        return decisions

    decisions = asyncio.run(_run())

    assert not any(decision.is_sacrificial for decision in decisions)
    assert dispatcher.credits == 1


def test_dispatcher_spends_preserved_tail_budget_on_later_long_request() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )

    async def _run():
        for _ in range(20):
            await dispatcher._choose_backend(
                "/v1/videos",
                {
                    "width": "854",
                    "height": "480",
                    "num_inference_steps": "3",
                    "num_frames": "80",
                },
            )
        return await dispatcher._choose_backend(
            "/v1/videos",
            {
                "width": "1280",
                "height": "720",
                "num_inference_steps": "6",
                "num_frames": "80",
            },
        )

    decision = asyncio.run(_run())

    assert decision.is_sacrificial is True
    assert dispatcher.credits == 0
    assert dispatcher.tail_admitted_count == 1

    asyncio.run(dispatcher._mark_failed_response(decision, elapsed_s=1.0))

    assert dispatcher.credits == 0
    assert dispatcher.tail_admitted_count == 1
