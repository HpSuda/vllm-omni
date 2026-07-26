# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

import benchmarks.diffusion.super_p95_dispatcher as dispatcher_module
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


def test_central_pull_defers_normal_binding_until_backend_capacity_opens() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy="central_pull_max_risk",
    )
    body = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }

    async def _run():
        first = await dispatcher._choose_backend("/v1/videos", body)
        second = await dispatcher._choose_backend("/v1/videos", body)
        third_task = asyncio.create_task(dispatcher._choose_backend("/v1/videos", body))
        await asyncio.sleep(0)

        assert {first.backend_index, second.backend_index} == {0, 1}
        assert not third_task.done()
        assert len(dispatcher._pending_normal_dispatches) == 1

        await dispatcher._mark_failed_response(first, elapsed_s=1.0)
        third = await third_task
        return first, second, third

    first, second, third = asyncio.run(_run())

    assert third.backend_index == first.backend_index
    assert third.central_wait_s >= 0.0
    assert dispatcher.central_pull_dispatches == 3
    assert dispatcher.central_queue_max_depth == 1
    assert dispatcher.backends[third.backend_index].inflight_normal_requests == 1
    assert dispatcher.backends[second.backend_index].inflight_normal_requests == 1


def test_central_pull_selects_max_online_wait_plus_estimate_risk() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy="central_pull_max_risk",
    )
    short = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }
    long = {
        "width": "1280",
        "height": "720",
        "num_inference_steps": "6",
        "num_frames": "80",
    }

    async def _run():
        incumbent = await dispatcher._choose_backend("/v1/videos", short)
        older_short = asyncio.create_task(dispatcher._choose_backend("/v1/videos", short))
        newer_long = asyncio.create_task(dispatcher._choose_backend("/v1/videos", long))
        await asyncio.sleep(0)

        await dispatcher._mark_failed_response(incumbent, elapsed_s=1.0)
        await asyncio.sleep(0)
        assert newer_long.done()
        assert not older_short.done()

        selected_long = await newer_long
        await dispatcher._mark_failed_response(selected_long, elapsed_s=1.0)
        selected_short = await older_short
        return selected_long, selected_short

    selected_long, selected_short = asyncio.run(_run())

    assert selected_long.estimated_service_s > selected_short.estimated_service_s
    assert selected_long.arrival_counter > selected_short.arrival_counter


def test_cost_damped_risk_balances_age_against_estimated_service(monkeypatch) -> None:
    now_s = 0.0
    monkeypatch.setattr(dispatcher_module.time, "perf_counter", lambda: now_s)
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy="central_pull_cost_damped_risk",
        central_pull_risk_beta=0.5,
    )
    short = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }
    long = {
        "width": "1280",
        "height": "720",
        "num_inference_steps": "6",
        "num_frames": "80",
    }

    async def _run():
        nonlocal now_s
        incumbent = await dispatcher._choose_backend("/v1/videos", short)
        older_short = asyncio.create_task(dispatcher._choose_backend("/v1/videos", short))
        await asyncio.sleep(0)

        now_s = 50.0
        newer_long = asyncio.create_task(dispatcher._choose_backend("/v1/videos", long))
        await asyncio.sleep(0)
        await dispatcher._mark_failed_response(incumbent, elapsed_s=1.0)
        await asyncio.sleep(0)

        assert older_short.done()
        assert not newer_long.done()
        selected_short = await older_short
        await dispatcher._mark_failed_response(selected_short, elapsed_s=1.0)
        selected_long = await newer_long
        return selected_short, selected_long

    selected_short, selected_long = asyncio.run(_run())

    assert selected_short.arrival_counter < selected_long.arrival_counter
    assert selected_short.central_risk_beta == pytest.approx(0.5)
    assert selected_short.central_risk_score == pytest.approx(50.0 + 0.5 * selected_short.estimated_service_s)


@pytest.mark.parametrize("beta", [-0.1, float("inf"), float("nan")])
def test_cost_damped_risk_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="central_pull_risk_beta"):
        SuperP95Dispatcher(
            backend_urls=["http://backend-0"],
            backend_hardware_profiles=None,
            quota_every=20,
            quota_amount=1,
            threshold_ratio=0.8,
            sacrificial_load_factor=0.1,
            request_timeout_s=30.0,
            normal_routing_policy="central_pull_cost_damped_risk",
            central_pull_risk_beta=beta,
        )


def test_central_pull_still_dispatches_tail_without_normal_capacity() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy="central_pull_max_risk",
    )
    short = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }
    long = {
        "width": "1280",
        "height": "720",
        "num_inference_steps": "6",
        "num_frames": "80",
    }

    async def _run():
        await dispatcher._choose_backend("/v1/videos", short)
        dispatcher.credits = 1
        return await dispatcher._choose_backend("/v1/videos", long)

    tail = asyncio.run(_run())

    assert tail.is_sacrificial is True
    assert dispatcher.backends[0].inflight_normal_requests == 1
    assert dispatcher.backends[0].inflight_sacrificial_requests == 1


def test_tail_pack_reuses_existing_tail_backend() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=[
            "http://backend-0",
            "http://backend-1",
            "http://backend-2",
        ],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        tail_routing_mode="pack",
        normal_routing_policy="central_pull_cost_damped_risk",
        central_pull_risk_beta=0.95,
    )
    body = {
        "width": "1280",
        "height": "720",
        "num_inference_steps": "6",
        "num_frames": "80",
    }
    estimate = dispatcher._estimate_service_s("/v1/videos", body, "910B2")
    dispatcher.global_min_service_s = 1.0
    dispatcher.global_max_service_s = estimate
    dispatcher.credits = 1
    dispatcher.backends[1].sacrificial_load_s = estimate
    dispatcher.backends[1].inflight_sacrificial_requests = 1

    decision = asyncio.run(dispatcher._choose_backend("/v1/videos", body))

    assert decision.is_sacrificial is True
    assert decision.backend_index == 1
    assert dispatcher.tail_routing_mode == "pack"
    assert dispatcher.central_pull_risk_beta == pytest.approx(0.95)


def test_dispatcher_rejects_unknown_tail_routing_mode() -> None:
    with pytest.raises(ValueError, match="tail_routing_mode"):
        SuperP95Dispatcher(
            backend_urls=["http://backend-0"],
            backend_hardware_profiles=None,
            quota_every=20,
            quota_amount=1,
            threshold_ratio=0.8,
            sacrificial_load_factor=0.1,
            request_timeout_s=30.0,
            tail_routing_mode="random",
        )
