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


def test_managed_launcher_refuses_to_reuse_occupied_backend_ports(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "managed-backends"
    specs = [
        dispatcher_module.ManagedBackendSpec(
            device_id=str(index),
            port=8091 + index,
            base_url=f"http://127.0.0.1:{8091 + index}",
            hardware_profile="910B3",
        )
        for index in range(2)
    ]
    launcher = dispatcher_module.ManagedBackendLauncher(
        specs=specs,
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        backend_args=[],
        backend_env={},
        backend_scheduler="super_p95_step",
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        health_timeout_s=30.0,
        health_poll_interval_s=1.0,
        log_dir=str(log_dir),
    )
    monkeypatch.setattr(
        launcher,
        "_port_has_listener",
        lambda spec: spec.port == 8092,
    )

    with pytest.raises(RuntimeError, match=r"occupied ports: 8092"):
        launcher.start_all()

    assert launcher._processes == []
    assert not log_dir.exists()


def test_managed_launcher_starts_backend_without_intermediate_shell(tmp_path, monkeypatch) -> None:
    spec = dispatcher_module.ManagedBackendSpec(
        device_id="0",
        port=8091,
        base_url="http://127.0.0.1:8091",
        hardware_profile="910B3",
    )
    launcher = dispatcher_module.ManagedBackendLauncher(
        specs=[spec],
        model="local-model",
        backend_args=["--omni"],
        backend_env={},
        backend_scheduler="super_p95_step",
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        health_timeout_s=30.0,
        health_poll_interval_s=1.0,
        log_dir=str(tmp_path),
    )
    captured = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(dispatcher_module.subprocess, "Popen", fake_popen)

    managed = launcher._start_one(spec)
    try:
        assert captured["command"] == [
            dispatcher_module.sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            "local-model",
            "--port",
            "8091",
            "--omni",
        ]
        assert captured["kwargs"]["stdout"] is managed.log_file
        assert captured["kwargs"]["stderr"] is dispatcher_module.subprocess.STDOUT
    finally:
        managed.log_file.close()


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


def test_queue_band_risk_uses_band_beta_only_inside_configured_depth(
    monkeypatch,
) -> None:
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
        normal_routing_policy="central_pull_queue_band_risk",
        central_pull_risk_beta=0.85,
        central_pull_band_risk_beta=0.625,
        central_pull_band_min_pending=2,
        central_pull_band_max_pending=2,
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
        now_s = 60.0
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

    assert selected_short.central_risk_beta == pytest.approx(0.625)
    assert selected_short.central_queue_depth == 2
    assert selected_short.central_risk_band_active is True
    assert selected_long.central_queue_depth == 1
    assert selected_long.central_risk_band_active is False
    assert selected_short.arrival_counter < selected_long.arrival_counter
    assert dispatcher._central_pull_beta(1) == pytest.approx(0.85)
    assert dispatcher._central_pull_beta(2) == pytest.approx(0.625)
    assert dispatcher._central_pull_beta(3) == pytest.approx(0.85)


@pytest.mark.parametrize(
    ("pending_types", "expected_fraction", "expected_beta", "expected_active"),
    [
        (("short", "short", "short", "long"), 0.25, 0.4, True),
        (("short", "short", "long", "long"), 0.5, 0.625, False),
    ],
)
def test_queue_mix_risk_uses_online_long_composition(
    monkeypatch,
    pending_types,
    expected_fraction,
    expected_beta,
    expected_active,
) -> None:
    now_s = 0.0
    monkeypatch.setattr(dispatcher_module.time, "perf_counter", lambda: now_s)
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        long_request_ratio=1.5,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy="central_pull_queue_mix_risk",
        central_pull_risk_beta=0.85,
        central_pull_band_risk_beta=0.625,
        central_pull_band_min_pending=4,
        central_pull_band_max_pending=4,
        central_pull_mix_risk_beta=0.4,
        central_pull_mix_min_pending=4,
        central_pull_mix_max_pending=4,
        central_pull_mix_max_long_fraction=0.32,
    )
    bodies = {
        "short": {
            "width": "854",
            "height": "480",
            "num_inference_steps": "3",
            "num_frames": "80",
        },
        "long": {
            "width": "1280",
            "height": "720",
            "num_inference_steps": "6",
            "num_frames": "80",
        },
    }

    async def _run():
        incumbent = await dispatcher._choose_backend("/v1/videos", bodies["short"])
        tasks = [
            asyncio.create_task(dispatcher._choose_backend("/v1/videos", bodies[request_type]))
            for request_type in pending_types
        ]
        await asyncio.sleep(0)
        await dispatcher._mark_failed_response(incumbent, elapsed_s=1.0)
        await asyncio.sleep(0)
        selected_task = next(task for task in tasks if task.done())
        selected = await selected_task
        first_selected = selected
        remaining = [task for task in tasks if task is not selected_task]
        while remaining:
            await dispatcher._mark_failed_response(selected, elapsed_s=1.0)
            await asyncio.sleep(0)
            selected_task = next(task for task in remaining if task.done())
            remaining.remove(selected_task)
            selected = await selected_task
        await dispatcher._mark_failed_response(selected, elapsed_s=1.0)
        return first_selected

    selected = asyncio.run(_run())

    assert selected.central_queue_depth == 4
    assert selected.central_long_fraction == pytest.approx(expected_fraction)
    assert selected.central_risk_beta == pytest.approx(expected_beta)
    assert selected.central_mix_active is expected_active
    assert selected.central_risk_band_active is True


def test_tail_aware_release_calendar_beam_records_online_plan(
    monkeypatch,
) -> None:
    now_s = 0.0
    monkeypatch.setattr(
        dispatcher_module.time,
        "perf_counter",
        lambda: now_s,
    )
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
        central_pull_risk_beta=0.85,
        central_pull_band_risk_beta=0.625,
        central_pull_band_min_pending=2,
        central_pull_band_max_pending=3,
        central_pull_beam_horizon=2,
        central_pull_beam_width=4,
        central_pull_beam_branch_width=2,
        central_pull_beam_risk_slack_s=1000.0,
        central_pull_beam_min_pending=2,
        central_pull_beam_max_pending=3,
        central_pull_beam_history_size=8,
        central_pull_beam_candidate_cap=20,
    )

    def body(request_id, *, long=False):
        return {
            "request_id": request_id,
            "width": "1280" if long else "854",
            "height": "720" if long else "480",
            "num_inference_steps": "6" if long else "3",
            "num_frames": "80",
        }

    async def _run():
        nonlocal now_s
        incumbent_0 = await dispatcher._choose_backend(
            "/v1/videos",
            body("incumbent-0"),
        )
        incumbent_1 = await dispatcher._choose_backend(
            "/v1/videos",
            body("incumbent-1"),
        )
        tasks = [
            asyncio.create_task(
                dispatcher._choose_backend(
                    "/v1/videos",
                    body(f"pending-{index}", long=index == 2),
                )
            )
            for index in range(3)
        ]
        await asyncio.sleep(0)
        now_s = 20.0
        await dispatcher._mark_failed_response(
            incumbent_0,
            elapsed_s=20.0,
        )
        await asyncio.sleep(0)
        selected_task = next(task for task in tasks if task.done())
        selected = await selected_task

        remaining = [task for task in tasks if task is not selected_task]
        current = selected
        while remaining:
            now_s += 10.0
            await dispatcher._mark_failed_response(
                current,
                elapsed_s=10.0,
            )
            await asyncio.sleep(0)
            selected_task = next(task for task in remaining if task.done())
            remaining.remove(selected_task)
            current = await selected_task
        await dispatcher._mark_failed_response(
            current,
            elapsed_s=10.0,
        )
        await dispatcher._mark_failed_response(
            incumbent_1,
            elapsed_s=50.0,
        )
        return selected

    selected = asyncio.run(_run())

    assert selected.planner_used_beam is True
    assert selected.planner_candidate_count > 0
    assert selected.planner_predicted_before_mean_s is not None
    assert selected.planner_predicted_after_mean_s is not None
    assert selected.planner_prefix[0] == selected.request_id
    assert len(selected.planner_release_calendar_s) == 2
    assert selected.planner_active_normal_count == 1
    assert selected.planner_outstanding_tail_count == 0
    assert selected.planner_projected_cohort_size == 4
    assert dispatcher.release_calendar_beam_plans >= 1


def test_release_calendar_beam_falls_back_when_tail_eta_is_unknown(
    monkeypatch,
) -> None:
    now_s = 0.0
    monkeypatch.setattr(
        dispatcher_module.time,
        "perf_counter",
        lambda: now_s,
    )
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
        central_pull_beam_min_pending=2,
        central_pull_beam_max_pending=2,
    )
    dispatcher.backends[1].inflight_sacrificial_requests = 1
    dispatcher.backends[1].inflight_normal_requests = 1
    body = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }

    async def _run():
        incumbent = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "incumbent"},
        )
        tasks = [
            asyncio.create_task(
                dispatcher._choose_backend(
                    "/v1/videos",
                    {**body, "request_id": f"pending-{index}"},
                )
            )
            for index in range(2)
        ]
        await asyncio.sleep(0)
        await dispatcher._mark_failed_response(
            incumbent,
            elapsed_s=1.0,
        )
        await asyncio.sleep(0)
        selected_task = next(task for task in tasks if task.done())
        selected = await selected_task
        for task in tasks:
            if task is selected_task:
                continue
            await dispatcher._mark_failed_response(
                selected,
                elapsed_s=1.0,
            )
            selected = await task
        await dispatcher._mark_failed_response(
            selected,
            elapsed_s=1.0,
        )
        return await selected_task

    selected = asyncio.run(_run())

    assert selected.planner_used_beam is False
    assert selected.planner_fallback_reason == "running_tail_eta_unavailable"


def test_release_calendar_beam_treats_preemptible_tail_as_available(
    monkeypatch,
) -> None:
    now_s = 0.0
    monkeypatch.setattr(
        dispatcher_module.time,
        "perf_counter",
        lambda: now_s,
    )
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
        central_pull_beam_min_pending=2,
        central_pull_beam_max_pending=2,
        running_tail_preemptible_for_normal=True,
    )
    dispatcher.backends[1].inflight_sacrificial_requests = 1
    body = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }

    async def _run():
        incumbent = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "incumbent"},
        )
        tasks = [
            asyncio.create_task(
                dispatcher._choose_backend(
                    "/v1/videos",
                    {**body, "request_id": f"pending-{index}"},
                )
            )
            for index in range(3)
        ]
        await asyncio.sleep(0)
        running_on_tail_backend_task = next(task for task in tasks if task.done())
        running_on_tail_backend = await running_on_tail_backend_task
        queued_tasks = [
            task for task in tasks if task is not running_on_tail_backend_task
        ]
        await dispatcher._mark_failed_response(
            incumbent,
            elapsed_s=1.0,
        )
        await asyncio.sleep(0)
        selected_task = next(task for task in queued_tasks if task.done())
        selected = await selected_task
        await dispatcher._mark_failed_response(
            running_on_tail_backend,
            elapsed_s=1.0,
        )
        remaining_task = next(task for task in queued_tasks if task is not selected_task)
        remaining = await remaining_task
        await dispatcher._mark_failed_response(
            selected,
            elapsed_s=1.0,
        )
        await dispatcher._mark_failed_response(
            remaining,
            elapsed_s=1.0,
        )
        return selected

    selected = asyncio.run(_run())

    assert selected.planner_used_beam is True
    assert selected.planner_fallback_reason is None
    assert dispatcher.release_calendar_beam_plans >= 1


def test_release_calendar_busy_epoch_excludes_quiescent_warmup_history(
    monkeypatch,
) -> None:
    now_s = 0.0
    monkeypatch.setattr(
        dispatcher_module.time,
        "perf_counter",
        lambda: now_s,
    )
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
    )
    body = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }

    async def _run():
        nonlocal now_s
        warmup = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "warmup"},
        )
        now_s = 10.0
        await dispatcher._apply_response_feedback(
            warmup,
            dispatcher_module.httpx.Headers(),
            elapsed_s=10.0,
        )
        assert list(dispatcher._completed_latency_history_s) == [10.0]

        now_s = 20.0
        measured = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "measured"},
        )
        assert not dispatcher._completed_latency_history_s
        assert dispatcher.release_calendar_busy_epoch == 2
        assert dispatcher.arrival_counter == 2
        await dispatcher._mark_failed_response(
            measured,
            elapsed_s=1.0,
        )

    asyncio.run(_run())


def test_release_calendar_busy_epoch_does_not_reset_while_work_is_active(
    monkeypatch,
) -> None:
    now_s = 0.0
    monkeypatch.setattr(
        dispatcher_module.time,
        "perf_counter",
        lambda: now_s,
    )
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
    )
    body = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }

    async def _run():
        nonlocal now_s
        first = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "first"},
        )
        second = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "second"},
        )
        now_s = 10.0
        await dispatcher._apply_response_feedback(
            first,
            dispatcher_module.httpx.Headers(),
            elapsed_s=10.0,
        )
        assert list(dispatcher._completed_latency_history_s) == [10.0]

        now_s = 11.0
        third = await dispatcher._choose_backend(
            "/v1/videos",
            {**body, "request_id": "third"},
        )
        assert list(dispatcher._completed_latency_history_s) == [10.0]
        assert dispatcher.release_calendar_busy_epoch == 1
        await dispatcher._mark_failed_response(second, elapsed_s=1.0)
        await dispatcher._mark_failed_response(third, elapsed_s=1.0)

    asyncio.run(_run())


def test_failed_http_response_does_not_enter_release_calendar_history() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
    )

    class FailingClient:
        async def post(self, *args, **kwargs):
            del args, kwargs
            return dispatcher_module.httpx.Response(
                503,
                json={"error": "unavailable"},
            )

    async def _run():
        dispatcher._client = FailingClient()
        response = await dispatcher.dispatch_json(
            "/v1/videos",
            {
                "request_id": "failed",
                "width": 854,
                "height": 480,
                "num_inference_steps": 3,
                "num_frames": 80,
            },
            {},
        )
        assert response.status_code == 503

    asyncio.run(_run())

    assert not dispatcher._completed_latency_history_s
    assert dispatcher.backends[0].active_normal_prediction is None
    assert dispatcher.backends[0].inflight_normal_requests == 0


def test_cancelled_post_releases_release_calendar_state() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy=("central_pull_tail_aware_release_calendar_beam"),
    )

    class BlockingClient:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def post(self, *args, **kwargs):
            del args, kwargs
            self.entered.set()
            await asyncio.Future()

    async def _run():
        client = BlockingClient()
        dispatcher._client = client
        task = asyncio.create_task(
            dispatcher.dispatch_json(
                "/v1/videos",
                {
                    "request_id": "cancelled",
                    "width": 854,
                    "height": 480,
                    "num_inference_steps": 3,
                    "num_frames": 80,
                },
                {},
            )
        )
        await client.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())

    backend = dispatcher.backends[0]
    assert backend.inflight_normal_requests == 0
    assert backend.normal_load_s == pytest.approx(0.0)
    assert backend.active_normal_prediction is None
    assert not dispatcher._completed_latency_history_s


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


def test_protected_drain_tail_waits_for_target_backend_normal(monkeypatch) -> None:
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
        tail_dispatch_mode="protected_drain",
    )
    short = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }
    long = {
        "request_id": "tail-0",
        "width": "1280",
        "height": "720",
        "num_inference_steps": "6",
        "num_frames": "80",
    }

    async def _run():
        nonlocal now_s
        normal = await dispatcher._choose_backend("/v1/videos", short)
        dispatcher.credits = 1
        tail_task = asyncio.create_task(dispatcher._choose_backend("/v1/videos", long))
        await asyncio.sleep(0)

        assert not tail_task.done()
        assert len(dispatcher._pending_tail_dispatches) == 1
        assert dispatcher.backends[0].inflight_sacrificial_requests == 1

        now_s = 25.0
        await dispatcher._mark_failed_response(normal, elapsed_s=1.0)
        tail = await tail_task
        await dispatcher._mark_failed_response(tail, elapsed_s=1.0)
        return tail

    tail = asyncio.run(_run())

    assert tail.is_sacrificial is True
    assert tail.central_wait_s == pytest.approx(25.0)
    assert dispatcher.tail_gate_releases == 1
    assert dispatcher.tail_gate_wait_total_s == pytest.approx(25.0)
    assert dispatcher.backends[0].inflight_sacrificial_requests == 0


def test_protected_drain_tail_yields_to_pending_central_normal() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=None,
        quota_every=1000,
        quota_amount=0,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
        normal_routing_policy="central_pull_cost_damped_risk",
        tail_dispatch_mode="protected_drain",
    )
    short = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }
    long = {
        "request_id": "tail-0",
        "width": "1280",
        "height": "720",
        "num_inference_steps": "6",
        "num_frames": "80",
    }

    async def _run():
        incumbent = await dispatcher._choose_backend("/v1/videos", short)
        queued_normal_task = asyncio.create_task(dispatcher._choose_backend("/v1/videos", short))
        await asyncio.sleep(0)
        dispatcher.credits = 1
        tail_task = asyncio.create_task(dispatcher._choose_backend("/v1/videos", long))
        await asyncio.sleep(0)

        await dispatcher._mark_failed_response(incumbent, elapsed_s=1.0)
        queued_normal = await queued_normal_task
        assert not tail_task.done()
        assert len(dispatcher._pending_normal_dispatches) == 0
        assert len(dispatcher._pending_tail_dispatches) == 1

        await dispatcher._mark_failed_response(queued_normal, elapsed_s=1.0)
        tail = await tail_task
        await dispatcher._mark_failed_response(tail, elapsed_s=1.0)
        return tail

    tail = asyncio.run(_run())

    assert tail.is_sacrificial is True
    assert dispatcher.tail_gate_releases == 1
    assert dispatcher.central_pull_dispatches == 2


def test_tail_idle_backfill_parallelizes_protected_tail_drain() -> None:
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
        tail_dispatch_mode="protected_drain",
        tail_idle_backfill=True,
        normal_routing_policy="central_pull_cost_damped_risk",
    )
    short = {
        "width": "854",
        "height": "480",
        "num_inference_steps": "3",
        "num_frames": "80",
    }

    async def _run():
        normals = [
            await dispatcher._choose_backend(
                "/v1/videos",
                {**short, "request_id": f"normal-{index}"},
            )
            for index in range(3)
        ]
        assert {decision.backend_index for decision in normals} == {
            0,
            1,
            2,
        }

        tail_tasks = []
        for index in range(3):
            dispatcher.credits = 1
            tail_tasks.append(
                asyncio.create_task(
                    dispatcher._choose_backend(
                        "/v1/videos",
                        {
                            "request_id": f"tail-{index}",
                            "width": "1280",
                            "height": "720",
                            "num_inference_steps": "6",
                            "num_frames": "80",
                        },
                    )
                )
            )
            await asyncio.sleep(0)

        assert len(dispatcher._pending_tail_dispatches) == 3
        assert {pending.decision.backend_index for pending in dispatcher._pending_tail_dispatches} == {0}

        for completed_count, normal in enumerate(normals, start=1):
            await dispatcher._mark_failed_response(
                normal,
                elapsed_s=1.0,
            )
            await asyncio.sleep(0)
            assert sum(task.done() for task in tail_tasks) == completed_count

        tails = await asyncio.gather(*tail_tasks)
        assert {decision.backend_index for decision in tails} == {0, 1, 2}
        assert [decision.request_id for decision in tails] == [
            "tail-0",
            "tail-1",
            "tail-2",
        ]
        assert len(dispatcher._pending_tail_dispatches) == 0
        for tail in tails:
            await dispatcher._mark_failed_response(tail, elapsed_s=1.0)
        return tails

    tails = asyncio.run(_run())

    assert all(tail.is_sacrificial for tail in tails)
    assert dispatcher.tail_gate_releases == 3
    assert dispatcher.tail_idle_backfill_reassignments == 2
    assert all(backend.inflight_sacrificial_requests == 0 for backend in dispatcher.backends)


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


def test_dispatcher_rejects_unknown_tail_dispatch_mode() -> None:
    with pytest.raises(ValueError, match="tail_dispatch_mode"):
        SuperP95Dispatcher(
            backend_urls=["http://backend-0"],
            backend_hardware_profiles=None,
            quota_every=20,
            quota_amount=1,
            threshold_ratio=0.8,
            sacrificial_load_factor=0.1,
            request_timeout_s=30.0,
            tail_dispatch_mode="random",
        )


@pytest.mark.parametrize(
    ("tail_routing_mode", "tail_dispatch_mode"),
    [
        ("spread", "protected_drain"),
        ("pack", "immediate"),
    ],
)
def test_tail_idle_backfill_requires_pack_and_protected_drain(
    tail_routing_mode: str,
    tail_dispatch_mode: str,
) -> None:
    with pytest.raises(ValueError, match="tail_idle_backfill requires"):
        SuperP95Dispatcher(
            backend_urls=["http://backend-0"],
            backend_hardware_profiles=None,
            quota_every=20,
            quota_amount=1,
            threshold_ratio=0.8,
            sacrificial_load_factor=0.1,
            request_timeout_s=30.0,
            tail_routing_mode=tail_routing_mode,
            tail_dispatch_mode=tail_dispatch_mode,
            tail_idle_backfill=True,
        )
