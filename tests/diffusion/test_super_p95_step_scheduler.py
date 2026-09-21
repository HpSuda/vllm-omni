# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import super_p95_step_scheduler as policy_module
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus, StepBatchSamplingParamsKey
from vllm_omni.diffusion.sched.super_p95_step_scheduler import SuperP95StepScheduler
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(request_id, *, tail=False, cost=20.0, **overrides):
    # Use a lightweight request envelope: the scheduler owns no tensor work.
    params = {field.name: field.default for field in fields(StepBatchSamplingParamsKey)}
    params.update(
        num_inference_steps=4,
        step_index=None,
        sigmas=None,
        timesteps=None,
        width=512,
        height=512,
        resolution=None,
        seed=42,
        generator_device="cpu",
        lora_request=None,
        extra_args={"_super_p95_sacrificial": tail, "_super_p95_estimated_service_s": cost},
    )
    params.update(overrides)
    # Avoid OmniDiffusionRequest.__post_init__, which resolves model sampling
    # defaults. All scheduler-facing state is explicit in this unit fixture.
    request = object.__new__(OmniDiffusionRequest)
    request.request_id = request_id
    request.prompt = "test prompt"
    request.sampling_params = SimpleNamespace(**params)
    request.diffusion_kv_requests = None
    request.batch_compatibility_key = None
    request.kv_transfer_params = None
    request.scheduler_queue_wait_ms = None
    return request


def _scheduler(*, capacity=1, model="QwenImagePipeline"):
    scheduler = SuperP95StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=capacity, model_class_name=model))
    return scheduler


def _advance(scheduler, output, *, step=4, finished=True):
    outputs = [RunnerOutput(request_id=rid, step_index=step, finished=finished) for rid in output.scheduled_request_ids]
    return scheduler.update_from_output(output, BatchRunnerOutput(outputs))


def test_normal_fifo_and_tail_newest_first():
    scheduler = _scheduler()
    for name, tail in (("tail0", True), ("normal0", False), ("tail1", True), ("normal1", False)):
        scheduler.add_request(_request(name, tail=tail))
    for expected in ("normal0", "normal1", "tail1", "tail0"):
        output = scheduler.schedule()
        assert output.scheduled_request_ids == [expected]
        assert _advance(scheduler, output) == {expected}
        scheduler.pop_request_state(expected)
    assert not scheduler.has_requests()


def test_normal_preempts_tail_and_resumes_native_cached_state():
    scheduler = _scheduler()
    scheduler.add_request(_request("tail", tail=True))
    output = scheduler.schedule()
    _advance(scheduler, output, step=1, finished=False)
    scheduler.add_request(_request("normal"))
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["normal"]
    assert scheduler.get_request_state("tail").status == DiffusionRequestStatus.PREEMPTED
    assert scheduler.get_load_snapshot().sacrificial_load_s == 15.0
    _advance(scheduler, output)
    resumed = scheduler.schedule()
    assert resumed.scheduled_cached_reqs.request_ids == ["tail"]
    assert resumed.scheduled_new_reqs == []
    assert scheduler.get_request_state("tail").req.sampling_params.step_index == 1
    _advance(scheduler, resumed)
    assert scheduler.get_load_snapshot().sacrificial_load_s == 0.0


def test_running_normal_is_not_preempted():
    scheduler = _scheduler()
    scheduler.add_request(_request("first", cost=100))
    _advance(scheduler, scheduler.schedule(), step=1, finished=False)
    scheduler.add_request(_request("short", cost=1))
    assert scheduler.schedule().scheduled_request_ids == ["first"]


def test_newer_tail_does_not_preempt_running_tail():
    scheduler = _scheduler()
    scheduler.add_request(_request("tail0", tail=True))
    _advance(scheduler, scheduler.schedule(), step=1, finished=False)
    scheduler.add_request(_request("tail1", tail=True))
    assert scheduler.schedule().scheduled_request_ids == ["tail0"]


def test_native_kv_loading_blocks_preemption():
    scheduler = _scheduler()
    scheduler.add_request(_request("tail", tail=True))
    _advance(scheduler, scheduler.schedule(), step=1, finished=False)
    scheduler._kv_loading_request_ids.add("tail")
    scheduler.add_request(_request("normal"))
    assert scheduler.schedule().scheduled_request_ids == ["tail"]
    scheduler._kv_loading_request_ids.clear()
    assert scheduler.schedule().scheduled_request_ids == ["normal"]


def test_abort_removes_waiting_and_load():
    scheduler = _scheduler()
    scheduler.add_request(_request("aborted", tail=True))
    scheduler.finish_requests("aborted", DiffusionRequestStatus.FINISHED_ABORTED)
    assert scheduler.schedule().scheduled_request_ids == []
    assert scheduler.get_load_snapshot().sacrificial_load_s == 0.0
    scheduler.pop_request_state("aborted")
    assert "aborted" not in scheduler._metadata
    assert "aborted" not in scheduler._request_progress


def test_duplicate_identity_is_rejected_without_overwriting_metadata():
    scheduler = _scheduler()
    scheduler.add_request(_request("same", cost=20))
    with pytest.raises(ValueError, match="already active"):
        scheduler.add_request(_request("same", tail=True, cost=100))
    assert scheduler.get_load_snapshot().normal_load_s == 20.0


def test_batch2_searches_window_without_mixing_incompatible_shapes(monkeypatch):
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", "1")
    scheduler = _scheduler(capacity=2)
    scheduler.add_request(_request("first"))
    scheduler.add_request(_request("large", width=1024, height=1024))
    scheduler.add_request(_request("partner"))
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["first", "partner"]
    assert list(scheduler._waiting) == ["large"]
    _advance(scheduler, output)
    assert scheduler.schedule().scheduled_request_ids == ["large"]


@pytest.mark.parametrize("override", [{"quality": "lossless"}, {"num_inference_steps": 5}, {"lora_scale": 0.5}])
def test_batch2_respects_native_key_and_step_count(monkeypatch, override):
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", "1")
    scheduler = _scheduler(capacity=2)
    scheduler.add_request(_request("first"))
    scheduler.add_request(_request("incompatible", **override))
    assert scheduler.schedule().scheduled_request_ids == ["first"]


@pytest.mark.parametrize(
    "capacity,model,tail", [(1, "QwenImagePipeline", False), (2, "WanPipeline", False), (2, "QwenImagePipeline", True)]
)
def test_batch2_never_exceeds_capacity_or_batches_wan_or_tail(monkeypatch, capacity, model, tail):
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", "1")
    scheduler = _scheduler(capacity=capacity, model=model)
    scheduler.add_request(_request("a", tail=tail))
    scheduler.add_request(_request("b", tail=tail))
    assert len(scheduler.schedule().scheduled_request_ids) == 1


def test_batch2_has_no_midwave_admission_and_keeps_partial_completion(monkeypatch):
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", "1")
    scheduler = _scheduler(capacity=2)
    scheduler.add_request(_request("first"))
    output = scheduler.schedule()
    _advance(scheduler, output, step=1, finished=False)
    scheduler.add_request(_request("later"))
    assert scheduler.schedule().scheduled_request_ids == ["first"]
    _advance(scheduler, output)
    scheduler.pop_request_state("first")
    scheduler.add_request(_request("partner"))
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["later", "partner"]
    completed = scheduler.update_from_output(
        output,
        BatchRunnerOutput(
            [RunnerOutput("later", step_index=4, finished=True), RunnerOutput("partner", step_index=1, finished=False)]
        ),
    )
    assert completed == {"later"}
    assert scheduler.schedule().scheduled_request_ids == ["partner"]


def test_missing_output_uses_native_error_cleanup():
    scheduler = _scheduler()
    scheduler.add_request(_request("missing"))
    output = scheduler.schedule()
    assert scheduler.update_from_output(output, BatchRunnerOutput([])) == {"missing"}
    assert scheduler.get_request_state("missing").status == DiffusionRequestStatus.FINISHED_ERROR
    assert scheduler.get_load_snapshot().normal_load_s == 0.0


def test_trace_retains_native_request_identity_across_preempt_and_resume(monkeypatch):
    events = []
    monkeypatch.setattr(policy_module, "write_trace_event", lambda path, event, **data: events.append((event, data)))
    scheduler = _scheduler()
    scheduler.add_request(_request("tail", tail=True))
    _advance(scheduler, scheduler.schedule(), step=1, finished=False)
    scheduler.add_request(_request("normal"))
    _advance(scheduler, scheduler.schedule())
    _advance(scheduler, scheduler.schedule())
    tail_events = [(event, data) for event, data in events if data["request_id"] == "tail"]
    assert [event for event, _ in tail_events] == [
        "scheduler_enqueue",
        "scheduler_select",
        "scheduler_preempt",
        "scheduler_select",
        "scheduler_complete",
    ]
    assert tail_events[3][1]["selection_kind"] == "resume"
    assert tail_events[3][1]["completed_steps"] == 1


@pytest.mark.parametrize("mode", [DiffusionKVCacheMode.PAGED_SCHEDULER, "paged_scheduler", "paged_worker_local"])
def test_paged_kv_is_explicitly_rejected_even_for_injected_scheduler(mode):
    scheduler = SuperP95StepScheduler()
    with pytest.raises(ValueError, match="dense_legacy"):
        scheduler.initialize(SimpleNamespace(diffusion_kv_mode=mode))


def test_native_kv_transfer_is_rejected():
    scheduler = SuperP95StepScheduler()
    with pytest.raises(ValueError, match="without native KV transfer"):
        scheduler.initialize(SimpleNamespace(kv_transfer_config=object()))
