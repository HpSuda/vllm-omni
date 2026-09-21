# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.sched import StepScheduler, SuperP95StepScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("name,expected", [("super_p95_step", SuperP95StepScheduler), ("step_baseline", StepScheduler)])
def test_opt_in_selects_native_step_execution_before_executor_setup(monkeypatch, name, expected):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_SCHEDULER", name)
    monkeypatch.delenv("SUPER_P95_QWEN_SMALL_BATCH2", raising=False)
    engine = object.__new__(DiffusionEngine)
    config = SimpleNamespace(step_execution=False, streaming_output=False, max_num_seqs=1)
    engine.execution_mode = engine._resolve_execution_mode(config)
    assert config.step_execution
    assert engine.execution_mode == DiffusionExecutionMode.STEP_BATCH
    engine._init_scheduler(config)
    assert isinstance(engine.scheduler, expected)


@pytest.mark.parametrize("model,capacity", [("QwenImagePipeline", 2), ("WanPipeline", 1)])
def test_qwen_small_batch_capacity_is_resolved_before_worker_construction(monkeypatch, model, capacity):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_SCHEDULER", "super_p95_step")
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", "1")
    engine = object.__new__(DiffusionEngine)
    config = SimpleNamespace(step_execution=False, streaming_output=False, max_num_seqs=1, model_class_name=model)
    engine._resolve_execution_mode(config)
    assert config.max_num_seqs == capacity


def test_unselected_policy_does_not_change_native_step_scheduler(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_SCHEDULER", raising=False)
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", "1")
    engine = object.__new__(DiffusionEngine)
    config = SimpleNamespace(
        step_execution=True, streaming_output=False, max_num_seqs=1, model_class_name="QwenImagePipeline"
    )
    engine.execution_mode = engine._resolve_execution_mode(config)
    engine._init_scheduler(config)
    assert type(engine.scheduler) is StepScheduler
    assert config.max_num_seqs == 1


@pytest.mark.parametrize(
    "model,capacity,batch2", [("Wan22Pipeline", 2, "1"), ("QwenImagePipeline", 2, "0"), ("QwenImagePipeline", 3, "1")]
)
def test_unsupported_profile_batch_capacity_fails_before_worker_setup(monkeypatch, model, capacity, batch2):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_SCHEDULER", "super_p95_step")
    monkeypatch.setenv("SUPER_P95_QWEN_SMALL_BATCH2", batch2)
    engine = object.__new__(DiffusionEngine)
    config = SimpleNamespace(
        step_execution=False, streaming_output=False, max_num_seqs=capacity, model_class_name=model
    )
    with pytest.raises(ValueError, match="queue length is independent"):
        engine._resolve_execution_mode(config)


def test_paged_kv_rejected_before_worker_creation(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_SCHEDULER", "super_p95_step")
    engine = object.__new__(DiffusionEngine)
    config = SimpleNamespace(
        step_execution=False, streaming_output=False, max_num_seqs=1, diffusion_kv_mode="paged_scheduler"
    )
    with pytest.raises(ValueError, match="paged-KV admission/preemption is not supported"):
        engine._resolve_execution_mode(config)
