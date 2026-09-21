# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests: preemption must preserve Wan's per-request solver state."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    build_wan_scheduler,
    get_wan22_pre_process_func,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _sampling(*, steps=4, seed=42, solver="unipc", guidance=4.0, guidance_2=1.0, outputs=1):
    # Only the fields consumed by the pipeline; no device/model loading needed.
    return SimpleNamespace(
        height=32,
        width=48,
        num_frames=8,
        num_inference_steps=steps,
        output_type="latent",
        num_outputs_per_prompt=outputs,
        guidance_scale=guidance,
        guidance_scale_provided=True,
        guidance_scale_2=guidance_2,
        guidance_scale_2_provided=True,
        boundary_ratio=None,
        max_sequence_length=4,
        generator=torch.Generator().manual_seed(seed),
        latents=None,
        extra_args={"sample_solver": solver},
    )


@pytest.fixture
def pipe(monkeypatch):
    from vllm_omni.diffusion.models.wan2_2 import pipeline_wan2_2 as module

    instance = Wan22Pipeline.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(instance)
    instance.device = torch.device("cpu")
    instance.is_dmd = False
    instance.expand_timesteps = False
    instance.boundary_ratio = 0.875
    instance.transformer = SimpleNamespace(dtype=torch.float32, label="high")
    instance.transformer_2 = SimpleNamespace(dtype=torch.float32, label="low")
    instance.transformer_config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=2)
    instance.vae_scale_factor_spatial = 8
    instance.vae_scale_factor_temporal = 4
    instance._sample_solver = "unipc"
    instance._flow_shift = 5.0
    instance.scheduler = build_wan_scheduler("unipc", 5.0)
    instance.od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        streaming_output=False,
        flow_shift=5.0,
    )
    instance.encode_prompt = MagicMock(
        side_effect=lambda **kw: (
            torch.ones(kw["num_videos_per_prompt"], 4, 3),
            torch.zeros(kw["num_videos_per_prompt"], 4, 3) if kw["do_classifier_free_guidance"] else None,
        )
    )
    instance.record_denoise_step = MagicMock()
    instance.progress_bar = lambda **kw: nullcontext(SimpleNamespace(update=lambda: None))
    instance.predict_noise_maybe_with_cfg = MagicMock(
        side_effect=lambda **kw: (
            kw["positive_kwargs"]["hidden_states"] * 0.1
            + (0.2 if kw["positive_kwargs"]["current_model"].label == "high" else 0.3)
            + kw["true_cfg_scale"] * 0.01
        )
    )
    instance.scheduler_step_maybe_with_cfg = lambda noise, t, latents, do_cfg, per_request_scheduler=None: (
        per_request_scheduler if per_request_scheduler is not None else instance.scheduler
    ).step(noise, t, latents, return_dict=False)[0]
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: False)
    return instance


def _state(pipe, request_id="req", **kwargs):
    state = StepRequestState(request_id=request_id, sampling=_sampling(**kwargs), prompt="a mountain")
    return pipe.prepare_encode(state)


def _step(pipe, state):
    noise = pipe.denoise_step(SimpleNamespace(states=[state]))
    pipe.step_scheduler(state, noise)


@pytest.mark.parametrize("solver", ["unipc", "euler"])
def test_interleaving_preserves_solver_history_and_output(pipe, solver):
    reference = _state(pipe, "reference", solver=solver)
    while not reference.denoise_completed:
        _step(pipe, reference)

    tail = _state(pipe, "tail", solver=solver)
    _step(pipe, tail)
    saved_latents = tail.latents.clone()
    saved_index = tail.scheduler.step_index
    normal = _state(pipe, "normal", steps=3, seed=7, solver=solver)
    assert normal.scheduler is not tail.scheduler
    assert normal.scheduler is not pipe.scheduler
    while not normal.denoise_completed:
        _step(pipe, normal)
    torch.testing.assert_close(tail.latents, saved_latents)
    assert tail.scheduler.step_index == saved_index
    while not tail.denoise_completed:
        _step(pipe, tail)
    torch.testing.assert_close(tail.latents, reference.latents, rtol=0, atol=0)


@pytest.mark.parametrize("solver", ["unipc", "euler"])
def test_step_mode_matches_request_mode_denoising(pipe, solver):
    state = _state(pipe, solver=solver, outputs=2)
    original_latents = state.latents.clone()
    pipe.scheduler = build_wan_scheduler(solver, 5.0)
    if solver == "unipc":
        pipe.scheduler.set_timesteps(4, device="cpu", shift=5.0)
    else:
        pipe.scheduler.set_timesteps(4, device="cpu")
    expected = pipe.diffuse(
        latents=original_latents,
        timesteps=pipe.scheduler.timesteps,
        prompt_embeds=state.prompt_embeds,
        negative_prompt_embeds=state.negative_prompt_embeds,
        **state.extra["wan_t2v"],
    )
    while not state.denoise_completed:
        _step(pipe, state)
    torch.testing.assert_close(state.latents, expected, rtol=0, atol=0)
    assert state.latents.shape == (2, 2, 3, 4, 6)
    output = pipe.post_decode(state)
    torch.testing.assert_close(output.output, expected)
    assert output.media is None


@pytest.mark.parametrize("timestep,expert,cfg,scale", [(900.0, "high", True, 4.0), (800.0, "low", False, 1.0)])
def test_expert_boundary_and_cfg_are_request_local(pipe, timestep, expert, cfg, scale):
    state = _state(pipe)
    state.timesteps = torch.tensor([timestep])
    _state(pipe, "other", guidance=9.0, guidance_2=7.0)
    pipe.denoise_step(SimpleNamespace(states=[state]))
    call = pipe.predict_noise_maybe_with_cfg.call_args.kwargs
    assert call["positive_kwargs"]["current_model"].label == expert
    assert call["do_true_cfg"] is cfg
    assert call["true_cfg_scale"] == scale
    assert state.do_true_cfg is cfg


def test_prepare_respects_request_solver_shift_and_does_not_mutate_guidance_flags(pipe):
    sampling = _sampling(solver="euler")
    sampling.guidance_scale_provided = False
    sampling.guidance_scale_2_provided = False
    sampling.extra_args["flow_shift"] = 12.0
    state = pipe.prepare_encode(StepRequestState("req", sampling, "a mountain"))
    assert state.scheduler._shift == 12.0
    assert state.extra["wan_t2v"]["guidance_low"] == 4.0
    assert not sampling.guidance_scale_provided
    assert not sampling.guidance_scale_2_provided


@pytest.mark.parametrize("missing_expert,timestep", [("transformer", 900.0), ("transformer_2", 800.0)])
def test_single_expert_deployment_uses_remaining_transformer(pipe, missing_expert, timestep):
    setattr(pipe, missing_expert, None)
    state = _state(pipe)
    state.timesteps = torch.tensor([timestep])
    pipe.denoise_step(SimpleNamespace(states=[state]))
    assert pipe.predict_noise_maybe_with_cfg.call_args.kwargs["positive_kwargs"]["current_model"] is not None


@pytest.mark.parametrize("feature", ["dmd", "i2v", "pp", "streaming", "multimodal"])
def test_unsupported_modes_fail_explicitly(pipe, feature):
    prompt = "a mountain"
    if feature == "dmd":
        pipe.is_dmd = True
    elif feature == "i2v":
        pipe.expand_timesteps = True
    elif feature == "pp":
        pipe.od_config.parallel_config.pipeline_parallel_size = 2
    elif feature == "streaming":
        pipe.od_config.streaming_output = True
    else:
        prompt = {"prompt": prompt, "multi_modal_data": {"image": object()}}
    with pytest.raises(ValueError):
        pipe.prepare_encode(StepRequestState("req", _sampling(), prompt))
    pipe.encode_prompt.assert_not_called()


def test_multiple_requests_per_step_are_rejected(pipe):
    with pytest.raises(ValueError, match="single request"):
        pipe.denoise_step(SimpleNamespace(states=[_state(pipe, "a"), _state(pipe, "b")]))


def test_upstream_input_batch_can_switch_between_requests(pipe):
    tail = _state(pipe, "tail", outputs=2)
    batch = InputBatch.make_batch([tail])
    pipe.step_scheduler(tail, pipe.denoise_step(batch))
    normal = _state(pipe, "normal", steps=3)
    batch = InputBatch.make_batch([normal], cached_batch=batch)
    while not normal.denoise_completed:
        pipe.step_scheduler(normal, pipe.denoise_step(batch))
        if not normal.denoise_completed:
            batch = InputBatch.make_batch([normal], cached_batch=batch)
    batch = InputBatch.make_batch([tail], cached_batch=batch)
    assert batch.request_ids == ["tail"]
    assert batch.latents.shape[0] == 2
    torch.testing.assert_close(batch.latents, tail.latents)
    while not tail.denoise_completed:
        pipe.step_scheduler(tail, pipe.denoise_step(batch))
        if not tail.denoise_completed:
            batch = InputBatch.make_batch([tail], cached_batch=batch)
    assert tail.step_index == 4


@pytest.mark.parametrize("output_owner", [True, False])
def test_decode_preserves_vae_normalization_and_rank_ownership(pipe, output_owner):
    state = _state(pipe)
    state.sampling.output_type = "np"
    decoded = torch.ones(1, 3, 9, 32, 48) if output_owner else torch.empty(0)
    pipe.vae = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(z_dim=2, latents_mean=[1.0, -1.0], latents_std=[2.0, 3.0]),
        decode=MagicMock(return_value=(decoded,)),
    )
    result = pipe.post_decode(state)
    actual = pipe.vae.decode.call_args.args[0]
    expected = state.latents * torch.tensor([2.0, 3.0]).view(1, 2, 1, 1, 1)
    expected += torch.tensor([1.0, -1.0]).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(actual, expected)
    if output_owner:
        assert result.output is None
        torch.testing.assert_close(result.media.video.tensor, decoded)
    else:
        assert result.media is None
        assert result.output.numel() == 0


def test_invalid_preencoded_output_fails_before_encoding(pipe):
    sampling = _sampling()
    sampling.extra_args["preencode_mp4"] = True
    with pytest.raises(ValueError, match="output_type"):
        pipe.prepare_encode(StepRequestState("req", sampling, "a mountain"))
    pipe.encode_prompt.assert_not_called()


@pytest.mark.parametrize("solver", ["unipc", "euler"])
@pytest.mark.parametrize("outputs", [1, 2])
def test_step_preparation_matches_full_forward(pipe, solver, outputs):
    request = SimpleNamespace(prompt="a mountain", sampling_params=_sampling(solver=solver, outputs=outputs))
    expected = pipe.forward(DiffusionRequestBatch([request]))[0].output
    state = _state(pipe, solver=solver, outputs=outputs)
    while not state.denoise_completed:
        _step(pipe, state)
    torch.testing.assert_close(state.latents, expected, rtol=0, atol=0)


def test_precomputed_embeddings_follow_request_batch_collation(pipe):
    prompt_embeds = torch.ones(4, 3)
    negative_embeds = torch.zeros(4, 3)
    state = StepRequestState(
        "req",
        _sampling(outputs=2),
        {
            "additional_information": {
                "prompt_embeds": [prompt_embeds],
                "negative_prompt_embeds": [negative_embeds],
            }
        },
    )
    pipe.prepare_encode(state)
    assert state.prompt_embeds.shape == (2, 4, 3)
    assert state.negative_prompt_embeds.shape == (2, 4, 3)
    pipe.encode_prompt.assert_not_called()


@pytest.mark.parametrize(
    "step_mode,use_step,steps,expected",
    [
        (True, True, None, 40),
        (False, True, None, None),
        (True, False, None, 40),
        (True, True, 7, 7),
        (True, True, 2, 2),
    ],
)
def test_preprocessor_materializes_only_step_request_defaults(step_mode, use_step, steps, expected):
    sampling = _sampling(steps=steps)
    sampling.num_frames = 1  # Native startup warmup must remain a single frame.
    request = SimpleNamespace(prompt="a mountain", sampling_params=sampling, use_step_execution=use_step)
    preprocess = get_wan22_pre_process_func(SimpleNamespace(step_execution=step_mode))
    assert preprocess(request) is request
    assert sampling.num_inference_steps == expected
    assert sampling.num_frames == 1


def test_direct_step_preparation_materializes_default_steps(pipe):
    state = _state(pipe, steps=None)
    assert state.sampling.num_inference_steps == 40
    assert state.total_steps == 40


@pytest.mark.parametrize(
    "field,value",
    [
        ("step_index", 1),
        ("step_index", -1),
        ("timesteps", torch.tensor([900.0, 800.0])),
        ("sigmas", [1.0, 0.5]),
        ("num_inference_steps", 0),
        ("num_inference_steps", True),
    ],
)
@pytest.mark.parametrize("through_preprocessor", [True, False])
def test_external_partial_schedules_fail_before_encoding(pipe, field, value, through_preprocessor):
    sampling = _sampling()
    setattr(sampling, field, value)
    with pytest.raises(ValueError):
        if through_preprocessor:
            preprocess = get_wan22_pre_process_func(SimpleNamespace(step_execution=True))
            preprocess(SimpleNamespace(prompt="a mountain", sampling_params=sampling))
        else:
            pipe.prepare_encode(StepRequestState("req", sampling, "a mountain"))
    pipe.encode_prompt.assert_not_called()


def test_prepare_rejects_reinitializing_advanced_cached_state(pipe):
    state = _state(pipe)
    _step(pipe, state)
    with pytest.raises(ValueError, match="new request state"):
        pipe.prepare_encode(state)
