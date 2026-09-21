# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-private T2V state for the upstream step-execution runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def prepare_wan_t2v_step_sampling(sampling: OmniDiffusionSamplingParams) -> None:
    """Materialize the forward default before native step-scheduler admission.

    External partial-schedule restarts are not equivalent to resuming a cached
    request: UniPC needs its derivative history, not only an initial latent.
    """
    if getattr(sampling, "timesteps", None) is not None or getattr(sampling, "sigmas", None) is not None:
        raise ValueError("Wan T2V step execution does not support custom timesteps or sigmas.")
    if getattr(sampling, "step_index", None) not in (None, 0):
        raise ValueError(
            "Wan T2V step execution requires initial step_index=0; only cached in-flight requests can resume."
        )
    if sampling.num_inference_steps is None:
        sampling.num_inference_steps = 40
    if (
        isinstance(sampling.num_inference_steps, bool)
        or not isinstance(sampling.num_inference_steps, int)
        or sampling.num_inference_steps <= 0
    ):
        raise ValueError("Wan T2V num_inference_steps must be a positive integer.")


class Wan22StepExecutionMixin:
    supports_step_execution = True

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        # Import lazily because the main pipeline also inherits this mixin.
        from vllm_omni.diffusion.models.wan2_2.chunked_mp4 import (
            resolve_wan_preencode_batch_frames,
            resolve_wan_preencode_mp4,
        )
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
            Wan22Pipeline,
            build_wan_scheduler,
            resolve_wan_flow_shift,
            resolve_wan_guidance_scales,
            resolve_wan_sample_solver,
        )
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

        # Derived VACE / distilled pipelines must not accidentally inherit T2V
        # semantics. Pipeline parallelism has a different asynchronous return
        # contract; neither is safe to silently execute as ordinary T2V.
        if type(self) is not Wan22Pipeline or self.is_dmd or self.expand_timesteps:
            raise ValueError("Wan step execution currently supports the standard Wan22 T2V pipeline only.")
        if self.od_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Wan step execution does not support pipeline parallelism.")
        if self.od_config.streaming_output:
            raise ValueError("Wan step execution does not support streaming output.")

        sampling = state.sampling
        prepare_wan_t2v_step_sampling(sampling)
        if state.step_index != 0:
            raise ValueError("Wan prepare_encode requires new request state with step_index=0.")
        prompt_data = state.prompt if isinstance(state.prompt, dict) else {}
        if prompt_data.get("multi_modal_data"):
            raise ValueError("Wan T2V step execution does not accept image, audio, or video conditions.")
        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            [state.prompt], {"prompt_embeds": None, "negative_prompt_embeds": None}
        )
        prompt_embeds = prompt_fields["prompt_embeds"]
        negative_prompt_embeds = prompt_fields["negative_prompt_embeds"]
        prompt = state.prompt if isinstance(state.prompt, str) else prompt_data.get("prompt")
        negative_prompt = prompt_data.get("negative_prompt") if negative_prompt_embeds is None else None
        if prompt_embeds is None and not prompt:
            raise ValueError("Prompt is required for Wan2.2 generation when prompt_embeds are not provided.")
        if prompt_embeds is not None:
            prompt = None

        mod_value = self.vae_scale_factor_spatial * self.transformer_config.patch_size[1]
        height = ((sampling.height or 480) // mod_value) * mod_value
        width = ((sampling.width or 832) // mod_value) * mod_value
        num_frames = sampling.num_frames or 81
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        num_steps = sampling.num_inference_steps
        if resolve_wan_preencode_mp4(sampling, output_type=sampling.output_type or "np"):
            resolve_wan_preencode_batch_frames(sampling)
        guidance_low, guidance_high = resolve_wan_guidance_scales(sampling, default_guidance_scale=4.0)
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else sampling.boundary_ratio
        boundary_ratio = 0.875 if boundary_ratio is None else boundary_ratio
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high,
            boundary_ratio=boundary_ratio,
        )
        model = self.transformer if self.transformer is not None else self.transformer_2
        if model is None:
            raise RuntimeError("No transformer available for Wan step execution")
        dtype = model.dtype
        num_outputs = sampling.num_outputs_per_prompt or 1
        do_cfg = guidance_low > 1.0 or guidance_high > 1.0
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=num_outputs,
                max_sequence_length=sampling.max_sequence_length or 512,
                device=self.device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=self.device, dtype=dtype).repeat_interleave(num_outputs, dim=0)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=dtype)
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_outputs, dim=0)
            elif do_cfg:
                _, negative_prompt_embeds = self.encode_prompt(
                    prompt=[""],
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=num_outputs,
                    max_sequence_length=sampling.max_sequence_length or 512,
                    device=self.device,
                    dtype=dtype,
                )

        # Do not reconstruct OmniDiffusionRequest: its __post_init__ changes
        # explicit-guidance flags on the already-normalized sampling object.
        request = SimpleNamespace(sampling_params=sampling)
        sample_solver = resolve_wan_sample_solver(request, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(request, self.od_config)
        # UniPC carries multi-step derivative history. Reconstructing it when a
        # request resumes (or sharing self.scheduler) changes generated output.
        scheduler = build_wan_scheduler(sample_solver, flow_shift)
        if sample_solver == "unipc":
            scheduler.set_timesteps(num_steps, device=self.device, shift=flow_shift)
        else:
            scheduler.set_timesteps(num_steps, device=self.device)

        state.prompt_embeds = prompt_embeds
        state.negative_prompt_embeds = negative_prompt_embeds
        state.latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=self.transformer_config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=DiffusionRequestBatch.collate_sampling_param_generators([sampling], num_outputs, None),
            latents=sampling.latents,
        )
        state.scheduler = scheduler
        state.timesteps = scheduler.timesteps
        state.step_index = 0
        state.do_true_cfg = do_cfg and negative_prompt_embeds is not None
        state.extra["wan_t2v"] = {
            "guidance_low": guidance_low,
            "guidance_high": guidance_high,
            "boundary_timestep": boundary_ratio * scheduler.config.num_train_timesteps,
            "dtype": dtype,
            "attention_kwargs": kwargs.get("attention_kwargs") or {},
        }
        return state

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor:
        states = input_batch.states
        if len(states) != 1:
            raise ValueError("Wan step execution requires a single request per step batch.")
        state = states[0]
        context = state.extra["wan_t2v"]
        timestep = state.current_timestep
        self._current_timestep = timestep
        self._num_timesteps = state.total_steps
        self._guidance_scale = context["guidance_low"]
        self._guidance_scale_2 = context["guidance_high"]
        self.record_denoise_step(state.step_index, timestep, scheduler=state.scheduler, total_steps=state.total_steps)
        noise_pred, do_cfg = self._predict_wan_step(
            latents=state.latents,
            timestep=timestep,
            prompt_embeds=state.prompt_embeds,
            negative_prompt_embeds=state.negative_prompt_embeds,
            **context,
        )
        # The two experts may use different CFG scales. The scheduler CFG path
        # must follow the expert selected at this step, not the previous request.
        state.do_true_cfg = do_cfg
        return noise_pred

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            state.current_timestep,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        return self._decode_wan_latents(state.latents, state.sampling)
