# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from benchmarks.diffusion.simulator.models import ExperimentConfig, RequestTypeConfig


@dataclass(frozen=True)
class ServiceProfile:
    """Coarse phase-level work for one request before runtime noise."""

    text_encode_s: float
    latent_prepare_s: float
    denoise_compute_s: float
    denoise_overhead_s: float
    usp_communication_s: float
    hsdp_communication_s: float
    vae_decode_s: float
    postprocess_s: float
    estimate_encode_fraction: float
    estimate_denoise_fraction: float
    estimate_decode_fraction: float
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        times = (
            self.text_encode_s,
            self.latent_prepare_s,
            self.denoise_compute_s,
            self.denoise_overhead_s,
            self.usp_communication_s,
            self.hsdp_communication_s,
            self.vae_decode_s,
            self.postprocess_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in times):
            raise ValueError("service profile times must be finite and non-negative")
        if self.total_s <= 0.0:
            raise ValueError("service profile total time must be positive")
        estimate_fractions = (
            self.estimate_encode_fraction,
            self.estimate_denoise_fraction,
            self.estimate_decode_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in estimate_fractions):
            raise ValueError("service profile estimate fractions must be finite and non-negative")
        if not math.isclose(sum(estimate_fractions), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("service profile estimate fractions must sum to one")

    @property
    def encode_s(self) -> float:
        return self.text_encode_s + self.latent_prepare_s

    @property
    def denoise_s(self) -> float:
        return self.denoise_compute_s + self.denoise_overhead_s + self.usp_communication_s + self.hsdp_communication_s

    @property
    def decode_s(self) -> float:
        return self.vae_decode_s + self.postprocess_s

    @property
    def total_s(self) -> float:
        return self.encode_s + self.denoise_s + self.decode_s

    def scaled(self, multiplier: float) -> ServiceProfile:
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("service profile multiplier must be positive and finite")
        return ServiceProfile(
            text_encode_s=self.text_encode_s * multiplier,
            latent_prepare_s=self.latent_prepare_s * multiplier,
            denoise_compute_s=self.denoise_compute_s * multiplier,
            denoise_overhead_s=self.denoise_overhead_s * multiplier,
            usp_communication_s=self.usp_communication_s * multiplier,
            hsdp_communication_s=self.hsdp_communication_s * multiplier,
            vae_decode_s=self.vae_decode_s * multiplier,
            postprocess_s=self.postprocess_s * multiplier,
            estimate_encode_fraction=self.estimate_encode_fraction,
            estimate_denoise_fraction=self.estimate_denoise_fraction,
            estimate_decode_fraction=self.estimate_decode_fraction,
            diagnostics=dict(self.diagnostics),
        )


class ServiceTimingModel(Protocol):
    def profile(self, request_type: RequestTypeConfig) -> ServiceProfile: ...


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} option(s): {', '.join(unknown)}")


_MISSING = object()


def _float_value(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: Any = _MISSING,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if key not in data:
        if default is _MISSING:
            raise ValueError(f"missing required option {path}.{key}")
        value = default
    else:
        value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}.{key} must be finite")
    if minimum is not None:
        invalid_minimum = result < minimum if minimum_inclusive else result <= minimum
        if invalid_minimum:
            comparison = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{path}.{key} must be {comparison} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{path}.{key} must be at most {maximum}")
    return result


def _optional_positive_float(
    data: Mapping[str, Any],
    key: str,
    path: str,
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _float_value(data, key, path, minimum=0.0, minimum_inclusive=False)


def _int_value(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: Any = _MISSING,
    minimum: int = 1,
) -> int:
    if key not in data:
        if default is _MISSING:
            raise ValueError(f"missing required option {path}.{key}")
        value = default
    else:
        value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{path}.{key} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{path}.{key} must be at least {minimum}")
    return result


def _choice_value(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: str,
    choices: set[str],
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a string")
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{path}.{key} must be one of: {allowed}")
    return value


def _patch_size(data: Mapping[str, Any], path: str) -> tuple[int, int, int]:
    raw = data.get("patch_size", [1, 2, 2])
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{path}.patch_size must contain three integers")
    values: list[int] = []
    for index, value in enumerate(raw):
        item_path = f"{path}.patch_size[{index}]"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{item_path} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{item_path} must be an integer")
        result = int(value)
        if result <= 0:
            raise ValueError(f"{item_path} must be positive")
        values.append(result)
    return values[0], values[1], values[2]


@dataclass(frozen=True)
class _Wan22ModelParameters:
    vae_spatial_scale: int
    vae_temporal_scale: int
    patch_size: tuple[int, int, int]
    num_layers: int
    num_attention_heads: int
    hidden_size: int
    ffn_dim: int
    text_tokens: int
    latent_channels: int
    output_channels: int
    activation_bytes: float
    latent_element_bytes: float
    parameter_bytes: float


@dataclass(frozen=True)
class _Wan22ExecutionParameters:
    usp_degree: int
    ulysses_mode: str
    cfg_passes: int
    parallel_compute_efficiency: float
    shape_efficiency_reference_tokens: int | None
    shape_efficiency_exponent: float
    min_shape_efficiency: float
    usp_overlap_fraction: float
    hsdp_overlap_fraction: float


@dataclass(frozen=True)
class _Wan22HardwareParameters:
    effective_compute_tflops_per_device: float
    effective_usp_bandwidth_gbytes_per_s: float | None
    effective_hsdp_bandwidth_gbytes_per_s: float | None
    usp_collective_latency_us: float
    hsdp_collective_latency_us: float


@dataclass(frozen=True)
class _Wan22StageParameters:
    text_encode_s: float
    latent_prepare_fixed_s: float
    latent_prepare_bandwidth_gbytes_per_s: float | None
    vae_decode_fixed_s: float
    vae_decode_gpixel_per_s: float | None
    denoise_step_overhead_s: float
    postprocess_s: float


class FixedAnchorTimingModel:
    def __init__(self, config: ExperimentConfig) -> None:
        if config.service.timing_model.options:
            options = ", ".join(sorted(config.service.timing_model.options))
            raise ValueError(f"unknown fixed_anchor timing model option(s): {options}")
        missing = [
            request_type.name
            for request_type in config.workload.request_types
            if request_type.nominal_service_s is None
        ]
        if missing:
            raise ValueError(
                "fixed_anchor timing model requires nominal_service_s for request type(s): " + ", ".join(missing)
            )
        self._service = config.service

    def profile(self, request_type: RequestTypeConfig) -> ServiceProfile:
        anchor_s = request_type.nominal_service_s
        if anchor_s is None:  # Defensive for programmatically modified configs.
            raise ValueError(f"request type {request_type.name!r} has no nominal_service_s")
        denoise_fraction = 1.0 - self._service.encode_fraction - self._service.decode_fraction
        hsdp_weight = self._service.hsdp.communication_overhead_weight if self._service.hsdp.enabled else 0.0
        return ServiceProfile(
            text_encode_s=anchor_s * self._service.encode_fraction,
            latent_prepare_s=0.0,
            denoise_compute_s=anchor_s * denoise_fraction,
            denoise_overhead_s=0.0,
            usp_communication_s=0.0,
            hsdp_communication_s=anchor_s * denoise_fraction * hsdp_weight,
            vae_decode_s=anchor_s * self._service.decode_fraction,
            postprocess_s=0.0,
            estimate_encode_fraction=self._service.encode_fraction,
            estimate_denoise_fraction=denoise_fraction,
            estimate_decode_fraction=self._service.decode_fraction,
            diagnostics={"timing_model": "fixed_anchor"},
        )


class AnalyticWan22TimingModel:
    """Input-derived Wan2.2 phase model using aggregate FLOPs and D2D work."""

    def __init__(self, config: ExperimentConfig) -> None:
        service = config.service
        if service.encode_fraction != 0.0 or service.decode_fraction != 0.0:
            raise ValueError(
                "analytic_wan22 timing uses stage parameters; "
                "service.encode_fraction and service.decode_fraction must be zero"
            )
        if service.hsdp.communication_overhead_weight != 0.0:
            raise ValueError(
                "analytic_wan22 computes HSDP from bytes and bandwidth; "
                "service.hsdp.communication_overhead_weight must be zero"
            )

        options = service.timing_model.options
        _reject_unknown(options, {"model", "execution", "hardware", "stages"}, "analytic_wan22")
        model_data = _mapping(options.get("model", {}), "service.timing_model.model")
        execution_data = _mapping(
            options.get("execution", {}),
            "service.timing_model.execution",
        )
        hardware_data = _mapping(
            options.get("hardware", {}),
            "service.timing_model.hardware",
        )
        stage_data = _mapping(options.get("stages", {}), "service.timing_model.stages")

        _reject_unknown(
            model_data,
            {
                "vae_spatial_scale",
                "vae_temporal_scale",
                "patch_size",
                "num_layers",
                "num_attention_heads",
                "hidden_size",
                "ffn_dim",
                "text_tokens",
                "latent_channels",
                "output_channels",
                "activation_bytes",
                "latent_element_bytes",
                "parameter_bytes",
            },
            "analytic_wan22.model",
        )
        self._model = _Wan22ModelParameters(
            vae_spatial_scale=_int_value(
                model_data,
                "vae_spatial_scale",
                "analytic_wan22.model",
                default=8,
            ),
            vae_temporal_scale=_int_value(
                model_data,
                "vae_temporal_scale",
                "analytic_wan22.model",
                default=4,
            ),
            patch_size=_patch_size(model_data, "analytic_wan22.model"),
            num_layers=_int_value(
                model_data,
                "num_layers",
                "analytic_wan22.model",
                default=40,
            ),
            num_attention_heads=_int_value(
                model_data,
                "num_attention_heads",
                "analytic_wan22.model",
                default=40,
            ),
            hidden_size=_int_value(
                model_data,
                "hidden_size",
                "analytic_wan22.model",
                default=5_120,
            ),
            ffn_dim=_int_value(
                model_data,
                "ffn_dim",
                "analytic_wan22.model",
                default=13_824,
            ),
            text_tokens=_int_value(
                model_data,
                "text_tokens",
                "analytic_wan22.model",
                default=512,
            ),
            latent_channels=_int_value(
                model_data,
                "latent_channels",
                "analytic_wan22.model",
                default=16,
            ),
            output_channels=_int_value(
                model_data,
                "output_channels",
                "analytic_wan22.model",
                default=16,
            ),
            activation_bytes=_float_value(
                model_data,
                "activation_bytes",
                "analytic_wan22.model",
                default=2.0,
                minimum=0.0,
                minimum_inclusive=False,
            ),
            latent_element_bytes=_float_value(
                model_data,
                "latent_element_bytes",
                "analytic_wan22.model",
                default=4.0,
                minimum=0.0,
                minimum_inclusive=False,
            ),
            parameter_bytes=_float_value(
                model_data,
                "parameter_bytes",
                "analytic_wan22.model",
                default=2.0,
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )

        _reject_unknown(
            execution_data,
            {
                "usp_degree",
                "ulysses_mode",
                "cfg_passes",
                "parallel_compute_efficiency",
                "shape_efficiency_reference_tokens",
                "shape_efficiency_exponent",
                "min_shape_efficiency",
                "usp_overlap_fraction",
                "hsdp_overlap_fraction",
            },
            "analytic_wan22.execution",
        )
        raw_reference_tokens = execution_data.get("shape_efficiency_reference_tokens")
        reference_tokens = (
            None
            if raw_reference_tokens is None
            else _int_value(
                execution_data,
                "shape_efficiency_reference_tokens",
                "analytic_wan22.execution",
            )
        )
        self._execution = _Wan22ExecutionParameters(
            usp_degree=_int_value(
                execution_data,
                "usp_degree",
                "analytic_wan22.execution",
            ),
            ulysses_mode=_choice_value(
                execution_data,
                "ulysses_mode",
                "analytic_wan22.execution",
                default="strict",
                choices={"strict", "advanced_uaa"},
            ),
            cfg_passes=_int_value(
                execution_data,
                "cfg_passes",
                "analytic_wan22.execution",
                default=2,
            ),
            parallel_compute_efficiency=_float_value(
                execution_data,
                "parallel_compute_efficiency",
                "analytic_wan22.execution",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
            shape_efficiency_reference_tokens=reference_tokens,
            shape_efficiency_exponent=_float_value(
                execution_data,
                "shape_efficiency_exponent",
                "analytic_wan22.execution",
                default=0.0,
                minimum=0.0,
            ),
            min_shape_efficiency=_float_value(
                execution_data,
                "min_shape_efficiency",
                "analytic_wan22.execution",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
            usp_overlap_fraction=_float_value(
                execution_data,
                "usp_overlap_fraction",
                "analytic_wan22.execution",
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            ),
            hsdp_overlap_fraction=_float_value(
                execution_data,
                "hsdp_overlap_fraction",
                "analytic_wan22.execution",
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            ),
        )

        _reject_unknown(
            hardware_data,
            {
                "effective_compute_tflops_per_device",
                "effective_usp_bandwidth_gbytes_per_s",
                "effective_hsdp_bandwidth_gbytes_per_s",
                "usp_collective_latency_us",
                "hsdp_collective_latency_us",
            },
            "analytic_wan22.hardware",
        )
        self._hardware = _Wan22HardwareParameters(
            effective_compute_tflops_per_device=_float_value(
                hardware_data,
                "effective_compute_tflops_per_device",
                "analytic_wan22.hardware",
                minimum=0.0,
                minimum_inclusive=False,
            ),
            effective_usp_bandwidth_gbytes_per_s=_optional_positive_float(
                hardware_data,
                "effective_usp_bandwidth_gbytes_per_s",
                "analytic_wan22.hardware",
            ),
            effective_hsdp_bandwidth_gbytes_per_s=_optional_positive_float(
                hardware_data,
                "effective_hsdp_bandwidth_gbytes_per_s",
                "analytic_wan22.hardware",
            ),
            usp_collective_latency_us=_float_value(
                hardware_data,
                "usp_collective_latency_us",
                "analytic_wan22.hardware",
                default=0.0,
                minimum=0.0,
            ),
            hsdp_collective_latency_us=_float_value(
                hardware_data,
                "hsdp_collective_latency_us",
                "analytic_wan22.hardware",
                default=0.0,
                minimum=0.0,
            ),
        )

        _reject_unknown(
            stage_data,
            {
                "text_encode_s",
                "latent_prepare_fixed_s",
                "latent_prepare_bandwidth_gbytes_per_s",
                "vae_decode_fixed_s",
                "vae_decode_gpixel_per_s",
                "denoise_step_overhead_s",
                "postprocess_s",
            },
            "analytic_wan22.stages",
        )
        self._stages = _Wan22StageParameters(
            text_encode_s=_float_value(
                stage_data,
                "text_encode_s",
                "analytic_wan22.stages",
                default=0.0,
                minimum=0.0,
            ),
            latent_prepare_fixed_s=_float_value(
                stage_data,
                "latent_prepare_fixed_s",
                "analytic_wan22.stages",
                default=0.0,
                minimum=0.0,
            ),
            latent_prepare_bandwidth_gbytes_per_s=_optional_positive_float(
                stage_data,
                "latent_prepare_bandwidth_gbytes_per_s",
                "analytic_wan22.stages",
            ),
            vae_decode_fixed_s=_float_value(
                stage_data,
                "vae_decode_fixed_s",
                "analytic_wan22.stages",
                default=0.0,
                minimum=0.0,
            ),
            vae_decode_gpixel_per_s=_optional_positive_float(
                stage_data,
                "vae_decode_gpixel_per_s",
                "analytic_wan22.stages",
            ),
            denoise_step_overhead_s=_float_value(
                stage_data,
                "denoise_step_overhead_s",
                "analytic_wan22.stages",
                default=0.0,
                minimum=0.0,
            ),
            postprocess_s=_float_value(
                stage_data,
                "postprocess_s",
                "analytic_wan22.stages",
                default=0.0,
                minimum=0.0,
            ),
        )
        self._service = service
        self._validate_topology(config)

    def _validate_topology(self, config: ExperimentConfig) -> None:
        usp_degree = self._execution.usp_degree
        attention_heads = self._model.num_attention_heads
        if self._model.hidden_size % attention_heads != 0:
            raise ValueError("analytic_wan22 model.hidden_size must be divisible by model.num_attention_heads")
        if self._execution.ulysses_mode == "strict" and attention_heads % usp_degree != 0:
            raise ValueError(
                "analytic_wan22 strict Ulysses requires "
                "model.num_attention_heads to be divisible by "
                "execution.usp_degree"
            )
        shard_size = self._service.hsdp.shard_size
        standalone_hsdp = self._service.hsdp.enabled and usp_degree == 1 and shard_size > 1
        expected_devices = shard_size if standalone_hsdp else usp_degree
        device_counts = {backend.devices for backend in config.topology.backends}
        if device_counts != {expected_devices}:
            raise ValueError(
                "analytic_wan22 requires every backend devices value to equal "
                f"{expected_devices} for the configured USP/HSDP mode; "
                f"got {sorted(device_counts)}"
            )
        if usp_degree > 1 and self._hardware.effective_usp_bandwidth_gbytes_per_s is None:
            raise ValueError(
                "analytic_wan22 with usp_degree > 1 requires hardware.effective_usp_bandwidth_gbytes_per_s"
            )
        if self._service.hsdp.enabled:
            if usp_degree > 1 and shard_size != usp_degree:
                raise ValueError(
                    "combined analytic_wan22 USP/HSDP requires service.hsdp.shard_size to equal execution.usp_degree"
                )
            if shard_size > 1 and self._hardware.effective_hsdp_bandwidth_gbytes_per_s is None:
                raise ValueError("analytic_wan22 HSDP requires hardware.effective_hsdp_bandwidth_gbytes_per_s")

    def _text_tokens(self, request_type: RequestTypeConfig) -> int:
        value = request_type.metadata.get("text_tokens", self._model.text_tokens)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"request type {request_type.name!r} metadata.text_tokens must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"request type {request_type.name!r} metadata.text_tokens must be an integer")
        result = int(value)
        if result <= 0:
            raise ValueError(f"request type {request_type.name!r} metadata.text_tokens must be positive")
        return result

    def profile(self, request_type: RequestTypeConfig) -> ServiceProfile:
        model = self._model
        execution = self._execution
        hardware = self._hardware
        stages = self._stages

        patch_t, patch_h, patch_w = model.patch_size
        height_multiple = model.vae_spatial_scale * patch_h
        width_multiple = model.vae_spatial_scale * patch_w
        normalized_height = request_type.height // height_multiple * height_multiple
        normalized_width = request_type.width // width_multiple * width_multiple
        if normalized_height <= 0 or normalized_width <= 0:
            raise ValueError(f"request type {request_type.name!r} is smaller than the Wan2.2 VAE/patch alignment")

        frames = request_type.num_frames
        temporal_scale = model.vae_temporal_scale
        normalized_frames = (
            frames if (frames - 1) % temporal_scale == 0 else frames // temporal_scale * temporal_scale + 1
        )
        latent_frames = (normalized_frames - 1) // temporal_scale + 1
        latent_height = normalized_height // model.vae_spatial_scale
        latent_width = normalized_width // model.vae_spatial_scale
        if latent_frames % patch_t != 0:
            raise ValueError(
                f"request type {request_type.name!r} latent frame count is not divisible by model.patch_size[0]"
            )

        transformer_tokens = latent_frames // patch_t * (latent_height // patch_h) * (latent_width // patch_w)
        usp_degree = execution.usp_degree
        padded_tokens = math.ceil(transformer_tokens / usp_degree) * usp_degree
        local_tokens = padded_tokens // usp_degree
        text_tokens = self._text_tokens(request_type)

        layers = model.num_layers
        hidden = model.hidden_size
        ffn = model.ffn_dim
        parallelizable_flops_per_forward = layers * (
            12 * padded_tokens * hidden**2 + 4 * padded_tokens * hidden * ffn + 4 * padded_tokens**2 * hidden
        )
        replicated_flops_per_forward = layers * (4 * text_tokens * hidden**2 + 4 * padded_tokens * text_tokens * hidden)
        transformer_forwards = request_type.num_inference_steps * execution.cfg_passes
        parallelizable_denoise_flops = parallelizable_flops_per_forward * transformer_forwards
        replicated_denoise_flops = replicated_flops_per_forward * transformer_forwards
        denoise_flops = parallelizable_denoise_flops + replicated_denoise_flops

        reference_tokens = execution.shape_efficiency_reference_tokens
        if reference_tokens is None:
            shape_efficiency = 1.0
        else:
            raw_shape_efficiency = (padded_tokens / reference_tokens) ** execution.shape_efficiency_exponent
            shape_efficiency = min(
                1.0,
                max(execution.min_shape_efficiency, raw_shape_efficiency),
            )
        effective_compute_tflops_per_device = (
            hardware.effective_compute_tflops_per_device * execution.parallel_compute_efficiency * shape_efficiency
        )
        aggregate_compute_tflops = usp_degree * effective_compute_tflops_per_device
        denoise_compute_s = parallelizable_denoise_flops / (
            aggregate_compute_tflops * 1e12
        ) + replicated_denoise_flops / (effective_compute_tflops_per_device * 1e12)
        executed_denoise_flops = parallelizable_denoise_flops + usp_degree * replicated_denoise_flops

        remote_fraction = (usp_degree - 1) / usp_degree
        usp_bytes_per_forward = 0.0
        usp_collective_calls_per_forward = 0
        usp_communication_s = 0.0
        if usp_degree > 1:
            self_attention_bytes = 4 * local_tokens * hidden * model.activation_bytes * remote_fraction * layers
            cross_attention_bytes = (
                (2 * local_tokens * hidden * model.activation_bytes + 2 * text_tokens * hidden * model.activation_bytes)
                * remote_fraction
                * layers
            )
            projected_width = model.output_channels * math.prod(model.patch_size)
            output_gather_bytes = local_tokens * projected_width * model.activation_bytes * (usp_degree - 1)
            usp_bytes_per_forward = self_attention_bytes + cross_attention_bytes + output_gather_bytes
            usp_collective_calls_per_forward = 8 * layers + 1
            usp_bytes = usp_bytes_per_forward * transformer_forwards
            assert hardware.effective_usp_bandwidth_gbytes_per_s is not None
            usp_raw_s = (
                usp_bytes / (hardware.effective_usp_bandwidth_gbytes_per_s * 1e9)
                + usp_collective_calls_per_forward * transformer_forwards * hardware.usp_collective_latency_us / 1e6
            )
            usp_communication_s = usp_raw_s * (1.0 - execution.usp_overlap_fraction)
        else:
            usp_bytes = 0.0

        hsdp_bytes_per_forward = 0.0
        hsdp_collective_calls_per_forward = 0
        hsdp_communication_s = 0.0
        shard_size = self._service.hsdp.shard_size
        if self._service.hsdp.enabled and shard_size > 1:
            hsdp_remote_fraction = (shard_size - 1) / shard_size
            block_parameter_elements = 8 * hidden**2 + 2 * hidden * ffn
            block_parameter_bytes = block_parameter_elements * model.parameter_bytes
            hsdp_bytes_per_forward = layers * block_parameter_bytes * hsdp_remote_fraction
            hsdp_collective_calls_per_forward = layers
            hsdp_bytes = hsdp_bytes_per_forward * transformer_forwards
            assert hardware.effective_hsdp_bandwidth_gbytes_per_s is not None
            hsdp_raw_s = (
                hsdp_bytes / (hardware.effective_hsdp_bandwidth_gbytes_per_s * 1e9)
                + hsdp_collective_calls_per_forward * transformer_forwards * hardware.hsdp_collective_latency_us / 1e6
            )
            hsdp_communication_s = hsdp_raw_s * (1.0 - execution.hsdp_overlap_fraction)
        else:
            block_parameter_bytes = (8 * hidden**2 + 2 * hidden * ffn) * model.parameter_bytes
            hsdp_bytes = 0.0

        latent_elements = model.latent_channels * latent_frames * latent_height * latent_width
        latent_bytes = latent_elements * model.latent_element_bytes
        latent_prepare_s = stages.latent_prepare_fixed_s
        if stages.latent_prepare_bandwidth_gbytes_per_s is not None:
            latent_prepare_s += latent_bytes / (stages.latent_prepare_bandwidth_gbytes_per_s * 1e9)

        output_pixel_frames = normalized_width * normalized_height * normalized_frames
        output_gpixel_frames = output_pixel_frames / 1e9
        vae_decode_s = stages.vae_decode_fixed_s
        if stages.vae_decode_gpixel_per_s is not None:
            vae_decode_s += output_gpixel_frames / stages.vae_decode_gpixel_per_s

        denoise_overhead_s = request_type.num_inference_steps * stages.denoise_step_overhead_s
        encode_s = stages.text_encode_s + latent_prepare_s
        denoise_s = denoise_compute_s + denoise_overhead_s + usp_communication_s + hsdp_communication_s
        decode_s = vae_decode_s + stages.postprocess_s
        total_s = encode_s + denoise_s + decode_s

        diagnostics: dict[str, Any] = {
            "timing_model": "analytic_wan22",
            "normalized_width": normalized_width,
            "normalized_height": normalized_height,
            "normalized_frames": normalized_frames,
            "latent_frames": latent_frames,
            "latent_elements": latent_elements,
            "latent_bytes": latent_bytes,
            "transformer_tokens": transformer_tokens,
            "padded_transformer_tokens": padded_tokens,
            "text_tokens": text_tokens,
            "transformer_forwards": transformer_forwards,
            "denoise_flops": denoise_flops,
            "denoise_petaflops": denoise_flops / 1e15,
            "parallelizable_denoise_flops": parallelizable_denoise_flops,
            "replicated_denoise_flops_per_rank": replicated_denoise_flops,
            "executed_denoise_flops_across_ranks": executed_denoise_flops,
            "shape_compute_efficiency": shape_efficiency,
            "effective_compute_tflops_per_device_after_efficiency": (effective_compute_tflops_per_device),
            "aggregate_effective_compute_tflops": aggregate_compute_tflops,
            "ulysses_mode": execution.ulysses_mode,
            "usp_communication_bytes_per_rank": usp_bytes,
            "usp_collective_calls_per_rank": (usp_collective_calls_per_forward * transformer_forwards),
            "hsdp_communication_bytes_per_rank": hsdp_bytes,
            "hsdp_collective_calls_per_rank": (hsdp_collective_calls_per_forward * transformer_forwards),
            "hsdp_block_parameter_bytes": block_parameter_bytes,
            "output_pixel_frames": output_pixel_frames,
            "output_gpixel_frames": output_gpixel_frames,
        }
        return ServiceProfile(
            text_encode_s=stages.text_encode_s,
            latent_prepare_s=latent_prepare_s,
            denoise_compute_s=denoise_compute_s,
            denoise_overhead_s=denoise_overhead_s,
            usp_communication_s=usp_communication_s,
            hsdp_communication_s=hsdp_communication_s,
            vae_decode_s=vae_decode_s,
            postprocess_s=stages.postprocess_s,
            estimate_encode_fraction=encode_s / total_s,
            estimate_denoise_fraction=denoise_s / total_s,
            estimate_decode_fraction=decode_s / total_s,
            diagnostics=diagnostics,
        )


def build_service_timing_model(config: ExperimentConfig) -> ServiceTimingModel:
    kind = config.service.timing_model.kind
    if kind == "fixed_anchor":
        return FixedAnchorTimingModel(config)
    if kind == "analytic_wan22":
        return AnalyticWan22TimingModel(config)
    raise ValueError(f"unknown service timing model type {kind!r}")
