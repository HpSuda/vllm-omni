# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


class Priority(str, Enum):
    NORMAL = "normal"
    SACRIFICIAL = "sacrificial"


class Phase(str, Enum):
    ENCODE = "encode"
    DENOISE = "denoise"
    DECODE = "decode"


class RequestStatus(str, Enum):
    NOT_ARRIVED = "not_arrived"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass(frozen=True)
class RequestTypeConfig:
    name: str
    weight: float
    width: int
    height: int
    num_inference_steps: int
    num_frames: int
    nominal_service_s: float | None = None
    estimated_service_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("request type name cannot be empty")
        if not _is_finite_number(self.weight) or self.weight <= 0.0:
            raise ValueError(f"request type {self.name!r} weight must be positive")
        if not _is_int(self.width) or not _is_int(self.height) or self.width <= 0 or self.height <= 0:
            raise ValueError(f"request type {self.name!r} dimensions must be positive")
        if (
            not _is_int(self.num_inference_steps)
            or not _is_int(self.num_frames)
            or self.num_inference_steps <= 0
            or self.num_frames <= 0
        ):
            raise ValueError(f"request type {self.name!r} steps and frames must be positive")
        if self.nominal_service_s is not None and (
            not _is_finite_number(self.nominal_service_s) or self.nominal_service_s <= 0.0
        ):
            raise ValueError(f"request type {self.name!r} nominal_service_s must be positive")
        if self.estimated_service_s is not None and (
            not _is_finite_number(self.estimated_service_s) or self.estimated_service_s <= 0.0
        ):
            raise ValueError(f"request type {self.name!r} estimated_service_s must be positive")


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int
    request_types: tuple[RequestTypeConfig, ...]
    utilization: float | None = None
    request_rate: float | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.num_requests) or self.num_requests <= 0:
            raise ValueError("workload.num_requests must be positive")
        if not self.request_types:
            raise ValueError("workload.request_types cannot be empty")
        if (self.utilization is None) == (self.request_rate is None):
            raise ValueError("set exactly one of workload.utilization or workload.request_rate")
        if self.utilization is not None and (not _is_finite_number(self.utilization) or self.utilization <= 0.0):
            raise ValueError("workload.utilization must be positive")
        if self.request_rate is not None and (not _is_finite_number(self.request_rate) or self.request_rate <= 0.0):
            raise ValueError("workload.request_rate must be positive")


@dataclass(frozen=True)
class HsdpServiceConfig:
    """Profile-level approximation of HSDP communication work."""

    enabled: bool = False
    shard_size: int = 1
    communication_overhead_weight: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("service.hsdp.enabled must be a boolean")
        if not _is_int(self.shard_size) or self.shard_size <= 0:
            raise ValueError("service.hsdp.shard_size must be positive")
        if not _is_finite_number(self.communication_overhead_weight) or self.communication_overhead_weight < 0.0:
            raise ValueError("service.hsdp.communication_overhead_weight cannot be negative")


@dataclass(frozen=True)
class TimingModelConfig:
    kind: str = "fixed_anchor"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("service.timing_model.type cannot be empty")


@dataclass(frozen=True)
class ServiceConfig:
    encode_fraction: float = 0.0
    decode_fraction: float = 0.0
    actual_jitter_sigma: float = 0.0
    estimate_error_sigma: float = 0.0
    preemption_cost_s: float = 0.0
    actual_service_scale: float = 1.0
    hsdp: HsdpServiceConfig = field(default_factory=HsdpServiceConfig)
    timing_model: TimingModelConfig = field(default_factory=TimingModelConfig)

    def __post_init__(self) -> None:
        if not _is_finite_number(self.encode_fraction) or not _is_finite_number(self.decode_fraction):
            raise ValueError("service phase fractions must be finite")
        if self.encode_fraction < 0.0 or self.decode_fraction < 0.0:
            raise ValueError("service phase fractions cannot be negative")
        if self.encode_fraction + self.decode_fraction >= 1.0:
            raise ValueError("encode_fraction + decode_fraction must be less than 1")
        if not _is_finite_number(self.actual_jitter_sigma) or not _is_finite_number(self.estimate_error_sigma):
            raise ValueError("service noise sigmas must be finite")
        if self.actual_jitter_sigma < 0.0 or self.estimate_error_sigma < 0.0:
            raise ValueError("service noise sigmas cannot be negative")
        if not _is_finite_number(self.preemption_cost_s) or self.preemption_cost_s < 0.0:
            raise ValueError("service.preemption_cost_s cannot be negative")
        if not _is_finite_number(self.actual_service_scale) or self.actual_service_scale <= 0.0:
            raise ValueError("service.actual_service_scale must be positive")


@dataclass(frozen=True)
class BackendConfig:
    name: str
    speed: float = 1.0
    devices: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backend name cannot be empty")
        if not _is_finite_number(self.speed) or self.speed <= 0.0:
            raise ValueError(f"backend {self.name!r} speed must be positive")
        if not _is_int(self.devices) or self.devices <= 0:
            raise ValueError(f"backend {self.name!r} devices must be positive")


@dataclass(frozen=True)
class TopologyConfig:
    name: str
    backends: tuple[BackendConfig, ...]

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("topology.backends cannot be empty")
        names = [backend.name for backend in self.backends]
        if len(names) != len(set(names)):
            raise ValueError("topology backend names must be unique")


@dataclass(frozen=True)
class ComponentConfig:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("policy component type cannot be empty")


@dataclass(frozen=True)
class PolicyConfig:
    classifier: ComponentConfig
    router: ComponentConfig
    scheduler: ComponentConfig


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    runs: int = 20

    def __post_init__(self) -> None:
        if not _is_int(self.seed):
            raise ValueError("simulation.seed must be an integer")
        if not _is_int(self.runs) or self.runs <= 0:
            raise ValueError("simulation.runs must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    simulation: SimulationConfig
    workload: WorkloadConfig
    service: ServiceConfig
    topology: TopologyConfig
    policy: PolicyConfig

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name cannot be empty")


@dataclass
class SimRequest:
    request_id: str
    arrival_seq: int
    arrival_time_s: float
    request_type: RequestTypeConfig
    actual_encode_s: float
    actual_step_s: float
    actual_decode_s: float
    actual_text_encode_s: float
    actual_latent_prepare_s: float
    actual_denoise_compute_s: float
    actual_denoise_overhead_s: float
    actual_usp_communication_s: float
    actual_hsdp_communication_s: float
    actual_vae_decode_s: float
    actual_postprocess_s: float
    estimated_encode_s: float
    estimated_step_s: float
    estimated_decode_s: float
    service_diagnostics: dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    status: RequestStatus = RequestStatus.NOT_ARRIVED
    backend_name: str | None = None
    encode_completed: bool = False
    completed_steps: int = 0
    decode_completed: bool = False
    first_start_time_s: float | None = None
    completion_time_s: float | None = None
    preemptions: int = 0
    resumes: int = 0
    dispatch_count: int = 0

    @property
    def total_steps(self) -> int:
        return self.request_type.num_inference_steps

    @property
    def actual_total_s(self) -> float:
        return self.actual_encode_s + self.actual_step_s * self.total_steps + self.actual_decode_s

    @property
    def estimated_total_s(self) -> float:
        return self.estimated_encode_s + self.estimated_step_s * self.total_steps + self.estimated_decode_s

    @property
    def remaining_estimated_base_s(self) -> float:
        remaining = 0.0
        if not self.encode_completed:
            remaining += self.estimated_encode_s
        remaining += self.estimated_step_s * max(self.total_steps - self.completed_steps, 0)
        if not self.decode_completed:
            remaining += self.estimated_decode_s
        return remaining

    @property
    def next_phase(self) -> Phase:
        if not self.encode_completed:
            return Phase.ENCODE
        if self.completed_steps < self.total_steps:
            return Phase.DENOISE
        if not self.decode_completed:
            return Phase.DECODE
        raise RuntimeError(f"request {self.request_id!r} has no remaining phase")

    def actual_phase_duration_s(self, phase: Phase, speed: float) -> float:
        base_duration = {
            Phase.ENCODE: self.actual_encode_s,
            Phase.DENOISE: self.actual_step_s,
            Phase.DECODE: self.actual_decode_s,
        }[phase]
        return base_duration / speed

    def remaining_estimated_s(self, speed: float) -> float:
        return self.remaining_estimated_base_s / speed

    def estimated_total_on_backend_s(self, speed: float) -> float:
        return self.estimated_total_s / speed

    @property
    def latency_s(self) -> float:
        if self.completion_time_s is None:
            raise RuntimeError(f"request {self.request_id!r} has not completed")
        return self.completion_time_s - self.arrival_time_s

    def to_view(self, backend_speed: float | None = None) -> RequestView:
        speed = backend_speed or 1.0
        return RequestView(
            request_id=self.request_id,
            arrival_seq=self.arrival_seq,
            arrival_time_s=self.arrival_time_s,
            request_type_name=self.request_type.name,
            width=self.request_type.width,
            height=self.request_type.height,
            num_inference_steps=self.request_type.num_inference_steps,
            num_frames=self.request_type.num_frames,
            priority=self.priority,
            status=self.status,
            backend_name=self.backend_name,
            encode_completed=self.encode_completed,
            completed_steps=self.completed_steps,
            decode_completed=self.decode_completed,
            estimated_total_s=self.estimated_total_on_backend_s(speed),
            estimated_remaining_s=self.remaining_estimated_s(speed),
            has_started=self.first_start_time_s is not None,
        )


@dataclass(frozen=True)
class RequestView:
    """Policy-visible request state; deliberately excludes true service time."""

    request_id: str
    arrival_seq: int
    arrival_time_s: float
    request_type_name: str
    width: int
    height: int
    num_inference_steps: int
    num_frames: int
    priority: Priority
    status: RequestStatus
    backend_name: str | None
    encode_completed: bool
    completed_steps: int
    decode_completed: bool
    estimated_total_s: float
    estimated_remaining_s: float
    has_started: bool


@dataclass(frozen=True)
class BackendView:
    name: str
    speed: float
    normal_load_s: float
    sacrificial_load_s: float
    inflight_normal: int
    inflight_sacrificial: int
    latency_ema_s: float
    running_request_id: str | None
    pending_requests: int


@dataclass
class BackendState:
    config: BackendConfig
    pending_ids: list[str] = field(default_factory=list)
    running_id: str | None = None
    assigned_normal_load_s: float = 0.0
    assigned_sacrificial_load_s: float = 0.0
    inflight_normal: int = 0
    inflight_sacrificial: int = 0
    latency_ema_s: float = 0.0
    busy_time_s: float = 0.0
    completed_requests: int = 0

    @property
    def name(self) -> str:
        return self.config.name
