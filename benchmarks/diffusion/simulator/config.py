# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.diffusion.simulator.models import (
    BackendConfig,
    ComponentConfig,
    ExperimentConfig,
    HsdpServiceConfig,
    PolicyConfig,
    RequestTypeConfig,
    ServiceConfig,
    SimulationConfig,
    TimingModelConfig,
    TopologyConfig,
    WorkloadConfig,
)


def _yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "Loading simulator YAML requires PyYAML. Install the repository dependencies or `pip install pyyaml`."
        ) from exc
    return yaml


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _sequence(value: Any, path: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a list")
    return list(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} field(s): {', '.join(unknown)}")


def _validate_config_value(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_config_value(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_config_value(item, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{path} has unsupported value type {type(value).__name__!r}")


def _required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ValueError(f"missing required field {path}.{key}")
    return data[key]


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _parse_component(value: Any, path: str) -> ComponentConfig:
    data = _mapping(value, path)
    kind = str(_required(data, "type", path)).strip()
    return ComponentConfig(kind=kind, options={key: copy.deepcopy(item) for key, item in data.items() if key != "type"})


def _parse_timing_model(value: Any) -> TimingModelConfig:
    path = "service.timing_model"
    data = _mapping(value, path)
    kind = str(_required(data, "type", path)).strip()
    return TimingModelConfig(
        kind=kind,
        options={key: copy.deepcopy(item) for key, item in data.items() if key != "type"},
    )


def _parse_request_type(value: Any, index: int) -> RequestTypeConfig:
    path = f"workload.request_types[{index}]"
    data = _mapping(value, path)
    _reject_unknown(
        data,
        {
            "name",
            "weight",
            "width",
            "height",
            "num_inference_steps",
            "num_frames",
            "nominal_service_s",
            "estimated_service_s",
            "metadata",
        },
        path,
    )
    nominal = data.get("nominal_service_s")
    estimated = data.get("estimated_service_s")
    return RequestTypeConfig(
        name=str(_required(data, "name", path)),
        weight=_as_float(_required(data, "weight", path), f"{path}.weight"),
        width=_as_int(_required(data, "width", path), f"{path}.width"),
        height=_as_int(_required(data, "height", path), f"{path}.height"),
        num_inference_steps=_as_int(_required(data, "num_inference_steps", path), f"{path}.num_inference_steps"),
        num_frames=_as_int(_required(data, "num_frames", path), f"{path}.num_frames"),
        nominal_service_s=None if nominal is None else _as_float(nominal, f"{path}.nominal_service_s"),
        estimated_service_s=None if estimated is None else _as_float(estimated, f"{path}.estimated_service_s"),
        metadata=_mapping(data.get("metadata", {}), f"{path}.metadata"),
    )


def _parse_topology(value: Any) -> TopologyConfig:
    path = "topology"
    data = _mapping(value, path)
    _reject_unknown(
        data,
        {"name", "backends", "backend_count", "devices_per_backend", "speed_factors"},
        path,
    )
    name = str(data.get("name", "topology"))
    if "backends" in data:
        if any(key in data for key in ("backend_count", "devices_per_backend", "speed_factors")):
            raise ValueError("topology.backends cannot be combined with topology shorthand fields")
        backends: list[BackendConfig] = []
        for index, raw_backend in enumerate(_sequence(data["backends"], "topology.backends")):
            backend_path = f"topology.backends[{index}]"
            backend = _mapping(raw_backend, backend_path)
            _reject_unknown(backend, {"name", "speed", "devices"}, backend_path)
            backends.append(
                BackendConfig(
                    name=str(_required(backend, "name", backend_path)),
                    speed=_as_float(backend.get("speed", 1.0), f"{backend_path}.speed"),
                    devices=_as_int(backend.get("devices", 1), f"{backend_path}.devices"),
                )
            )
        return TopologyConfig(name=name, backends=tuple(backends))

    backend_count = _as_int(_required(data, "backend_count", path), "topology.backend_count")
    devices = _as_int(data.get("devices_per_backend", 1), "topology.devices_per_backend")
    raw_speeds = data.get("speed_factors")
    speeds = (
        [1.0] * backend_count
        if raw_speeds is None
        else [
            _as_float(item, f"topology.speed_factors[{index}]")
            for index, item in enumerate(_sequence(raw_speeds, "topology.speed_factors"))
        ]
    )
    if len(speeds) != backend_count:
        raise ValueError("topology.speed_factors must contain one entry per backend")
    return TopologyConfig(
        name=name,
        backends=tuple(
            BackendConfig(name=f"backend-{index}", speed=speeds[index], devices=devices)
            for index in range(backend_count)
        ),
    )


def parse_experiment_config(value: Any) -> ExperimentConfig:
    data = _mapping(value, "config")
    _validate_config_value(data, "config")
    _reject_unknown(data, {"version", "name", "simulation", "workload", "service", "topology", "policy"}, "config")
    version = _as_int(data.get("version", 1), "config.version")
    if version != 1:
        raise ValueError(f"unsupported simulator config version: {version}")

    simulation_data = _mapping(data.get("simulation", {}), "simulation")
    _reject_unknown(simulation_data, {"seed", "runs"}, "simulation")
    simulation = SimulationConfig(
        seed=_as_int(simulation_data.get("seed", 42), "simulation.seed"),
        runs=_as_int(simulation_data.get("runs", 20), "simulation.runs"),
    )

    workload_data = _mapping(_required(data, "workload", "config"), "workload")
    _reject_unknown(workload_data, {"num_requests", "utilization", "request_rate", "request_types"}, "workload")
    utilization = workload_data.get("utilization")
    request_rate = workload_data.get("request_rate")
    request_types = tuple(
        _parse_request_type(item, index)
        for index, item in enumerate(
            _sequence(_required(workload_data, "request_types", "workload"), "workload.request_types")
        )
    )
    workload = WorkloadConfig(
        num_requests=_as_int(_required(workload_data, "num_requests", "workload"), "workload.num_requests"),
        utilization=None if utilization is None else _as_float(utilization, "workload.utilization"),
        request_rate=None if request_rate is None else _as_float(request_rate, "workload.request_rate"),
        request_types=request_types,
    )

    service_data = _mapping(data.get("service", {}), "service")
    _reject_unknown(
        service_data,
        {
            "encode_fraction",
            "decode_fraction",
            "actual_jitter_sigma",
            "estimate_error_sigma",
            "preemption_cost_s",
            "actual_service_scale",
            "hsdp",
            "timing_model",
        },
        "service",
    )
    hsdp_data = _mapping(service_data.get("hsdp", {}), "service.hsdp")
    _reject_unknown(
        hsdp_data,
        {"enabled", "shard_size", "communication_overhead_weight"},
        "service.hsdp",
    )
    service = ServiceConfig(
        encode_fraction=_as_float(service_data.get("encode_fraction", 0.0), "service.encode_fraction"),
        decode_fraction=_as_float(service_data.get("decode_fraction", 0.0), "service.decode_fraction"),
        actual_jitter_sigma=_as_float(service_data.get("actual_jitter_sigma", 0.0), "service.actual_jitter_sigma"),
        estimate_error_sigma=_as_float(service_data.get("estimate_error_sigma", 0.0), "service.estimate_error_sigma"),
        preemption_cost_s=_as_float(service_data.get("preemption_cost_s", 0.0), "service.preemption_cost_s"),
        actual_service_scale=_as_float(
            service_data.get("actual_service_scale", 1.0),
            "service.actual_service_scale",
        ),
        hsdp=HsdpServiceConfig(
            enabled=_as_bool(hsdp_data.get("enabled", False), "service.hsdp.enabled"),
            shard_size=_as_int(hsdp_data.get("shard_size", 1), "service.hsdp.shard_size"),
            communication_overhead_weight=_as_float(
                hsdp_data.get("communication_overhead_weight", 0.0),
                "service.hsdp.communication_overhead_weight",
            ),
        ),
        timing_model=_parse_timing_model(service_data.get("timing_model", {"type": "fixed_anchor"})),
    )

    policy_data = _mapping(_required(data, "policy", "config"), "policy")
    _reject_unknown(policy_data, {"classifier", "router", "scheduler"}, "policy")
    policy = PolicyConfig(
        classifier=_parse_component(_required(policy_data, "classifier", "policy"), "policy.classifier"),
        router=_parse_component(_required(policy_data, "router", "policy"), "policy.router"),
        scheduler=_parse_component(_required(policy_data, "scheduler", "policy"), "policy.scheduler"),
    )

    experiment = ExperimentConfig(
        name=str(data.get("name", "diffusion-simulation")),
        simulation=simulation,
        workload=workload,
        service=service,
        topology=_parse_topology(_required(data, "topology", "config")),
        policy=policy,
    )

    # Component options are intentionally free-form at the dataclass layer so
    # new policies stay cheap to add. Resolve their factories here so that
    # loading a config (and especially ``--validate-only``) still rejects an
    # unknown policy or misspelled option before a simulation starts.
    from benchmarks.diffusion.simulator.policies import (
        build_classifier,
        build_router,
        build_scheduler,
    )

    build_classifier(experiment.policy.classifier)
    build_router(experiment.policy.router)
    build_scheduler(experiment.policy.scheduler)
    from benchmarks.diffusion.simulator.service_models import (
        build_service_timing_model,
    )

    timing_model = build_service_timing_model(experiment)
    for request_type in experiment.workload.request_types:
        timing_model.profile(request_type)
    return experiment


def _parse_override_value(raw: str) -> Any:
    yaml = _yaml_module()
    return yaml.safe_load(raw)


def apply_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"invalid override {override!r}; expected dotted.path=value")
        dotted_path, raw_value = override.split("=", 1)
        parts = [part for part in dotted_path.split(".") if part]
        if not parts:
            raise ValueError(f"invalid override path in {override!r}")
        cursor: Any = updated
        for part in parts[:-1]:
            if isinstance(cursor, dict):
                if part not in cursor:
                    raise ValueError(f"override path {dotted_path!r} does not exist")
                cursor = cursor[part]
            elif isinstance(cursor, list):
                try:
                    index = int(part)
                except ValueError as exc:
                    raise ValueError(f"override path {dotted_path!r} requires a list index at {part!r}") from exc
                if index < 0 or index >= len(cursor):
                    raise ValueError(f"override path {dotted_path!r} has an out-of-range list index")
                cursor = cursor[index]
            else:
                raise ValueError(f"override path {dotted_path!r} crosses a scalar value")

        final_part = parts[-1]
        parsed_value = _parse_override_value(raw_value)
        if isinstance(cursor, dict):
            if final_part not in cursor:
                raise ValueError(f"override path {dotted_path!r} does not exist")
            cursor[final_part] = parsed_value
        elif isinstance(cursor, list):
            try:
                index = int(final_part)
            except ValueError as exc:
                raise ValueError(f"override path {dotted_path!r} requires a list index at {final_part!r}") from exc
            if index < 0 or index >= len(cursor):
                raise ValueError(f"override path {dotted_path!r} has an out-of-range list index")
            cursor[index] = parsed_value
        else:
            raise ValueError(f"override path {dotted_path!r} crosses a scalar value")
    return updated


def load_experiment_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    yaml = _yaml_module()
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if loaded is None:
        raise ValueError(f"simulator config {config_path} is empty")
    mapping = _mapping(loaded, str(config_path))
    return parse_experiment_config(apply_overrides(mapping, overrides))
