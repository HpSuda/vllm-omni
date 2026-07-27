# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.diffusion.simulator.config import (  # noqa: E402
    apply_overrides,
    load_experiment_config,
    parse_experiment_config,
)
from benchmarks.diffusion.simulator.engine import (  # noqa: E402
    Simulator,
    config_to_dict,
    percentile,
)
from benchmarks.diffusion.simulator.models import (  # noqa: E402
    BackendView,
    Priority,
    RequestStatus,
    RequestView,
)
from benchmarks.diffusion.simulator.policies import (  # noqa: E402
    LeastInflightRouter,
    OnlineCreditTailClassifier,
    ProjectedCompletionRouter,
    QuantileRiskTailClassifier,
    QuotaTailClassifier,
    RoundRobinRouter,
    TwoQueueScheduler,
    WeightedLeastLoadRouter,
    build_classifier,
)
from benchmarks.diffusion.simulator.runner import run_experiment  # noqa: E402
from benchmarks.diffusion.simulator.service_models import (  # noqa: E402
    build_service_timing_model,
)

_SIMULATOR_CONFIGS = _REPO_ROOT / "benchmarks" / "diffusion" / "simulator" / "configs"


def _raw_config(
    *,
    num_requests: int = 2,
    request_rate: float = 1.0,
    service_s: float = 9.0,
    steps: int = 3,
    encode_fraction: float = 0.0,
    decode_fraction: float = 0.0,
    preemption_cost_s: float = 0.0,
    actual_service_scale: float = 1.0,
    hsdp_enabled: bool = False,
    hsdp_shard_size: int = 1,
    hsdp_communication_overhead_weight: float = 0.0,
    backend_count: int = 1,
    speed_factors: list[float] | None = None,
    runs: int = 1,
    seed: int = 42,
    classifier: dict[str, Any] | None = None,
    router: dict[str, Any] | None = None,
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if speed_factors is None:
        speed_factors = [1.0] * backend_count
    if classifier is None:
        classifier = {"type": "all_normal"}
    if router is None:
        router = {"type": "round_robin"}
    if scheduler is None:
        scheduler = {"type": "fifo"}
    return {
        "version": 1,
        "name": "test-experiment",
        "simulation": {"seed": seed, "runs": runs},
        "workload": {
            "num_requests": num_requests,
            "request_rate": request_rate,
            "request_types": [
                {
                    "name": "only",
                    "weight": 1.0,
                    "width": 64,
                    "height": 64,
                    "num_inference_steps": steps,
                    "num_frames": 1,
                    "nominal_service_s": service_s,
                }
            ],
        },
        "service": {
            "encode_fraction": encode_fraction,
            "decode_fraction": decode_fraction,
            "actual_jitter_sigma": 0.0,
            "estimate_error_sigma": 0.0,
            "preemption_cost_s": preemption_cost_s,
            "actual_service_scale": actual_service_scale,
            "hsdp": {
                "enabled": hsdp_enabled,
                "shard_size": hsdp_shard_size,
                "communication_overhead_weight": hsdp_communication_overhead_weight,
            },
        },
        "topology": {
            "name": "test-topology",
            "backend_count": backend_count,
            "devices_per_backend": 1,
            "speed_factors": speed_factors,
        },
        "policy": {
            "classifier": classifier,
            "router": router,
            "scheduler": scheduler,
        },
    }


def _raw_config_with_all_policy_numbers() -> dict[str, Any]:
    raw = _raw_config(
        classifier={
            "type": "quota_tail",
            "quota_every": 20,
            "quota_amount": 1,
            "threshold_ratio": 0.8,
            "long_request_ratio": 1.5,
            "initial_arrival_counter": 0,
            "initial_credits": 0,
            "initial_min_service_s": 1.0,
            "initial_max_service_s": 1.0,
            "max_sacrificial": 2,
        },
        router={
            "type": "weighted_least_load",
            "sacrificial_load_factor": 0.1,
            "load_view": "assigned",
            "update_latency_ema": False,
        },
    )
    raw["workload"]["utilization"] = None
    raw["workload"]["request_types"][0]["estimated_service_s"] = 9.0
    return raw


def _raw_config_for_numeric_path(path: str) -> dict[str, Any]:
    raw = _raw_config_with_all_policy_numbers()
    if path == "workload.utilization":
        raw["workload"]["utilization"] = 0.9
        raw["workload"]["request_rate"] = None
    if path.startswith("topology.backends."):
        raw["topology"] = {
            "name": "explicit-test-topology",
            "backends": [{"name": "backend", "speed": 1.0, "devices": 1}],
        }
    if path == "policy.router.nonpreemptible_weight":
        raw["policy"]["router"] = {
            "type": "projected_completion",
            "nonpreemptible_weight": 0.1,
            "tail_mode": "spread",
            "load_view": "assigned",
            "update_latency_ema": False,
        }
    if path.startswith("policy.scheduler."):
        raw["policy"]["scheduler"] = {
            "type": "two_queue",
            "normal_order": "bounded_srpt",
            "sacrificial_order": "lifo",
            "preempt_normal_over_sacrificial": True,
            "max_bypass": 2,
            "aging_s": 10.0,
            "preemption_hysteresis_s": 1.0,
            "global_work_stealing": True,
            "steal_order": "max_risk",
            "steal_hysteresis_s": 2.0,
            "steal_cost_s": 0.5,
            "global_protected_pull": False,
            "protected_pull_order": "fifo",
            "protected_pull_risk_beta": 0.5,
            "protected_pull_cost_s": 0.25,
        }
    return raw


def _preemptive_config(
    *,
    encode_fraction: float = 0.0,
    decode_fraction: float = 0.0,
    service_s: float = 9.0,
    preemption_cost_s: float = 0.0,
) -> Any:
    """Make request zero sacrificial and all later requests normal."""

    return parse_experiment_config(
        _raw_config(
            service_s=service_s,
            encode_fraction=encode_fraction,
            decode_fraction=decode_fraction,
            preemption_cost_s=preemption_cost_s,
            classifier={
                "type": "quota_tail",
                "quota_every": 1_000,
                "quota_amount": 0,
                "threshold_ratio": 0.0,
                "long_request_ratio": None,
                "initial_credits": 1,
            },
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
            },
        )
    )


def _set_arrivals(simulator: Simulator, arrivals: list[float]) -> None:
    assert len(arrivals) == len(simulator.requests)
    for request, arrival_time_s in zip(simulator.requests.values(), arrivals):
        request.arrival_time_s = arrival_time_s


def _write_yaml_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    config_path = tmp_path / "simulator.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _request_view(
    sequence: int,
    priority: Priority,
    *,
    status: RequestStatus = RequestStatus.WAITING,
    request_type_name: str = "test",
    estimated_total_s: float = 10.0,
    estimated_remaining_s: float = 10.0,
    arrival_time_s: float | None = None,
) -> RequestView:
    return RequestView(
        request_id=f"request-{sequence}",
        arrival_seq=sequence,
        arrival_time_s=float(sequence) if arrival_time_s is None else arrival_time_s,
        request_type_name=request_type_name,
        width=64,
        height=64,
        num_inference_steps=3,
        num_frames=1,
        priority=priority,
        status=status,
        backend_name="backend",
        encode_completed=False,
        completed_steps=0,
        decode_completed=False,
        estimated_total_s=estimated_total_s,
        estimated_remaining_s=estimated_remaining_s,
        has_started=status == RequestStatus.RUNNING,
    )


def _backend_view(
    name: str,
    *,
    speed: float = 1.0,
    normal_load_s: float = 0.0,
    sacrificial_load_s: float = 0.0,
    inflight_normal: int = 0,
    inflight_sacrificial: int = 0,
    latency_ema_s: float = 0.0,
) -> BackendView:
    return BackendView(
        name=name,
        speed=speed,
        normal_load_s=normal_load_s,
        sacrificial_load_s=sacrificial_load_s,
        inflight_normal=inflight_normal,
        inflight_sacrificial=inflight_sacrificial,
        latency_ema_s=latency_ema_s,
        running_request_id=None,
        pending_requests=inflight_normal + inflight_sacrificial,
    )


def test_config_is_strict_and_overrides_only_existing_fields() -> None:
    raw = _raw_config()
    original = copy.deepcopy(raw)

    overridden = apply_overrides(
        raw,
        [
            "simulation.seed=7",
            "simulation.runs=3",
            "topology.speed_factors=[2.5]",
            "service.preemption_cost_s=0.125",
        ],
    )
    parsed = parse_experiment_config(overridden)

    assert raw == original, "overrides must not mutate the caller's mapping"
    assert parsed.simulation.seed == 7
    assert parsed.simulation.runs == 3
    assert parsed.topology.backends[0].speed == 2.5
    assert parsed.service.preemption_cost_s == pytest.approx(0.125)

    with pytest.raises(ValueError, match="does not exist"):
        apply_overrides(raw, ["simulation.typo=1"])
    with pytest.raises(ValueError, match="override path"):
        apply_overrides(raw, ["name.value=1"])

    unknown = copy.deepcopy(raw)
    unknown["workload"]["request_typo"] = 1
    with pytest.raises(ValueError, match=r"unknown workload field\(s\): request_typo"):
        parse_experiment_config(unknown)

    unknown_option = copy.deepcopy(raw)
    unknown_option["policy"]["classifier"]["typo"] = True
    with pytest.raises(ValueError, match=r"unknown classifier option\(s\): typo"):
        parse_experiment_config(unknown_option)


@pytest.mark.parametrize(
    ("component", "definition", "error"),
    [
        ("classifier", {"type": "unknown"}, "unknown classifier type"),
        (
            "classifier",
            {"type": "all_normal", "misspelled": True},
            r"unknown classifier option\(s\): misspelled",
        ),
        ("router", {"type": "unknown"}, "unknown router type"),
        (
            "router",
            {"type": "round_robin", "misspelled": True},
            r"unknown router option\(s\): misspelled",
        ),
        ("scheduler", {"type": "unknown"}, "unknown scheduler type"),
        (
            "scheduler",
            {"type": "fifo", "misspelled": True},
            r"unknown scheduler option\(s\): misspelled",
        ),
    ],
)
def test_load_rejects_unknown_policy_kinds_and_options(
    tmp_path: Path,
    component: str,
    definition: dict[str, Any],
    error: str,
) -> None:
    raw = _raw_config()
    raw["policy"][component] = definition

    with pytest.raises(ValueError, match=error):
        load_experiment_config(_write_yaml_config(tmp_path, raw))


def test_overrides_address_list_indices_and_reject_invalid_indices() -> None:
    raw = _raw_config()
    overridden = apply_overrides(
        raw,
        [
            "workload.request_types.0.name=renamed",
            "workload.request_types.0.num_inference_steps=7",
            "topology.speed_factors.0=2.0",
        ],
    )
    parsed = parse_experiment_config(overridden)

    assert parsed.workload.request_types[0].name == "renamed"
    assert parsed.workload.request_types[0].num_inference_steps == 7
    assert parsed.topology.backends[0].speed == 2.0
    assert raw["workload"]["request_types"][0]["name"] == "only"

    for override, error in [
        ("workload.request_types.1.name=bad", "out-of-range list index"),
        ("workload.request_types.-1.name=bad", "out-of-range list index"),
        ("workload.request_types.first.name=bad", "requires a list index"),
        ("workload.request_types.0.unknown=bad", "does not exist"),
    ]:
        with pytest.raises(ValueError, match=error):
            apply_overrides(raw, [override])


@pytest.mark.parametrize(
    "path",
    [
        "version",
        "simulation.seed",
        "simulation.runs",
        "workload.num_requests",
        "workload.request_types.0.width",
        "workload.request_types.0.height",
        "workload.request_types.0.num_inference_steps",
        "workload.request_types.0.num_frames",
        "topology.backend_count",
        "topology.devices_per_backend",
        "topology.backends.0.devices",
        "service.hsdp.shard_size",
        "policy.classifier.max_sacrificial",
        "policy.scheduler.max_bypass",
    ],
)
@pytest.mark.parametrize("invalid_literal", ["1.5", "true"], ids=["fractional", "boolean"])
def test_all_schema_integer_fields_reject_fractional_floats_and_bools(
    path: str,
    invalid_literal: str,
) -> None:
    raw = _raw_config_for_numeric_path(path)
    invalid = apply_overrides(raw, [f"{path}={invalid_literal}"])

    with pytest.raises(ValueError, match="must be an integer"):
        parse_experiment_config(invalid)


@pytest.mark.parametrize(
    "option",
    [
        "quota_every",
        "quota_amount",
        "initial_arrival_counter",
        "initial_credits",
        "max_sacrificial",
    ],
)
@pytest.mark.parametrize("invalid_value", [1.5, True], ids=["fractional", "boolean"])
def test_all_classifier_integer_options_reject_fractional_floats_and_bools(
    option: str,
    invalid_value: Any,
) -> None:
    raw = _raw_config_with_all_policy_numbers()
    raw["policy"]["classifier"][option] = invalid_value

    with pytest.raises(ValueError, match=rf"classifier\.{option} must be an integer"):
        parse_experiment_config(raw)


def test_integral_float_integer_fields_resolve_to_ints() -> None:
    raw = _raw_config_with_all_policy_numbers()
    overridden = apply_overrides(
        raw,
        [
            "simulation.runs=2.0",
            "workload.request_types.0.width=2.0",
            "policy.classifier.quota_every=2.0",
            "policy.classifier.quota_amount=2.0",
        ],
    )

    parsed = parse_experiment_config(overridden)
    assert parsed.simulation.runs == 2
    assert parsed.workload.request_types[0].width == 2
    classifier = build_classifier(parsed.policy.classifier)
    assert isinstance(classifier, QuotaTailClassifier)
    assert classifier.quota_every == 2
    assert classifier.quota_amount == 2
    assert isinstance(classifier.quota_every, int)
    assert isinstance(classifier.quota_amount, int)


def test_new_policy_components_build_from_config() -> None:
    raw = _raw_config(
        classifier={
            "type": "quota_tail",
            "quota_every": 20,
            "quota_amount": 1,
            "threshold_ratio": 0.8,
            "long_request_ratio": 1.5,
            "max_sacrificial": 2,
            "eligible_request_types": ["long"],
        },
        router={
            "type": "projected_completion",
            "nonpreemptible_weight": 0.25,
            "tail_mode": "sink",
            "load_view": "remaining",
            "update_latency_ema": True,
        },
        scheduler={
            "type": "two_queue",
            "normal_order": "bounded_srpt",
            "sacrificial_order": "lifo",
            "preempt_normal_over_sacrificial": True,
            "max_bypass": 2,
            "aging_s": 30.0,
            "global_work_stealing": True,
            "steal_order": "oldest",
            "steal_hysteresis_s": 5.0,
            "steal_cost_s": 0.25,
            "global_protected_pull": False,
            "protected_pull_order": "max_risk",
            "protected_pull_risk_beta": 0.25,
            "protected_pull_band_risk_beta": 0.625,
            "protected_pull_band_min_pending": 10,
            "protected_pull_band_max_pending": 27,
            "protected_pull_risk_slack_s": 30.0,
            "protected_pull_guard_fraction": 0.04,
            "protected_pull_guard_max": 2,
            "protected_pull_guard_min_pending": 10,
            "protected_pull_tail_head_start": True,
            "protected_pull_cost_s": 0.5,
        },
    )

    simulator = Simulator(parse_experiment_config(raw))

    assert isinstance(simulator.classifier, QuotaTailClassifier)
    assert simulator.classifier.max_sacrificial == 2
    assert simulator.classifier.eligible_request_types == frozenset({"long"})
    assert isinstance(simulator.router, ProjectedCompletionRouter)
    assert simulator.router.nonpreemptible_weight == pytest.approx(0.25)
    assert simulator.router.tail_mode == "sink"
    assert simulator.router.load_view == "remaining"
    assert simulator.router.update_latency_ema is True
    assert isinstance(simulator.scheduler, TwoQueueScheduler)
    assert simulator.scheduler.normal_order == "bounded_srpt"
    assert simulator.scheduler.max_bypass == 2
    assert simulator.scheduler.aging_s == pytest.approx(30.0)
    assert simulator.scheduler.global_work_stealing is True
    assert simulator.scheduler.steal_order == "oldest"
    assert simulator.scheduler.steal_hysteresis_s == pytest.approx(5.0)
    assert simulator.scheduler.steal_cost_s == pytest.approx(0.25)
    assert simulator.scheduler.global_protected_pull is False
    assert simulator.scheduler.protected_pull_order == "max_risk"
    assert simulator.scheduler.protected_pull_risk_beta == pytest.approx(0.25)
    assert simulator.scheduler.protected_pull_band_risk_beta == pytest.approx(0.625)
    assert simulator.scheduler.protected_pull_band_min_pending == 10
    assert simulator.scheduler.protected_pull_band_max_pending == 27
    assert simulator.scheduler.protected_pull_risk_slack_s == pytest.approx(30.0)
    assert simulator.scheduler.protected_pull_guard_fraction == pytest.approx(0.04)
    assert simulator.scheduler.protected_pull_guard_max == 2
    assert simulator.scheduler.protected_pull_guard_min_pending == 10
    assert simulator.scheduler.protected_pull_tail_head_start is True
    assert simulator.scheduler.protected_pull_cost_s == pytest.approx(0.5)


@pytest.mark.parametrize(
    "path",
    [
        "workload.request_rate",
        "workload.utilization",
        "workload.request_types.0.weight",
        "workload.request_types.0.nominal_service_s",
        "workload.request_types.0.estimated_service_s",
        "service.encode_fraction",
        "service.decode_fraction",
        "service.actual_jitter_sigma",
        "service.estimate_error_sigma",
        "service.preemption_cost_s",
        "service.actual_service_scale",
        "service.hsdp.communication_overhead_weight",
        "topology.speed_factors.0",
        "topology.backends.0.speed",
        "policy.classifier.threshold_ratio",
        "policy.classifier.long_request_ratio",
        "policy.classifier.initial_min_service_s",
        "policy.classifier.initial_max_service_s",
        "policy.router.sacrificial_load_factor",
        "policy.router.nonpreemptible_weight",
        "policy.scheduler.aging_s",
        "policy.scheduler.preemption_hysteresis_s",
        "policy.scheduler.steal_hysteresis_s",
        "policy.scheduler.steal_cost_s",
        "policy.scheduler.protected_pull_risk_beta",
        "policy.scheduler.protected_pull_cost_s",
    ],
)
def test_all_numeric_float_fields_reject_bools(path: str) -> None:
    raw = _raw_config_for_numeric_path(path)
    invalid = apply_overrides(raw, [f"{path}=true"])

    with pytest.raises(ValueError, match="must be a number"):
        parse_experiment_config(invalid)


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_parse_rejects_non_finite_values_at_any_depth(non_finite: float) -> None:
    raw = _raw_config()
    raw["workload"]["request_types"][0]["metadata"] = {"nested": {"bad": non_finite}}

    with pytest.raises(ValueError, match=r"config\.workload\.request_types\[0\]\.metadata\.nested\.bad must be finite"):
        parse_experiment_config(raw)


def test_hsdp_enabled_requires_a_boolean() -> None:
    raw = _raw_config()
    raw["service"]["hsdp"]["enabled"] = 1

    with pytest.raises(ValueError, match=r"service\.hsdp\.enabled must be a boolean"):
        parse_experiment_config(raw)


@pytest.mark.parametrize(
    ("unsafe_value", "case_name"),
    [
        (date(2026, 7, 23), "timestamp"),
        ({"set-member"}, "set"),
        (b"binary-data", "binary"),
    ],
)
def test_load_rejects_non_json_native_yaml_values(
    tmp_path: Path,
    unsafe_value: Any,
    case_name: str,
) -> None:
    raw = _raw_config()
    raw["workload"]["request_types"][0]["metadata"] = {case_name: unsafe_value}

    with pytest.raises(ValueError, match="unsupported value type"):
        load_experiment_config(_write_yaml_config(tmp_path, raw))


def test_parse_rejects_non_json_native_tuple_metadata() -> None:
    raw = _raw_config()
    raw["workload"]["request_types"][0]["metadata"] = {"tuple": ("not", "a", "list")}

    with pytest.raises(ValueError, match="unsupported value type 'tuple'"):
        parse_experiment_config(raw)


def test_percentile_uses_hyndman_fan_type_7() -> None:
    values = [30.0, 0.0, 20.0, 10.0]

    assert percentile(values, 0) == 0.0
    assert percentile(values, 25) == pytest.approx(7.5)
    assert percentile(values, 50) == pytest.approx(15.0)
    assert percentile(values, 95) == pytest.approx(28.5)
    assert percentile(values, 100) == 30.0

    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile(values, 101)


def test_workload_matches_benchmark_seeded_choices_and_poisson_arrivals() -> None:
    seed = 17
    request_rate = 0.4
    raw = _raw_config(num_requests=12, request_rate=request_rate, seed=seed)
    raw["workload"]["request_types"] = [
        {
            "name": "short",
            "weight": 1.0,
            "width": 64,
            "height": 64,
            "num_inference_steps": 2,
            "num_frames": 1,
            "nominal_service_s": 2.0,
        },
        {
            "name": "long",
            "weight": 3.0,
            "width": 128,
            "height": 128,
            "num_inference_steps": 4,
            "num_frames": 1,
            "nominal_service_s": 8.0,
        },
    ]
    simulator = Simulator(parse_experiment_config(raw))

    profile_rng = random.Random(seed)
    expected_profiles = profile_rng.choices(
        ["short", "long"],
        weights=[1.0, 3.0],
        k=12,
    )
    arrival_rng = random.Random(seed)
    expected_arrivals = [0.0]
    for _ in range(1, 12):
        expected_arrivals.append(expected_arrivals[-1] + arrival_rng.expovariate(request_rate))

    assert [request.request_type.name for request in simulator.requests.values()] == expected_profiles
    assert [request.arrival_time_s for request in simulator.requests.values()] == pytest.approx(expected_arrivals)

    replay = Simulator(parse_experiment_config(raw))
    assert [request.request_type.name for request in replay.requests.values()] == expected_profiles
    assert [request.arrival_time_s for request in replay.requests.values()] == pytest.approx(expected_arrivals)


def test_current_minimum_guard_defers_quota_while_idealized_uses_it() -> None:
    current = QuotaTailClassifier(
        quota_every=2,
        quota_amount=1,
        threshold_ratio=0.8,
        long_request_ratio=1.5,
    )
    idealized = QuotaTailClassifier(
        quota_every=2,
        quota_amount=1,
        threshold_ratio=0.8,
        long_request_ratio=None,
    )
    requests = [_request_view(index, Priority.NORMAL) for index in range(3)]

    assert current.classify(requests[0], 10.0).priority == Priority.NORMAL
    current_second = current.classify(requests[1], 12.0)
    assert current_second.priority == Priority.NORMAL
    assert current_second.details["clearly_long"] is False
    assert current_second.details["credits_after"] == 1
    current_third = current.classify(requests[2], 20.0)
    assert current_third.priority == Priority.SACRIFICIAL
    assert current_third.details["credits_after"] == 0

    assert idealized.classify(requests[0], 10.0).priority == Priority.NORMAL
    idealized_second = idealized.classify(requests[1], 12.0)
    assert idealized_second.priority == Priority.SACRIFICIAL
    assert idealized_second.details["clearly_long"] is True
    assert idealized_second.details["credits_after"] == 0
    assert idealized.classify(requests[2], 20.0).priority == Priority.NORMAL


def test_quota_tail_can_cap_and_restrict_sacrificial_request_types() -> None:
    classifier = QuotaTailClassifier(
        quota_every=1,
        quota_amount=1,
        threshold_ratio=0.0,
        long_request_ratio=None,
        max_sacrificial=2,
        eligible_request_types=["long"],
    )
    short = _request_view(0, Priority.NORMAL, request_type_name="short")
    long_requests = [_request_view(index, Priority.NORMAL, request_type_name="long") for index in range(1, 4)]

    short_decision = classifier.classify(short, 10.0)
    decisions = [classifier.classify(request, 10.0) for request in long_requests]

    assert short_decision.priority == Priority.NORMAL
    assert short_decision.details["eligible_request_type"] is False
    assert [decision.priority for decision in decisions] == [
        Priority.SACRIFICIAL,
        Priority.SACRIFICIAL,
        Priority.NORMAL,
    ]
    assert decisions[-1].details["below_sacrificial_cap"] is False
    assert decisions[-1].details["sacrificial_count"] == 2


def test_quota_tail_explicit_release_points_ignore_initial_dispatcher_counter() -> None:
    classifier = QuotaTailClassifier(
        quota_every=1_000,
        quota_amount=1,
        threshold_ratio=0.0,
        long_request_ratio=None,
        initial_arrival_counter=17,
        max_sacrificial=2,
        eligible_request_types=["long"],
        credit_release_requests=[2, 4],
    )
    requests = [_request_view(index, Priority.NORMAL, request_type_name="long") for index in range(5)]

    decisions = [classifier.classify(request, 10.0) for request in requests]

    assert [decision.priority for decision in decisions] == [
        Priority.NORMAL,
        Priority.SACRIFICIAL,
        Priority.NORMAL,
        Priority.SACRIFICIAL,
        Priority.NORMAL,
    ]
    assert [decision.details["classified_requests"] for decision in decisions] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("max_sacrificial", -1, "max_sacrificial cannot be negative"),
        ("max_sacrificial", True, "max_sacrificial must be an integer"),
        (
            "eligible_request_types",
            "long",
            "eligible_request_types must be a list or null",
        ),
        (
            "eligible_request_types",
            [""],
            "eligible_request_types entries must be non-empty strings",
        ),
        (
            "credit_release_requests",
            [0],
            "credit_release_requests entries must be positive integers",
        ),
        (
            "credit_release_requests",
            [2, 2],
            "credit_release_requests entries must be unique",
        ),
    ],
)
def test_quota_tail_rejects_invalid_cap_and_request_type_options(
    option: str,
    value: Any,
    error: str,
) -> None:
    kwargs: dict[str, Any] = {
        "quota_every": 20,
        "quota_amount": 1,
        "threshold_ratio": 0.8,
        "long_request_ratio": 1.5,
        option: value,
    }

    with pytest.raises(ValueError, match=error):
        QuotaTailClassifier(**kwargs)


def test_online_prefix_safe_tail_never_needs_a_finite_horizon() -> None:
    classifier = OnlineCreditTailClassifier(
        credit_mode="prefix_safe",
        target_quantile=0.95,
        token_rate=0.04,
        credit_capacity=None,
        threshold_ratio=0.0,
        long_request_ratio=None,
    )

    decisions = [classifier.classify(_request_view(index, Priority.NORMAL), 10.0) for index in range(1, 101)]
    tail_positions = [
        index for index, decision in enumerate(decisions, start=1) if decision.priority == Priority.SACRIFICIAL
    ]

    assert tail_positions == [21, 41, 61, 81]
    for prefix, decision in enumerate(decisions, start=1):
        assert decision.details["sacrificial_count"] <= math.floor(0.05 * (prefix - 1) + 1e-12)


@pytest.mark.parametrize(
    ("initial_credits", "expected_positions"),
    [
        (0.0, [25, 50, 75, 100]),
        (1.0, [1, 26, 51, 76]),
    ],
)
def test_online_smooth_bucket_bounds_unknown_measurement_windows(
    initial_credits: float,
    expected_positions: list[int],
) -> None:
    classifier = OnlineCreditTailClassifier(
        credit_mode="smooth_token_bucket",
        target_quantile=0.95,
        token_rate=0.04,
        credit_capacity=1.0,
        threshold_ratio=0.0,
        long_request_ratio=None,
        initial_credits=initial_credits,
    )

    decisions = [classifier.classify(_request_view(index, Priority.NORMAL), 10.0) for index in range(1, 126)]
    tail_positions = [
        index for index, decision in enumerate(decisions, start=1) if decision.priority == Priority.SACRIFICIAL
    ]

    assert tail_positions[:4] == expected_positions
    for window_size, limit in [(50, 2), (100, 4)]:
        for start in range(1, len(decisions) - window_size + 2):
            stop = start + window_size
            assert sum(start <= position < stop for position in tail_positions) <= limit


def test_online_prefix_safe_congestion_gate_preserves_credit() -> None:
    classifier = OnlineCreditTailClassifier(
        credit_mode="prefix_safe",
        target_quantile=0.5,
        token_rate=0.04,
        credit_capacity=None,
        threshold_ratio=0.0,
        long_request_ratio=None,
        congestion_threshold=0.5,
    )
    empty = (_backend_view("backend"),)
    congested = (_backend_view("backend", normal_load_s=10.0),)

    first = classifier.classify(_request_view(1, Priority.NORMAL), 10.0, backends=empty)
    second = classifier.classify(_request_view(2, Priority.NORMAL), 10.0, backends=empty)
    third = classifier.classify(_request_view(3, Priority.NORMAL), 10.0, backends=empty)
    fourth = classifier.classify(_request_view(4, Priority.NORMAL), 10.0, backends=congested)

    assert first.priority == Priority.NORMAL
    assert second.priority == Priority.NORMAL
    assert third.priority == Priority.NORMAL
    assert third.details["reason"] == "below_congestion_threshold"
    assert third.details["credits_after"] == pytest.approx(1.0)
    assert fourth.priority == Priority.SACRIFICIAL
    assert fourth.details["credits_after"] == pytest.approx(0.0)


def test_online_credit_tail_builds_from_strict_config() -> None:
    classifier = build_classifier(
        parse_experiment_config(
            _raw_config(
                classifier={
                    "type": "online_credit_tail",
                    "credit_mode": "smooth_token_bucket",
                    "target_quantile": 0.95,
                    "token_rate": 0.04,
                    "credit_capacity": 1.0,
                    "threshold_ratio": 0.8,
                    "long_request_ratio": 1.5,
                    "initial_credits": 0.5,
                    "initial_min_service_s": None,
                    "initial_max_service_s": 0.0,
                    "eligible_request_types": None,
                    "congestion_threshold": 0.5,
                    "cooldown_requests": 25,
                },
            )
        ).policy.classifier
    )

    assert isinstance(classifier, OnlineCreditTailClassifier)
    assert classifier.credit_mode == "smooth_token_bucket"
    assert classifier.credit_capacity == pytest.approx(1.0)
    assert classifier.congestion_threshold == pytest.approx(0.5)
    assert classifier.cooldown_requests == 25


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("credit_mode", "future_aware", "credit_mode must be one of"),
        ("target_quantile", 1.0, "target_quantile must be between"),
        ("token_rate", 0.0, "token_rate must be between"),
        ("credit_capacity", 0.5, "credit_capacity must be at least"),
        ("congestion_threshold", -1.0, "congestion_threshold cannot be negative"),
        ("cooldown_requests", -1, "cooldown_requests cannot be negative"),
    ],
)
def test_online_credit_tail_rejects_invalid_options(option: str, value: Any, error: str) -> None:
    kwargs: dict[str, Any] = {
        "credit_mode": "prefix_safe",
        "target_quantile": 0.95,
        "token_rate": 0.04,
        "credit_capacity": 1.0,
        "threshold_ratio": 0.8,
        "long_request_ratio": 1.5,
        option: value,
    }

    with pytest.raises(ValueError, match=error):
        OnlineCreditTailClassifier(**kwargs)


def test_quantile_risk_tail_uses_prior_online_risks_and_hard_budget() -> None:
    classifier = QuantileRiskTailClassifier(
        tail_budget=2,
        risk_quantile=0.5,
        min_observations=2,
    )
    backends = (
        _backend_view("fast-empty", speed=1.0),
        _backend_view("fast-busy", speed=2.0, normal_load_s=20.0),
    )
    estimates = [10.0, 20.0, 14.0, 30.0, 40.0, 50.0]
    decisions = [
        classifier.classify(
            _request_view(index, Priority.NORMAL, estimated_total_s=estimate),
            estimate,
            now_s=float(index),
            backends=backends,
        )
        for index, estimate in enumerate(estimates)
    ]

    assert [decision.priority for decision in decisions] == [
        Priority.NORMAL,
        Priority.NORMAL,
        Priority.NORMAL,
        Priority.SACRIFICIAL,
        Priority.SACRIFICIAL,
        Priority.NORMAL,
    ]
    assert decisions[1].details["risk_threshold_s"] is None
    assert decisions[1].details["risk_history_size_before"] == 1
    assert decisions[2].details["risk_threshold_s"] == pytest.approx(15.0)
    assert decisions[2].details["protected_risk_s"] == pytest.approx(14.0)
    assert decisions[2].details["reason"] == "below_risk_threshold"
    assert decisions[3].details["risk_threshold_s"] == pytest.approx(14.0)
    assert decisions[5].details["reason"] == "tail_budget_exhausted"
    assert decisions[5].details["sacrificial_count"] == 2


def test_quantile_risk_tail_accounts_for_visible_protected_backend_load() -> None:
    request = _request_view(0, Priority.NORMAL, estimated_total_s=10.0)
    backends = (
        _backend_view("tail-loaded", normal_load_s=10.0, sacrificial_load_s=100.0),
        _backend_view("normal-loaded", normal_load_s=30.0),
    )
    ignores_preemptible_tail = QuantileRiskTailClassifier(
        tail_budget=0,
        risk_quantile=0.5,
        min_observations=1,
        tail_load_weight=0.0,
    )
    counts_tail = QuantileRiskTailClassifier(
        tail_budget=0,
        risk_quantile=0.5,
        min_observations=1,
        tail_load_weight=1.0,
    )

    ignored = ignores_preemptible_tail.classify(request, 10.0, now_s=0.0, backends=backends)
    counted = counts_tail.classify(request, 10.0, now_s=0.0, backends=backends)

    assert ignored.details["protected_risk_s"] == pytest.approx(20.0)
    assert counted.details["protected_risk_s"] == pytest.approx(40.0)


def test_quantile_risk_tail_history_uses_eligible_shadow_normal_risks() -> None:
    classifier = QuantileRiskTailClassifier(
        tail_budget=2,
        risk_quantile=0.5,
        min_observations=2,
        eligible_request_types=["long"],
    )
    inputs = [
        ("short", 1_000.0),
        ("medium", 900.0),
        ("long", 10.0),
        ("long", 20.0),
        ("long", 30.0),
        ("long", 18.0),
    ]
    decisions = [
        classifier.classify(
            _request_view(
                index,
                Priority.NORMAL,
                request_type_name=request_type,
                estimated_total_s=estimate,
            ),
            estimate,
            now_s=float(index),
            backends=(_backend_view("backend"),),
        )
        for index, (request_type, estimate) in enumerate(inputs)
    ]

    assert decisions[0].details["risk_observed"] is False
    assert decisions[1].details["risk_observed"] is False
    assert decisions[2].details["risk_history_size_before"] == 0
    assert decisions[4].priority == Priority.SACRIFICIAL
    assert decisions[4].details["risk_threshold_s"] == pytest.approx(15.0)
    assert decisions[4].details["risk_observed"] is True
    # The previous Tail request contributes its counterfactual Normal risk of
    # 30 seconds, lifting the next median from 15 to 20 seconds.
    assert decisions[5].details["risk_threshold_s"] == pytest.approx(20.0)
    assert decisions[5].priority == Priority.NORMAL


def test_quantile_risk_tail_release_window_and_forced_end_consumption() -> None:
    classifier = QuantileRiskTailClassifier(
        tail_budget=2,
        risk_quantile=1.0,
        min_observations=1,
        admission_start_request=2,
        admission_end_request=5,
        credit_release_requests=[2, 4],
        eligible_request_types=["long"],
        force_consume_at_end=True,
    )
    estimates = [100.0, 1.0, 1.0, 1.0, 1.0]
    decisions = [
        classifier.classify(
            _request_view(
                index,
                Priority.NORMAL,
                request_type_name="long",
                estimated_total_s=estimate,
            ),
            estimate,
            now_s=float(index),
            backends=(_backend_view("backend"),),
        )
        for index, estimate in enumerate(estimates)
    ]

    assert [decision.priority for decision in decisions] == [
        Priority.NORMAL,
        Priority.NORMAL,
        Priority.NORMAL,
        Priority.SACRIFICIAL,
        Priority.SACRIFICIAL,
    ]
    assert decisions[0].details["reason"] == "outside_admission_window"
    assert decisions[1].details["quota_added"] == 1
    assert decisions[1].details["reason"] == "below_risk_threshold"
    assert decisions[3].details["quota_added"] == 1
    assert decisions[3].details["remaining_admission_positions"] == 2
    assert decisions[3].details["reason"] == "forced_end_budget"
    assert decisions[4].details["reason"] == "forced_end_budget"
    assert decisions[4].details["credits_after"] == 0


def test_quantile_risk_tail_builds_from_strict_config_and_engine_supplies_online_views() -> None:
    raw = _raw_config(
        num_requests=3,
        request_rate=1.0,
        classifier={
            "type": "quantile_risk_tail",
            "tail_budget": 1,
            "risk_quantile": 0.5,
            "min_observations": 1,
            "history_window": 2,
            "tail_load_weight": 0.0,
            "risk_threshold_multiplier": 1.0,
            "risk_margin_s": 0.0,
            "admission_start_request": 2,
            "admission_end_request": 3,
            "credit_release_requests": [2],
            "credits_per_release": 1,
            "eligible_request_types": ["only"],
            "history_eligible_only": True,
            "force_consume_at_end": True,
        },
    )
    simulator = Simulator(parse_experiment_config(raw), collect_events=True)

    assert isinstance(simulator.classifier, QuantileRiskTailClassifier)
    result = simulator.run()
    arrivals = [event for event in result.events if event["event"] == "arrival"]

    assert len(arrivals) == 3
    assert arrivals[0]["classifier"]["protected_risk_s"] == pytest.approx(9.0)
    assert arrivals[0]["classifier"]["risk_history_size_before"] == 0
    assert arrivals[1]["classifier"]["risk_history_size_before"] == 1
    assert all(event["classifier"]["risk_history_size_before"] <= 2 for event in arrivals)
    assert all(event["classifier"]["protected_risk_s"] >= 0.0 for event in arrivals)


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("tail_budget", -1, "tail_budget cannot be negative"),
        ("risk_quantile", 1.1, "risk_quantile must be between 0 and 1"),
        ("min_observations", 0, "min_observations must be positive"),
        ("history_window", 1, "history_window cannot be smaller than min_observations"),
        ("admission_start_request", 0, "admission_start_request must be positive"),
        ("admission_end_request", 1, "admission_end_request cannot precede"),
        ("credits_per_release", 0, "credits_per_release must be positive"),
        ("credit_release_requests", [1], "cannot precede admission_start_request"),
    ],
)
def test_quantile_risk_tail_rejects_invalid_options(option: str, value: Any, error: str) -> None:
    kwargs: dict[str, Any] = {
        "tail_budget": 2,
        "risk_quantile": 0.8,
        "min_observations": 2,
        "history_window": 2,
        "admission_start_request": 2,
        "admission_end_request": 5,
        option: value,
    }

    with pytest.raises(ValueError, match=error):
        QuantileRiskTailClassifier(**kwargs)


def test_current_two_queue_is_normal_fifo_sacrificial_lifo_and_selectively_preemptive() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
    )
    backend = _backend_view("backend")
    normal_1 = _request_view(1, Priority.NORMAL)
    sacrificial_2 = _request_view(2, Priority.SACRIFICIAL)
    normal_3 = _request_view(3, Priority.NORMAL)
    sacrificial_4 = _request_view(4, Priority.SACRIFICIAL)

    assert (
        scheduler.choose(
            now_s=0.0,
            backend=backend,
            incumbent=None,
            pending=(sacrificial_4, normal_3, sacrificial_2, normal_1),
        )
        == normal_1.request_id
    )
    assert (
        scheduler.choose(
            now_s=0.0,
            backend=backend,
            incumbent=None,
            pending=(sacrificial_2, sacrificial_4),
        )
        == sacrificial_4.request_id
    )

    running_sacrificial = _request_view(0, Priority.SACRIFICIAL, status=RequestStatus.RUNNING)
    assert (
        scheduler.choose(
            now_s=1.0,
            backend=backend,
            incumbent=running_sacrificial,
            pending=(normal_3, normal_1),
        )
        == normal_1.request_id
    )

    running_normal = _request_view(5, Priority.NORMAL, status=RequestStatus.RUNNING)
    assert (
        scheduler.choose(
            now_s=1.0,
            backend=backend,
            incumbent=running_normal,
            pending=(normal_1, sacrificial_4),
        )
        == running_normal.request_id
    )
    assert (
        scheduler.choose(
            now_s=1.0,
            backend=backend,
            incumbent=running_sacrificial,
            pending=(sacrificial_4,),
        )
        == running_sacrificial.request_id
    )


def test_two_queue_srpt_orders_normal_requests_by_estimated_remaining_time() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="srpt",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
    )
    backend = _backend_view("backend")
    long = _request_view(0, Priority.NORMAL, estimated_remaining_s=30.0)
    short = _request_view(1, Priority.NORMAL, estimated_remaining_s=3.0)

    assert (
        scheduler.choose(
            now_s=2.0,
            backend=backend,
            incumbent=None,
            pending=(long, short),
        )
        == short.request_id
    )


@pytest.mark.parametrize("normal_order", ["fifo", "lifo"])
def test_two_queue_arrival_orders_do_not_preempt_a_normal_incumbent(
    normal_order: str,
) -> None:
    scheduler = TwoQueueScheduler(
        normal_order=normal_order,
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
    )
    incumbent = _request_view(
        0,
        Priority.NORMAL,
        status=RequestStatus.RUNNING,
        estimated_remaining_s=30.0,
    )
    pending_short = _request_view(
        1,
        Priority.NORMAL,
        estimated_remaining_s=1.0,
    )

    assert (
        scheduler.choose(
            now_s=3.0,
            backend=_backend_view("backend"),
            incumbent=incumbent,
            pending=(pending_short,),
        )
        == incumbent.request_id
    )


def test_two_queue_arrival_plus_cost_balances_age_and_service_cost() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="arrival_plus_cost",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
    )
    backend = _backend_view("backend")
    old_long = _request_view(
        0,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_total_s=100.0,
    )
    new_short = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=10.0,
        estimated_total_s=20.0,
    )

    assert (
        scheduler.choose(
            now_s=10.0,
            backend=backend,
            incumbent=None,
            pending=(old_long, new_short),
        )
        == new_short.request_id
    )

    much_older_long = _request_view(
        2,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_total_s=50.0,
    )
    much_newer_short = _request_view(
        3,
        Priority.NORMAL,
        arrival_time_s=40.0,
        estimated_total_s=20.0,
    )
    assert (
        scheduler.choose(
            now_s=40.0,
            backend=backend,
            incumbent=None,
            pending=(much_older_long, much_newer_short),
        )
        == much_older_long.request_id
    )


def test_two_queue_arrival_plus_cost_does_not_preempt_a_normal_incumbent() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="arrival_plus_cost",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
    )
    incumbent = _request_view(
        0,
        Priority.NORMAL,
        status=RequestStatus.RUNNING,
        arrival_time_s=0.0,
        estimated_total_s=100.0,
    )
    pending_short = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=1.0,
        estimated_total_s=1.0,
    )

    assert (
        scheduler.choose(
            now_s=3.0,
            backend=_backend_view("backend"),
            incumbent=incumbent,
            pending=(pending_short,),
        )
        == incumbent.request_id
    )


def test_two_queue_size_class_fifo_prioritizes_classes_and_preserves_class_fifo() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="size_class_fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        size_class_order=["short", "medium", "long"],
    )
    backend = _backend_view("backend")
    long = _request_view(0, Priority.NORMAL, request_type_name="long")
    medium = _request_view(1, Priority.NORMAL, request_type_name="medium")
    earlier_short = _request_view(2, Priority.NORMAL, request_type_name="short")
    later_short = _request_view(3, Priority.NORMAL, request_type_name="short")

    assert (
        scheduler.choose(
            now_s=4.0,
            backend=backend,
            incumbent=None,
            pending=(later_short, medium, long, earlier_short),
        )
        == earlier_short.request_id
    )


def test_two_queue_size_class_fifo_does_not_preempt_a_normal_incumbent() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="size_class_fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        size_class_order=["short", "medium", "long"],
    )
    incumbent_long = _request_view(
        0,
        Priority.NORMAL,
        status=RequestStatus.RUNNING,
        request_type_name="long",
    )
    pending_short = _request_view(
        1,
        Priority.NORMAL,
        request_type_name="short",
    )

    assert (
        scheduler.choose(
            now_s=3.0,
            backend=_backend_view("backend"),
            incumbent=incumbent_long,
            pending=(pending_short,),
        )
        == incumbent_long.request_id
    )


def test_two_queue_bounded_size_class_fifo_limits_bypass() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="bounded_size_class_fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        size_class_order=["short", "medium", "long"],
        max_bypass=1,
        aging_s=None,
    )
    backend = _backend_view("backend")
    old_long = _request_view(0, Priority.NORMAL, request_type_name="long")
    short_1 = _request_view(1, Priority.NORMAL, request_type_name="short")
    short_2 = _request_view(2, Priority.NORMAL, request_type_name="short")

    assert (
        scheduler.choose(
            now_s=2.0,
            backend=backend,
            incumbent=None,
            pending=(old_long, short_1),
        )
        == short_1.request_id
    )
    assert (
        scheduler.choose(
            now_s=3.0,
            backend=backend,
            incumbent=None,
            pending=(old_long, short_2),
        )
        == old_long.request_id
    )


@pytest.mark.parametrize("normal_order", ["srpt", "bounded_srpt"])
def test_two_queue_remaining_time_orders_preempt_normal_at_step_boundary(
    normal_order: str,
) -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=2,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            scheduler={
                "type": "two_queue",
                "normal_order": normal_order,
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 1.0])
    short = simulator.requests["request-00001"]
    short.estimated_encode_s = 0.0
    short.estimated_step_s = 1.0 / 3.0
    short.estimated_decode_s = 0.0

    result = simulator.run()
    preempt = next(event for event in result.events if event["event"] == "preempt")

    assert preempt["time_s"] == pytest.approx(3.0)
    assert preempt["request_id"] == "request-00000"
    assert preempt["selected_request_id"] == "request-00001"
    assert result.metrics["preemptions"] >= 1


def test_two_queue_bounded_srpt_limits_how_often_an_old_request_is_bypassed() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="bounded_srpt",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        max_bypass=1,
        aging_s=None,
    )
    backend = _backend_view("backend")
    old_long = _request_view(0, Priority.NORMAL, estimated_remaining_s=30.0)
    short_1 = _request_view(1, Priority.NORMAL, estimated_remaining_s=3.0)
    short_2 = _request_view(2, Priority.NORMAL, estimated_remaining_s=2.0)

    assert (
        scheduler.choose(
            now_s=2.0,
            backend=backend,
            incumbent=None,
            pending=(old_long, short_1),
        )
        == short_1.request_id
    )
    assert (
        scheduler.choose(
            now_s=3.0,
            backend=backend,
            incumbent=None,
            pending=(old_long, short_2),
        )
        == old_long.request_id
    )


def test_two_queue_bounded_srpt_aging_forces_an_old_waiter() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="bounded_srpt",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        max_bypass=None,
        aging_s=5.0,
    )
    backend = _backend_view("backend")
    aged_long = _request_view(
        0,
        Priority.NORMAL,
        estimated_remaining_s=30.0,
        arrival_time_s=0.0,
    )
    new_short = _request_view(
        1,
        Priority.NORMAL,
        estimated_remaining_s=1.0,
        arrival_time_s=9.0,
    )

    assert (
        scheduler.choose(
            now_s=10.0,
            backend=backend,
            incumbent=None,
            pending=(aged_long, new_short),
        )
        == aged_long.request_id
    )


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        (
            "global_work_stealing",
            1,
            "global_work_stealing must be a boolean",
        ),
        (
            "global_protected_pull",
            1,
            "global_protected_pull must be a boolean",
        ),
        ("steal_order", "random", "scheduler.steal_order must be"),
        ("protected_pull_order", "random", "scheduler.protected_pull_order must be"),
        (
            "steal_hysteresis_s",
            -1.0,
            "steal_hysteresis_s cannot be negative",
        ),
        ("steal_cost_s", -1.0, "steal_cost_s cannot be negative"),
        (
            "protected_pull_risk_beta",
            -1.0,
            "protected_pull_risk_beta cannot be negative",
        ),
        (
            "protected_pull_band_risk_beta",
            -1.0,
            "protected_pull_band_risk_beta cannot be negative",
        ),
        (
            "protected_pull_band_min_pending",
            0,
            "protected_pull_band_min_pending must be positive",
        ),
        (
            "protected_pull_risk_slack_s",
            -1.0,
            "protected_pull_risk_slack_s cannot be negative",
        ),
        (
            "protected_pull_guard_fraction",
            1.0,
            "protected_pull_guard_fraction must be in",
        ),
        (
            "protected_pull_guard_max",
            -1,
            "protected_pull_guard_max cannot be negative",
        ),
        (
            "protected_pull_guard_min_pending",
            0,
            "protected_pull_guard_min_pending must be positive",
        ),
        (
            "protected_pull_tail_head_start",
            1,
            "protected_pull_tail_head_start must be a boolean",
        ),
        ("protected_pull_cost_s", -1.0, "protected_pull_cost_s cannot be negative"),
    ],
)
def test_two_queue_rejects_invalid_work_stealing_options(
    option: str,
    value: Any,
    error: str,
) -> None:
    kwargs: dict[str, Any] = {
        "normal_order": "fifo",
        "sacrificial_order": "lifo",
        "preempt_normal_over_sacrificial": True,
        option: value,
    }

    with pytest.raises(ValueError, match=error):
        TwoQueueScheduler(**kwargs)


def test_work_stealing_and_global_protected_pull_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TwoQueueScheduler(
            normal_order="fifo",
            sacrificial_order="lifo",
            preempt_normal_over_sacrificial=True,
            global_work_stealing=True,
            global_protected_pull=True,
        )


def test_idle_backend_steals_unstarted_protected_work_and_charges_cost() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=3,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            backend_count=2,
            speed_factors=[1.0, 10.0],
            router={"type": "round_robin"},
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_work_stealing": True,
                "steal_order": "max_risk",
                "steal_hysteresis_s": 0.5,
                "steal_cost_s": 1.0,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 0.0, 0.0])

    result = simulator.run()
    steal = next(event for event in result.events if event["event"] == "steal")
    stolen = next(request for request in result.requests if request["request_id"] == steal["request_id"])

    assert steal["time_s"] == pytest.approx(0.9)
    assert steal["source_backend"] == "backend-0"
    assert steal["backend"] == "backend-1"
    assert steal["benefit_s"] == pytest.approx(17.1)
    assert steal["cost_s"] == pytest.approx(1.0)
    assert stolen["request_id"] == "request-00002"
    assert stolen["backend"] == "backend-1"
    assert stolen["first_start_time_s"] == pytest.approx(1.9)
    assert result.metrics["steals"] == 1
    assert result.metrics["steal_cost_total_s"] == pytest.approx(1.0)
    assert result.metrics["switch_cost_total_s"] == pytest.approx(0.0)


def test_tail_step_boundary_steals_remote_protected_before_tail_continues() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=3,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            backend_count=2,
            classifier={
                "type": "quota_tail",
                "quota_every": 1_000,
                "quota_amount": 0,
                "threshold_ratio": 0.0,
                "long_request_ratio": None,
                "initial_credits": 1,
            },
            router={
                "type": "projected_completion",
                "nonpreemptible_weight": 100.0,
                "tail_mode": "sink",
                "load_view": "assigned",
                "update_latency_ema": False,
            },
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_work_stealing": True,
                "steal_order": "max_risk",
                "steal_hysteresis_s": 0.0,
                "steal_cost_s": 0.0,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 0.0, 0.0])

    result = simulator.run()
    steal = next(event for event in result.events if event["event"] == "steal")
    preempt = next(event for event in result.events if event["event"] == "preempt")

    assert steal["time_s"] == pytest.approx(3.0)
    assert steal["request_id"] == "request-00002"
    assert steal["source_backend"] == "backend-1"
    assert steal["backend"] == "backend-0"
    assert preempt["request_id"] == "request-00000"
    assert preempt["selected_request_id"] == "request-00002"
    assert result.requests[0]["priority"] == Priority.SACRIFICIAL.value
    assert result.requests[0]["backend"] == "backend-0"
    assert result.requests[1]["backend"] == "backend-1"
    assert result.metrics["steals"] == 1
    assert result.metrics["preemptions"] == 1


def test_work_stealing_hysteresis_can_keep_original_backend_binding() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=3,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            backend_count=2,
            speed_factors=[1.0, 10.0],
            router={"type": "round_robin"},
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_work_stealing": True,
                "steal_order": "max_risk",
                "steal_hysteresis_s": 100.0,
                "steal_cost_s": 0.0,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 0.0, 0.0])

    result = simulator.run()

    assert not [event for event in result.events if event["event"] == "steal"]
    assert result.requests[2]["backend"] == "backend-0"
    assert result.metrics["steals"] == 0
    assert result.metrics["steal_cost_total_s"] == pytest.approx(0.0)


def test_work_stealing_ignores_tail_and_already_started_requests() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=2,
            backend_count=2,
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_work_stealing": True,
            },
        )
    )
    simulator = Simulator(config)
    target, donor = simulator.backends.values()
    started_normal = simulator.requests["request-00000"]
    tail = simulator.requests["request-00001"]
    started_normal.status = RequestStatus.WAITING
    started_normal.backend_name = donor.name
    started_normal.first_start_time_s = 0.0
    tail.status = RequestStatus.WAITING
    tail.backend_name = donor.name
    tail.priority = Priority.SACRIFICIAL
    donor.pending_ids[:] = [started_normal.request_id, tail.request_id]
    donor.assigned_normal_load_s = started_normal.estimated_total_on_backend_s(donor.config.speed)
    donor.assigned_sacrificial_load_s = tail.estimated_total_on_backend_s(donor.config.speed)
    donor.inflight_normal = 1
    donor.inflight_sacrificial = 1

    assert simulator._maybe_steal_protected(target, incumbent=None) is None
    assert donor.pending_ids == [started_normal.request_id, tail.request_id]
    assert target.pending_ids == []
    assert simulator._steals == 0


@pytest.mark.parametrize(
    ("steal_order", "expected_request_id"),
    [
        ("oldest", "request-00000"),
        ("max_risk", "request-00001"),
    ],
)
def test_work_stealing_candidate_order_is_configurable(
    steal_order: str,
    expected_request_id: str,
) -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=2,
            backend_count=2,
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_work_stealing": True,
                "steal_order": steal_order,
            },
        )
    )
    simulator = Simulator(config)
    target, donor = simulator.backends.values()
    old_short = simulator.requests["request-00000"]
    new_long = simulator.requests["request-00001"]
    old_short.arrival_time_s = 0.0
    old_short.estimated_encode_s = 0.0
    old_short.estimated_step_s = 1.0 / old_short.total_steps
    old_short.estimated_decode_s = 0.0
    new_long.arrival_time_s = 10.0
    new_long.estimated_encode_s = 0.0
    new_long.estimated_step_s = 20.0 / new_long.total_steps
    new_long.estimated_decode_s = 0.0
    for request in (old_short, new_long):
        request.status = RequestStatus.WAITING
        request.backend_name = donor.name
    donor.pending_ids[:] = [old_short.request_id, new_long.request_id]
    donor.assigned_normal_load_s = sum(
        request.estimated_total_on_backend_s(donor.config.speed) for request in (old_short, new_long)
    )
    donor.inflight_normal = 2
    simulator._now_s = 20.0

    assert simulator._maybe_steal_protected(target, incumbent=None) == expected_request_id
    assert simulator.requests[expected_request_id].backend_name == target.name
    assert donor.inflight_normal == 1
    assert target.inflight_normal == 1


def test_deferred_protected_binding_uses_fastest_idle_backend_and_charges_pull_cost() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=1,
            service_s=9.0,
            steps=3,
            backend_count=2,
            speed_factors=[1.0, 2.0],
            router={"type": "round_robin"},
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_protected_pull": True,
                "protected_pull_order": "fifo",
                "protected_pull_cost_s": 2.0,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0])

    result = simulator.run()
    arrival = next(event for event in result.events if event["event"] == "arrival")
    pull = next(event for event in result.events if event["event"] == "protected_pull")
    request = result.requests[0]

    assert arrival["binding"] == "deferred"
    assert "backend" not in arrival
    assert pull["backend"] == "backend-1"
    assert pull["protected_pull_order"] == "fifo"
    assert pull["cost_s"] == pytest.approx(2.0)
    assert request["backend"] == "backend-1"
    assert request["first_start_time_s"] == pytest.approx(2.0)
    assert request["central_wait_s"] == pytest.approx(0.0)
    assert result.metrics["protected_pulls"] == 1
    assert result.metrics["protected_pull_cost_total_s"] == pytest.approx(2.0)
    assert result.metrics["central_normal_queue_max_depth"] == 1
    assert result.metrics["central_wait_mean_s"] == pytest.approx(0.0)
    assert result.metrics["central_wait_p50_s"] == pytest.approx(0.0)
    assert result.metrics["central_wait_p95_s"] == pytest.approx(0.0)
    assert result.metrics["steals"] == 0


def test_disabled_global_protected_pull_preserves_legacy_runtime_behavior() -> None:
    base_raw = _raw_config(
        num_requests=5,
        request_rate=0.5,
        service_s=9.0,
        steps=3,
        backend_count=2,
        router={"type": "round_robin"},
        scheduler={
            "type": "two_queue",
            "normal_order": "fifo",
            "sacrificial_order": "lifo",
            "preempt_normal_over_sacrificial": True,
        },
    )
    explicit_off_raw = copy.deepcopy(base_raw)
    explicit_off_raw["policy"]["scheduler"].update(
        {
            "global_protected_pull": False,
            "protected_pull_order": "max_risk",
            "protected_pull_cost_s": 2.0,
        }
    )

    legacy = Simulator(parse_experiment_config(base_raw), collect_events=True).run()
    explicit_off = Simulator(parse_experiment_config(explicit_off_raw), collect_events=True).run()

    assert explicit_off.metrics == legacy.metrics
    assert explicit_off.requests == legacy.requests
    assert explicit_off.events == legacy.events


@pytest.mark.parametrize(
    ("protected_pull_order", "expected_pull_ids"),
    [
        ("fifo", ["request-00000", "request-00001", "request-00002"]),
        ("max_risk", ["request-00000", "request-00002", "request-00001"]),
        (
            "cost_damped_risk",
            ["request-00000", "request-00002", "request-00001"],
        ),
        (
            "queue_band_risk",
            ["request-00000", "request-00002", "request-00001"],
        ),
        ("risk_slack_srpt", ["request-00000", "request-00002", "request-00001"]),
        ("guarded_max_risk", ["request-00000", "request-00002", "request-00001"]),
        ("arrival_plus_cost", ["request-00000", "request-00001", "request-00002"]),
        ("highest_response_ratio", ["request-00000", "request-00001", "request-00002"]),
    ],
)
def test_global_protected_pull_order_is_online_and_configurable(
    protected_pull_order: str,
    expected_pull_ids: list[str],
) -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=3,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            backend_count=1,
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_protected_pull": True,
                "protected_pull_order": protected_pull_order,
                "protected_pull_cost_s": 0.0,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 1.0, 2.0])
    queued_short = simulator.requests["request-00001"]
    queued_long = simulator.requests["request-00002"]
    queued_short.estimated_encode_s = 0.0
    queued_short.estimated_step_s = 1.0 / queued_short.total_steps
    queued_short.estimated_decode_s = 0.0
    queued_long.estimated_encode_s = 0.0
    queued_long.estimated_step_s = 20.0 / queued_long.total_steps
    queued_long.estimated_decode_s = 0.0

    result = simulator.run()
    pulls = [event for event in result.events if event["event"] == "protected_pull"]

    assert [event["request_id"] for event in pulls] == expected_pull_ids
    assert result.metrics["protected_pulls"] == 3
    assert result.metrics["central_normal_queue_max_depth"] == 2
    assert all(event["queue_depth_after"] == event["queue_depth_before"] - 1 for event in pulls)
    if protected_pull_order == "max_risk":
        assert pulls[1]["online_risk_s"] > pulls[2]["online_risk_s"]


@pytest.mark.parametrize(
    ("protected_pull_order", "expected_request_id"),
    [
        ("fifo", "request-0"),
        ("max_risk", "request-0"),
        ("cost_damped_risk", "request-0"),
        ("queue_band_risk", "request-0"),
        ("risk_slack_srpt", "request-0"),
        ("guarded_max_risk", "request-0"),
        ("arrival_plus_cost", "request-1"),
        ("highest_response_ratio", "request-1"),
    ],
)
def test_protected_pull_order_uses_only_online_age_and_estimate(
    protected_pull_order: str,
    expected_request_id: str,
) -> None:
    scheduler = TwoQueueScheduler(
        normal_order="fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        global_protected_pull=True,
        protected_pull_order=protected_pull_order,
    )
    old_long = _request_view(
        0,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_total_s=100.0,
        estimated_remaining_s=100.0,
    )
    newer_short = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=5.0,
        estimated_total_s=1.0,
        estimated_remaining_s=1.0,
    )

    assert (
        scheduler.choose_protected_pull(
            now_s=10.0,
            pending=(old_long, newer_short),
        )
        == expected_request_id
    )


def test_cost_damped_risk_reduces_service_estimate_advantage() -> None:
    old_short = _request_view(
        0,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_total_s=1.0,
        estimated_remaining_s=1.0,
    )
    newer_long = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=10.0,
        estimated_total_s=20.0,
        estimated_remaining_s=20.0,
    )
    common = {
        "normal_order": "fifo",
        "sacrificial_order": "lifo",
        "preempt_normal_over_sacrificial": True,
        "global_protected_pull": True,
    }
    max_risk = TwoQueueScheduler(
        **common,
        protected_pull_order="max_risk",
    )
    damped = TwoQueueScheduler(
        **common,
        protected_pull_order="cost_damped_risk",
    )

    assert (
        max_risk.choose_protected_pull(
            now_s=10.0,
            pending=(old_short, newer_long),
        )
        == newer_long.request_id
    )
    assert (
        damped.choose_protected_pull(
            now_s=10.0,
            pending=(old_short, newer_long),
        )
        == old_short.request_id
    )
    assert damped.protected_pull_risk_beta == pytest.approx(0.5)


def test_cost_damped_risk_beta_one_matches_max_risk_for_unstarted_work() -> None:
    requests = (
        _request_view(
            0,
            Priority.NORMAL,
            arrival_time_s=0.0,
            estimated_total_s=1.0,
            estimated_remaining_s=1.0,
        ),
        _request_view(
            1,
            Priority.NORMAL,
            arrival_time_s=10.0,
            estimated_total_s=20.0,
            estimated_remaining_s=20.0,
        ),
    )
    common = {
        "normal_order": "fifo",
        "sacrificial_order": "lifo",
        "preempt_normal_over_sacrificial": True,
        "global_protected_pull": True,
    }
    max_risk = TwoQueueScheduler(
        **common,
        protected_pull_order="max_risk",
    )
    damped = TwoQueueScheduler(
        **common,
        protected_pull_order="cost_damped_risk",
        protected_pull_risk_beta=1.0,
    )

    assert damped.choose_protected_pull(now_s=10.0, pending=requests) == max_risk.choose_protected_pull(
        now_s=10.0,
        pending=requests,
    )


def test_queue_band_risk_changes_beta_only_inside_pending_band() -> None:
    old_short = _request_view(
        0,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_total_s=1.0,
        estimated_remaining_s=1.0,
    )
    newer_long = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=9.6,
        estimated_total_s=20.0,
        estimated_remaining_s=20.0,
    )
    filler = _request_view(
        2,
        Priority.NORMAL,
        arrival_time_s=10.0,
        estimated_total_s=1.0,
        estimated_remaining_s=1.0,
    )
    scheduler = TwoQueueScheduler(
        normal_order="fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        global_protected_pull=True,
        protected_pull_order="queue_band_risk",
        protected_pull_risk_beta=1.0,
        protected_pull_band_risk_beta=0.5,
        protected_pull_band_min_pending=2,
        protected_pull_band_max_pending=2,
    )

    assert scheduler.choose_protected_pull(
        now_s=10.0,
        pending=(old_short, newer_long),
    ) == old_short.request_id
    assert scheduler.choose_protected_pull(
        now_s=10.0,
        pending=(old_short, newer_long, filler),
    ) == newer_long.request_id


def test_risk_slack_srpt_uses_shortest_request_inside_urgent_band() -> None:
    older_short = _request_view(
        0,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_total_s=1.0,
        estimated_remaining_s=1.0,
    )
    newer_long = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=90.0,
        estimated_total_s=100.0,
        estimated_remaining_s=100.0,
    )
    common = {
        "normal_order": "fifo",
        "sacrificial_order": "lifo",
        "preempt_normal_over_sacrificial": True,
        "global_protected_pull": True,
    }
    max_risk = TwoQueueScheduler(
        **common,
        protected_pull_order="max_risk",
    )
    slack_srpt = TwoQueueScheduler(
        **common,
        protected_pull_order="risk_slack_srpt",
        protected_pull_risk_slack_s=10.0,
    )

    assert (
        max_risk.choose_protected_pull(
            now_s=100.0,
            pending=(older_short, newer_long),
        )
        == newer_long.request_id
    )
    assert (
        slack_srpt.choose_protected_pull(
            now_s=100.0,
            pending=(older_short, newer_long),
        )
        == older_short.request_id
    )


def test_guarded_max_risk_skips_bounded_online_risk_prefix() -> None:
    requests = tuple(
        _request_view(
            index,
            Priority.NORMAL,
            arrival_time_s=float(index),
            estimated_total_s=1.0,
            estimated_remaining_s=1.0,
        )
        for index in range(20)
    )
    scheduler = TwoQueueScheduler(
        normal_order="fifo",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        global_protected_pull=True,
        protected_pull_order="guarded_max_risk",
        protected_pull_guard_fraction=0.05,
        protected_pull_guard_max=1,
        protected_pull_guard_min_pending=20,
    )

    assert (
        scheduler.choose_protected_pull(
            now_s=20.0,
            pending=requests,
        )
        == "request-1"
    )


def test_global_protected_pull_does_not_preempt_a_normal_incumbent() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=2,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            backend_count=1,
            scheduler={
                "type": "two_queue",
                "normal_order": "srpt",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_protected_pull": True,
                "protected_pull_order": "max_risk",
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 0.1])
    first = simulator.requests["request-00000"]
    second = simulator.requests["request-00001"]
    first.estimated_step_s = 100.0 / first.total_steps
    second.estimated_step_s = 1.0 / second.total_steps

    result = simulator.run()
    pulls = [event for event in result.events if event["event"] == "protected_pull"]

    assert not [event for event in result.events if event["event"] == "preempt"]
    assert [event["request_id"] for event in pulls] == ["request-00000", "request-00001"]
    assert pulls[1]["time_s"] == pytest.approx(9.0)
    assert result.requests[0]["completion_time_s"] == pytest.approx(9.0)


def test_protected_pull_tail_head_start_matches_backend_dispatch_race() -> None:
    def run(tail_head_start: bool):
        config = parse_experiment_config(
            _raw_config(
                num_requests=3,
                request_rate=1.0,
                service_s=9.0,
                steps=3,
                backend_count=1,
                classifier={
                    "type": "quota_tail",
                    "quota_every": 2,
                    "quota_amount": 1,
                    "threshold_ratio": 0.0,
                    "long_request_ratio": None,
                },
                router={
                    "type": "weighted_least_load",
                    "sacrificial_load_factor": 0.1,
                    "load_view": "assigned",
                    "update_latency_ema": False,
                },
                scheduler={
                    "type": "two_queue",
                    "normal_order": "fifo",
                    "sacrificial_order": "lifo",
                    "preempt_normal_over_sacrificial": True,
                    "global_protected_pull": True,
                    "protected_pull_order": "fifo",
                    "protected_pull_tail_head_start": tail_head_start,
                    "protected_pull_cost_s": 0.0,
                },
            )
        )
        simulator = Simulator(config, collect_events=True)
        _set_arrivals(simulator, [0.0, 0.1, 0.2])
        return simulator.run()

    legacy = run(False)
    modeled = run(True)
    legacy_requests = {request["request_id"]: request for request in legacy.requests}
    modeled_requests = {request["request_id"]: request for request in modeled.requests}

    assert legacy_requests["request-00001"]["priority"] == "sacrificial"
    assert legacy_requests["request-00001"]["preemptions"] == 0
    assert modeled_requests["request-00001"]["preemptions"] == 1
    assert (
        modeled_requests["request-00001"]["first_start_time_s"]
        < modeled_requests["request-00002"]["first_start_time_s"]
    )
    assert (
        legacy_requests["request-00001"]["first_start_time_s"]
        > legacy_requests["request-00002"]["first_start_time_s"]
    )


def test_tail_step_boundary_pulls_global_normal_without_moving_tail() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=3,
            request_rate=1.0,
            service_s=9.0,
            steps=3,
            preemption_cost_s=0.5,
            backend_count=2,
            classifier={
                "type": "quota_tail",
                "quota_every": 1_000,
                "quota_amount": 0,
                "threshold_ratio": 0.0,
                "long_request_ratio": None,
                "initial_credits": 1,
            },
            router={
                "type": "projected_completion",
                "nonpreemptible_weight": 100.0,
                "tail_mode": "sink",
                "load_view": "assigned",
                "update_latency_ema": False,
            },
            scheduler={
                "type": "two_queue",
                "normal_order": "fifo",
                "sacrificial_order": "lifo",
                "preempt_normal_over_sacrificial": True,
                "global_protected_pull": True,
                "protected_pull_order": "fifo",
                "protected_pull_cost_s": 1.0,
            },
        )
    )
    simulator = Simulator(config, collect_events=True)
    _set_arrivals(simulator, [0.0, 0.0, 0.0])

    result = simulator.run()
    pulls = [event for event in result.events if event["event"] == "protected_pull"]
    preempt = next(event for event in result.events if event["event"] == "preempt")
    pull_switch = next(
        event for event in result.events if event["event"] == "switch_start" and event["request_id"] == "request-00002"
    )

    assert [(event["request_id"], event["backend"]) for event in pulls] == [
        ("request-00001", "backend-1"),
        ("request-00002", "backend-0"),
    ]
    assert pulls[1]["time_s"] == pytest.approx(3.0)
    assert preempt["request_id"] == "request-00000"
    assert preempt["selected_request_id"] == "request-00002"
    assert result.requests[0]["priority"] == Priority.SACRIFICIAL.value
    assert result.requests[0]["backend"] == "backend-0"
    assert result.requests[0]["preemptions"] == 1
    assert pull_switch["duration_s"] == pytest.approx(1.5)
    assert pull_switch["preemption_cost_s"] == pytest.approx(0.5)
    assert pull_switch["protected_pull_cost_s"] == pytest.approx(1.0)
    assert result.metrics["protected_pulls"] == 2
    assert result.metrics["protected_pull_cost_total_s"] == pytest.approx(2.0)
    assert result.metrics["switch_cost_total_s"] == pytest.approx(0.5)


def test_two_queue_least_laxity_prioritizes_highest_quantile_risk() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="least_laxity",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
    )
    old_large = _request_view(
        0,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_remaining_s=12.0,
    )
    new_small = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=9.0,
        estimated_remaining_s=2.0,
    )

    assert (
        scheduler.choose(
            now_s=10.0,
            backend=_backend_view("backend"),
            incumbent=None,
            pending=(new_small, old_large),
        )
        == old_large.request_id
    )


def test_two_queue_least_laxity_respects_preemption_hysteresis() -> None:
    scheduler = TwoQueueScheduler(
        normal_order="least_laxity",
        sacrificial_order="lifo",
        preempt_normal_over_sacrificial=True,
        preemption_hysteresis_s=2.0,
    )
    incumbent = _request_view(
        0,
        Priority.NORMAL,
        status=RequestStatus.RUNNING,
        arrival_time_s=0.0,
        estimated_remaining_s=10.0,
    )
    close_candidate = _request_view(
        1,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_remaining_s=11.0,
    )
    clear_candidate = _request_view(
        2,
        Priority.NORMAL,
        arrival_time_s=0.0,
        estimated_remaining_s=13.0,
    )

    assert (
        scheduler.choose(
            now_s=10.0,
            backend=_backend_view("backend"),
            incumbent=incumbent,
            pending=(close_candidate,),
        )
        == incumbent.request_id
    )
    assert (
        scheduler.choose(
            now_s=10.0,
            backend=_backend_view("backend"),
            incumbent=incumbent,
            pending=(clear_candidate,),
        )
        == clear_candidate.request_id
    )


def test_preemption_occurs_at_step_boundary_without_reexecuting_progress() -> None:
    simulator = Simulator(_preemptive_config(), collect_events=True)
    _set_arrivals(simulator, [0.0, 1.0])

    result = simulator.run()
    first = result.requests[0]
    preemptions = [event for event in result.events if event["event"] == "preempt"]
    first_phase_starts = [
        event
        for event in result.events
        if event["event"] == "phase_start" and event["request_id"] == first["request_id"]
    ]

    assert len(preemptions) == 1
    assert preemptions[0]["time_s"] == pytest.approx(3.0)
    assert preemptions[0]["completed_steps"] == 1
    assert first["preemptions"] == 1
    assert first["resumes"] == 1
    assert [event["phase"] for event in first_phase_starts].count("encode") == 1
    assert [event["step_index"] for event in first_phase_starts if event["phase"] == "denoise"] == [
        0,
        1,
        2,
    ]
    assert [event["phase"] for event in first_phase_starts].count("decode") == 1


def test_encode_and_first_step_are_atomic_for_scheduling() -> None:
    simulator = Simulator(
        _preemptive_config(encode_fraction=0.2, decode_fraction=0.2, service_s=10.0),
        collect_events=True,
    )
    _set_arrivals(simulator, [0.0, 1.0])

    result = simulator.run()
    events = list(result.events)
    encode_complete_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "phase_complete" and event["request_id"] == "request-00000" and event["phase"] == "encode"
    )

    assert events[encode_complete_index]["time_s"] == pytest.approx(2.0)
    first_step_start = events[encode_complete_index + 1]
    assert first_step_start["time_s"] == pytest.approx(2.0)
    assert first_step_start["event"] == "phase_start"
    assert first_step_start["request_id"] == "request-00000"
    assert first_step_start["phase"] == "denoise"
    assert first_step_start["step_index"] == 0
    preempt = next(event for event in events if event["event"] == "preempt")
    assert preempt["time_s"] == pytest.approx(4.0)
    assert preempt["completed_steps"] == 1


def test_final_step_and_decode_are_atomic_for_scheduling() -> None:
    simulator = Simulator(
        _preemptive_config(encode_fraction=0.2, decode_fraction=0.2, service_s=10.0),
        collect_events=True,
    )
    _set_arrivals(simulator, [0.0, 7.0])

    result = simulator.run()
    events = list(result.events)
    final_step_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "phase_complete"
        and event["request_id"] == "request-00000"
        and event["phase"] == "denoise"
        and event["completed_steps"] == 3
    )

    assert events[final_step_index]["time_s"] == pytest.approx(8.0)
    assert events[final_step_index + 1]["event"] == "phase_start"
    assert events[final_step_index + 1]["request_id"] == "request-00000"
    assert events[final_step_index + 1]["phase"] == "decode"
    assert events[final_step_index + 1]["time_s"] == pytest.approx(8.0)
    assert not [event for event in events if event["event"] == "preempt"]
    assert result.requests[0]["completion_time_s"] == pytest.approx(10.0)
    assert result.requests[1]["first_start_time_s"] == pytest.approx(10.0)


def test_simulation_conserves_all_service_and_switch_work() -> None:
    simulator = Simulator(_preemptive_config(preemption_cost_s=0.5), collect_events=True)
    _set_arrivals(simulator, [0.0, 1.0])

    result = simulator.run()
    makespan_s = result.metrics["makespan_s"]
    accounted_busy_s = sum(result.metrics["backend_utilization"].values()) * makespan_s
    expected_busy_s = sum(request["service_s"] for request in result.requests)
    expected_busy_s += result.metrics["switch_cost_total_s"]

    assert result.metrics["completed_requests"] == len(result.requests) == 2
    assert result.metrics["failed_requests"] == 0
    assert result.metrics["switch_cost_total_s"] == pytest.approx(0.5)
    assert accounted_busy_s == pytest.approx(expected_busy_s)
    assert all(request["completed_steps"] == 3 for request in result.requests)
    assert len([event for event in result.events if event["event"] == "complete"]) == 2


def test_hsdp_weight_adds_only_denoise_communication_to_actual_service() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=1,
            service_s=10.0,
            steps=2,
            encode_fraction=0.2,
            decode_fraction=0.2,
            actual_service_scale=2.0,
            hsdp_enabled=True,
            hsdp_shard_size=2,
            hsdp_communication_overhead_weight=0.5,
        )
    )

    result = Simulator(config, collect_events=True).run()
    request = result.requests[0]
    phase_starts = [event for event in result.events if event["event"] == "phase_start"]
    denoise_starts = [event for event in phase_starts if event["phase"] == "denoise"]

    # Base compute is 10s * 2.0 = 20s: encode=4, denoise=12, decode=4.
    # HSDP adds 50% of denoise compute (6s), split across the two steps.
    assert request["service_s"] == pytest.approx(26.0)
    assert request["hsdp_communication_s"] == pytest.approx(6.0)
    assert request["estimated_service_s"] == pytest.approx(10.0)
    assert request["completion_time_s"] == pytest.approx(26.0)
    assert [event["duration_s"] for event in phase_starts] == pytest.approx([4.0, 9.0, 9.0, 4.0])
    assert [event["hsdp_communication_s"] for event in denoise_starts] == pytest.approx([3.0, 3.0])
    assert result.metrics["service_total_s"] == pytest.approx(26.0)
    assert result.metrics["hsdp_communication_total_s"] == pytest.approx(6.0)
    assert result.metrics["hsdp_communication_fraction_of_service"] == pytest.approx(6.0 / 26.0)


def test_utilization_rate_includes_actual_scale_and_hsdp_overhead() -> None:
    raw = _raw_config(
        num_requests=1,
        service_s=10.0,
        actual_service_scale=2.0,
        hsdp_enabled=True,
        hsdp_shard_size=2,
        hsdp_communication_overhead_weight=0.5,
    )
    raw["workload"]["request_rate"] = None
    raw["workload"]["utilization"] = 0.9

    simulator = Simulator(parse_experiment_config(raw))

    # All work is denoise: effective service is 10 * 2 * (1 + 0.5) = 30s.
    assert simulator.request_rate == pytest.approx(0.9 / 30.0)


def test_routing_weights_loads_and_breaks_ties_deterministically() -> None:
    request = _request_view(0, Priority.NORMAL)
    weighted = WeightedLeastLoadRouter(
        sacrificial_load_factor=0.1,
        load_view="assigned",
        update_latency_ema=False,
    )
    normal_loaded = _backend_view("normal", normal_load_s=1.0)
    sacrificial_loaded = _backend_view("sacrificial", sacrificial_load_s=5.0)
    assert weighted.choose(request, (normal_loaded, sacrificial_loaded)).backend_name == "sacrificial"

    backend_z = _backend_view("backend-z")
    backend_a = _backend_view("backend-a")
    assert weighted.choose(request, (backend_z, backend_a)).backend_name == "backend-a"
    assert LeastInflightRouter().choose(request, (backend_z, backend_a)).backend_name == "backend-a"

    round_robin = RoundRobinRouter()
    assert [round_robin.choose(request, (backend_z, backend_a)).backend_name for _ in range(3)] == [
        "backend-z",
        "backend-a",
        "backend-z",
    ]


def test_projected_completion_routes_normal_by_load_tail_weight_and_service() -> None:
    request = _request_view(
        0,
        Priority.NORMAL,
        estimated_total_s=10.0,
    )
    router = ProjectedCompletionRouter(
        nonpreemptible_weight=0.1,
        tail_mode="spread",
        load_view="remaining",
        update_latency_ema=True,
    )
    faster = _backend_view(
        "faster",
        speed=2.0,
        normal_load_s=5.0,
        sacrificial_load_s=100.0,
    )
    slower = _backend_view(
        "slower",
        speed=1.0,
        normal_load_s=18.0,
    )

    decision = router.choose(request, (slower, faster))

    assert decision.backend_name == "faster"
    assert decision.details["score"][0] == pytest.approx(20.0)
    assert decision.details["request_service_s"] == pytest.approx(5.0)
    assert decision.details["load_view"] == "remaining"
    assert router.update_latency_ema is True


def test_projected_completion_uses_ema_as_a_deterministic_tie_breaker() -> None:
    request = _request_view(0, Priority.NORMAL)
    router = ProjectedCompletionRouter(
        nonpreemptible_weight=0.1,
        tail_mode="spread",
        load_view="assigned",
        update_latency_ema=True,
    )
    high_ema = _backend_view("a", latency_ema_s=20.0)
    low_ema = _backend_view("z", latency_ema_s=10.0)

    assert router.choose(request, (high_ema, low_ema)).backend_name == "z"


def test_projected_completion_tail_modes_spread_pack_and_sink() -> None:
    tail = _request_view(0, Priority.SACRIFICIAL)
    spread = ProjectedCompletionRouter(
        nonpreemptible_weight=0.1,
        tail_mode="spread",
        load_view="assigned",
        update_latency_ema=False,
    )
    pack = ProjectedCompletionRouter(
        nonpreemptible_weight=0.1,
        tail_mode="pack",
        load_view="assigned",
        update_latency_ema=False,
    )
    sink = ProjectedCompletionRouter(
        nonpreemptible_weight=0.1,
        tail_mode="sink",
        load_view="assigned",
        update_latency_ema=False,
    )
    high_normal = _backend_view(
        "high-normal",
        normal_load_s=20.0,
        sacrificial_load_s=1.0,
    )
    high_tail = _backend_view(
        "high-tail",
        normal_load_s=5.0,
        sacrificial_load_s=8.0,
    )
    empty = _backend_view("empty")

    assert spread.choose(tail, (high_normal, high_tail, empty)).backend_name == "empty"
    assert pack.choose(tail, (high_normal, high_tail, empty)).backend_name == "high-tail"
    assert sink.choose(tail, (high_normal, high_tail, empty)).backend_name == "high-normal"


def test_multi_seed_aggregation_uses_consecutive_seeds_and_confidence_interval() -> None:
    config = parse_experiment_config(
        _raw_config(
            num_requests=8,
            request_rate=3.0,
            service_s=1.0,
            steps=2,
            runs=3,
            seed=101,
        )
    )

    experiment = run_experiment(config)
    p95_values = [run.metrics["latency_p95_s"] for run in experiment.runs]
    aggregate = experiment.aggregate
    p95_aggregate = aggregate["metrics"]["latency_p95_s"]

    assert aggregate["num_runs"] == 3
    assert aggregate["seeds"] == [101, 102, 103]
    assert p95_aggregate["mean"] == pytest.approx(statistics.fmean(p95_values))
    assert p95_aggregate["median"] == pytest.approx(statistics.median(p95_values))
    expected_stddev = statistics.stdev(p95_values)
    expected_margin = 1.96 * expected_stddev / len(p95_values) ** 0.5
    assert p95_aggregate["stddev"] == pytest.approx(expected_stddev)
    assert p95_aggregate["ci95_low"] == pytest.approx(statistics.fmean(p95_values) - expected_margin)
    assert p95_aggregate["ci95_high"] == pytest.approx(statistics.fmean(p95_values) + expected_margin)


def test_serialized_config_round_trips_and_experiment_records_effective_overrides() -> None:
    config = _preemptive_config(preemption_cost_s=0.25)

    serialized_config = config_to_dict(config)
    json_config = json.loads(json.dumps(serialized_config, allow_nan=False))
    assert parse_experiment_config(json_config) == config

    experiment = run_experiment(config, seed=700, runs=2)
    output = experiment.to_dict()
    json_output = json.loads(json.dumps(output, allow_nan=False))
    resolved = parse_experiment_config(json_output["config"])

    assert output["config"]["simulation"] == {"seed": 700, "runs": 2}
    assert output["aggregate"]["seeds"] == [700, 701]
    assert resolved.simulation.seed == 700
    assert resolved.simulation.runs == 2
    assert resolved.policy == config.policy
    assert config.simulation.seed == 42
    assert config.simulation.runs == 1


def test_seed42_benchmark_preset_sacrifices_documented_tail_indices() -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / "wan22_benchmark_rps005_current.yaml")

    result = Simulator(config).run()
    sacrificial = [request for request in result.requests if request["priority"] == Priority.SACRIFICIAL.value]

    assert [request["arrival_seq"] for request in sacrificial] == [18, 38]
    assert [request["request_type"] for request in sacrificial] == ["long", "long"]
    assert result.metrics["sacrificial_requests"] == 2


@pytest.mark.parametrize(
    ("filename", "backend_count", "devices", "actual_anchors"),
    [
        (
            "wan22_2xusp4_calibrated.yaml",
            2,
            4,
            [46.461192143149674, 86.86919855233282, 188.72025074996054],
        ),
        (
            "wan22_4xusp2_calibrated.yaml",
            4,
            2,
            [68.60214959550649, 132.73277402296662, 357.1262867404148],
        ),
    ],
)
def test_measured_service_presets_keep_actual_and_policy_estimates_separate(
    filename: str,
    backend_count: int,
    devices: int,
    actual_anchors: list[float],
) -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / filename)

    assert len(config.topology.backends) == backend_count
    assert {backend.devices for backend in config.topology.backends} == {devices}
    assert [request.nominal_service_s for request in config.workload.request_types] == pytest.approx(actual_anchors)
    assert [request.estimated_service_s for request in config.workload.request_types] == pytest.approx(
        [38.07, 71.34, 119.71]
    )
    expected_scale = 0.958 if filename == "wan22_2xusp4_calibrated.yaml" else 0.9689
    assert config.service.actual_service_scale == pytest.approx(expected_scale)
    assert config.policy.classifier.options["initial_arrival_counter"] == 1
    assert config.policy.router.options["sacrificial_load_factor"] == pytest.approx(0.1)
    assert config.policy.scheduler.options["normal_order"] == "fifo"

    result = Simulator(config).run()
    sacrificial = [request for request in result.requests if request["priority"] == Priority.SACRIFICIAL.value]
    assert [request["arrival_seq"] for request in sacrificial] == [18, 38]


@pytest.mark.parametrize(
    ("filename", "backend_count", "devices", "shard_size"),
    [
        ("wan22_hsdp4_sensitivity.yaml", 2, 4, 4),
        ("wan22_hsdp2_sensitivity.yaml", 4, 2, 2),
    ],
)
def test_hsdp_sensitivity_presets_load_with_expected_topology(
    filename: str,
    backend_count: int,
    devices: int,
    shard_size: int,
) -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / filename)

    assert len(config.topology.backends) == backend_count
    assert {backend.devices for backend in config.topology.backends} == {devices}
    assert config.service.hsdp.enabled is True
    assert config.service.hsdp.shard_size == shard_size
    assert config.service.hsdp.communication_overhead_weight == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("filename", "num_requests", "tail_budget"),
    [
        ("wan22_central_pull_max_risk_50.yaml", 50, 2),
        ("wan22_central_pull_max_risk_100.yaml", 100, 4),
    ],
)
def test_central_pull_recommendation_presets_load_with_frozen_policy(
    filename: str,
    num_requests: int,
    tail_budget: int,
) -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / filename)
    scheduler = config.policy.scheduler.options

    assert config.workload.num_requests == num_requests
    assert len(config.topology.backends) == 4
    assert {backend.devices for backend in config.topology.backends} == {2}
    assert config.policy.classifier.options["max_sacrificial"] == tail_budget
    assert scheduler["global_work_stealing"] is False
    assert scheduler["global_protected_pull"] is True
    assert scheduler["protected_pull_order"] == "max_risk"
    assert scheduler["protected_pull_cost_s"] == pytest.approx(0.5)


def test_cost_damped_risk_4xusp2_preset_uses_new_estimator() -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / "wan22_4xusp2_cost_damped_risk_50.yaml")
    scheduler = config.policy.scheduler.options

    assert config.workload.num_requests == 50
    assert config.workload.request_rate == pytest.approx(0.05)
    assert len(config.topology.backends) == 4
    assert {backend.devices for backend in config.topology.backends} == {2}
    assert [request_type.estimated_service_s for request_type in config.workload.request_types] == pytest.approx(
        [
            68.60214959550649,
            132.73277402296662,
            357.1262867404148,
        ]
    )
    assert scheduler["global_protected_pull"] is True
    assert scheduler["protected_pull_order"] == "cost_damped_risk"
    assert scheduler["protected_pull_risk_beta"] == pytest.approx(0.5)
    assert scheduler["protected_pull_cost_s"] == pytest.approx(0.5)


def test_tail_gate_8xusp1_preset_uses_latest_trace_calibration() -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / "wan22_8xusp1_tail_gate_50.yaml")
    scheduler = config.policy.scheduler.options

    assert config.workload.num_requests == 50
    assert config.workload.request_rate == pytest.approx(0.05)
    assert len(config.topology.backends) == 8
    assert {backend.devices for backend in config.topology.backends} == {1}
    assert [request_type.nominal_service_s for request_type in config.workload.request_types] == pytest.approx(
        [
            106.19037452572957,
            221.95944084071866,
            628.089346267283,
        ]
    )
    assert [request_type.estimated_service_s for request_type in config.workload.request_types] == pytest.approx(
        [
            110.724,
            219.548,
            612.299,
        ]
    )
    assert config.service.actual_jitter_sigma == pytest.approx(0.011468315167067403)
    assert config.policy.router.options["tail_mode"] == "pack"
    assert scheduler["global_protected_pull"] is True
    assert scheduler["protected_pull_order"] == "cost_damped_risk"
    assert scheduler["protected_pull_risk_beta"] == pytest.approx(0.85)
    assert scheduler["protected_pull_tail_head_start"] is False


def _analytic_small_config() -> dict[str, Any]:
    raw = _raw_config(
        num_requests=1,
        request_rate=1.0,
        service_s=1.0,
        steps=2,
        hsdp_enabled=True,
        hsdp_shard_size=2,
        backend_count=1,
    )
    del raw["workload"]["request_types"][0]["nominal_service_s"]
    raw["workload"]["request_types"][0].update(
        {
            "width": 4,
            "height": 4,
            "num_frames": 1,
        }
    )
    raw["topology"]["devices_per_backend"] = 2
    raw["service"]["timing_model"] = {
        "type": "analytic_wan22",
        "model": {
            "vae_spatial_scale": 1,
            "vae_temporal_scale": 1,
            "patch_size": [1, 1, 1],
            "num_layers": 1,
            "num_attention_heads": 2,
            "hidden_size": 4,
            "ffn_dim": 8,
            "text_tokens": 2,
            "latent_channels": 1,
            "output_channels": 1,
            "activation_bytes": 2,
            "latent_element_bytes": 2,
            "parameter_bytes": 2,
        },
        "execution": {
            "usp_degree": 2,
            "ulysses_mode": "strict",
            "cfg_passes": 1,
            "parallel_compute_efficiency": 1.0,
            "shape_efficiency_reference_tokens": None,
            "shape_efficiency_exponent": 0.0,
            "min_shape_efficiency": 1.0,
            "usp_overlap_fraction": 0.0,
            "hsdp_overlap_fraction": 0.0,
        },
        "hardware": {
            # 1e-9 TFLOPS = 1000 FLOP/s per device.
            "effective_compute_tflops_per_device": 1e-9,
            # 1e-6 GB/s = 1000 byte/s.
            "effective_usp_bandwidth_gbytes_per_s": 1e-6,
            "effective_hsdp_bandwidth_gbytes_per_s": 1e-6,
            "usp_collective_latency_us": 10_000,
            "hsdp_collective_latency_us": 10_000,
        },
        "stages": {
            "text_encode_s": 0.25,
            "latent_prepare_fixed_s": 0.0,
            "latent_prepare_bandwidth_gbytes_per_s": None,
            "vae_decode_fixed_s": 0.5,
            "vae_decode_gpixel_per_s": None,
            "denoise_step_overhead_s": 0.1,
            "postprocess_s": 0.0,
        },
    }
    return raw


def test_analytic_wan22_small_model_has_exact_phase_and_d2d_accounting() -> None:
    config = parse_experiment_config(_analytic_small_config())
    assert parse_experiment_config(config_to_dict(config)) == config
    result = Simulator(config, collect_events=True).run()
    request = result.requests[0]

    assert request["transformer_tokens"] == 16
    assert request["padded_transformer_tokens"] == 16
    assert request["transformer_forwards"] == 2
    assert request["denoise_flops"] == 19_712
    assert request["parallelizable_denoise_flops"] == 18_432
    assert request["replicated_denoise_flops_per_rank"] == 1_280
    assert request["executed_denoise_flops_across_ranks"] == 20_992
    assert request["usp_communication_bytes_per_rank"] == pytest.approx(448.0)
    assert request["hsdp_communication_bytes_per_rank"] == pytest.approx(384.0)
    assert request["denoise_compute_s"] == pytest.approx(10.496)
    assert request["usp_communication_s"] == pytest.approx(0.628)
    assert request["hsdp_communication_s"] == pytest.approx(0.404)
    assert request["denoise_overhead_s"] == pytest.approx(0.2)
    assert request["service_s"] == pytest.approx(12.478)
    assert request["estimated_service_s"] == pytest.approx(12.478)

    component_total_s = sum(
        request[field]
        for field in (
            "text_encode_s",
            "latent_prepare_s",
            "denoise_compute_s",
            "denoise_overhead_s",
            "usp_communication_s",
            "hsdp_communication_s",
            "vae_decode_s",
            "postprocess_s",
        )
    )
    assert component_total_s == pytest.approx(request["service_s"])
    assert result.metrics["service_total_s"] == pytest.approx(component_total_s)
    assert result.metrics["denoise_flops_total"] == 19_712
    assert result.metrics["executed_denoise_flops_across_ranks_total"] == 20_992

    denoise_events = [
        event for event in result.events if event["event"] == "phase_start" and event["phase"] == "denoise"
    ]
    assert len(denoise_events) == 2
    for event in denoise_events:
        assert event["duration_s"] == pytest.approx(
            event["denoise_compute_s"]
            + event["denoise_overhead_s"]
            + event["usp_communication_s"]
            + event["hsdp_communication_s"]
        )


def test_analytic_wan22_matches_production_input_normalization() -> None:
    config = load_experiment_config(_SIMULATOR_CONFIGS / "wan22_analytic_usp4.yaml")
    model = build_service_timing_model(config)
    diagnostics = {
        request_type.name: model.profile(request_type).diagnostics for request_type in config.workload.request_types
    }

    assert (
        diagnostics["short"]["normalized_width"],
        diagnostics["short"]["normalized_height"],
        diagnostics["short"]["normalized_frames"],
        diagnostics["short"]["latent_frames"],
        diagnostics["short"]["transformer_tokens"],
        diagnostics["short"]["padded_transformer_tokens"],
    ) == (848, 480, 81, 21, 33_390, 33_392)
    assert (
        diagnostics["medium"]["normalized_width"],
        diagnostics["medium"]["normalized_height"],
        diagnostics["medium"]["normalized_frames"],
        diagnostics["medium"]["latent_frames"],
        diagnostics["medium"]["transformer_tokens"],
        diagnostics["medium"]["padded_transformer_tokens"],
    ) == (848, 480, 121, 31, 49_290, 49_292)
    assert (
        diagnostics["long"]["normalized_width"],
        diagnostics["long"]["normalized_height"],
        diagnostics["long"]["normalized_frames"],
        diagnostics["long"]["latent_frames"],
        diagnostics["long"]["transformer_tokens"],
        diagnostics["long"]["padded_transformer_tokens"],
    ) == (1280, 720, 81, 21, 75_600, 75_600)


def test_analytic_wan22_rejects_legacy_double_counting_and_unknown_options() -> None:
    raw = _analytic_small_config()
    raw["service"]["hsdp"]["communication_overhead_weight"] = 0.25
    with pytest.raises(ValueError, match="communication_overhead_weight must be zero"):
        parse_experiment_config(raw)

    raw = _analytic_small_config()
    raw["service"]["timing_model"]["hardware"]["bandwith_typo"] = 1.0
    with pytest.raises(ValueError, match=r"unknown analytic_wan22\.hardware option"):
        parse_experiment_config(raw)

    raw = _analytic_small_config()
    raw["workload"]["request_types"][0]["metadata"] = {"text_tokens": 0}
    with pytest.raises(ValueError, match=r"metadata\.text_tokens must be positive"):
        parse_experiment_config(raw)


def test_analytic_wan22_supports_standalone_hsdp() -> None:
    raw = _analytic_small_config()
    raw["service"]["timing_model"]["execution"]["usp_degree"] = 1
    raw["service"]["hsdp"]["shard_size"] = 4
    raw["topology"]["devices_per_backend"] = 4

    result = Simulator(parse_experiment_config(raw)).run()
    request = result.requests[0]

    assert request["usp_communication_bytes_per_rank"] == 0.0
    assert request["usp_communication_s"] == 0.0
    assert request["hsdp_communication_bytes_per_rank"] > 0.0
    assert request["hsdp_communication_s"] > 0.0


def test_analytic_wan22_validates_strict_ulysses_head_divisibility() -> None:
    raw = _analytic_small_config()
    raw["service"]["hsdp"]["enabled"] = False
    raw["service"]["hsdp"]["shard_size"] = 1
    raw["service"]["timing_model"]["execution"]["usp_degree"] = 3
    raw["topology"]["devices_per_backend"] = 3

    with pytest.raises(ValueError, match="strict Ulysses"):
        parse_experiment_config(raw)

    raw["service"]["timing_model"]["execution"]["ulysses_mode"] = "advanced_uaa"
    parse_experiment_config(raw)


def test_timing_model_validation_keeps_fixed_anchor_requirements_explicit() -> None:
    raw = _raw_config()
    del raw["workload"]["request_types"][0]["nominal_service_s"]
    with pytest.raises(ValueError, match="fixed_anchor timing model requires nominal_service_s"):
        parse_experiment_config(raw)

    raw = _raw_config()
    raw["service"]["timing_model"] = {"type": "unknown"}
    with pytest.raises(ValueError, match="unknown service timing model type"):
        parse_experiment_config(raw)
