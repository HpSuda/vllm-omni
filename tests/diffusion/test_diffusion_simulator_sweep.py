# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmarks.diffusion.simulator.config import load_experiment_config
from benchmarks.diffusion.simulator.sweep import (
    _paired_p95_summary,
    load_sweep_matrix,
    main,
    run_sweep,
)

_SIMULATOR_CONFIGS = Path(__file__).resolve().parents[2] / "benchmarks" / "diffusion" / "simulator" / "configs"


def _raw_config() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "sweep-test",
        "simulation": {"seed": 999, "runs": 1},
        "workload": {
            "num_requests": 4,
            "request_rate": 1.0,
            "request_types": [
                {
                    "name": "only",
                    "weight": 1.0,
                    "width": 64,
                    "height": 64,
                    "num_inference_steps": 2,
                    "num_frames": 1,
                    "nominal_service_s": 4.0,
                }
            ],
        },
        "service": {
            "encode_fraction": 0.0,
            "decode_fraction": 0.0,
            "actual_jitter_sigma": 0.0,
            "estimate_error_sigma": 0.0,
            "preemption_cost_s": 0.0,
            "actual_service_scale": 1.0,
            "hsdp": {
                "enabled": False,
                "shard_size": 1,
                "communication_overhead_weight": 0.0,
            },
        },
        "topology": {
            "name": "single",
            "backend_count": 1,
            "devices_per_backend": 1,
            "speed_factors": [1.0],
        },
        "policy": {
            "classifier": {"type": "all_normal"},
            "router": {"type": "round_robin"},
            "scheduler": {"type": "fifo"},
        },
    }


def _write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _raw_matrix() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "paired-test",
        "seed": 7,
        "runs": 3,
        "baseline": "baseline",
        "scenarios": [
            {
                "name": "tiny",
                "config": "simulator.yaml",
                "overrides": ["workload.request_rate=1.0"],
            }
        ],
        "variants": [
            {
                "name": "baseline",
                "overrides": ["service.actual_service_scale=1.0"],
            },
            {
                "name": "slow",
                "overrides": ["service.actual_service_scale=2.0"],
            },
        ],
    }


def test_sweep_resolves_relative_paths_pairs_seeds_and_ranks_variants(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "simulator.yaml", _raw_config())
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(matrix_path, _raw_matrix())

    matrix = load_sweep_matrix(matrix_path)
    assert matrix.scenarios[0].config_path == (tmp_path / "simulator.yaml").resolve()

    result = run_sweep(matrix)
    assert result["matrix"]["seeds"] == [7, 8, 9]
    ranking = result["scenarios"][0]["ranking"]
    assert [row["variant"] for row in ranking] == ["baseline", "slow"]
    assert ranking[0]["mean_reduction_pct"] == pytest.approx(0.0)
    assert ranking[0]["tie_rate_pct"] == pytest.approx(100.0)
    assert ranking[1]["mean_reduction_pct"] < 0.0
    assert [pair["seed"] for pair in ranking[1]["pairs"]] == [7, 8, 9]


def test_sweep_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_yaml(tmp_path / "simulator.yaml", _raw_config())
    matrix_path = tmp_path / "matrix.yaml"
    output_path = tmp_path / "result.json"
    _write_yaml(matrix_path, _raw_matrix())

    main(["--matrix", str(matrix_path), "--output", str(output_path)])

    stdout = capsys.readouterr().out
    assert "Scenario   : tiny" in stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["matrix"]["baseline_variant"] == "baseline"
    assert payload["scenarios"][0]["ranking"][0]["rank"] == 1


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda matrix: matrix["variants"].append({"name": "baseline", "overrides": []}),
            "duplicate variant name",
        ),
        (
            lambda matrix: matrix.update({"baseline": "missing"}),
            "is not present in variants",
        ),
        (
            lambda matrix: matrix["variants"][0]["overrides"].append("simulation.seed=9"),
            "cannot set simulation.seed",
        ),
    ],
)
def test_sweep_matrix_rejects_ambiguous_pairing(
    tmp_path: Path,
    mutation: Any,
    error: str,
) -> None:
    _write_yaml(tmp_path / "simulator.yaml", _raw_config())
    raw_matrix = _raw_matrix()
    mutation(raw_matrix)
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(matrix_path, raw_matrix)

    with pytest.raises(ValueError, match=error):
        load_sweep_matrix(matrix_path)


def test_paired_summary_rejects_seed_mismatch() -> None:
    with pytest.raises(ValueError, match="seed mismatch"):
        _paired_p95_summary({7: 10.0, 8: 12.0}, {7: 9.0, 9: 11.0})


def test_bundled_wan22_policy_comparison_resolves_every_pair() -> None:
    matrix = load_sweep_matrix(_SIMULATOR_CONFIGS / "wan22_p95_policy_comparison.yaml")

    assert matrix.seed == 162
    assert matrix.runs == 400
    assert len(matrix.scenarios) == 4
    assert len(matrix.variants) == 5

    for scenario in matrix.scenarios:
        for variant in matrix.variants:
            config_path = variant.config_path or scenario.config_path
            assert config_path is not None
            config = load_experiment_config(
                config_path,
                [*scenario.overrides, *variant.overrides],
            )
            assert sum(backend.devices for backend in config.topology.backends) == 8
            assert config.service.preemption_cost_s == pytest.approx(2.0)


def test_bundled_wan22_quantile_risk_training_resolves_every_pair() -> None:
    matrix = load_sweep_matrix(_SIMULATOR_CONFIGS / "wan22_quantile_risk_training.yaml")

    assert matrix.seed == 562
    assert matrix.runs == 100
    assert len(matrix.scenarios) == 4
    assert len(matrix.variants) == 7

    for scenario in matrix.scenarios:
        for variant in matrix.variants:
            config_path = variant.config_path or scenario.config_path
            assert config_path is not None
            config = load_experiment_config(
                config_path,
                [*scenario.overrides, *variant.overrides],
            )
            assert sum(backend.devices for backend in config.topology.backends) == 8
            assert config.policy.classifier.kind == "quantile_risk_tail"
            assert config.policy.scheduler.options["global_work_stealing"] is False


def test_bundled_wan22_online_credit_holdout_reuses_horizon_free_policy() -> None:
    matrix = load_sweep_matrix(_SIMULATOR_CONFIGS / "wan22_online_credit_holdout.yaml")

    assert matrix.seed == 1762
    assert matrix.runs == 400
    assert len(matrix.scenarios) == 4
    assert len(matrix.variants) == 5

    for scenario in matrix.scenarios:
        for variant in matrix.variants:
            config_path = variant.config_path or scenario.config_path
            assert config_path is not None
            config = load_experiment_config(
                config_path,
                [*scenario.overrides, *variant.overrides],
            )
            assert sum(backend.devices for backend in config.topology.backends) == 8
            if config.policy.classifier.kind != "online_credit_tail":
                continue
            classifier = config.policy.classifier.options
            assert "credit_release_requests" not in classifier
            assert "max_sacrificial" not in classifier
            assert "tail_budget" not in classifier


def test_bundled_wan22_central_pull_training_resolves_every_pair() -> None:
    matrix = load_sweep_matrix(_SIMULATOR_CONFIGS / "wan22_central_pull_training.yaml")

    assert matrix.seed == 662
    assert matrix.runs == 100
    assert len(matrix.scenarios) == 4
    assert len(matrix.variants) == 6

    for scenario in matrix.scenarios:
        for variant in matrix.variants:
            config_path = variant.config_path or scenario.config_path
            assert config_path is not None
            config = load_experiment_config(
                config_path,
                [*scenario.overrides, *variant.overrides],
            )
            assert sum(backend.devices for backend in config.topology.backends) == 8
            scheduler = config.policy.scheduler.options
            assert not (scheduler["global_work_stealing"] and scheduler["global_protected_pull"])


def test_bundled_4xusp2_cost_damped_risk_robustness_matrix_is_online() -> None:
    matrix = load_sweep_matrix(_SIMULATOR_CONFIGS / "wan22_4xusp2_cost_damped_risk_50_robustness.yaml")

    assert matrix.seed == 50042
    assert matrix.runs == 800
    assert len(matrix.scenarios) == 5
    assert len(matrix.variants) == 4

    for scenario in matrix.scenarios:
        for variant in matrix.variants:
            config_path = variant.config_path or scenario.config_path
            assert config_path is not None
            config = load_experiment_config(
                config_path,
                [*scenario.overrides, *variant.overrides],
            )
            classifier = config.policy.classifier.options
            scheduler = config.policy.scheduler.options
            assert config.workload.num_requests == 50
            assert len(config.topology.backends) == 4
            assert {backend.devices for backend in config.topology.backends} == {2}
            assert [
                request_type.estimated_service_s for request_type in config.workload.request_types
            ] == pytest.approx(
                [
                    68.60214959550649,
                    132.73277402296662,
                    357.1262867404148,
                ]
            )
            assert "max_sacrificial" not in classifier
            assert "credit_release_requests" not in classifier
            if variant.name == "Central Pull + Cost-Damped Risk beta0.5":
                assert scheduler["global_protected_pull"] is True
                assert scheduler["protected_pull_order"] == "cost_damped_risk"
                assert scheduler["protected_pull_risk_beta"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("filename", "seed", "runs", "variant_count"),
    [
        ("wan22_topology_conditional_central_pull_training.yaml", 1162, 100, 4),
        ("wan22_topology_conditional_central_pull_holdout.yaml", 1262, 400, 3),
    ],
)
def test_bundled_topology_conditional_central_pull_matrices_are_valid(
    filename: str,
    seed: int,
    runs: int,
    variant_count: int,
) -> None:
    matrix = load_sweep_matrix(_SIMULATOR_CONFIGS / filename)

    assert matrix.seed == seed
    assert matrix.runs == runs
    assert len(matrix.scenarios) == 4
    assert len(matrix.variants) == variant_count

    for scenario in matrix.scenarios:
        for variant in matrix.variants:
            config_path = variant.config_path or scenario.config_path
            assert config_path is not None
            config = load_experiment_config(
                config_path,
                [*scenario.overrides, *variant.overrides],
            )
            scheduler = config.policy.scheduler.options
            assert sum(backend.devices for backend in config.topology.backends) == 8
            assert not (scheduler["global_work_stealing"] and scheduler["global_protected_pull"])
