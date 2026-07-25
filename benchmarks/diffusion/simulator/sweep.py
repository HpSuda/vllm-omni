# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit, paired sweeps for diffusion simulator policy variants."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.diffusion.simulator.config import load_experiment_config
from benchmarks.diffusion.simulator.runner import ExperimentResult, run_experiment


@dataclass(frozen=True)
class SweepItem:
    name: str
    config_path: Path | None
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class SweepMatrix:
    source_path: Path
    name: str
    seed: int
    runs: int
    baseline_variant: str
    scenarios: tuple[SweepItem, ...]
    variants: tuple[SweepItem, ...]


def _yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "Loading a sweep matrix requires PyYAML. Install the repository dependencies or `pip install pyyaml`."
        ) from exc
    return yaml


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _parse_item(value: Any, *, path: str, matrix_dir: Path) -> SweepItem:
    data = _mapping(value, path)
    unknown = sorted(set(data) - {"name", "config", "overrides"})
    if unknown:
        raise ValueError(f"unknown {path} field(s): {', '.join(unknown)}")

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}.name must be a non-empty string")

    raw_config = data.get("config")
    config_path: Path | None = None
    if raw_config is not None:
        if not isinstance(raw_config, str) or not raw_config.strip():
            raise ValueError(f"{path}.config must be a non-empty path string or null")
        config_path = Path(raw_config)
        if not config_path.is_absolute():
            config_path = matrix_dir / config_path
        config_path = config_path.resolve()

    raw_overrides = data.get("overrides", [])
    if isinstance(raw_overrides, (str, bytes)) or not isinstance(raw_overrides, Sequence):
        raise ValueError(f"{path}.overrides must be a list of strings")
    overrides: list[str] = []
    for index, override in enumerate(raw_overrides):
        if not isinstance(override, str) or not override.strip():
            raise ValueError(f"{path}.overrides[{index}] must be a non-empty string")
        dotted_path = override.split("=", 1)[0].strip()
        if dotted_path in {"simulation.seed", "simulation.runs"}:
            raise ValueError(f"{path}.overrides[{index}] cannot set {dotted_path}; use the matrix-level seed/runs")
        overrides.append(override)

    return SweepItem(name=name.strip(), config_path=config_path, overrides=tuple(overrides))


def _reject_duplicate_names(items: Sequence[SweepItem], path: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item.name in seen:
            duplicates.add(item.name)
        seen.add(item.name)
    if duplicates:
        raise ValueError(f"duplicate {path} name(s): {', '.join(sorted(duplicates))}")


def load_sweep_matrix(path: str | Path) -> SweepMatrix:
    matrix_path = Path(path).resolve()
    yaml = _yaml_module()
    with matrix_path.open(encoding="utf-8") as matrix_file:
        loaded = yaml.safe_load(matrix_file)
    if loaded is None:
        raise ValueError(f"sweep matrix {matrix_path} is empty")

    data = _mapping(loaded, str(matrix_path))
    unknown = sorted(set(data) - {"version", "name", "seed", "runs", "baseline", "scenarios", "variants"})
    if unknown:
        raise ValueError(f"unknown sweep matrix field(s): {', '.join(unknown)}")

    version = _integer(data.get("version", 1), "version")
    if version != 1:
        raise ValueError(f"unsupported sweep matrix version: {version}")
    seed = _integer(data.get("seed"), "seed")
    runs = _integer(data.get("runs"), "runs")
    if runs <= 0:
        raise ValueError("runs must be positive")

    name = data.get("name", matrix_path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    baseline = data.get("baseline")
    if not isinstance(baseline, str) or not baseline.strip():
        raise ValueError("baseline must name one variant")
    baseline = baseline.strip()

    raw_scenarios = data.get("scenarios")
    raw_variants = data.get("variants")
    if isinstance(raw_scenarios, (str, bytes)) or not isinstance(raw_scenarios, Sequence) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty list")
    if isinstance(raw_variants, (str, bytes)) or not isinstance(raw_variants, Sequence) or not raw_variants:
        raise ValueError("variants must be a non-empty list")

    matrix_dir = matrix_path.parent
    scenarios = tuple(
        _parse_item(value, path=f"scenarios[{index}]", matrix_dir=matrix_dir)
        for index, value in enumerate(raw_scenarios)
    )
    variants = tuple(
        _parse_item(value, path=f"variants[{index}]", matrix_dir=matrix_dir) for index, value in enumerate(raw_variants)
    )
    _reject_duplicate_names(scenarios, "scenario")
    _reject_duplicate_names(variants, "variant")

    variant_names = {variant.name for variant in variants}
    if baseline not in variant_names:
        raise ValueError(f"baseline variant {baseline!r} is not present in variants")

    for scenario in scenarios:
        for variant in variants:
            config_path = variant.config_path or scenario.config_path
            if config_path is None:
                raise ValueError(
                    f"scenario {scenario.name!r} and variant {variant.name!r} do not specify a config path"
                )
            if not config_path.is_file():
                raise ValueError(
                    f"config for scenario {scenario.name!r}, variant {variant.name!r} does not exist: {config_path}"
                )

    return SweepMatrix(
        source_path=matrix_path,
        name=name.strip(),
        seed=seed,
        runs=runs,
        baseline_variant=baseline,
        scenarios=scenarios,
        variants=variants,
    )


def _seed_p95(result: ExperimentResult) -> dict[int, float]:
    seed_p95: dict[int, float] = {}
    for run in result.runs:
        if run.seed in seed_p95:
            raise ValueError(f"duplicate run seed in experiment result: {run.seed}")
        seed_p95[run.seed] = float(run.metrics["latency_p95_s"])
    return seed_p95


def _paired_p95_summary(
    baseline_by_seed: Mapping[int, float],
    candidate_by_seed: Mapping[int, float],
) -> dict[str, Any]:
    baseline_seeds = tuple(baseline_by_seed)
    candidate_seeds = tuple(candidate_by_seed)
    if baseline_seeds != candidate_seeds:
        raise ValueError(
            f"paired sweep seed mismatch: baseline has {list(baseline_seeds)}, candidate has {list(candidate_seeds)}"
        )
    if not baseline_seeds:
        raise ValueError("paired sweep requires at least one run")

    pairs: list[dict[str, Any]] = []
    for seed in baseline_seeds:
        baseline_p95 = float(baseline_by_seed[seed])
        candidate_p95 = float(candidate_by_seed[seed])
        if not math.isfinite(baseline_p95) or baseline_p95 <= 0.0:
            raise ValueError(f"baseline P95 for seed {seed} must be positive and finite")
        if not math.isfinite(candidate_p95) or candidate_p95 < 0.0:
            raise ValueError(f"candidate P95 for seed {seed} must be non-negative and finite")
        delta_s = baseline_p95 - candidate_p95
        reduction_pct = delta_s / baseline_p95 * 100.0
        pairs.append(
            {
                "seed": seed,
                "baseline_p95_s": baseline_p95,
                "candidate_p95_s": candidate_p95,
                "delta_s": delta_s,
                "reduction_pct": reduction_pct,
            }
        )

    reductions = [pair["reduction_pct"] for pair in pairs]
    deltas = [pair["delta_s"] for pair in pairs]
    candidate_values = [pair["candidate_p95_s"] for pair in pairs]
    tolerance = 1e-12
    wins = sum(delta > tolerance for delta in deltas)
    ties = sum(abs(delta) <= tolerance for delta in deltas)
    worst_pair = min(pairs, key=lambda pair: (pair["reduction_pct"], pair["seed"]))
    best_pair = max(pairs, key=lambda pair: (pair["reduction_pct"], -pair["seed"]))

    return {
        "mean_p95_s": statistics.fmean(candidate_values),
        "median_p95_s": statistics.median(candidate_values),
        "max_p95_s": max(candidate_values),
        "mean_delta_s": statistics.fmean(deltas),
        "mean_reduction_pct": statistics.fmean(reductions),
        "median_reduction_pct": statistics.median(reductions),
        "win_rate_pct": wins / len(pairs) * 100.0,
        "tie_rate_pct": ties / len(pairs) * 100.0,
        "worst_seed": worst_pair["seed"],
        "worst_reduction_pct": worst_pair["reduction_pct"],
        "best_seed": best_pair["seed"],
        "best_reduction_pct": best_pair["reduction_pct"],
        "pairs": pairs,
    }


def _run_item(matrix: SweepMatrix, scenario: SweepItem, variant: SweepItem) -> tuple[Path, ExperimentResult]:
    config_path = variant.config_path or scenario.config_path
    assert config_path is not None
    overrides = [*scenario.overrides, *variant.overrides]
    config = load_experiment_config(config_path, overrides)
    result = run_experiment(config, seed=matrix.seed, runs=matrix.runs)
    expected_seeds = list(range(matrix.seed, matrix.seed + matrix.runs))
    actual_seeds = [run.seed for run in result.runs]
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"seed mismatch for scenario {scenario.name!r}, variant {variant.name!r}: "
            f"expected {expected_seeds}, got {actual_seeds}"
        )
    return config_path, result


def run_sweep(matrix: SweepMatrix) -> dict[str, Any]:
    scenario_results: list[dict[str, Any]] = []
    for scenario in matrix.scenarios:
        executed: dict[str, tuple[Path, ExperimentResult]] = {}
        for variant in matrix.variants:
            executed[variant.name] = _run_item(matrix, scenario, variant)

        _, baseline_result = executed[matrix.baseline_variant]
        baseline_by_seed = _seed_p95(baseline_result)
        rows: list[dict[str, Any]] = []
        for declaration_index, variant in enumerate(matrix.variants):
            config_path, result = executed[variant.name]
            paired = _paired_p95_summary(baseline_by_seed, _seed_p95(result))
            rows.append(
                {
                    "variant": variant.name,
                    "is_baseline": variant.name == matrix.baseline_variant,
                    "config": str(config_path),
                    "scenario_overrides": list(scenario.overrides),
                    "variant_overrides": list(variant.overrides),
                    "declaration_index": declaration_index,
                    **paired,
                }
            )

        rows.sort(
            key=lambda row: (
                -row["mean_reduction_pct"],
                row["mean_p95_s"],
                row["declaration_index"],
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row.pop("declaration_index")
        scenario_results.append(
            {
                "scenario": scenario.name,
                "baseline_variant": matrix.baseline_variant,
                "ranking": rows,
            }
        )

    return {
        "matrix": {
            "name": matrix.name,
            "source": str(matrix.source_path),
            "seed": matrix.seed,
            "runs": matrix.runs,
            "seeds": list(range(matrix.seed, matrix.seed + matrix.runs)),
            "baseline_variant": matrix.baseline_variant,
        },
        "scenarios": scenario_results,
    }


def _print_sweep(result: Mapping[str, Any]) -> None:
    matrix = result["matrix"]
    seeds = matrix["seeds"]
    print(f"Sweep      : {matrix['name']}")
    print(f"Runs       : {matrix['runs']}")
    print(f"Seeds      : {seeds[0]}..{seeds[-1]}")
    print(f"Baseline   : {matrix['baseline_variant']}")
    for scenario in result["scenarios"]:
        rows = scenario["ranking"]
        variant_width = max(7, *(len(row["variant"]) for row in rows))
        print()
        print(f"Scenario   : {scenario['scenario']}")
        print(
            f"{'Rank':>4}  {'Variant':<{variant_width}}  {'Mean P95 (s)':>12}  "
            f"{'Paired reduction':>16}  {'Win rate':>8}  {'Worst seed':>10}"
        )
        for row in rows:
            print(
                f"{row['rank']:>4}  {row['variant']:<{variant_width}}  {row['mean_p95_s']:>12.3f}  "
                f"{row['mean_reduction_pct']:>+15.3f}%  {row['win_rate_pct']:>7.1f}%  "
                f"{row['worst_seed']:>10}"
            )


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an explicit paired matrix of diffusion simulator variants")
    parser.add_argument("--matrix", type=Path, required=True, help="Sweep matrix YAML")
    parser.add_argument("--output", type=Path, help="Write paired rankings and per-seed values as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    matrix = load_sweep_matrix(args.matrix)
    result = run_sweep(matrix)
    _print_sweep(result)
    if args.output is not None:
        _write_json(args.output, result)


if __name__ == "__main__":
    main()
