# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from benchmarks.diffusion.simulator.engine import SimulationResult, Simulator, config_to_dict
from benchmarks.diffusion.simulator.models import ExperimentConfig


@dataclass(frozen=True)
class ExperimentResult:
    config: dict[str, Any]
    runs: tuple[SimulationResult, ...]
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "aggregate": self.aggregate,
            "runs": [run.to_dict(include_requests=False, include_events=False) for run in self.runs],
        }


def _aggregate_numeric(values: list[float]) -> dict[str, float | None]:
    stddev = statistics.stdev(values) if len(values) > 1 else None
    ci_margin = None if stddev is None else 1.96 * stddev / math.sqrt(len(values))
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "stddev": stddev,
        "ci95_low": None if ci_margin is None else mean - ci_margin,
        "ci95_high": None if ci_margin is None else mean + ci_margin,
        "min": min(values),
        "max": max(values),
    }


def _build_aggregate(runs: tuple[SimulationResult, ...]) -> dict[str, Any]:
    metric_names = (
        "makespan_s",
        "throughput_rps",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "latency_max_s",
        "sacrificial_requests",
        "preemptions",
        "steals",
        "steal_cost_total_s",
        "protected_pulls",
        "protected_pull_cost_total_s",
        "central_normal_queue_max_depth",
        "central_wait_mean_s",
        "central_wait_p50_s",
        "central_wait_p95_s",
        "service_total_s",
        "text_encode_total_s",
        "latent_prepare_total_s",
        "denoise_compute_total_s",
        "denoise_overhead_total_s",
        "usp_communication_total_s",
        "hsdp_communication_total_s",
        "vae_decode_total_s",
        "postprocess_total_s",
        "communication_total_s",
        "communication_fraction_of_service",
        "usp_communication_fraction_of_service",
        "hsdp_communication_fraction_of_service",
        "denoise_flops_total",
        "executed_denoise_flops_across_ranks_total",
        "usp_communication_bytes_per_rank_total",
        "hsdp_communication_bytes_per_rank_total",
    )
    return {
        "num_runs": len(runs),
        "seeds": [run.seed for run in runs],
        "metrics": {
            metric_name: _aggregate_numeric([float(run.metrics[metric_name]) for run in runs])
            for metric_name in metric_names
        },
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    seed: int | None = None,
    runs: int | None = None,
    collect_events: bool = False,
) -> ExperimentResult:
    base_seed = config.simulation.seed if seed is None else seed
    num_runs = config.simulation.runs if runs is None else runs
    if num_runs <= 0:
        raise ValueError("runs must be positive")
    simulation_runs = tuple(
        Simulator(config, seed=base_seed + run_index, collect_events=collect_events).run()
        for run_index in range(num_runs)
    )
    resolved_config = config_to_dict(config)
    resolved_config["simulation"]["seed"] = base_seed
    resolved_config["simulation"]["runs"] = num_runs
    return ExperimentResult(
        config=resolved_config,
        runs=simulation_runs,
        aggregate=_build_aggregate(simulation_runs),
    )
