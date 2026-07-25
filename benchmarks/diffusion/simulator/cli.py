# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.diffusion.simulator.config import load_experiment_config
from benchmarks.diffusion.simulator.runner import ExperimentResult, run_experiment

_DEFAULT_CONFIG = Path(__file__).with_name("configs") / "wan22_super_p95_current.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only discrete-event simulator for diffusion scheduling")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG, help="Experiment YAML configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override an existing YAML field; repeat as needed",
    )
    parser.add_argument("--seed", type=int, help="Override the first run seed")
    parser.add_argument("--runs", type=int, help="Override the number of Monte Carlo runs")
    parser.add_argument("--output", type=Path, help="Write resolved config and aggregate metrics as JSON")
    parser.add_argument("--requests-output", type=Path, help="Write per-request records as CSV")
    parser.add_argument("--trace-output", type=Path, help="Write detailed events as JSONL")
    parser.add_argument("--validate-only", action="store_true", help="Validate the config without running")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _write_requests(path: Path, result: ExperimentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"seed": run.seed, **request} for run in result.runs for request in run.requests]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, result: ExperimentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for run in result.runs:
            for event in run.events:
                output_file.write(json.dumps({"seed": run.seed, **event}, sort_keys=True, allow_nan=False))
                output_file.write("\n")


def _print_summary(result: ExperimentResult) -> None:
    aggregate = result.aggregate
    p95 = aggregate["metrics"]["latency_p95_s"]
    throughput = aggregate["metrics"]["throughput_rps"]
    print(f"Experiment : {result.config['name']}")
    print(f"Runs       : {aggregate['num_runs']}")
    print(f"Seeds      : {aggregate['seeds'][0]}..{aggregate['seeds'][-1]}")
    print(f"P95 mean   : {p95['mean']:.6f} s")
    if p95["ci95_low"] is not None:
        print(f"P95 CI95   : [{p95['ci95_low']:.6f}, {p95['ci95_high']:.6f}] s")
    print(f"Throughput : {throughput['mean']:.6f} req/s")


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config, args.overrides)
    if args.validate_only:
        print(f"Valid simulator config: {args.config}")
        return

    result = run_experiment(
        config,
        seed=args.seed,
        runs=args.runs,
        collect_events=args.trace_output is not None,
    )
    _print_summary(result)
    if args.output is not None:
        _write_json(args.output, result.to_dict())
    if args.requests_output is not None:
        _write_requests(args.requests_output, result)
    if args.trace_output is not None:
        _write_trace(args.trace_output, result)


if __name__ == "__main__":
    main()
