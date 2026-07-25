# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Statistical post-processing for explicit paired simulator sweeps.

``sweep.py`` deliberately records every per-seed comparison without making
inferential claims.  This module validates that those comparisons are truly
paired across every scenario and variant, then treats a seed -- rather than a
scenario row -- as the resampling unit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_RANDOMIZATION_SAMPLES = 100_000
DEFAULT_RANDOM_SEED = 20_260_723
DEFAULT_EXACT_SIGN_FLIP_MAX_N = 16


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _sequence(value: Any, path: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a list")
    return list(value)


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _validate_positive_int(value: int, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")


def _derived_seed(base_seed: int, *parts: str) -> int:
    material = "\0".join((str(base_seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big")


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    """Return a Hyndman-Fan type-7 quantile."""

    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def _bootstrap_mean_summary(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    _validate_positive_int(samples, "bootstrap_samples")
    if not values:
        raise ValueError("bootstrap requires at least one paired seed")

    finite_values = tuple(_finite_float(value, "bootstrap value") for value in values)
    observed_mean = statistics.fmean(finite_values)
    if all(value == finite_values[0] for value in finite_values):
        bootstrap_means = [observed_mean]
    else:
        rng = random.Random(seed)
        count = len(finite_values)
        bootstrap_means = [sum(rng.choices(finite_values, k=count)) / count for _ in range(samples)]

    return {
        "mean_reduction_pct": observed_mean,
        "ci95_low": _linear_quantile(bootstrap_means, 0.025),
        "ci95_high": _linear_quantile(bootstrap_means, 0.975),
        "one_sided_95_lcb": _linear_quantile(bootstrap_means, 0.05),
        # Bonferroni simultaneous lower bound for four pre-registered
        # scenarios: family alpha=0.05 => per-scenario alpha=0.0125.
        "one_sided_98_75_lcb": _linear_quantile(bootstrap_means, 0.0125),
    }


def _exact_sign_flip_p_value(values: Sequence[float], observed_sum: float) -> tuple[float, int]:
    tolerance = 1e-12
    exceedances = 0
    permutations = 1 << len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        randomized_sum = sum(sign * value for sign, value in zip(signs, values))
        if randomized_sum >= observed_sum - tolerance:
            exceedances += 1
    return exceedances / permutations, permutations


def _monte_carlo_sign_flip_p_value(
    values: Sequence[float],
    observed_sum: float,
    *,
    samples: int,
    seed: int,
) -> tuple[float, int]:
    _validate_positive_int(samples, "randomization_samples")
    tolerance = 1e-12
    exceedances = 0
    rng = random.Random(seed)
    count = len(values)
    all_negative_sum = -sum(values)
    for _ in range(samples):
        positive_bits = rng.getrandbits(count)
        randomized_sum = all_negative_sum
        while positive_bits:
            least_significant_bit = positive_bits & -positive_bits
            index = least_significant_bit.bit_length() - 1
            randomized_sum += 2.0 * values[index]
            positive_bits ^= least_significant_bit
        if randomized_sum >= observed_sum - tolerance:
            exceedances += 1

    # The plus-one correction prevents a Monte Carlo estimate of zero.
    return (exceedances + 1) / (samples + 1), samples


def _sign_flip_summary(
    values: Sequence[float],
    *,
    exact_max_n: int,
    monte_carlo_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not values:
        raise ValueError("sign-flip test requires at least one paired seed")
    if isinstance(exact_max_n, bool) or not isinstance(exact_max_n, int) or exact_max_n < 0:
        raise ValueError("exact_sign_flip_max_n must be a non-negative integer")

    finite_values = tuple(_finite_float(value, "sign-flip value") for value in values)
    observed_sum = sum(finite_values)
    if len(finite_values) <= exact_max_n:
        p_value, permutations = _exact_sign_flip_p_value(finite_values, observed_sum)
        method = "exact"
    else:
        p_value, permutations = _monte_carlo_sign_flip_p_value(
            finite_values,
            observed_sum,
            samples=monte_carlo_samples,
            seed=seed,
        )
        method = "monte_carlo"
    return {
        "alternative": "mean_reduction_pct > 0",
        "method": method,
        "samples": permutations,
        "p_value_one_sided": p_value,
    }


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, excluding no hypotheses implicitly."""

    ordered: list[tuple[str, float]] = []
    for name, value in p_values.items():
        p_value = _finite_float(value, f"p_values[{name!r}]")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p_values[{name!r}] must be between zero and one")
        ordered.append((name, p_value))
    ordered.sort(key=lambda item: (item[1], item[0]))

    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    hypothesis_count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running_maximum = max(running_maximum, (hypothesis_count - index) * p_value)
        adjusted[name] = min(1.0, running_maximum)
    return adjusted


def _parse_sweep(
    payload: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[int, ...], dict[str, dict[str, tuple[float, ...]]]]:
    root = _mapping(payload, "sweep")
    matrix = _mapping(root.get("matrix"), "sweep.matrix")
    baseline_variant = _non_empty_string(matrix.get("baseline_variant"), "sweep.matrix.baseline_variant")
    raw_seeds = _sequence(matrix.get("seeds"), "sweep.matrix.seeds")
    seeds = tuple(_integer(seed, f"sweep.matrix.seeds[{index}]") for index, seed in enumerate(raw_seeds))
    if not seeds:
        raise ValueError("sweep.matrix.seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("sweep.matrix.seeds contains duplicates")
    if "runs" in matrix and _integer(matrix["runs"], "sweep.matrix.runs") != len(seeds):
        raise ValueError("sweep.matrix.runs does not match sweep.matrix.seeds")

    raw_scenarios = _sequence(root.get("scenarios"), "sweep.scenarios")
    if not raw_scenarios:
        raise ValueError("sweep.scenarios must not be empty")

    scenario_names: list[str] = []
    effects: dict[str, dict[str, tuple[float, ...]]] = {}
    expected_variants: set[str] | None = None

    for scenario_index, raw_scenario in enumerate(raw_scenarios):
        scenario_path = f"sweep.scenarios[{scenario_index}]"
        scenario = _mapping(raw_scenario, scenario_path)
        scenario_name = _non_empty_string(scenario.get("scenario"), f"{scenario_path}.scenario")
        if scenario_name in effects:
            raise ValueError(f"duplicate scenario name: {scenario_name!r}")
        scenario_names.append(scenario_name)
        scenario_baseline = _non_empty_string(
            scenario.get("baseline_variant"),
            f"{scenario_path}.baseline_variant",
        )
        if scenario_baseline != baseline_variant:
            raise ValueError(
                f"scenario {scenario_name!r} baseline {scenario_baseline!r} "
                f"does not match matrix baseline {baseline_variant!r}"
            )

        ranking = _sequence(scenario.get("ranking"), f"{scenario_path}.ranking")
        if not ranking:
            raise ValueError(f"{scenario_path}.ranking must not be empty")
        scenario_effects: dict[str, tuple[float, ...]] = {}
        baselines_by_variant: dict[str, tuple[float, ...]] = {}
        for row_index, raw_row in enumerate(ranking):
            row_path = f"{scenario_path}.ranking[{row_index}]"
            row = _mapping(raw_row, row_path)
            variant = _non_empty_string(row.get("variant"), f"{row_path}.variant")
            if variant in scenario_effects:
                raise ValueError(f"scenario {scenario_name!r} contains duplicate variant {variant!r}")
            pairs = _sequence(row.get("pairs"), f"{row_path}.pairs")
            pair_seeds: list[int] = []
            reductions: list[float] = []
            baseline_values: list[float] = []
            for pair_index, raw_pair in enumerate(pairs):
                pair_path = f"{row_path}.pairs[{pair_index}]"
                pair = _mapping(raw_pair, pair_path)
                pair_seed = _integer(pair.get("seed"), f"{pair_path}.seed")
                baseline_p95 = _finite_float(pair.get("baseline_p95_s"), f"{pair_path}.baseline_p95_s")
                candidate_p95 = _finite_float(pair.get("candidate_p95_s"), f"{pair_path}.candidate_p95_s")
                reduction = _finite_float(pair.get("reduction_pct"), f"{pair_path}.reduction_pct")
                if baseline_p95 <= 0.0:
                    raise ValueError(f"{pair_path}.baseline_p95_s must be positive")
                if candidate_p95 < 0.0:
                    raise ValueError(f"{pair_path}.candidate_p95_s cannot be negative")
                expected_reduction = (baseline_p95 - candidate_p95) / baseline_p95 * 100.0
                if not math.isclose(reduction, expected_reduction, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(
                        f"{pair_path}.reduction_pct is inconsistent with baseline_p95_s and candidate_p95_s"
                    )
                pair_seeds.append(pair_seed)
                baseline_values.append(baseline_p95)
                reductions.append(reduction)

            if tuple(pair_seeds) != seeds:
                raise ValueError(
                    f"seed alignment mismatch for scenario {scenario_name!r}, variant {variant!r}: "
                    f"expected {list(seeds)}, got {pair_seeds}"
                )
            scenario_effects[variant] = tuple(reductions)
            baselines_by_variant[variant] = tuple(baseline_values)

        variants = set(scenario_effects)
        if baseline_variant not in variants:
            raise ValueError(f"scenario {scenario_name!r} does not contain baseline variant {baseline_variant!r}")
        if expected_variants is None:
            expected_variants = variants
        elif variants != expected_variants:
            missing = sorted(expected_variants - variants)
            extra = sorted(variants - expected_variants)
            raise ValueError(
                f"variant alignment mismatch for scenario {scenario_name!r}: missing={missing}, extra={extra}"
            )

        canonical_baseline = baselines_by_variant[baseline_variant]
        for variant, variant_baseline in baselines_by_variant.items():
            if variant_baseline != canonical_baseline:
                raise ValueError(f"baseline P95 alignment mismatch for scenario {scenario_name!r}, variant {variant!r}")
        baseline_reductions = scenario_effects[baseline_variant]
        if any(abs(value) > 1e-10 for value in baseline_reductions):
            raise ValueError(f"baseline variant {baseline_variant!r} has non-zero reduction in {scenario_name!r}")

        effects[scenario_name] = scenario_effects

    assert expected_variants is not None
    return baseline_variant, tuple(scenario_names), seeds, effects


def analyze_sweep(
    payload: Mapping[str, Any],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    randomization_samples: int = DEFAULT_RANDOMIZATION_SAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    exact_sign_flip_max_n: int = DEFAULT_EXACT_SIGN_FLIP_MAX_N,
    hypothesis_variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Analyze one ``sweep.py`` result using seed-block paired statistics."""

    _validate_positive_int(bootstrap_samples, "bootstrap_samples")
    _validate_positive_int(randomization_samples, "randomization_samples")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if isinstance(exact_sign_flip_max_n, bool) or not isinstance(exact_sign_flip_max_n, int):
        raise ValueError("exact_sign_flip_max_n must be an integer")
    if exact_sign_flip_max_n < 0:
        raise ValueError("exact_sign_flip_max_n cannot be negative")

    baseline_variant, scenario_names, seeds, effects = _parse_sweep(payload)
    variant_names = tuple(sorted(next(iter(effects.values()))))
    if hypothesis_variants is None:
        resolved_hypothesis_variants = tuple(variant for variant in variant_names if variant != baseline_variant)
        explicit_hypothesis_family = False
    else:
        if isinstance(hypothesis_variants, (str, bytes)) or not isinstance(hypothesis_variants, Sequence):
            raise ValueError("hypothesis_variants must be a list of variant names or null")
        resolved: list[str] = []
        seen: set[str] = set()
        for index, raw_variant in enumerate(hypothesis_variants):
            variant = _non_empty_string(raw_variant, f"hypothesis_variants[{index}]")
            if variant in seen:
                raise ValueError(f"hypothesis_variants contains duplicate variant {variant!r}")
            if variant == baseline_variant:
                raise ValueError(f"hypothesis_variants cannot contain baseline variant {baseline_variant!r}")
            if variant not in variant_names:
                raise ValueError(f"hypothesis_variants contains unknown variant {variant!r}")
            seen.add(variant)
            resolved.append(variant)
        resolved_hypothesis_variants = tuple(resolved)
        explicit_hypothesis_family = True
    hypothesis_family = frozenset(resolved_hypothesis_variants)

    rows: list[dict[str, Any]] = []
    raw_p_values: dict[str, float] = {}

    for variant in variant_names:
        scenario_values = {scenario: effects[scenario][variant] for scenario in scenario_names}
        global_values = tuple(
            statistics.fmean(scenario_values[scenario][seed_index] for scenario in scenario_names)
            for seed_index in range(len(seeds))
        )
        global_summary = _bootstrap_mean_summary(
            global_values,
            samples=bootstrap_samples,
            seed=_derived_seed(random_seed, "bootstrap", variant, "global"),
        )
        global_summary["per_seed"] = [
            {"seed": seed, "reduction_pct": value} for seed, value in zip(seeds, global_values)
        ]

        scenario_summaries: list[dict[str, Any]] = []
        for scenario in scenario_names:
            summary = _bootstrap_mean_summary(
                scenario_values[scenario],
                samples=bootstrap_samples,
                seed=_derived_seed(random_seed, "bootstrap", variant, scenario),
            )
            scenario_summaries.append({"scenario": scenario, **summary})

        is_baseline = variant == baseline_variant
        if is_baseline:
            randomization = {
                "alternative": "mean_reduction_pct > 0",
                "method": "not_applicable_baseline",
                "samples": 0,
                "p_value_one_sided": 1.0,
            }
        elif variant in hypothesis_family:
            randomization = _sign_flip_summary(
                global_values,
                exact_max_n=exact_sign_flip_max_n,
                monte_carlo_samples=randomization_samples,
                seed=_derived_seed(random_seed, "sign_flip", variant),
            )
            raw_p_values[variant] = randomization["p_value_one_sided"]
        else:
            randomization = {
                "alternative": "mean_reduction_pct > 0",
                "method": "diagnostic_not_in_family",
                "samples": 0,
                "p_value_one_sided": 1.0,
            }

        rows.append(
            {
                "variant": variant,
                "is_baseline": is_baseline,
                "in_hypothesis_family": variant in hypothesis_family,
                "global_effect": global_summary,
                "scenario_effects": scenario_summaries,
                "randomization_test": randomization,
            }
        )

    adjusted_p_values = _holm_adjust(raw_p_values)
    for row in rows:
        variant = row["variant"]
        row["randomization_test"]["holm_adjusted_p"] = adjusted_p_values.get(variant, 1.0)

    rows.sort(
        key=lambda row: (
            -row["global_effect"]["mean_reduction_pct"],
            row["variant"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    matrix = _mapping(payload, "sweep").get("matrix")
    matrix_name = _mapping(matrix, "sweep.matrix").get("name")
    return {
        "analysis": {
            "sweep_name": matrix_name,
            "baseline_variant": baseline_variant,
            "scenario_names": list(scenario_names),
            "seeds": list(seeds),
            "seed_blocks": len(seeds),
            "scenario_weighting": "equal_within_seed",
            "bootstrap": {
                "method": "paired_seed_block_percentile",
                "samples": bootstrap_samples,
                "two_sided_confidence": 0.95,
                "one_sided_confidence": 0.95,
            },
            "randomization": {
                "method": "paired_sign_flip",
                "monte_carlo_samples": randomization_samples,
                "exact_max_seed_blocks": exact_sign_flip_max_n,
                "alternative": "mean_reduction_pct > 0",
                "multiple_comparison_adjustment": "holm_bonferroni",
                "hypothesis_variants": list(resolved_hypothesis_variants),
                "explicit_hypothesis_family": explicit_hypothesis_family,
            },
            "random_seed": random_seed,
        },
        "variants": rows,
    }


def _print_summary(result: Mapping[str, Any]) -> None:
    analysis = _mapping(result.get("analysis"), "analysis")
    print(f"Sweep      : {analysis.get('sweep_name')}")
    print(f"Baseline   : {analysis['baseline_variant']}")
    print(f"Seed blocks: {analysis['seed_blocks']}")
    print(f"Scenarios  : {len(analysis['scenario_names'])}")
    print()
    print(f"{'Rank':>4}  {'Variant':<56}  {'Global effect':>14}  {'95% CI':>23}  {'LCB95':>9}  {'Holm p':>9}")
    for row in _sequence(result.get("variants"), "variants"):
        variant = row["variant"]
        effect = row["global_effect"]
        randomization = row["randomization_test"]
        interval = f"[{effect['ci95_low']:+.3f}, {effect['ci95_high']:+.3f}]"
        print(
            f"{row['rank']:>4}  {variant:<56}  {effect['mean_reduction_pct']:>+13.3f}%  "
            f"{interval:>23}  {effect['one_sided_95_lcb']:>+8.3f}%  "
            f"{randomization['holm_adjusted_p']:>9.6f}"
        )


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process a paired diffusion simulator sweep")
    parser.add_argument("--input", type=Path, required=True, help="JSON output written by simulator.sweep")
    parser.add_argument("--output", type=Path, help="Write block-bootstrap and randomization statistics as JSON")
    parser.add_argument(
        "--hypothesis-variant",
        action="append",
        dest="hypothesis_variants",
        help="Variant included in the Holm/sign-flip family; repeat to include more than one",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    with args.input.open(encoding="utf-8") as input_file:
        loaded = json.load(input_file)
    result = analyze_sweep(
        _mapping(loaded, str(args.input)),
        hypothesis_variants=args.hypothesis_variants,
    )
    _print_summary(result)
    if args.output is not None:
        _write_json(args.output, result)


if __name__ == "__main__":
    main()
