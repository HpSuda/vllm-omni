# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.diffusion.simulator.sweep_stats import analyze_sweep, main


def _pair(seed: int, reduction_pct: float) -> dict[str, float | int]:
    baseline_p95_s = 100.0
    candidate_p95_s = baseline_p95_s * (1.0 - reduction_pct / 100.0)
    return {
        "seed": seed,
        "baseline_p95_s": baseline_p95_s,
        "candidate_p95_s": candidate_p95_s,
        "delta_s": baseline_p95_s - candidate_p95_s,
        "reduction_pct": reduction_pct,
    }


def _sweep_payload(seed_count: int = 5) -> dict[str, Any]:
    seeds = list(range(562, 562 + seed_count))
    scenarios = []
    for scenario_index in range(4):
        good_reductions = [float(seed_index + scenario_index + 1) for seed_index in range(seed_count)]
        bad_reductions = [-value for value in good_reductions]
        scenarios.append(
            {
                "scenario": f"scenario-{scenario_index}",
                "baseline_variant": "incumbent",
                # Deliberately vary ranking order: alignment is by variant name.
                "ranking": [
                    {
                        "variant": "good" if scenario_index % 2 else "incumbent",
                        "pairs": (
                            [_pair(seed, reduction) for seed, reduction in zip(seeds, good_reductions)]
                            if scenario_index % 2
                            else [_pair(seed, 0.0) for seed in seeds]
                        ),
                    },
                    {
                        "variant": "incumbent" if scenario_index % 2 else "bad",
                        "pairs": (
                            [_pair(seed, 0.0) for seed in seeds]
                            if scenario_index % 2
                            else [_pair(seed, reduction) for seed, reduction in zip(seeds, bad_reductions)]
                        ),
                    },
                    {
                        "variant": "bad" if scenario_index % 2 else "good",
                        "pairs": (
                            [_pair(seed, reduction) for seed, reduction in zip(seeds, bad_reductions)]
                            if scenario_index % 2
                            else [_pair(seed, reduction) for seed, reduction in zip(seeds, good_reductions)]
                        ),
                    },
                ],
            }
        )
    return {
        "matrix": {
            "name": "stats-test",
            "runs": seed_count,
            "seeds": seeds,
            "baseline_variant": "incumbent",
        },
        "scenarios": scenarios,
    }


def _by_variant(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["variant"]: row for row in result["variants"]}


def test_analyze_sweep_uses_equal_weight_seed_blocks_and_holm_adjustment() -> None:
    result = analyze_sweep(
        _sweep_payload(),
        bootstrap_samples=500,
        randomization_samples=1_000,
        random_seed=7,
    )
    rows = _by_variant(result)

    # Per seed, the four scenario effects are [i+1, i+2, i+3, i+4].
    assert [item["reduction_pct"] for item in rows["good"]["global_effect"]["per_seed"]] == pytest.approx(
        [2.5, 3.5, 4.5, 5.5, 6.5]
    )
    assert rows["good"]["global_effect"]["mean_reduction_pct"] == pytest.approx(4.5)
    assert rows["good"]["global_effect"]["one_sided_98_75_lcb"] <= rows["good"]["global_effect"]["one_sided_95_lcb"]
    assert rows["bad"]["global_effect"]["mean_reduction_pct"] == pytest.approx(-4.5)
    assert rows["incumbent"]["global_effect"]["mean_reduction_pct"] == pytest.approx(0.0)

    good_test = rows["good"]["randomization_test"]
    assert good_test["method"] == "exact"
    assert good_test["samples"] == 32
    assert good_test["p_value_one_sided"] == pytest.approx(1 / 32)
    # Two candidate hypotheses are adjusted; the baseline is excluded.
    assert good_test["holm_adjusted_p"] == pytest.approx(2 / 32)
    assert rows["bad"]["randomization_test"]["holm_adjusted_p"] == pytest.approx(1.0)
    assert rows["incumbent"]["randomization_test"]["holm_adjusted_p"] == pytest.approx(1.0)

    assert result["analysis"]["scenario_weighting"] == "equal_within_seed"
    assert result["analysis"]["seed_blocks"] == 5
    assert len(rows["good"]["scenario_effects"]) == 4


def test_bootstrap_and_monte_carlo_are_deterministic() -> None:
    payload = _sweep_payload(seed_count=8)
    options = {
        "bootstrap_samples": 300,
        "randomization_samples": 500,
        "random_seed": 19,
        "exact_sign_flip_max_n": 3,
    }

    first = analyze_sweep(payload, **options)
    second = analyze_sweep(payload, **options)

    assert first == second
    test = _by_variant(first)["good"]["randomization_test"]
    assert test["method"] == "monte_carlo"
    assert test["samples"] == 500
    assert 0.0 < test["p_value_one_sided"] <= 1.0


def test_explicit_hypothesis_family_excludes_diagnostic_from_sign_flip_and_holm() -> None:
    result = analyze_sweep(
        _sweep_payload(),
        bootstrap_samples=100,
        randomization_samples=100,
        hypothesis_variants=["good"],
    )
    rows = _by_variant(result)

    assert rows["good"]["in_hypothesis_family"] is True
    assert rows["good"]["randomization_test"]["method"] == "exact"
    assert rows["good"]["randomization_test"]["holm_adjusted_p"] == pytest.approx(1 / 32)

    diagnostic = rows["bad"]
    assert diagnostic["in_hypothesis_family"] is False
    assert diagnostic["global_effect"]["mean_reduction_pct"] == pytest.approx(-4.5)
    assert len(diagnostic["scenario_effects"]) == 4
    assert diagnostic["randomization_test"] == {
        "alternative": "mean_reduction_pct > 0",
        "method": "diagnostic_not_in_family",
        "samples": 0,
        "p_value_one_sided": 1.0,
        "holm_adjusted_p": 1.0,
    }

    metadata = result["analysis"]["randomization"]
    assert metadata["hypothesis_variants"] == ["good"]
    assert metadata["explicit_hypothesis_family"] is True


@pytest.mark.parametrize(
    ("hypothesis_variants", "error"),
    [
        (["incumbent"], "cannot contain baseline"),
        (["missing"], "unknown variant"),
        (["good", "good"], "duplicate variant"),
        ("good", "must be a list"),
    ],
)
def test_explicit_hypothesis_family_is_validated(hypothesis_variants: Any, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        analyze_sweep(
            _sweep_payload(),
            bootstrap_samples=20,
            randomization_samples=20,
            hypothesis_variants=hypothesis_variants,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["scenarios"][0]["ranking"][0]["pairs"].pop(),
            "seed alignment mismatch",
        ),
        (
            lambda payload: payload["scenarios"][1]["ranking"].pop(),
            "variant alignment mismatch",
        ),
        (
            lambda payload: payload["scenarios"][0]["ranking"][2]["pairs"][0].update({"baseline_p95_s": 101.0}),
            "reduction_pct is inconsistent",
        ),
    ],
)
def test_analyze_sweep_rejects_unpaired_inputs(mutation: Any, error: str) -> None:
    payload = _sweep_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=error):
        analyze_sweep(payload, bootstrap_samples=20, randomization_samples=20)


def test_analyze_sweep_rejects_baseline_p95_mismatch_between_variants() -> None:
    payload = _sweep_payload()
    pair = payload["scenarios"][0]["ranking"][2]["pairs"][0]
    pair["baseline_p95_s"] = 200.0
    pair["candidate_p95_s"] = 198.0
    pair["reduction_pct"] = 1.0

    with pytest.raises(ValueError, match="baseline P95 alignment mismatch"):
        analyze_sweep(payload, bootstrap_samples=20, randomization_samples=20)


def test_sweep_stats_cli_prints_summary_and_optionally_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "sweep.json"
    output_path = tmp_path / "stats.json"
    input_path.write_text(json.dumps(_sweep_payload()), encoding="utf-8")

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--hypothesis-variant",
            "good",
        ]
    )

    stdout = capsys.readouterr().out
    assert "Seed blocks: 5" in stdout
    assert "Holm p" in stdout
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["analysis"]["baseline_variant"] == "incumbent"
    assert output["analysis"]["randomization"]["hypothesis_variants"] == ["good"]
    assert _by_variant(output)["good"]["global_effect"]["mean_reduction_pct"] == pytest.approx(4.5)
    assert _by_variant(output)["bad"]["randomization_test"]["method"] == "diagnostic_not_in_family"


def test_input_is_not_mutated() -> None:
    payload = _sweep_payload()
    original = copy.deepcopy(payload)

    analyze_sweep(payload, bootstrap_samples=20, randomization_samples=20)

    assert payload == original
