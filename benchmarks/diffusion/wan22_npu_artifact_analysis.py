# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate and summarize a Wan2.2 benchmark artifact without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_MEASURED_REQUEST_RE = re.compile(r"^request-(\d{5})$")
_WORKLOAD_ORDER = {"short": 0, "medium": 1, "long": 2}
_REQUIRED_LIFECYCLE_EVENTS = (
    "client_arrive",
    "client_job_accepted",
    "client_finish",
    "dispatcher_arrive",
    "dispatch",
    "dispatcher_job_accepted",
    "dispatcher_complete",
    "backend_arrive",
    "backend_job_accepted",
    "backend_start",
    "scheduler_enqueue",
    "scheduler_complete",
    "backend_complete",
)
_CLASS_METRICS = {
    "scheduler_s": "scheduler_elapsed_s",
    "backend_s": "backend_inference_time_s",
    "central_wait_s": "central_wait_s",
    "e2e_s": "e2e_latency_s",
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact is incomplete or internally inconsistent."""


def analyze_artifact(
    artifact_path: str | Path,
    *,
    slowest_limit: int = 5,
) -> dict[str, Any]:
    """Read, strictly validate, and analyze one ``tar.gz`` benchmark artifact."""
    if slowest_limit < 1:
        raise ValueError("slowest_limit must be positive")

    path = Path(artifact_path)
    if not path.is_file():
        raise ArtifactValidationError(f"Artifact does not exist: {path}")

    sha256 = _sha256(path)
    with tarfile.open(path, mode="r:gz") as archive:
        members = _regular_members_by_name(archive)
        result = _load_json_member(archive, members, "result.json")
        trace_payload = _load_json_member(archive, members, "request_trace.json")
        raw_events, raw_files = _load_raw_trace(archive, members)

    if not isinstance(result, dict):
        raise ArtifactValidationError("result.json must contain a JSON object")
    if not isinstance(trace_payload, dict):
        raise ArtifactValidationError("request_trace.json must contain a JSON object")
    raw_rows = trace_payload.get("requests")
    trace_summary = trace_payload.get("summary")
    if not isinstance(raw_rows, list):
        raise ArtifactValidationError("request_trace.json.requests must be a JSON array")
    if not isinstance(trace_summary, dict):
        raise ArtifactValidationError("request_trace.json.summary must be a JSON object")
    if any(not isinstance(row, dict) for row in raw_rows):
        raise ArtifactValidationError("Every request_trace.json.requests item must be an object")

    measured_rows = _measured_rows(raw_rows)
    request_ids = [str(row["request_id"]) for row in measured_rows]
    _validate_measured_ids(request_ids)
    _validate_success_and_result(measured_rows, result, trace_summary)
    _add_exclusive_service(measured_rows)

    counts = {
        "queue_class": _counts(measured_rows, "queue_class"),
        "workload_class": _counts(measured_rows, "workload_class"),
        "backend": _counts(measured_rows, "backend"),
    }
    _validate_reported_counts(trace_summary, counts)
    raw_validation = _validate_raw_trace(raw_events, measured_rows, raw_files)

    latencies = [_finite_number(row, "e2e_latency_s") for row in measured_rows]
    p95 = percentile_type7_boundary(measured_rows, "e2e_latency_s", 0.95)
    _validate_latency_aggregates(result, trace_summary, latencies, p95["value_s"])

    class_metrics: dict[str, Any] = {}
    for workload_class in _sorted_classes(measured_rows):
        class_rows = [row for row in measured_rows if row.get("workload_class") == workload_class]
        metrics = {
            output_name: _metric_summary([_finite_number(row, input_name) for row in class_rows])
            for output_name, input_name in _CLASS_METRICS.items()
        }
        metrics["exclusive_service_s"] = _metric_summary(
            [_finite_number(row, "exclusive_service_s") for row in class_rows]
        )
        class_metrics[workload_class] = {
            "request_count": len(class_rows),
            **metrics,
        }

    slowest = [
        _request_snapshot(row)
        for row in sorted(
            measured_rows,
            key=lambda row: _finite_number(row, "e2e_latency_s"),
            reverse=True,
        )[:slowest_limit]
    ]
    tail_rows = [row for row in measured_rows if str(row.get("queue_class", "")).lower() == "tail"]
    preempted_rows = [row for row in measured_rows if _nonnegative_int(row, "scheduler_preempt_count") > 0]

    return {
        "schema_version": 1,
        "artifact": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256,
        },
        "validation": {
            "status": "passed",
            "measured_request_ids": {
                "count": len(request_ids),
                "first": request_ids[0],
                "last": request_ids[-1],
                "contiguous": True,
            },
            "successful_requests": len(measured_rows),
            "failed_requests": 0,
            "raw_trace": raw_validation,
        },
        "summary": {
            "duration_s": _finite_number(result, "duration"),
            "completed_requests": _nonnegative_int(result, "completed_requests"),
            "failed_requests": _nonnegative_int(result, "failed_requests"),
            "throughput_requests_per_s": _finite_number(result, "throughput_qps"),
            "latency_s": {
                "mean": statistics.fmean(latencies),
                "median": statistics.median(latencies),
                "p95": p95["value_s"],
                "p99": _percentile_type7(latencies, 0.99),
            },
        },
        "counts": counts,
        "p95_type7": p95,
        "slowest_requests": slowest,
        "per_workload_class": class_metrics,
        "tail": {
            "request_count": len(tail_rows),
            "preemption_count": sum(_nonnegative_int(row, "scheduler_preempt_count") for row in tail_rows),
            "requests": [_request_snapshot(row) for row in tail_rows],
        },
        "preemption": {
            "request_count": len(preempted_rows),
            "event_count": sum(_nonnegative_int(row, "scheduler_preempt_count") for row in preempted_rows),
            "request_ids": [str(row["request_id"]) for row in preempted_rows],
        },
        "clean_normal_calibration": _clean_normal_calibration(measured_rows),
    }


def percentile_type7_boundary(
    rows: list[dict[str, Any]],
    field: str,
    quantile: float,
) -> dict[str, Any]:
    """Return NumPy's default/type-7 percentile and its two boundary rows."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    if not rows:
        raise ArtifactValidationError("Cannot calculate a percentile with no rows")
    ordered = sorted(rows, key=lambda row: (_finite_number(row, field), str(row.get("request_id"))))
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    upper_weight = position - lower_index
    lower_weight = 1.0 - upper_weight
    lower_row = ordered[lower_index]
    upper_row = ordered[upper_index]
    lower_value = _finite_number(lower_row, field)
    upper_value = _finite_number(upper_row, field)
    value = lower_value * lower_weight + upper_value * upper_weight
    return {
        "method": "NumPy linear / Hyndman-Fan type 7",
        "quantile": quantile,
        "sample_count": len(ordered),
        "zero_based_position": position,
        "value_s": value,
        "lower": {
            "zero_based_index": lower_index,
            "one_based_rank": lower_index + 1,
            "weight": lower_weight,
            **_request_snapshot(lower_row),
        },
        "upper": {
            "zero_based_index": upper_index,
            "one_based_rank": upper_index + 1,
            "weight": upper_weight,
            **_request_snapshot(upper_row),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the compact, human-readable portion of an analysis report."""
    summary = report["summary"]
    p95 = report["p95_type7"]
    lines = [
        "# Wan2.2 NPU Artifact Analysis",
        "",
        f"- Artifact: `{report['artifact']['path']}`",
        f"- SHA256: `{report['artifact']['sha256']}`",
        f"- Validation: **{report['validation']['status']}**",
        f"- Success: {summary['completed_requests']}/{summary['completed_requests']}",
        f"- Duration: {summary['duration_s']:.3f}s",
        f"- P95: **{summary['latency_s']['p95']:.3f}s**",
        "",
        "## Counts",
        "",
        "| Dimension | Counts |",
        "|---|---|",
    ]
    for dimension, counts in report["counts"].items():
        rendered = ", ".join(f"{key}={value}" for key, value in counts.items())
        lines.append(f"| {dimension} | {rendered} |")

    lines.extend(
        [
            "",
            "## NumPy Type-7 P95 Boundary",
            "",
            "| Rank | Request | Class | Queue | E2E (s) | Weight |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for boundary in (p95["lower"], p95["upper"]):
        lines.append(
            f"| {boundary['one_based_rank']} | `{boundary['request_id']}` | "
            f"{boundary['workload_class']} | {boundary['queue_class']} | "
            f"{boundary['e2e_latency_s']:.3f} | {boundary['weight']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Class Latency Components",
            "",
            "Each cell is mean / P95 in seconds. `exclusive = backend_inference - scheduler_queue_wait`.",
            "",
            "| Class | N | Scheduler | Backend | Exclusive | Central wait | E2E |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for workload_class, class_report in report["per_workload_class"].items():
        cells = [
            _mean_p95_cell(class_report[name])
            for name in (
                "scheduler_s",
                "backend_s",
                "exclusive_service_s",
                "central_wait_s",
                "e2e_s",
            )
        ]
        lines.append(f"| {workload_class} | {class_report['request_count']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Slowest Requests",
            "",
            "| Request | Class | Queue | Backend | E2E (s) | Central wait (s) | Preemptions |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in report["slowest_requests"]:
        lines.append(
            f"| `{row['request_id']}` | {row['workload_class']} | {row['queue_class']} | "
            f"{row['backend']} | {row['e2e_latency_s']:.3f} | "
            f"{row['central_wait_s']:.3f} | {row['scheduler_preempt_count']} |"
        )

    calibration = report["clean_normal_calibration"]
    lines.extend(
        [
            "",
            "## Clean Normal Calibration",
            "",
            "Filter: `queue=normal AND preempt_count=0 AND select_count=1`. "
            "Statistics use exclusive service after symmetric 10% trimming.",
            "",
            "| Class | Kept / Raw | Trim each side | Trimmed mean (s) | Median (s) | CV |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workload_class, class_report in calibration["per_workload_class"].items():
        lines.append(
            f"| {workload_class} | {class_report['trimmed_count']} / {class_report['raw_count']} | "
            f"{class_report['trim_each_side']} | {class_report['trimmed_mean_s']:.3f} | "
            f"{class_report['trimmed_median_s']:.3f} | "
            f"{class_report['trimmed_cv_population']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Pooled within-class log sigma (MLE):",
            "",
            f"- All clean samples, for `actual_jitter_sigma`: "
            f"**{calibration['pooled_within_class_log_sigma_mle']:.6f}**",
            f"- Per-class 10%-trimmed samples: "
            f"**{calibration['pooled_within_class_log_sigma_10pct_trimmed_mle']:.6f}**",
            "",
        ]
    )
    return "\n".join(lines)


def _regular_members_by_name(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        if member.name in members:
            raise ArtifactValidationError(f"Duplicate archive member: {member.name}")
        members[member.name] = member
    return members


def _load_json_member(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    name: str,
) -> Any:
    member = members.get(name)
    if member is None:
        raise ArtifactValidationError(f"Missing archive member: {name}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ArtifactValidationError(f"Cannot read archive member: {name}")
    return _strict_json_loads(extracted.read(), name)


def _load_raw_trace(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_files = sorted(name for name in members if name.startswith("trace/") and name.endswith(".jsonl"))
    if not raw_files:
        raise ArtifactValidationError("Artifact has no trace/*.jsonl members")
    events: list[dict[str, Any]] = []
    for name in raw_files:
        extracted = archive.extractfile(members[name])
        if extracted is None:
            raise ArtifactValidationError(f"Cannot read archive member: {name}")
        for line_number, line in enumerate(extracted, start=1):
            if not line.strip():
                continue
            event = _strict_json_loads(line, f"{name}:{line_number}")
            if not isinstance(event, dict):
                raise ArtifactValidationError(f"{name}:{line_number} must contain a JSON object")
            events.append(event)
    if not events:
        raise ArtifactValidationError("Raw trace has no JSON events")
    return events, raw_files


def _strict_json_loads(data: bytes, source: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ArtifactValidationError(f"Non-standard JSON constant {value!r} in {source}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactValidationError(f"Duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactValidationError(f"Non-UTF-8 JSON in {source}") from exc
    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"Invalid JSON at {source}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc


def _measured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row in rows:
        request_id = source_row.get("request_id")
        if not isinstance(request_id, str) or _MEASURED_REQUEST_RE.fullmatch(request_id) is None:
            continue
        if request_id in seen:
            raise ArtifactValidationError(f"Duplicate measured request row: {request_id}")
        seen.add(request_id)
        measured.append(dict(source_row))
    measured.sort(key=lambda row: int(_MEASURED_REQUEST_RE.fullmatch(str(row["request_id"])).group(1)))
    if not measured:
        raise ArtifactValidationError("request_trace.json contains no measured request rows")
    return measured


def _validate_measured_ids(request_ids: list[str]) -> None:
    expected = [f"request-{index:05d}" for index in range(len(request_ids))]
    if request_ids != expected:
        missing = sorted(set(expected) - set(request_ids))
        unexpected = sorted(set(request_ids) - set(expected))
        raise ArtifactValidationError(
            f"Measured request IDs are not contiguous from request-00000: missing={missing}, unexpected={unexpected}"
        )


def _validate_success_and_result(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    trace_summary: dict[str, Any],
) -> None:
    failed = [str(row["request_id"]) for row in rows if row.get("success") is not True]
    if failed:
        raise ArtifactValidationError(f"Measured requests are not all successful: {failed}")
    for row in rows:
        _finite_number(row, "e2e_latency_s")

    completed = _nonnegative_int(result, "completed_requests")
    failed_count = _nonnegative_int(result, "failed_requests")
    if completed != len(rows) or failed_count != 0:
        raise ArtifactValidationError(
            f"result.json success mismatch: rows={len(rows)}, completed={completed}, failed={failed_count}"
        )
    expected_summary = {
        "measured_request_count": len(rows),
        "successful_request_count": len(rows),
        "failed_request_count": 0,
    }
    for key, expected in expected_summary.items():
        actual = _nonnegative_int(trace_summary, key)
        if actual != expected:
            raise ArtifactValidationError(f"request_trace.json.summary.{key}={actual}, expected {expected}")


def _add_exclusive_service(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        backend_s = _finite_number(row, "backend_inference_time_s")
        scheduler_queue_s = _finite_number(row, "scheduler_queue_wait_s")
        exclusive_s = backend_s - scheduler_queue_s
        if exclusive_s <= 0.0:
            raise ArtifactValidationError(
                f"{row['request_id']} has non-positive exclusive service: {backend_s} - {scheduler_queue_s}"
            )
        row["exclusive_service_s"] = exclusive_s


def _validate_reported_counts(
    trace_summary: dict[str, Any],
    counts: dict[str, dict[str, int]],
) -> None:
    reported_keys = {
        "queue_class": "queue_class_counts",
        "workload_class": "workload_class_counts",
        "backend": "backend_counts",
    }
    for dimension, summary_key in reported_keys.items():
        reported = trace_summary.get(summary_key)
        if reported != counts[dimension]:
            raise ArtifactValidationError(
                f"request_trace.json.summary.{summary_key}={reported!r}, recomputed={counts[dimension]!r}"
            )


def _validate_raw_trace(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    raw_files: list[str],
) -> dict[str, Any]:
    measured_ids = {str(row["request_id"]) for row in rows}
    raw_measured_ids = {
        str(event["request_id"])
        for event in events
        if isinstance(event.get("request_id"), str)
        and _MEASURED_REQUEST_RE.fullmatch(str(event["request_id"])) is not None
    }
    if raw_measured_ids != measured_ids:
        raise ArtifactValidationError(
            "Raw and aggregated measured request IDs disagree: "
            f"missing_from_raw={sorted(measured_ids - raw_measured_ids)}, "
            f"unexpected_in_raw={sorted(raw_measured_ids - measured_ids)}"
        )
    video_to_request: dict[str, str] = {}
    request_to_video: dict[str, set[str]] = defaultdict(set)
    for event in events:
        request_id = event.get("request_id")
        video_id = event.get("video_id")
        if request_id in measured_ids and isinstance(video_id, str) and video_id:
            owner = video_to_request.setdefault(video_id, str(request_id))
            if owner != request_id:
                raise ArtifactValidationError(f"video_id {video_id!r} maps to both {owner} and {request_id}")
            request_to_video[str(request_id)].add(video_id)

    missing_video_ids = sorted(request_id for request_id in measured_ids if not request_to_video[request_id])
    ambiguous_video_ids = sorted(request_id for request_id in measured_ids if len(request_to_video[request_id]) != 1)
    if missing_video_ids or ambiguous_video_ids:
        raise ArtifactValidationError(
            "Raw trace request/video mapping is incomplete: "
            f"missing={missing_video_ids}, ambiguous={ambiguous_video_ids}"
        )

    event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    raw_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        raw_request_id = event.get("request_id")
        canonical_id = (
            str(raw_request_id) if raw_request_id in measured_ids else video_to_request.get(str(raw_request_id))
        )
        if canonical_id is None:
            continue
        event_name = event.get("event")
        if not isinstance(event_name, str) or not event_name:
            raise ArtifactValidationError(f"Raw event for {canonical_id} has no event name")
        event_counts[canonical_id][event_name] += 1
        raw_by_request[canonical_id].append(event)

    for row in rows:
        request_id = str(row["request_id"])
        counts = event_counts[request_id]
        for event_name in _REQUIRED_LIFECYCLE_EVENTS:
            if counts[event_name] != 1:
                raise ArtifactValidationError(
                    f"{request_id} raw lifecycle has {counts[event_name]} {event_name} events; expected 1"
                )
        if counts["scheduler_select"] != _nonnegative_int(row, "scheduler_select_count"):
            raise ArtifactValidationError(f"{request_id} scheduler_select count disagrees with request trace")
        if counts["scheduler_preempt"] != _nonnegative_int(row, "scheduler_preempt_count"):
            raise ArtifactValidationError(f"{request_id} scheduler_preempt count disagrees with request trace")
        failure_events = sum(
            counts[name] for name in ("client_failed", "dispatcher_failed", "backend_failed", "scheduler_failed")
        )
        if failure_events:
            raise ArtifactValidationError(f"{request_id} has {failure_events} raw failure events")
        client_finish = next(event for event in raw_by_request[request_id] if event.get("event") == "client_finish")
        if client_finish.get("success") is not True:
            raise ArtifactValidationError(f"{request_id} raw client_finish is not successful")
        raw_latency = _finite_number(client_finish, "latency_s")
        _assert_close(
            raw_latency,
            _finite_number(row, "e2e_latency_s"),
            f"{request_id} raw/client aggregate latency",
        )

    return {
        "jsonl_file_count": len(raw_files),
        "nodes": sorted({str(event.get("node")) for event in events if isinstance(event.get("node"), str)}),
        "event_count": len(events),
        "measured_request_lifecycles_complete": len(rows),
        "request_video_mapping": "one-to-one",
    }


def _validate_latency_aggregates(
    result: dict[str, Any],
    trace_summary: dict[str, Any],
    latencies: list[float],
    p95: float,
) -> None:
    calculated = {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p95": p95,
        "p99": _percentile_type7(latencies, 0.99),
    }
    result_keys = {
        "mean": "latency_mean",
        "median": "latency_median",
        "p95": "latency_p95",
        "p99": "latency_p99",
    }
    summary_keys = {
        "mean": "latency_mean_s",
        "p95": "latency_p95_s",
        "p99": "latency_p99_s",
    }
    for name, key in result_keys.items():
        _assert_close(_finite_number(result, key), calculated[name], f"result.json.{key}")
    for name, key in summary_keys.items():
        _assert_close(
            _finite_number(trace_summary, key),
            calculated[name],
            f"request_trace.json.summary.{key}",
        )


def _clean_normal_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_rows = [
        row
        for row in rows
        if str(row.get("queue_class", "")).lower() == "normal"
        and _nonnegative_int(row, "scheduler_preempt_count") == 0
        and _nonnegative_int(row, "scheduler_select_count") == 1
    ]
    if not clean_rows:
        raise ArtifactValidationError("No clean Normal calibration rows match preempt_count=0 and select_count=1")
    per_class: dict[str, Any] = {}
    all_by_class: dict[str, list[float]] = {}
    trimmed_by_class: dict[str, list[float]] = {}
    for workload_class in _sorted_classes(clean_rows):
        values = sorted(
            _finite_number(row, "exclusive_service_s")
            for row in clean_rows
            if row.get("workload_class") == workload_class
        )
        trim_each_side = math.floor(len(values) * 0.10)
        trimmed = values[trim_each_side : len(values) - trim_each_side] if trim_each_side else values
        if not trimmed:
            raise ArtifactValidationError(f"10% trimming removed all {workload_class} calibration samples")
        mean = statistics.fmean(trimmed)
        population_stddev = statistics.pstdev(trimmed)
        all_by_class[workload_class] = values
        trimmed_by_class[workload_class] = trimmed
        per_class[workload_class] = {
            "raw_count": len(values),
            "trim_fraction_each_side": 0.10,
            "trim_each_side": trim_each_side,
            "trimmed_count": len(trimmed),
            "trimmed_mean_s": mean,
            "trimmed_median_s": statistics.median(trimmed),
            "trimmed_cv_population": population_stddev / mean,
            "trimmed_min_s": min(trimmed),
            "trimmed_max_s": max(trimmed),
        }

    return {
        "filter": {
            "queue_class": "normal",
            "scheduler_preempt_count": 0,
            "scheduler_select_count": 1,
            "success": True,
        },
        "exclusive_service_formula": "backend_inference_time_s - scheduler_queue_wait_s",
        "request_count_before_trimming": len(clean_rows),
        "request_count_after_trimming": sum(len(values) for values in trimmed_by_class.values()),
        "per_workload_class": per_class,
        "pooled_within_class_log_sigma_mle": _pooled_within_class_log_sigma(all_by_class),
        "pooled_within_class_log_sigma_10pct_trimmed_mle": _pooled_within_class_log_sigma(trimmed_by_class),
        "pooled_log_sigma_formula": (
            "sqrt(mean((log(exclusive_service_s) - mean_log_within_workload_class)^2)) "
            "over all clean Normal samples; use as actual_jitter_sigma"
        ),
        "trimmed_pooled_log_sigma_formula": (
            "the same within-class MLE over the per-class symmetric 10%-trimmed samples"
        ),
    }


def _pooled_within_class_log_sigma(values_by_class: dict[str, list[float]]) -> float:
    centered_logs: list[float] = []
    for values in values_by_class.values():
        logs = [math.log(value) for value in values]
        class_log_mean = statistics.fmean(logs)
        centered_logs.extend(value - class_log_mean for value in logs)
    if not centered_logs:
        raise ArtifactValidationError("Cannot calculate pooled log sigma without calibration samples")
    return math.sqrt(statistics.fmean(value * value for value in centered_logs))


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ArtifactValidationError("Metric group is empty")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile_type7(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _request_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "workload_class": row.get("workload_class"),
        "queue_class": row.get("queue_class"),
        "backend": row.get("backend"),
        "arrival_counter": row.get("arrival_counter"),
        "scheduler_select_count": _nonnegative_int(row, "scheduler_select_count"),
        "scheduler_preempt_count": _nonnegative_int(row, "scheduler_preempt_count"),
        "scheduler_elapsed_s": _finite_number(row, "scheduler_elapsed_s"),
        "scheduler_queue_wait_s": _finite_number(row, "scheduler_queue_wait_s"),
        "backend_inference_time_s": _finite_number(row, "backend_inference_time_s"),
        "exclusive_service_s": _finite_number(row, "exclusive_service_s"),
        "central_wait_s": _finite_number(row, "central_wait_s"),
        "e2e_latency_s": _finite_number(row, "e2e_latency_s"),
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ArtifactValidationError(f"{row.get('request_id')} has invalid {key}: {value!r}")
        counter[value] += 1
    return dict(sorted(counter.items()))


def _sorted_classes(rows: Iterable[dict[str, Any]]) -> list[str]:
    classes = {str(row.get("workload_class")) for row in rows}
    return sorted(classes, key=lambda value: (_WORKLOAD_ORDER.get(value, 100), value))


def _finite_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{mapping.get('request_id', 'payload')}.{key} is not a number: {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ArtifactValidationError(f"{mapping.get('request_id', 'payload')}.{key} is not finite: {value!r}")
    return converted


def _nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactValidationError(
            f"{mapping.get('request_id', 'payload')}.{key} is not a non-negative integer: {value!r}"
        )
    return value


def _percentile_type7(values: list[float], quantile: float) -> float:
    if not values:
        raise ArtifactValidationError("Cannot calculate a percentile with no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
        raise ArtifactValidationError(f"{label}={actual}, recomputed={expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_p95_cell(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.3f} / {summary['p95']:.3f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Path to the Wan2.2 benchmark artifact tar.gz.")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. JSON is the default.",
    )
    parser.add_argument(
        "--output",
        help="Write the report to this path instead of stdout. The input artifact is never modified.",
    )
    parser.add_argument(
        "--slowest-limit",
        type=int,
        default=5,
        help="Number of slowest requests to include (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze_artifact(args.artifact, slowest_limit=args.slowest_limit)
        output = (
            render_markdown(report)
            if args.format == "markdown"
            else json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except (ArtifactValidationError, tarfile.TarError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
