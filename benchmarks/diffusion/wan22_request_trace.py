# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Merge Wan2.2 client, dispatcher, and backend JSONL request traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROW_FIELDS = (
    "request_id",
    "workload_class",
    "width",
    "height",
    "num_frames",
    "num_inference_steps",
    "fps",
    "arrival_counter",
    "queue_class",
    "estimated_service_s",
    "normal_routing_policy",
    "tail_routing_mode",
    "tail_dispatch_mode",
    "tail_gate_wait_s",
    "central_risk_beta",
    "central_risk_score",
    "central_queue_depth",
    "central_wait_s",
    "backend",
    "video_id",
    "sched_req_id",
    "dispatcher_arrival_offset_s",
    "api_backend_start_offset_s",
    "scheduler_start_offset_s",
    "backend_start_offset_s",
    "scheduler_queue_wait_s",
    "scheduler_elapsed_s",
    "scheduler_select_count",
    "scheduler_preempt_count",
    "backend_inference_time_s",
    "prediction_error_percent",
    "dispatcher_e2e_s",
    "e2e_latency_s",
    "success",
    "error",
)


def load_trace_events(trace_dir: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(trace_dir)
    events: list[dict[str, Any]] = []
    for path in sorted(trace_path.glob("*.jsonl")):
        with path.open(encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(event, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                event["_trace_file"] = path.name
                events.append(event)
    events.sort(key=lambda event: (int(event.get("ts_ns", 0)), str(event.get("event", ""))))
    return events


def build_request_rows(
    events: list[dict[str, Any]],
    *,
    include_warmups: bool = False,
) -> list[dict[str, Any]]:
    request_id_by_video_id: dict[str, str] = {}
    for event in events:
        video_id = event.get("video_id")
        request_id = event.get("request_id")
        if (
            isinstance(video_id, str)
            and video_id
            and isinstance(request_id, str)
            and request_id
            and not str(event.get("event", "")).startswith("scheduler_")
        ):
            request_id_by_video_id[video_id] = request_id

    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        canonical_request_id = request_id_by_video_id.get(request_id, request_id)
        if not include_warmups and canonical_request_id.startswith("warmup-"):
            continue
        by_request[canonical_request_id].append(event)

    rows = [_build_request_row(request_id, request_events) for request_id, request_events in by_request.items()]
    return sorted(
        rows,
        key=lambda row: (
            _sortable_number(row.get("arrival_counter")),
            str(row["request_id"]),
        ),
    )


def summarize_request_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if isinstance(row.get("e2e_latency_s"), (int, float))]
    successful = [row for row in measured if row.get("success") is True]
    latencies = [float(row["e2e_latency_s"]) for row in successful]
    prediction_errors = [
        float(row["prediction_error_percent"])
        for row in rows
        if isinstance(row.get("prediction_error_percent"), (int, float))
    ]
    return {
        "request_count": len(rows),
        "measured_request_count": len(measured),
        "successful_request_count": len(successful),
        "failed_request_count": sum(row.get("success") is False for row in measured),
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "prediction_error_mean_percent": (
            sum(prediction_errors) / len(prediction_errors) if prediction_errors else None
        ),
        "queue_class_counts": _count_values(rows, "queue_class"),
        "workload_class_counts": _count_values(rows, "workload_class"),
        "backend_counts": _count_values(rows, "backend"),
    }


def write_trace_report(
    rows: list[dict[str, Any]],
    *,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")

    payload = {
        "summary": summarize_request_rows(rows),
        "requests": rows,
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _build_request_row(request_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    first = events[0]
    client_arrive = _first_event(events, "client_arrive")
    dispatcher_arrive = _first_event(events, "dispatcher_arrive")
    central_enqueue = _first_event(events, "central_enqueue")
    dispatch = _first_event(events, "dispatch")
    job_accepted = _first_event(events, "client_job_accepted") or _first_event(events, "dispatcher_job_accepted")
    backend_start = _first_event(events, "backend_start")
    backend_terminal = _first_event(events, "backend_complete") or _first_event(events, "backend_failed")
    scheduler_selects = _events_named(events, "scheduler_select")
    scheduler_start = scheduler_selects[0] if scheduler_selects else None
    scheduler_terminal = _first_event(events, "scheduler_complete") or _first_event(events, "scheduler_failed")
    scheduler_preempts = _events_named(events, "scheduler_preempt")
    dispatcher_terminal = _first_event(events, "dispatcher_complete") or _first_event(events, "dispatcher_failed")
    client_finish = _first_event(events, "client_finish")
    input_event = client_arrive or dispatcher_arrive or dispatch or backend_start or first

    arrival_ts = _number(client_arrive, "ts")
    dispatcher_arrival_ts = _number(dispatcher_arrive, "ts")
    api_backend_start_ts = _number(backend_start, "ts")
    scheduler_start_ts = _number(scheduler_start, "ts")
    scheduler_terminal_ts = _number(scheduler_terminal, "ts")
    actual_backend_start_ts = scheduler_start_ts or api_backend_start_ts
    estimated_service_s = _coalesce_number(dispatch, "estimated_service_s", central_enqueue)
    backend_inference_time_s = _coalesce_number(
        backend_terminal,
        "inference_time_s",
        dispatcher_terminal,
        client_finish,
        fallback_key="backend_inference_time_s",
    )

    success = client_finish.get("success") if client_finish is not None else None
    if success is None and backend_terminal is not None:
        success = backend_terminal.get("event") == "backend_complete"
    error = _coalesce(
        client_finish,
        "error",
        dispatcher_terminal,
        backend_terminal,
    )

    return {
        "request_id": request_id,
        "workload_class": _coalesce(input_event, "workload_class", dispatch, backend_start),
        "width": _coalesce(input_event, "width", dispatch, backend_start),
        "height": _coalesce(input_event, "height", dispatch, backend_start),
        "num_frames": _coalesce(input_event, "num_frames", dispatch, backend_start),
        "num_inference_steps": _coalesce(
            input_event,
            "num_inference_steps",
            dispatch,
            backend_start,
        ),
        "fps": _coalesce(input_event, "fps", dispatch, backend_start),
        "arrival_counter": _coalesce(dispatch, "arrival_counter", central_enqueue),
        "queue_class": _coalesce(dispatch, "queue_class", backend_start),
        "estimated_service_s": estimated_service_s,
        "normal_routing_policy": _coalesce(dispatch, "normal_routing_policy", central_enqueue),
        "tail_routing_mode": _coalesce(dispatch, "tail_routing_mode"),
        "tail_dispatch_mode": _coalesce(dispatch, "tail_dispatch_mode"),
        "tail_gate_wait_s": _coalesce_number(dispatch, "tail_gate_wait_s"),
        "central_risk_beta": _coalesce_number(dispatch, "central_risk_beta", central_enqueue),
        "central_risk_score": _coalesce_number(dispatch, "central_risk_score"),
        "central_queue_depth": _coalesce(central_enqueue, "queue_depth"),
        "central_wait_s": _coalesce_number(dispatch, "central_wait_s"),
        "backend": _coalesce(dispatch, "backend", backend_start, fallback_key="node"),
        "video_id": _coalesce(job_accepted, "video_id", backend_start),
        "sched_req_id": _coalesce(scheduler_start, "sched_req_id", scheduler_terminal),
        "dispatcher_arrival_offset_s": _difference(dispatcher_arrival_ts, arrival_ts),
        "api_backend_start_offset_s": _difference(api_backend_start_ts, arrival_ts),
        "scheduler_start_offset_s": _difference(scheduler_start_ts, arrival_ts),
        "backend_start_offset_s": _difference(actual_backend_start_ts, arrival_ts),
        "scheduler_queue_wait_s": _difference(scheduler_start_ts, api_backend_start_ts),
        "scheduler_elapsed_s": _difference(scheduler_terminal_ts, scheduler_start_ts),
        "scheduler_select_count": len(scheduler_selects),
        "scheduler_preempt_count": len(scheduler_preempts),
        "backend_inference_time_s": backend_inference_time_s,
        "prediction_error_percent": _prediction_error_percent(
            estimated_service_s,
            backend_inference_time_s,
        ),
        "dispatcher_e2e_s": _coalesce_number(dispatcher_terminal, "dispatcher_e2e_s"),
        "e2e_latency_s": _coalesce_number(client_finish, "latency_s"),
        "success": success,
        "error": error,
    }


def _first_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("event") == event_name), None)


def _events_named(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == event_name]


def _coalesce(
    first: dict[str, Any] | None,
    key: str,
    *fallback_events: dict[str, Any] | None,
    fallback_key: str | None = None,
) -> Any:
    keys = (key, fallback_key) if fallback_key and fallback_key != key else (key,)
    for event in (first, *fallback_events):
        if event is None:
            continue
        for candidate_key in keys:
            value = event.get(candidate_key)
            if value is not None:
                return value
    return None


def _number(event: dict[str, Any] | None, key: str) -> float | None:
    if event is None:
        return None
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _coalesce_number(
    first: dict[str, Any] | None,
    key: str,
    *fallback_events: dict[str, Any] | None,
    fallback_key: str | None = None,
) -> float | None:
    value = _coalesce(first, key, *fallback_events, fallback_key=fallback_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _difference(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    return max(end - start, 0.0)


def _prediction_error_percent(estimate: float | None, actual: float | None) -> float | None:
    if estimate is None or actual is None or estimate <= 0.0:
        return None
    return (actual - estimate) / estimate * 100.0


def _sortable_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.inf
    return float(value)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing client/dispatcher/backend JSONL files.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output path without extension. Defaults to TRACE_DIR/wan22_request_trace.",
    )
    parser.add_argument(
        "--include-warmups",
        action="store_true",
        help="Include request ids prefixed with warmup- in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix or str(Path(args.trace_dir) / "wan22_request_trace")
    events = load_trace_events(args.trace_dir)
    rows = build_request_rows(events, include_warmups=args.include_warmups)
    json_path, csv_path = write_trace_report(rows, output_prefix=output_prefix)
    print(json.dumps(summarize_request_rows(rows), indent=2, ensure_ascii=False))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
