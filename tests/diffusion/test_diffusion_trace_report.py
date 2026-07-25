# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.diffusion.simulator.trace_report import (  # noqa: E402
    bubble_intervals,
    load_events,
    load_requests,
    phase_intervals,
    request_intervals,
    write_report,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    events = [
        {
            "seed": 7,
            "event": "phase_start",
            "backend": "backend-0",
            "request_id": "request-0",
            "phase": "denoise",
            "step_index": 0,
            "time_s": 0.0,
            "duration_s": 2.0,
        },
        {
            "seed": 7,
            "event": "phase_start",
            "backend": "backend-1",
            "request_id": "request-2",
            "phase": "denoise",
            "step_index": 0,
            "time_s": 0.0,
            "duration_s": 8.0,
        },
        {
            "seed": 7,
            "event": "phase_start",
            "backend": "backend-0",
            "request_id": "request-1",
            "phase": "decode",
            "step_index": None,
            "time_s": 4.0,
            "duration_s": 2.0,
        },
    ]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    fieldnames = [
        "seed",
        "request_id",
        "request_type",
        "priority",
        "backend",
        "arrival_time_s",
        "first_start_time_s",
        "completion_time_s",
        "latency_s",
        "queue_before_first_start_s",
        "service_s",
        "preemptions",
        "resumes",
    ]
    rows = [
        [7, "request-0", "short", "normal", "backend-0", 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0, 0],
        [7, "request-1", "long", "sacrificial", "backend-0", 0.5, 4.0, 6.0, 5.5, 3.5, 2.0, 0, 0],
        [7, "request-2", "medium", "normal", "backend-1", 0.0, 0.0, 8.0, 8.0, 0.0, 8.0, 0, 0],
    ]
    requests_path = tmp_path / "requests.csv"
    with requests_path.open("w", encoding="utf-8", newline="") as request_file:
        writer = csv.writer(request_file)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return events_path, requests_path


def test_trace_report_classifies_local_and_drain_bubbles(tmp_path: Path) -> None:
    events_path, requests_path = _write_fixture(tmp_path)
    event_seed, events = load_events(events_path)
    request_seed, requests = load_requests(requests_path)
    intervals = phase_intervals(events)
    bubbles = bubble_intervals(intervals, requests, threshold_s=0.5)

    assert event_seed == request_seed == 7
    assert [
        (bubble.backend, bubble.start_s, bubble.end_s, bubble.kind)
        for bubble in bubbles
        if bubble.backend == "backend-0"
    ] == [
        ("backend-0", 2.0, 4.0, "local-pending"),
        ("backend-0", 6.0, 8.0, "drain-imbalance"),
    ]


def test_request_intervals_collapse_each_request_to_one_block(tmp_path: Path) -> None:
    _, requests_path = _write_fixture(tmp_path)
    _, requests = load_requests(requests_path)

    intervals = request_intervals(requests)

    assert [
        (interval.request_id, interval.start_s, interval.end_s, interval.phase)
        for interval in intervals
    ] == [
        ("request-0", 0.0, 2.0, "request"),
        ("request-1", 4.0, 6.0, "request"),
        ("request-2", 0.0, 8.0, "request"),
    ]


def test_trace_report_writes_html_svg_and_json(tmp_path: Path) -> None:
    events_path, requests_path = _write_fixture(tmp_path)
    output_path = tmp_path / "report.html"

    paths = write_report(
        events_path=events_path,
        requests_path=requests_path,
        output_path=output_path,
        seed=None,
        bubble_threshold_s=0.5,
        max_requests=3,
    )

    assert set(paths) == {
        "report",
        "backend_timeline",
        "request_waits",
        "summary",
    }
    assert all(path.is_file() for path in paths.values())
    assert "local-pending" in output_path.read_text(encoding="utf-8")
    timeline_svg = paths["backend_timeline"].read_text(encoding="utf-8")
    waits_svg = paths["request_waits"].read_text(encoding="utf-8")
    assert 'fill="#ffffff"' in timeline_svg
    assert 'fill="#ffffff"' in waits_svg
    assert "execution=" in timeline_svg
    assert "phase=denoise" not in timeline_svg
    assert "request-1" in waits_svg
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["seed"] == 7
    assert summary["request_count"] == 3
    assert summary["queue_wait_p95_s"] == pytest.approx(3.15)
