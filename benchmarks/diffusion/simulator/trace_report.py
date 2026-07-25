# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Render simulator JSONL/CSV traces as a self-contained diagnostic report.

The report deliberately uses only the Python standard library so it can run in
the same CPU-only environment as the simulator.  It produces:

* a backend timeline with one continuous block per request and idle/bubble
  intervals;
* a longest-wait request chart;
* an HTML report with backend and request tables;
* a machine-readable JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhaseInterval:
    backend: str
    request_id: str
    phase: str
    start_s: float
    end_s: float
    step_index: int | None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    request_type: str
    priority: str
    backend: str
    arrival_time_s: float
    first_start_time_s: float
    completion_time_s: float
    latency_s: float
    queue_before_first_start_s: float
    service_s: float
    preemptions: int
    resumes: int

    @property
    def total_wait_s(self) -> float:
        return max(self.latency_s - self.service_s, 0.0)

    @property
    def wait_after_first_start_s(self) -> float:
        return max(self.total_wait_s - self.queue_before_first_start_s, 0.0)


@dataclass(frozen=True)
class BubbleInterval:
    backend: str
    start_s: float
    end_s: float
    kind: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    return _integer(value, field)


def load_events(path: str | Path, seed: int | None = None) -> tuple[int, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    seeds: set[int] = set()
    with Path(path).open(encoding="utf-8") as event_file:
        for line_number, raw_line in enumerate(event_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"event at {path}:{line_number} must be an object")
            event_seed = _integer(event.get("seed"), f"event[{line_number}].seed")
            seeds.add(event_seed)
            if seed is None or event_seed == seed:
                events.append(event)
    selected_seed = _select_seed(seeds, seed, "events")
    if not events:
        raise ValueError(f"no events found for seed {selected_seed}")
    return selected_seed, events


def load_requests(path: str | Path, seed: int | None = None) -> tuple[int, list[RequestRecord]]:
    requests: list[RequestRecord] = []
    seeds: set[int] = set()
    with Path(path).open(encoding="utf-8", newline="") as request_file:
        reader = csv.DictReader(request_file)
        required = {
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
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"request CSV is missing field(s): {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            row_seed = _integer(row["seed"], f"request[{row_number}].seed")
            seeds.add(row_seed)
            if seed is not None and row_seed != seed:
                continue
            requests.append(
                RequestRecord(
                    request_id=str(row["request_id"]),
                    request_type=str(row["request_type"]),
                    priority=str(row["priority"]),
                    backend=str(row["backend"]),
                    arrival_time_s=_finite_float(row["arrival_time_s"], f"request[{row_number}].arrival"),
                    first_start_time_s=_finite_float(
                        row["first_start_time_s"],
                        f"request[{row_number}].first_start",
                    ),
                    completion_time_s=_finite_float(
                        row["completion_time_s"],
                        f"request[{row_number}].completion",
                    ),
                    latency_s=_finite_float(row["latency_s"], f"request[{row_number}].latency"),
                    queue_before_first_start_s=_finite_float(
                        row["queue_before_first_start_s"],
                        f"request[{row_number}].queue_wait",
                    ),
                    service_s=_finite_float(row["service_s"], f"request[{row_number}].service"),
                    preemptions=_integer(row["preemptions"], f"request[{row_number}].preemptions"),
                    resumes=_integer(row["resumes"], f"request[{row_number}].resumes"),
                )
            )
    selected_seed = _select_seed(seeds, seed, "requests")
    if not requests:
        raise ValueError(f"no requests found for seed {selected_seed}")
    return selected_seed, requests


def _select_seed(seeds: set[int], requested: int | None, source: str) -> int:
    if not seeds:
        raise ValueError(f"{source} contain no seeds")
    if requested is not None:
        if requested not in seeds:
            raise ValueError(f"seed {requested} is not present in {source}")
        return requested
    if len(seeds) != 1:
        raise ValueError(f"{source} contain multiple seeds {sorted(seeds)}; pass --seed")
    return next(iter(seeds))


def phase_intervals(events: Iterable[dict[str, Any]]) -> list[PhaseInterval]:
    intervals: list[PhaseInterval] = []
    for index, event in enumerate(events):
        if event.get("event") != "phase_start":
            continue
        start_s = _finite_float(event.get("time_s"), f"events[{index}].time_s")
        duration_s = _finite_float(event.get("duration_s"), f"events[{index}].duration_s")
        if duration_s < 0.0:
            raise ValueError(f"events[{index}].duration_s cannot be negative")
        backend = event.get("backend")
        request_id = event.get("request_id")
        phase = event.get("phase")
        if not all(isinstance(value, str) and value for value in (backend, request_id, phase)):
            raise ValueError(f"events[{index}] phase_start is missing backend/request_id/phase")
        intervals.append(
            PhaseInterval(
                backend=backend,
                request_id=request_id,
                phase=phase,
                start_s=start_s,
                end_s=start_s + duration_s,
                step_index=_optional_integer(event.get("step_index"), f"events[{index}].step_index"),
            )
        )
    if not intervals:
        raise ValueError("trace contains no phase_start events")
    return intervals


def request_intervals(requests: Sequence[RequestRecord]) -> list[PhaseInterval]:
    """Represent each request as one non-preemptive first-start-to-end block."""
    intervals: list[PhaseInterval] = []
    for request in requests:
        if request.first_start_time_s < request.arrival_time_s:
            raise ValueError(
                f"{request.request_id} starts before it arrives"
            )
        if request.completion_time_s < request.first_start_time_s:
            raise ValueError(
                f"{request.request_id} completes before it starts"
            )
        intervals.append(
            PhaseInterval(
                backend=request.backend,
                request_id=request.request_id,
                phase="request",
                start_s=request.first_start_time_s,
                end_s=request.completion_time_s,
                step_index=None,
            )
        )
    return intervals


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _active_at(intervals: Sequence[PhaseInterval], backend: str, time_s: float) -> bool:
    return any(
        interval.backend == backend and interval.start_s <= time_s < interval.end_s
        for interval in intervals
    )


def _bubble_kind(
    backend: str,
    start_s: float,
    end_s: float,
    requests: Sequence[RequestRecord],
    intervals: Sequence[PhaseInterval],
) -> str:
    midpoint = (start_s + end_s) / 2.0
    local_pending = any(
        request.backend == backend
        and request.arrival_time_s <= midpoint < request.completion_time_s
        and not _active_at(intervals, backend, midpoint)
        for request in requests
    )
    if local_pending:
        return "local-pending"
    global_waiting = any(
        request.arrival_time_s <= midpoint < request.first_start_time_s
        for request in requests
    )
    if global_waiting:
        return "global-waiting"
    global_unfinished = any(
        request.arrival_time_s <= midpoint < request.completion_time_s
        for request in requests
    )
    if global_unfinished:
        if midpoint < max(request.arrival_time_s for request in requests):
            return "arrival-gap"
        return "drain-imbalance"
    return "no-work"


def bubble_intervals(
    intervals: Sequence[PhaseInterval],
    requests: Sequence[RequestRecord],
    *,
    threshold_s: float,
) -> list[BubbleInterval]:
    if threshold_s < 0.0:
        raise ValueError("bubble threshold cannot be negative")
    start_s = min(request.arrival_time_s for request in requests)
    end_s = max(request.completion_time_s for request in requests)
    backends = sorted(
        {request.backend for request in requests if request.backend}
        | {interval.backend for interval in intervals},
        key=_natural_key,
    )
    bubbles: list[BubbleInterval] = []
    for backend in backends:
        backend_intervals = sorted(
            (interval for interval in intervals if interval.backend == backend),
            key=lambda interval: (interval.start_s, interval.end_s),
        )
        cursor = start_s
        for interval in backend_intervals:
            gap_end = min(max(interval.start_s, start_s), end_s)
            if gap_end - cursor >= threshold_s:
                bubbles.append(
                    BubbleInterval(
                        backend=backend,
                        start_s=cursor,
                        end_s=gap_end,
                        kind=_bubble_kind(backend, cursor, gap_end, requests, intervals),
                    )
                )
            cursor = max(cursor, min(interval.end_s, end_s))
        if end_s - cursor >= threshold_s:
            bubbles.append(
                BubbleInterval(
                    backend=backend,
                    start_s=cursor,
                    end_s=end_s,
                    kind=_bubble_kind(backend, cursor, end_s, requests, intervals),
                )
            )
    return bubbles


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_summary(
    intervals: Sequence[PhaseInterval],
    requests: Sequence[RequestRecord],
    bubbles: Sequence[BubbleInterval],
) -> dict[str, Any]:
    start_s = min(request.arrival_time_s for request in requests)
    end_s = max(request.completion_time_s for request in requests)
    makespan_s = max(end_s - start_s, 0.0)
    backends = sorted(
        {request.backend for request in requests if request.backend}
        | {interval.backend for interval in intervals},
        key=_natural_key,
    )
    backend_rows: list[dict[str, Any]] = []
    for backend in backends:
        busy_s = sum(interval.duration_s for interval in intervals if interval.backend == backend)
        backend_bubbles = [bubble for bubble in bubbles if bubble.backend == backend]
        diagnostic_bubbles = [
            bubble
            for bubble in backend_bubbles
            if bubble.kind != "no-work"
        ]
        backend_rows.append(
            {
                "backend": backend,
                "busy_s": busy_s,
                "utilization_pct": 0.0 if makespan_s <= 0.0 else busy_s / makespan_s * 100.0,
                "diagnostic_bubble_s": sum(bubble.duration_s for bubble in diagnostic_bubbles),
                "largest_diagnostic_bubble_s": max(
                    (bubble.duration_s for bubble in diagnostic_bubbles),
                    default=0.0,
                ),
                "bubble_count": len(diagnostic_bubbles),
            }
        )
    total_capacity_s = makespan_s * len(backends)
    total_busy_s = sum(row["busy_s"] for row in backend_rows)
    waits = [request.total_wait_s for request in requests]
    queue_waits = [request.queue_before_first_start_s for request in requests]
    return {
        "seed": None,
        "request_count": len(requests),
        "backend_count": len(backends),
        "start_s": start_s,
        "end_s": end_s,
        "makespan_s": makespan_s,
        "aggregate_utilization_pct": (
            0.0 if total_capacity_s <= 0.0 else total_busy_s / total_capacity_s * 100.0
        ),
        "diagnostic_bubble_s": sum(
            bubble.duration_s for bubble in bubbles if bubble.kind != "no-work"
        ),
        "queue_wait_p50_s": percentile(queue_waits, 0.50),
        "queue_wait_p95_s": percentile(queue_waits, 0.95),
        "total_wait_p95_s": percentile(waits, 0.95),
        "latency_p95_s": percentile([request.latency_s for request in requests], 0.95),
        "backend_rows": backend_rows,
        "largest_bubbles": [
            asdict(bubble)
            | {"duration_s": bubble.duration_s}
            for bubble in sorted(
                (bubble for bubble in bubbles if bubble.kind != "no-work"),
                key=lambda bubble: (-bubble.duration_s, _natural_key(bubble.backend)),
            )[:20]
        ],
        "longest_wait_requests": [
            {
                "request_id": request.request_id,
                "request_type": request.request_type,
                "priority": request.priority,
                "backend": request.backend,
                "queue_before_first_start_s": request.queue_before_first_start_s,
                "execution_s": max(
                    request.completion_time_s - request.first_start_time_s,
                    0.0,
                ),
                "latency_s": request.latency_s,
            }
            for request in sorted(
                requests,
                key=lambda request: (
                    -request.queue_before_first_start_s,
                    request.arrival_time_s,
                ),
            )[:20]
        ],
    }


_REQUEST_TYPE_COLORS = {
    "short": "#4e79a7",
    "medium": "#59a14f",
    "long": "#f28e2b",
}
_BUBBLE_COLORS = {
    "local-pending": "#e15759",
    "global-waiting": "#b07aa1",
    "arrival-gap": "#bab0ab",
    "drain-imbalance": "#edc948",
    "no-work": "#d9d9d9",
}


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg_header(width: int, height: int, label: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{_svg_escape(label)}">'
        ),
        "<style>",
        "text{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#202124}",
        ".axis{font-size:12px}.lane{font-size:13px;font-weight:600}.small{font-size:11px}",
        ".grid{stroke:#d8dce2;stroke-width:1}.outline{stroke:#343a40;stroke-width:1}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def render_backend_timeline_svg(
    intervals: Sequence[PhaseInterval],
    requests: Sequence[RequestRecord],
    bubbles: Sequence[BubbleInterval],
) -> str:
    start_s = min(request.arrival_time_s for request in requests)
    end_s = max(request.completion_time_s for request in requests)
    span_s = max(end_s - start_s, 1e-9)
    backends = sorted(
        {request.backend for request in requests if request.backend}
        | {interval.backend for interval in intervals},
        key=_natural_key,
    )
    width = 1440
    left = 120
    right = 30
    top = 60
    lane_height = 68
    plot_width = width - left - right
    height = top + lane_height * len(backends) + 90
    request_by_id = {request.request_id: request for request in requests}

    def x(time_s: float) -> float:
        return left + (time_s - start_s) / span_s * plot_width

    lines = _svg_header(width, height, "Backend request timeline and idle bubbles")
    lines.extend(
        [
            "<defs>",
            '<pattern id="tail-hatch" width="6" height="6" patternUnits="userSpaceOnUse">',
            '<path d="M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2" stroke="#5f2a62" stroke-width="1.5"/>',
            "</pattern>",
            "</defs>",
            f'<text x="{left}" y="24" class="lane">Backend request timeline</text>',
            (
                f'<text x="{left}" y="43" class="small">Range {start_s:.1f}s–{end_s:.1f}s; '
                "each request is one first-start-to-completion block; hatched blocks are Tail</text>"
            ),
        ]
    )
    tick_count = 10
    for tick in range(tick_count + 1):
        tick_time = start_s + span_s * tick / tick_count
        tick_x = x(tick_time)
        lines.append(
            f'<line x1="{tick_x:.2f}" y1="{top - 8}" x2="{tick_x:.2f}" '
            f'y2="{top + lane_height * len(backends)}" class="grid"/>'
        )
        lines.append(
            f'<text x="{tick_x:.2f}" y="{top + lane_height * len(backends) + 22}" '
            f'text-anchor="middle" class="axis">{tick_time:.0f}s</text>'
        )

    for lane_index, backend in enumerate(backends):
        y = top + lane_index * lane_height
        lines.append(f'<text x="{left - 12}" y="{y + 28}" text-anchor="end" class="lane">{_svg_escape(backend)}</text>')
        lines.append(
            f'<rect x="{left}" y="{y + 7}" width="{plot_width}" height="30" '
            'fill="#f6f7f9" stroke="#d8dce2"/>'
        )
        for bubble in bubbles:
            if bubble.backend != backend or bubble.kind == "no-work":
                continue
            bubble_x = x(bubble.start_s)
            bubble_width = max(x(bubble.end_s) - bubble_x, 1.0)
            color = _BUBBLE_COLORS[bubble.kind]
            lines.append(
                f'<rect x="{bubble_x:.2f}" y="{y + 7}" width="{bubble_width:.2f}" height="30" '
                f'fill="{color}" fill-opacity="0.34">'
                f"<title>{_svg_escape(bubble.kind)} bubble {bubble.duration_s:.2f}s "
                f"({bubble.start_s:.2f}s–{bubble.end_s:.2f}s)</title></rect>"
            )
        for interval in intervals:
            if interval.backend != backend:
                continue
            interval_x = x(interval.start_s)
            interval_width = max(x(interval.end_s) - interval_x, 1.0)
            request = request_by_id.get(interval.request_id)
            request_type = "unknown" if request is None else request.request_type
            color = _REQUEST_TYPE_COLORS.get(request_type, "#76b7b2")
            tail = request is not None and request.priority == "sacrificial"
            type_detail = "" if request is None else f" type={request.request_type}"
            priority = "" if request is None else f" priority={request.priority}"
            lines.append(
                f'<rect x="{interval_x:.2f}" y="{y + 8}" width="{interval_width:.2f}" height="28" '
                f'fill="{color}" class="outline">'
                f"<title>{_svg_escape(interval.request_id)}{type_detail}{priority} "
                f"execution={interval.duration_s:.2f}s</title></rect>"
            )
            if tail:
                lines.append(
                    f'<rect x="{interval_x:.2f}" y="{y + 8}" width="{interval_width:.2f}" '
                    'height="28" fill="url(#tail-hatch)" pointer-events="none"/>'
                )
            if interval_width >= 48:
                label = interval.request_id.removeprefix("request-")
                lines.append(
                    f'<text x="{interval_x + interval_width / 2:.2f}" y="{y + 27}" '
                    f'text-anchor="middle" class="small">{_svg_escape(label)}</text>'
                )

    legend_y = top + lane_height * len(backends) + 52
    legend_items = [
        ("Short", _REQUEST_TYPE_COLORS["short"]),
        ("Medium", _REQUEST_TYPE_COLORS["medium"]),
        ("Long", _REQUEST_TYPE_COLORS["long"]),
        ("Local pending", _BUBBLE_COLORS["local-pending"]),
        ("Global waiting", _BUBBLE_COLORS["global-waiting"]),
        ("Arrival gap", _BUBBLE_COLORS["arrival-gap"]),
        ("Drain imbalance", _BUBBLE_COLORS["drain-imbalance"]),
    ]
    cursor_x = left
    for label, color in legend_items:
        lines.append(f'<rect x="{cursor_x}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{cursor_x + 20}" y="{legend_y}" class="small">{_svg_escape(label)}</text>')
        cursor_x += max(112, len(label) * 7 + 34)
    lines.append(
        f'<rect x="{cursor_x}" y="{legend_y - 12}" width="14" height="14" '
        'fill="#ffffff" stroke="#343a40"/>'
    )
    lines.append(
        f'<rect x="{cursor_x}" y="{legend_y - 12}" width="14" height="14" '
        'fill="url(#tail-hatch)"/>'
    )
    lines.append(f'<text x="{cursor_x + 20}" y="{legend_y}" class="small">Tail</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def render_request_waits_svg(
    requests: Sequence[RequestRecord],
    *,
    max_requests: int,
) -> str:
    selected = sorted(
        requests,
        key=lambda request: (
            -request.queue_before_first_start_s,
            request.arrival_time_s,
        ),
    )[:max_requests]
    width = 1440
    left = 210
    right = 260
    top = 58
    row_height = 34
    plot_width = width - left - right
    height = top + row_height * len(selected) + 80
    max_latency = max((request.latency_s for request in selected), default=1.0)

    def bar_width(duration_s: float) -> float:
        return max(duration_s / max(max_latency, 1e-9) * plot_width, 0.0)

    lines = _svg_header(width, height, "Longest queue-wait requests")
    lines.extend(
        [
            f'<text x="{left}" y="24" class="lane">Longest queue-wait requests</text>',
            (
                f'<text x="{left}" y="43" class="small">Top {len(selected)} by time before first start; '
                "execution is treated as one continuous block through completion</text>"
            ),
        ]
    )
    for tick in range(6):
        value = max_latency * tick / 5
        tick_x = left + plot_width * tick / 5
        lines.append(
            f'<line x1="{tick_x:.2f}" y1="{top - 5}" x2="{tick_x:.2f}" '
            f'y2="{top + row_height * len(selected)}" class="grid"/>'
        )
        lines.append(
            f'<text x="{tick_x:.2f}" y="{top + row_height * len(selected) + 20}" '
            f'text-anchor="middle" class="axis">{value:.0f}s</text>'
        )
    for row_index, request in enumerate(selected):
        y = top + row_index * row_height
        label = (
            f"{request.request_id.removeprefix('request-')} "
            f"{request.request_type}/{request.priority}"
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" class="small">'
            f"{_svg_escape(label)}</text>"
        )
        segments = [
            ("Queue before first start", request.queue_before_first_start_s, "#f28e2b"),
            (
                "Execution",
                max(request.completion_time_s - request.first_start_time_s, 0.0),
                "#4e79a7",
            ),
        ]
        cursor_x = float(left)
        for segment_name, duration_s, color in segments:
            width_px = bar_width(duration_s)
            if width_px <= 0.0:
                continue
            lines.append(
                f'<rect x="{cursor_x:.2f}" y="{y + 5}" width="{max(width_px, 1.0):.2f}" '
                f'height="22" fill="{color}"><title>{_svg_escape(request.request_id)} '
                f"{segment_name}: {duration_s:.2f}s</title></rect>"
            )
            cursor_x += width_px
        lines.append(
            f'<text x="{width - 10}" y="{y + 20}" text-anchor="end" '
            f'class="small">queue {request.queue_before_first_start_s:.1f}s · '
            f'latency {request.latency_s:.1f}s</text>'
        )
    legend_y = top + row_height * len(selected) + 51
    legend_items = [
        ("Queue before first start", "#f28e2b"),
        ("Execution", "#4e79a7"),
    ]
    cursor_x = left
    for label, color in legend_items:
        lines.append(f'<rect x="{cursor_x}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{cursor_x + 20}" y="{legend_y}" class="small">{_svg_escape(label)}</text>')
        cursor_x += len(label) * 7 + 42
    lines.append("</svg>")
    return "\n".join(lines)


def _format_seconds(value: float) -> str:
    return f"{value:,.2f}s"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header_html
        + "</tr></thead><tbody>"
        + "".join(row_html)
        + "</tbody></table></div>"
    )


def render_html_report(
    *,
    seed: int,
    summary: dict[str, Any],
    timeline_svg: str,
    waits_svg: str,
) -> str:
    backend_table = _table(
        ("Backend", "Busy", "Utilization", "Diagnostic bubbles", "Largest bubble", "Count"),
        [
            (
                row["backend"],
                _format_seconds(row["busy_s"]),
                f'{row["utilization_pct"]:.1f}%',
                _format_seconds(row["diagnostic_bubble_s"]),
                _format_seconds(row["largest_diagnostic_bubble_s"]),
                row["bubble_count"],
            )
            for row in summary["backend_rows"]
        ],
    )
    bubble_table = _table(
        ("Backend", "Start", "End", "Duration", "Classification"),
        [
            (
                row["backend"],
                _format_seconds(row["start_s"]),
                _format_seconds(row["end_s"]),
                _format_seconds(row["duration_s"]),
                row["kind"],
            )
            for row in summary["largest_bubbles"][:12]
        ],
    )
    wait_table = _table(
        ("Request", "Type", "Priority", "Backend", "Queue before start", "Execution", "Latency"),
        [
            (
                row["request_id"],
                row["request_type"],
                row["priority"],
                row["backend"],
                _format_seconds(row["queue_before_first_start_s"]),
                _format_seconds(row["execution_s"]),
                _format_seconds(row["latency_s"]),
            )
            for row in summary["longest_wait_requests"][:12]
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diffusion simulator trace — seed {seed}</title>
<style>
:root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 24px; background: Canvas; color: CanvasText; }}
h1, h2 {{ font-weight: 600; }}
h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 30px; }}
.subtle {{ color: color-mix(in srgb, CanvasText 65%, Canvas); }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; }}
.metric {{ border: 1px solid color-mix(in srgb, CanvasText 18%, Canvas); border-radius: 8px; padding: 14px; }}
.metric b {{ display: block; font-size: 1.35rem; margin-top: 4px; }}
.chart {{ overflow-x: auto; background: white; padding: 8px; border-radius: 8px; }}
.chart svg {{ display: block; width: 100%; min-width: 900px; height: auto; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, Canvas);
  padding: 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
code {{ background: color-mix(in srgb, CanvasText 8%, Canvas); padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Diffusion simulator trace</h1>
<div class="subtle">Seed {seed} · {summary["request_count"]} requests · {summary["backend_count"]} backends</div>
<div class="metrics">
  <div class="metric">Makespan<b>{_format_seconds(summary["makespan_s"])}</b></div>
  <div class="metric">Aggregate utilization<b>{summary["aggregate_utilization_pct"]:.1f}%</b></div>
  <div class="metric">Queue-wait P95<b>{_format_seconds(summary["queue_wait_p95_s"])}</b></div>
  <div class="metric">Latency P95<b>{_format_seconds(summary["latency_p95_s"])}</b></div>
</div>
<h2>Backend requests and bubbles</h2>
<div class="chart">{timeline_svg}</div>
<p class="subtle"><code>local-pending</code>: this backend has assigned unfinished work;
<code>global-waiting</code>: an unstarted request exists elsewhere;
<code>arrival-gap</code>: no request is ready here before later arrivals;
<code>drain-imbalance</code>: other backends are still draining work.</p>
{backend_table}
<h2>Largest diagnostic bubbles</h2>
{bubble_table}
<h2>Requests with the longest queue wait</h2>
<div class="chart">{waits_svg}</div>
{wait_table}
</body>
</html>
"""


def write_report(
    *,
    events_path: str | Path,
    requests_path: str | Path,
    output_path: str | Path,
    seed: int | None,
    bubble_threshold_s: float,
    max_requests: int,
) -> dict[str, Path]:
    event_seed, _ = load_events(events_path, seed)
    request_seed, requests = load_requests(requests_path, event_seed)
    if event_seed != request_seed:
        raise ValueError(f"event seed {event_seed} does not match request seed {request_seed}")
    intervals = request_intervals(requests)
    bubbles = bubble_intervals(
        intervals,
        requests,
        threshold_s=bubble_threshold_s,
    )
    summary = build_summary(intervals, requests, bubbles)
    summary["seed"] = event_seed
    timeline_svg = render_backend_timeline_svg(intervals, requests, bubbles)
    waits_svg = render_request_waits_svg(requests, max_requests=max_requests)
    report_html = render_html_report(
        seed=event_seed,
        summary=summary,
        timeline_svg=timeline_svg,
        waits_svg=waits_svg,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline_path = output.with_name(f"{output.stem}.backend-timeline.svg")
    waits_path = output.with_name(f"{output.stem}.request-waits.svg")
    summary_path = output.with_name(f"{output.stem}.summary.json")
    output.write_text(report_html, encoding="utf-8")
    timeline_path.write_text(timeline_svg + "\n", encoding="utf-8")
    waits_path.write_text(waits_svg + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "report": output,
        "backend_timeline": timeline_path,
        "request_waits": waits_path,
        "summary": summary_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render diffusion simulator trace diagnostics")
    parser.add_argument("--events", type=Path, required=True, help="JSONL written by simulator --trace-output")
    parser.add_argument("--requests", type=Path, required=True, help="CSV written by simulator --requests-output")
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML report")
    parser.add_argument("--seed", type=int, help="Seed to render when input files contain multiple runs")
    parser.add_argument(
        "--bubble-threshold-s",
        type=float,
        default=1.0,
        help="Ignore idle intervals shorter than this many seconds (default: 1)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=20,
        help="Maximum requests in the longest-wait chart (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    paths = write_report(
        events_path=args.events,
        requests_path=args.requests,
        output_path=args.output,
        seed=args.seed,
        bubble_threshold_s=args.bubble_threshold_s,
        max_requests=args.max_requests,
    )
    for name, path in paths.items():
        print(f"{name:>18}: {path}")


if __name__ == "__main__":
    main()
