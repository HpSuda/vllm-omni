# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare Wan2.2 scheduler behavior from a common simulator request stream.

The script consumes the per-policy ``requests.csv`` and ``events.jsonl`` files
written by ``run_wan22_five_policy_behavior.sh``.  It writes a compact JSON
summary plus SVG figures that keep every policy on the same time and latency
scales.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICIES = (
    ("original", "原策略 8×USP1"),
    ("beta05_spread", "Central Pull + β=0.5 + Tail Spread"),
    ("beta095_pack", "Central Pull + β=0.95 + Tail Pack"),
    ("beta085_pack_gate", "Central Pull + β=0.85 + Tail Pack + Tail Gate"),
    ("beam_pack_gate", "Release-Calendar Beam + Tail Pack + Tail Gate"),
)

TYPE_COLORS = {
    "short": "#4E79A7",
    "medium": "#59A14F",
    "long": "#F28E2B",
}
TAIL_COLOR = "#B23A48"
POLICY_COLORS = ("#4E79A7", "#76B7B2", "#F28E2B", "#E15759", "#7B61A8")

NUMERIC_FIELDS = (
    "arrival_time_s",
    "first_start_time_s",
    "completion_time_s",
    "latency_s",
    "queue_before_first_start_s",
    "central_wait_s",
    "service_s",
    "estimated_service_s",
)
INTEGER_FIELDS = ("arrival_seq", "preemptions", "resumes", "dispatch_count")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _load_requests(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as request_file:
        rows = list(csv.DictReader(request_file))
    if not rows:
        raise ValueError(f"no requests in {path}")
    for row in rows:
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field])
        for field in INTEGER_FIELDS:
            row[field] = int(row[field])
    return rows


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as event_file:
        for line in event_file:
            if line.strip():
                events.append(json.loads(line))
    return events


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _phase_segments(
    events: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float, str]]]:
    """Merge adjacent internal phases into request-level active segments."""
    raw: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "phase_start":
            continue
        duration_s = float(event.get("duration_s", 0.0))
        if duration_s <= 0.0:
            continue
        start_s = float(event["time_s"])
        raw[(str(event["backend"]), str(event["request_id"]))].append(
            (start_s, start_s + duration_s)
        )

    by_backend: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for (backend, request_id), intervals in raw.items():
        merged: list[list[float]] = []
        for start_s, end_s in sorted(intervals):
            if merged and start_s <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], end_s)
            else:
                merged.append([start_s, end_s])
        by_backend[backend].extend(
            (start_s, end_s, request_id) for start_s, end_s in merged
        )
    for backend in by_backend:
        by_backend[backend].sort()
    return by_backend


def _analyze_policy(
    key: str,
    label: str,
    input_root: Path,
) -> dict[str, Any]:
    policy_root = input_root / key
    requests = _load_requests(policy_root / "requests.csv")
    events = _load_events(policy_root / "events.jsonl")
    request_by_id = {row["request_id"]: row for row in requests}
    ordered = sorted(requests, key=lambda row: (row["latency_s"], row["request_id"]))
    lower = ordered[94]
    upper = ordered[95]
    p95_s = lower["latency_s"] * 0.95 + upper["latency_s"] * 0.05
    tails = [row for row in requests if row["priority"] != "normal"]
    normals = [row for row in requests if row["priority"] == "normal"]
    start_s = min(row["arrival_time_s"] for row in requests)
    end_s = max(row["completion_time_s"] for row in requests)
    makespan_s = end_s - start_s
    backends = sorted({str(row["backend"]) for row in requests})
    backend_finish_s = {
        backend: max(
            row["completion_time_s"]
            for row in requests
            if row["backend"] == backend
        )
        for backend in backends
    }
    normal_backend_finish_s = {
        backend: max(
            row["completion_time_s"]
            for row in normals
            if row["backend"] == backend
        )
        for backend in backends
        if any(row["backend"] == backend for row in normals)
    }
    active_s = sum(
        float(event.get("duration_s", 0.0))
        for event in events
        if event.get("event") in {"phase_start", "switch_start"}
    )
    planner_pulls = [
        event for event in events if event.get("event") == "protected_pull"
    ]
    beam_pulls = [
        event for event in planner_pulls if event.get("planner_used_beam") is True
    ]
    fallback_reasons = Counter(
        str(event.get("planner_fallback_reason"))
        for event in planner_pulls
        if event.get("planner_fallback_reason")
    )
    predicted_improvements = [
        float(event["planner_predicted_before_p95_s"])
        - float(event["planner_predicted_after_p95_s"])
        for event in beam_pulls
        if event.get("planner_predicted_before_p95_s") is not None
        and event.get("planner_predicted_after_p95_s") is not None
    ]

    return {
        "key": key,
        "label": label,
        "requests": requests,
        "request_by_id": request_by_id,
        "events": events,
        "segments": _phase_segments(events),
        "request_count": len(requests),
        "normal_count": len(normals),
        "tail_count": len(tails),
        "p50_s": statistics.median(row["latency_s"] for row in requests),
        "p95_s": p95_s,
        "p99_s": _percentile([row["latency_s"] for row in requests], 0.99),
        "mean_s": statistics.fmean(row["latency_s"] for row in requests),
        "queue_wait_p95_s": _percentile(
            [row["queue_before_first_start_s"] for row in requests], 0.95
        ),
        "makespan_s": makespan_s,
        "throughput_rps": len(requests) / makespan_s,
        "active_utilization_pct": active_s / (makespan_s * len(backends)) * 100.0,
        "preemptions": sum(row["preemptions"] for row in requests),
        "preempted_request_count": sum(row["preemptions"] > 0 for row in requests),
        "normal_max_latency_s": max(row["latency_s"] for row in normals),
        "tail_min_latency_s": min(row["latency_s"] for row in tails),
        "tail_max_latency_s": max(row["latency_s"] for row in tails),
        "tail_first_start_s": min(row["first_start_time_s"] for row in tails),
        "tail_last_completion_s": max(row["completion_time_s"] for row in tails),
        "backend_finish_s": backend_finish_s,
        "backend_finish_spread_s": max(backend_finish_s.values())
        - min(backend_finish_s.values()),
        "normal_backend_finish_s": normal_backend_finish_s,
        "normal_backend_finish_spread_s": max(normal_backend_finish_s.values())
        - min(normal_backend_finish_s.values()),
        "p95_lower": _boundary_snapshot(lower, rank=95, weight=0.95),
        "p95_upper": _boundary_snapshot(upper, rank=96, weight=0.05),
        "planner": {
            "pull_count": len(planner_pulls),
            "beam_pull_count": len(beam_pulls),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "elapsed_ms_total": sum(
                float(event.get("planner_elapsed_ms") or 0.0)
                for event in planner_pulls
            ),
            "predicted_p95_improvement_s_total": sum(predicted_improvements),
            "predicted_p95_improvement_positive_count": sum(
                value > 1e-9 for value in predicted_improvements
            ),
        },
    }


def _boundary_snapshot(
    row: dict[str, Any],
    *,
    rank: int,
    weight: float,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "weight": weight,
        "request_id": row["request_id"],
        "request_type": row["request_type"],
        "priority": row["priority"],
        "backend": row["backend"],
        "arrival_s": row["arrival_time_s"],
        "first_start_s": row["first_start_time_s"],
        "completion_s": row["completion_time_s"],
        "queue_wait_s": row["queue_before_first_start_s"],
        "service_s": row["service_s"],
        "latency_s": row["latency_s"],
    }


def _svg_start(width: int, height: int, label: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{_escape(label)}">'
        ),
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',sans-serif;fill:#172033}",
        ".title{font-size:20px;font-weight:700}.subtitle{font-size:13px;fill:#4d586b}",
        ".policy{font-size:14px;font-weight:700}.lane{font-size:10px;fill:#657086}",
        ".axis{font-size:10px;fill:#657086}.small{font-size:10px}.note{font-size:11px;fill:#4d586b}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def _x(value: float, left: float, plot_width: float, max_value: float) -> float:
    return left + value / max_value * plot_width


def _render_timelines(policies: list[dict[str, Any]], output: Path) -> None:
    width = 1600
    left, right = 190, 35
    plot_width = width - left - right
    panel_height = 143
    top = 78
    height = top + len(policies) * panel_height + 74
    max_time = math.ceil(
        max(policy["makespan_s"] for policy in policies) / 500.0
    ) * 500.0
    lines = _svg_start(width, height, "五种 Wan2.2 调度策略的 backend 行为时间线")
    lines += [
        '<text x="32" y="32" class="title">同一请求流下的 backend 执行时间线</text>',
        (
            '<text x="32" y="55" class="subtitle">'
            "每个色块是一段真实运行；同一 Tail 被抢占时会分裂为多段。所有面板共用横轴。"
            "</text>"
        ),
    ]

    for policy_index, policy in enumerate(policies):
        y0 = top + policy_index * panel_height
        lines.append(
            f'<text x="20" y="{y0 + 13}" class="policy">{_escape(policy["label"])}</text>'
        )
        lines.append(
            (
                f'<text x="20" y="{y0 + 30}" class="note">'
                f'P95 {policy["p95_s"]:.1f}s · makespan {policy["makespan_s"]:.1f}s · '
                f'抢占 {policy["preemptions"]} 次</text>'
            )
        )
        request_by_id = policy["request_by_id"]
        for backend_index in range(8):
            backend = f"backend-{backend_index}"
            lane_y = y0 + 39 + backend_index * 12
            lines.append(
                f'<text x="{left - 9}" y="{lane_y + 9}" text-anchor="end" class="lane">B{backend_index}</text>'
            )
            lines.append(
                f'<rect x="{left}" y="{lane_y}" width="{plot_width}" height="9" rx="2" fill="#f1f3f7"/>'
            )
            for start_s, end_s, request_id in policy["segments"].get(backend, []):
                request = request_by_id[request_id]
                is_tail = request["priority"] != "normal"
                color = (
                    TAIL_COLOR
                    if is_tail
                    else TYPE_COLORS.get(request["request_type"], "#777777")
                )
                segment_x = _x(start_s, left, plot_width, max_time)
                segment_width = max(
                    _x(end_s, left, plot_width, max_time) - segment_x,
                    0.7,
                )
                boundary = request_id in {
                    policy["p95_lower"]["request_id"],
                    policy["p95_upper"]["request_id"],
                }
                stroke = "#111827" if boundary else color
                stroke_width = 1.2 if boundary else 0.2
                lines.append(
                    (
                        f'<rect x="{segment_x:.2f}" y="{lane_y + 1}" '
                        f'width="{segment_width:.2f}" height="7" rx="1.5" '
                        f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}">'
                        f"<title>{_escape(request_id)} · {_escape(request['request_type'])} · "
                        f"{'Tail' if is_tail else 'Normal'} · {start_s:.1f}–{end_s:.1f}s"
                        "</title></rect>"
                    )
                )
        last_arrival = max(row["arrival_time_s"] for row in policy["requests"])
        arrival_x = _x(last_arrival, left, plot_width, max_time)
        lines.append(
            (
                f'<line x1="{arrival_x:.2f}" y1="{y0 + 38}" x2="{arrival_x:.2f}" '
                f'y2="{y0 + 136}" stroke="#8a94a6" stroke-width="0.8" '
                'stroke-dasharray="3 3"/>'
            )
        )

    axis_y = top + len(policies) * panel_height + 4
    for tick in range(0, int(max_time) + 1, 1000):
        tick_x = _x(tick, left, plot_width, max_time)
        lines.append(
            f'<line x1="{tick_x:.2f}" y1="{top - 8}" x2="{tick_x:.2f}" y2="{axis_y}" stroke="#d8dde7" stroke-width="0.7"/>'
        )
        lines.append(
            f'<text x="{tick_x:.2f}" y="{axis_y + 17}" text-anchor="middle" class="axis">{tick}s</text>'
        )

    legend_y = height - 25
    legend = (
        ("Normal short", TYPE_COLORS["short"]),
        ("Normal medium", TYPE_COLORS["medium"]),
        ("Normal long", TYPE_COLORS["long"]),
        ("Tail", TAIL_COLOR),
        ("黑框：P95 rank 95/96 请求", "#111827"),
    )
    cursor = left
    for label, color in legend:
        lines.append(
            f'<rect x="{cursor}" y="{legend_y - 10}" width="14" height="9" rx="2" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{cursor + 20}" y="{legend_y - 2}" class="small">{_escape(label)}</text>'
        )
        cursor += 175 if "Normal" in label else 190
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_latency_profiles(
    policies: list[dict[str, Any]],
    output: Path,
) -> None:
    width, height = 1600, 680
    left, right, top, bottom = 90, 40, 86, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_latency = math.ceil(
        max(
            row["latency_s"]
            for policy in policies
            for row in policy["requests"]
        )
        / 1000.0
    ) * 1000.0
    lines = _svg_start(width, height, "五种 Wan2.2 调度策略的 E2E latency 排名曲线")
    lines += [
        '<text x="32" y="32" class="title">100 个请求的 E2E latency 排名曲线</text>',
        (
            '<text x="32" y="55" class="subtitle">'
            "横轴由快到慢排序；P95 由 rank 95 与 rank 96 按 95%/5% 线性插值得到。"
            "</text>"
        ),
    ]
    for tick in range(0, int(max_latency) + 1, 1000):
        tick_y = top + plot_height - tick / max_latency * plot_height
        lines.append(
            f'<line x1="{left}" y1="{tick_y:.2f}" x2="{width - right}" y2="{tick_y:.2f}" stroke="#d8dde7" stroke-width="0.8"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{tick_y + 4:.2f}" text-anchor="end" class="axis">{tick}s</text>'
        )
    for rank in (1, 20, 40, 60, 80, 95, 96, 100):
        rank_x = left + (rank - 1) / 99.0 * plot_width
        emphasized = rank in (95, 96)
        lines.append(
            (
                f'<line x1="{rank_x:.2f}" y1="{top}" x2="{rank_x:.2f}" '
                f'y2="{top + plot_height}" stroke="{"#9aa3b3" if emphasized else "#e6e9ef"}" '
                f'stroke-width="{"1.4" if emphasized else "0.7"}" '
                f'stroke-dasharray="{"4 3" if emphasized else "none"}"/>'
            )
        )
        lines.append(
            f'<text x="{rank_x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" class="axis">{rank}</text>'
        )

    for index, policy in enumerate(policies):
        ordered = sorted(row["latency_s"] for row in policy["requests"])
        points = []
        for offset, latency_s in enumerate(ordered):
            point_x = left + offset / 99.0 * plot_width
            point_y = top + plot_height - latency_s / max_latency * plot_height
            points.append(f"{point_x:.2f},{point_y:.2f}")
        color = POLICY_COLORS[index]
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        for rank, boundary in (
            (95, policy["p95_lower"]),
            (96, policy["p95_upper"]),
        ):
            point_x = left + (rank - 1) / 99.0 * plot_width
            point_y = (
                top
                + plot_height
                - boundary["latency_s"] / max_latency * plot_height
            )
            lines.append(
                (
                    f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="4.2" '
                    f'fill="{color}" stroke="#ffffff" stroke-width="1.2">'
                    f"<title>{_escape(policy['label'])} · rank {rank} · "
                    f"{_escape(boundary['request_id'])} · {boundary['latency_s']:.1f}s · "
                    f"{_escape(boundary['priority'])}</title></circle>"
                )
            )

    legend_y = height - 29
    cursor = 32
    for index, policy in enumerate(policies):
        color = POLICY_COLORS[index]
        lines.append(
            f'<line x1="{cursor}" y1="{legend_y - 4}" x2="{cursor + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{cursor + 31}" y="{legend_y}" class="small">{_escape(policy["label"])}</text>'
        )
        cursor += 300
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_p95_boundary(
    policies: list[dict[str, Any]],
    output: Path,
) -> None:
    width, height = 1600, 470
    left, right, top = 430, 60, 90
    plot_width = width - left - right
    row_height = 66
    max_latency = math.ceil(
        max(
            boundary["latency_s"]
            for policy in policies
            for boundary in (policy["p95_lower"], policy["p95_upper"])
        )
        / 500.0
    ) * 500.0
    lines = _svg_start(width, height, "五种 Wan2.2 调度策略的 P95 边界请求")
    lines += [
        '<text x="32" y="32" class="title">P95 边界：rank 95 / rank 96 是谁</text>',
        (
            '<text x="32" y="55" class="subtitle">'
            "圆点是两个边界请求，菱形是最终 P95；位置越靠左越好。"
            "</text>"
        ),
    ]
    for tick in range(0, int(max_latency) + 1, 500):
        tick_x = left + tick / max_latency * plot_width
        lines.append(
            f'<line x1="{tick_x:.2f}" y1="{top - 20}" x2="{tick_x:.2f}" y2="{top + row_height * 5 - 10}" stroke="#e0e4eb" stroke-width="0.8"/>'
        )
        lines.append(
            f'<text x="{tick_x:.2f}" y="{top - 29}" text-anchor="middle" class="axis">{tick}s</text>'
        )

    for index, policy in enumerate(policies):
        y = top + index * row_height
        color = POLICY_COLORS[index]
        lower = policy["p95_lower"]
        upper = policy["p95_upper"]
        lower_x = left + lower["latency_s"] / max_latency * plot_width
        upper_x = left + upper["latency_s"] / max_latency * plot_width
        p95_x = left + policy["p95_s"] / max_latency * plot_width
        lines.append(
            f'<text x="24" y="{y + 5}" class="policy">{_escape(policy["label"])}</text>'
        )
        lines.append(
            (
                f'<text x="24" y="{y + 23}" class="note">'
                f'r95 {lower["request_id"][-3:]} {lower["request_type"]}/{lower["priority"]} · '
                f'r96 {upper["request_id"][-3:]} {upper["request_type"]}/{upper["priority"]}'
                "</text>"
            )
        )
        lines.append(
            f'<line x1="{lower_x:.2f}" y1="{y}" x2="{upper_x:.2f}" y2="{y}" stroke="{color}" stroke-width="3" opacity="0.65"/>'
        )
        lines.append(
            f'<circle cx="{lower_x:.2f}" cy="{y}" r="7" fill="{color}"><title>rank 95: {lower["latency_s"]:.1f}s</title></circle>'
        )
        lines.append(
            f'<circle cx="{upper_x:.2f}" cy="{y}" r="7" fill="#ffffff" stroke="{color}" stroke-width="3"><title>rank 96: {upper["latency_s"]:.1f}s</title></circle>'
        )
        diamond = (
            f"{p95_x:.2f},{y - 10} {p95_x + 10:.2f},{y:.2f} "
            f"{p95_x:.2f},{y + 10} {p95_x - 10:.2f},{y:.2f}"
        )
        lines.append(
            f'<polygon points="{diamond}" fill="#111827"><title>P95: {policy["p95_s"]:.1f}s</title></polygon>'
        )
        lines.append(
            f'<text x="{p95_x + 14:.2f}" y="{y - 13}" class="small">P95 {policy["p95_s"]:.1f}s</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_tail_lifecycle(
    policies: list[dict[str, Any]],
    output: Path,
) -> None:
    width = 1600
    left, right, top = 205, 35, 82
    plot_width = width - left - right
    panel_height = 132
    height = top + panel_height * len(policies) + 65
    max_time = math.ceil(
        max(policy["makespan_s"] for policy in policies) / 500.0
    ) * 500.0
    lines = _svg_start(width, height, "五种 Wan2.2 调度策略的 Tail 生命周期")
    lines += [
        '<text x="32" y="32" class="title">五个 Tail 的等待、运行与抢占</text>',
        (
            '<text x="32" y="55" class="subtitle">'
            "浅色：到达后等待；深红：真实运行；运行段之间的空白表示被抢占或暂停。"
            "</text>"
        ),
    ]
    for policy_index, policy in enumerate(policies):
        y0 = top + policy_index * panel_height
        lines.append(
            f'<text x="20" y="{y0 + 12}" class="policy">{_escape(policy["label"])}</text>'
        )
        tail_rows = sorted(
            (
                row
                for row in policy["requests"]
                if row["priority"] != "normal"
            ),
            key=lambda row: row["arrival_seq"],
        )
        segment_lookup: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for segments in policy["segments"].values():
            for start_s, end_s, request_id in segments:
                segment_lookup[request_id].append((start_s, end_s))
        for row_index, request in enumerate(tail_rows):
            y = y0 + 25 + row_index * 18
            lines.append(
                f'<text x="{left - 10}" y="{y + 9}" text-anchor="end" class="lane">{_escape(request["request_id"][-3:])} · B{request["backend"].split("-")[-1]}</text>'
            )
            arrival_x = _x(
                request["arrival_time_s"], left, plot_width, max_time
            )
            completion_x = _x(
                request["completion_time_s"], left, plot_width, max_time
            )
            first_start_x = _x(
                request["first_start_time_s"], left, plot_width, max_time
            )
            lines.append(
                (
                    f'<rect x="{arrival_x:.2f}" y="{y + 2}" '
                    f'width="{max(first_start_x - arrival_x, 0.8):.2f}" height="7" '
                    'rx="2" fill="#F3C6CB">'
                    f"<title>等待 {request['queue_before_first_start_s']:.1f}s</title></rect>"
                )
            )
            lines.append(
                f'<line x1="{first_start_x:.2f}" y1="{y + 5.5}" x2="{completion_x:.2f}" y2="{y + 5.5}" stroke="#D9919A" stroke-width="2"/>'
            )
            for start_s, end_s in sorted(segment_lookup[request["request_id"]]):
                segment_x = _x(start_s, left, plot_width, max_time)
                segment_width = max(
                    _x(end_s, left, plot_width, max_time) - segment_x,
                    0.8,
                )
                lines.append(
                    (
                        f'<rect x="{segment_x:.2f}" y="{y}" '
                        f'width="{segment_width:.2f}" height="11" rx="2" '
                        f'fill="{TAIL_COLOR}"><title>运行 {start_s:.1f}–{end_s:.1f}s'
                        f"</title></rect>"
                    )
                )
            lines.append(
                f'<circle cx="{completion_x:.2f}" cy="{y + 5.5}" r="2.8" fill="#5A1620"/>'
            )
        lines.append(
            (
                f'<text x="{width - right}" y="{y0 + 15}" text-anchor="end" class="note">'
                f'Tail E2E {policy["tail_min_latency_s"]:.0f}–{policy["tail_max_latency_s"]:.0f}s · '
                f'抢占 {policy["preemptions"]} 次</text>'
            )
        )
    axis_y = top + panel_height * len(policies) + 4
    for tick in range(0, int(max_time) + 1, 1000):
        tick_x = _x(tick, left, plot_width, max_time)
        lines.append(
            f'<line x1="{tick_x:.2f}" y1="{top - 10}" x2="{tick_x:.2f}" y2="{axis_y}" stroke="#e1e5ec" stroke-width="0.7"/>'
        )
        lines.append(
            f'<text x="{tick_x:.2f}" y="{axis_y + 18}" text-anchor="middle" class="axis">{tick}s</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _serializable(policy: dict[str, Any]) -> dict[str, Any]:
    excluded = {"requests", "request_by_id", "events", "segments"}
    return {key: value for key, value in policy.items() if key not in excluded}


def _render_inline_fragment(
    policies: list[dict[str, Any]],
    output: Path,
) -> None:
    inline_policies = []
    for policy in policies:
        request_metadata = {
            request_id: {
                "type": row["request_type"],
                "priority": row["priority"],
            }
            for request_id, row in policy["request_by_id"].items()
        }
        segments = [
            [
                int(backend.split("-")[-1]),
                round(start_s, 3),
                round(end_s, 3),
                request_id,
            ]
            for backend, backend_segments in policy["segments"].items()
            for start_s, end_s, request_id in backend_segments
        ]
        tails = []
        for row in sorted(
            (
                request
                for request in policy["requests"]
                if request["priority"] != "normal"
            ),
            key=lambda request: request["arrival_seq"],
        ):
            tails.append(
                {
                    "id": row["request_id"],
                    "backend": int(row["backend"].split("-")[-1]),
                    "arrival": round(row["arrival_time_s"], 3),
                    "start": round(row["first_start_time_s"], 3),
                    "end": round(row["completion_time_s"], 3),
                    "preemptions": row["preemptions"],
                    "segments": [
                        [round(start_s, 3), round(end_s, 3)]
                        for backend_segments in policy["segments"].values()
                        for start_s, end_s, request_id in backend_segments
                        if request_id == row["request_id"]
                    ],
                }
            )
        inline_policies.append(
            {
                "key": policy["key"],
                "label": policy["label"],
                "p95": round(policy["p95_s"], 3),
                "makespan": round(policy["makespan_s"], 3),
                "preemptions": policy["preemptions"],
                "requests": request_metadata,
                "segments": segments,
                "tails": tails,
                "lower": policy["p95_lower"],
                "upper": policy["p95_upper"],
            }
        )
    encoded = json.dumps(
        {"policies": inline_policies},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fragment = r"""
<div id="wan22-five-policy-behavior">
  <div class="vis-caption">Backend execution · shared 0–8500s axis</div>
  <svg class="vis-chart" data-chart="timeline" viewBox="0 0 1200 770" role="img" aria-label="Five policies, each with eight backend execution lanes"></svg>
  <div class="vis-caption">P95 boundary · filled circle rank 95, open circle rank 96, diamond P95</div>
  <svg class="vis-chart vis-boundary" data-chart="boundary" viewBox="0 0 1200 285" role="img" aria-label="Rank 95, rank 96 and interpolated P95 latency for five policies"></svg>
  <div class="vis-caption">Tail lifecycle · pale wait, strong active segment, thin suspended interval</div>
  <svg class="vis-chart" data-chart="tails" viewBox="0 0 1200 700" role="img" aria-label="Arrival, wait, active service and completion of the five Tail requests under each policy"></svg>
  <div class="vis-alt" aria-label="Text summary of the plotted values"></div>
</div>
<style>
#wan22-five-policy-behavior {
  color: var(--foreground);
  background: transparent;
  width: 100%;
}
#wan22-five-policy-behavior .vis-caption {
  color: var(--muted-foreground);
  font-size: var(--font-size-base);
  font-weight: 500;
  margin: 0.75rem 0 0.25rem;
}
#wan22-five-policy-behavior .vis-chart {
  display: block;
  width: 100%;
  height: auto;
  color: var(--foreground);
}
#wan22-five-policy-behavior .vis-chart text {
  fill: var(--foreground);
  font-size: var(--font-size-base);
  font-weight: 400;
}
#wan22-five-policy-behavior .vis-chart .muted {
  fill: var(--muted-foreground);
}
#wan22-five-policy-behavior .vis-chart .policy {
  font-weight: 500;
}
#wan22-five-policy-behavior .vis-alt {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}
</style>
<script>
(() => {
  const root = document.getElementById("wan22-five-policy-behavior");
  const data = __INLINE_DATA__;
  const ns = "http://www.w3.org/2000/svg";
  const colors = {
    short: "var(--viz-series-1)",
    medium: "var(--viz-series-2)",
    long: "var(--viz-series-3)",
    tail: "var(--viz-series-4)",
    boundary: "var(--foreground)",
    grid: "var(--border)",
    lane: "var(--muted)"
  };
  const policyColors = [
    "var(--viz-series-1)",
    "var(--viz-series-2)",
    "var(--viz-series-3)",
    "var(--viz-series-4)",
    "var(--viz-series-5)"
  ];

  function node(tag, attrs = {}, text = "") {
    const item = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, value);
    if (text) item.textContent = text;
    return item;
  }
  function text(svg, x, y, value, cls = "", anchor = "start") {
    svg.appendChild(node("text", {x, y, class: cls, "text-anchor": anchor}, value));
  }
  function line(svg, x1, y1, x2, y2, attrs = {}) {
    svg.appendChild(node("line", {x1, y1, x2, y2, ...attrs}));
  }
  function scale(value, left, width, maximum) {
    return left + value / maximum * width;
  }

  const timeline = root.querySelector('[data-chart="timeline"]');
  const tl = {left: 250, right: 18, top: 28, panel: 143, lane: 12, max: 8500};
  tl.width = 1200 - tl.left - tl.right;
  for (let tick = 0; tick <= 8000; tick += 2000) {
    const x = scale(tick, tl.left, tl.width, tl.max);
    line(timeline, x, 16, x, 744, {stroke: colors.grid, "stroke-width": 1});
    text(timeline, x, 763, `${tick}s`, "muted", "middle");
  }
  data.policies.forEach((policy, policyIndex) => {
    const y0 = tl.top + policyIndex * tl.panel;
    text(timeline, 4, y0, policy.label, "policy");
    text(
      timeline,
      4,
      y0 + 19,
      `P95 ${policy.p95.toFixed(1)}s · makespan ${policy.makespan.toFixed(1)}s · preempt ${policy.preemptions}`,
      "muted"
    );
    for (let backend = 0; backend < 8; backend += 1) {
      const y = y0 + 34 + backend * tl.lane;
      text(timeline, tl.left - 8, y + 8, `B${backend}`, "muted", "end");
      timeline.appendChild(node("rect", {
        x: tl.left, y, width: tl.width, height: 8, rx: 2, fill: colors.lane
      }));
    }
    const boundaryIds = new Set([policy.lower.request_id, policy.upper.request_id]);
    policy.segments.forEach(([backend, start, end, requestId]) => {
      const meta = policy.requests[requestId];
      const y = y0 + 34 + backend * tl.lane;
      const x = scale(start, tl.left, tl.width, tl.max);
      const width = Math.max(scale(end, tl.left, tl.width, tl.max) - x, 0.8);
      timeline.appendChild(node("rect", {
        x, y: y + 1, width, height: 6, rx: 1,
        fill: meta.priority === "normal" ? colors[meta.type] : colors.tail,
        stroke: boundaryIds.has(requestId) ? colors.boundary : "none",
        "stroke-width": boundaryIds.has(requestId) ? 1.2 : 0
      }));
    });
  });

  const boundary = root.querySelector('[data-chart="boundary"]');
  const bd = {left: 390, right: 25, top: 35, row: 48, max: 3000};
  bd.width = 1200 - bd.left - bd.right;
  for (let tick = 0; tick <= bd.max; tick += 500) {
    const x = scale(tick, bd.left, bd.width, bd.max);
    line(boundary, x, 12, x, 260, {stroke: colors.grid, "stroke-width": 1});
    text(boundary, x, 278, `${tick}s`, "muted", "middle");
  }
  data.policies.forEach((policy, index) => {
    const y = bd.top + index * bd.row;
    const color = policyColors[index];
    const lx = scale(policy.lower.latency_s, bd.left, bd.width, bd.max);
    const ux = scale(policy.upper.latency_s, bd.left, bd.width, bd.max);
    const px = scale(policy.p95, bd.left, bd.width, bd.max);
    text(boundary, 4, y, policy.label, "policy");
    text(
      boundary,
      4,
      y + 17,
      `r95 ${policy.lower.request_id.slice(-3)} ${policy.lower.priority} · r96 ${policy.upper.request_id.slice(-3)} ${policy.upper.priority}`,
      "muted"
    );
    line(boundary, lx, y, ux, y, {stroke: color, "stroke-width": 3});
    boundary.appendChild(node("circle", {cx: lx, cy: y, r: 6, fill: color}));
    boundary.appendChild(node("circle", {
      cx: ux, cy: y, r: 6, fill: "var(--background)", stroke: color, "stroke-width": 3
    }));
    boundary.appendChild(node("polygon", {
      points: `${px},${y - 8} ${px + 8},${y} ${px},${y + 8} ${px - 8},${y}`,
      fill: colors.boundary
    }));
    text(boundary, px + 11, y - 9, `${policy.p95.toFixed(1)}s`, "muted");
  });

  const tails = root.querySelector('[data-chart="tails"]');
  const ta = {left: 245, right: 18, top: 26, panel: 133, row: 18, max: 8500};
  ta.width = 1200 - ta.left - ta.right;
  for (let tick = 0; tick <= 8000; tick += 2000) {
    const x = scale(tick, ta.left, ta.width, ta.max);
    line(tails, x, 12, x, 675, {stroke: colors.grid, "stroke-width": 1});
    text(tails, x, 696, `${tick}s`, "muted", "middle");
  }
  data.policies.forEach((policy, policyIndex) => {
    const y0 = ta.top + policyIndex * ta.panel;
    text(tails, 4, y0, policy.label, "policy");
    policy.tails.forEach((tail, tailIndex) => {
      const y = y0 + 24 + tailIndex * ta.row;
      const arrivalX = scale(tail.arrival, ta.left, ta.width, ta.max);
      const startX = scale(tail.start, ta.left, ta.width, ta.max);
      const endX = scale(tail.end, ta.left, ta.width, ta.max);
      text(tails, ta.left - 8, y + 8, `${tail.id.slice(-3)} · B${tail.backend}`, "muted", "end");
      tails.appendChild(node("rect", {
        x: arrivalX, y: y + 2, width: Math.max(startX - arrivalX, 0.8),
        height: 7, rx: 2, fill: colors.tail, opacity: 0.22
      }));
      line(tails, startX, y + 5.5, endX, y + 5.5, {
        stroke: colors.tail, "stroke-width": 2, opacity: 0.5
      });
      tail.segments.forEach(([start, end]) => {
        const x = scale(start, ta.left, ta.width, ta.max);
        tails.appendChild(node("rect", {
          x, y, width: Math.max(scale(end, ta.left, ta.width, ta.max) - x, 0.8),
          height: 11, rx: 2, fill: colors.tail
        }));
      });
      tails.appendChild(node("circle", {cx: endX, cy: y + 5.5, r: 3, fill: colors.boundary}));
    });
  });

  root.querySelector(".vis-alt").textContent = data.policies.map(policy =>
    `${policy.label}: P95 ${policy.p95.toFixed(1)} seconds, makespan ${policy.makespan.toFixed(1)} seconds, ${policy.preemptions} preemptions.`
  ).join(" ");
})();
</script>
""".strip().replace("__INLINE_DATA__", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment + "\n", encoding="utf-8")


def compare(
    input_root: Path,
    output_root: Path,
    *,
    inline_output: Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    policies = [
        _analyze_policy(key, label, input_root) for key, label in POLICIES
    ]
    original_p95 = policies[0]["p95_s"]
    for policy in policies:
        policy["p95_reduction_vs_original_pct"] = (
            (original_p95 - policy["p95_s"]) / original_p95 * 100.0
        )
        policy["p95_speedup_vs_original"] = original_p95 / policy["p95_s"]

    _render_timelines(policies, output_root / "backend-timelines.svg")
    _render_latency_profiles(policies, output_root / "latency-rank-profiles.svg")
    _render_p95_boundary(policies, output_root / "p95-boundary.svg")
    _render_tail_lifecycle(policies, output_root / "tail-lifecycle.svg")
    if inline_output is not None:
        _render_inline_fragment(policies, inline_output)
    payload = {
        "schema_version": 1,
        "workload": {
            "model": "Wan2.2",
            "topology": "8xUSP1",
            "request_count": 100,
            "request_rate_rps": 0.03,
            "seed": 42,
        },
        "policies": [_serializable(policy) for policy in policies],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "results/simulator/wan22_8xusp1_five_policy_behavior_100"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "benchmarks/diffusion/figures/"
            "wan22_five_policy_behavior_20260728"
        ),
    )
    parser.add_argument(
        "--inline-output",
        type=Path,
        help="Optional Codex inline-visualization HTML fragment.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = compare(
        args.input_root,
        args.output_root,
        inline_output=args.inline_output,
    )
    for policy in payload["policies"]:
        print(
            f"{policy['key']:>22}: "
            f"P95={policy['p95_s']:.3f}s, "
            f"vs original={policy['p95_reduction_vs_original_pct']:+.3f}%, "
            f"makespan={policy['makespan_s']:.3f}s"
        )


if __name__ == "__main__":
    main()
