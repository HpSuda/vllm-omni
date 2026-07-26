# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

import vllm_omni.trace_logging as trace_logging
from benchmarks.diffusion.wan22_request_trace import (
    build_request_rows,
    load_trace_events,
    summarize_request_rows,
    write_trace_report,
)
from vllm_omni.trace_logging import write_trace_event

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_trace_report_joins_client_dispatcher_and_backend_events(tmp_path) -> None:
    client_events = [
        {
            "ts": 100.0,
            "ts_ns": 100,
            "event": "client_arrive",
            "request_id": "request-00000",
            "width": 854,
            "height": 480,
            "num_frames": 80,
            "num_inference_steps": 3,
            "fps": 16,
        },
        {
            "ts": 180.0,
            "ts_ns": 600,
            "event": "client_finish",
            "request_id": "request-00000",
            "video_id": "video-1",
            "success": True,
            "latency_s": 80.0,
        },
    ]
    dispatcher_events = [
        {
            "ts": 101.0,
            "ts_ns": 200,
            "event": "dispatcher_arrive",
            "request_id": "request-00000",
        },
        {
            "ts": 105.0,
            "ts_ns": 300,
            "event": "central_enqueue",
            "request_id": "request-00000",
            "queue_depth": 4,
        },
        {
            "ts": 110.0,
            "ts_ns": 400,
            "event": "dispatch",
            "request_id": "request-00000",
            "arrival_counter": 7,
            "backend": "backend-2",
            "queue_class": "normal",
            "estimated_service_s": 68.0,
            "normal_routing_policy": "central_pull_cost_damped_risk",
            "tail_routing_mode": "pack",
            "tail_dispatch_mode": "protected_drain",
            "tail_gate_wait_s": None,
            "central_risk_beta": 0.5,
            "central_risk_score": 43.0,
            "central_wait_s": 9.0,
            "workload_class": "short",
        },
    ]
    backend_events = [
        {
            "ts": 112.0,
            "ts_ns": 500,
            "event": "backend_start",
            "request_id": "request-00000",
            "video_id": "video-1",
            "node": "backend-8093",
        },
        {
            "ts": 114.0,
            "ts_ns": 510,
            "event": "scheduler_select",
            "request_id": "video-1",
            "sched_req_id": "video-1",
            "selection_kind": "first",
        },
        {
            "ts": 176.0,
            "ts_ns": 540,
            "event": "scheduler_complete",
            "request_id": "video-1",
            "sched_req_id": "video-1",
        },
        {
            "ts": 178.0,
            "ts_ns": 550,
            "event": "backend_complete",
            "request_id": "request-00000",
            "video_id": "video-1",
            "inference_time_s": 66.0,
        },
    ]
    for name, events in (
        ("client.jsonl", client_events),
        ("dispatcher.jsonl", dispatcher_events),
        ("backend_8093.jsonl", backend_events),
    ):
        (tmp_path / name).write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    rows = build_request_rows(load_trace_events(tmp_path))

    assert rows == [
        {
            "request_id": "request-00000",
            "workload_class": "short",
            "width": 854,
            "height": 480,
            "num_frames": 80,
            "num_inference_steps": 3,
            "fps": 16,
            "arrival_counter": 7,
            "queue_class": "normal",
            "estimated_service_s": 68.0,
            "normal_routing_policy": "central_pull_cost_damped_risk",
            "tail_routing_mode": "pack",
            "tail_dispatch_mode": "protected_drain",
            "tail_gate_wait_s": None,
            "central_risk_beta": 0.5,
            "central_risk_score": 43.0,
            "central_queue_depth": 4,
            "central_wait_s": 9.0,
            "backend": "backend-2",
            "video_id": "video-1",
            "sched_req_id": "video-1",
            "dispatcher_arrival_offset_s": 1.0,
            "api_backend_start_offset_s": 12.0,
            "scheduler_start_offset_s": 14.0,
            "backend_start_offset_s": 14.0,
            "scheduler_queue_wait_s": 2.0,
            "scheduler_elapsed_s": 62.0,
            "scheduler_select_count": 1,
            "scheduler_preempt_count": 0,
            "backend_inference_time_s": 66.0,
            "prediction_error_percent": pytest.approx(-2.941176470588235),
            "dispatcher_e2e_s": None,
            "e2e_latency_s": 80.0,
            "success": True,
            "error": None,
        }
    ]

    summary = summarize_request_rows(rows)
    assert summary["latency_p95_s"] == 80.0
    assert summary["queue_class_counts"] == {"normal": 1}
    assert summary["backend_counts"] == {"backend-2": 1}

    json_path, csv_path = write_trace_report(rows, output_prefix=tmp_path / "report")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["successful_request_count"] == 1
    assert csv_path.read_text(encoding="utf-8").splitlines()[1].startswith("request-00000,")


def test_trace_report_excludes_warmups_by_default() -> None:
    events = [
        {
            "ts": 1.0,
            "ts_ns": 1,
            "event": "client_arrive",
            "request_id": "warmup-00000",
        },
        {
            "ts": 2.0,
            "ts_ns": 2,
            "event": "client_arrive",
            "request_id": "request-00000",
        },
    ]

    assert [row["request_id"] for row in build_request_rows(events)] == ["request-00000"]
    assert [row["request_id"] for row in build_request_rows(events, include_warmups=True)] == [
        "request-00000",
        "warmup-00000",
    ]


def test_write_trace_event_appends_jsonl(tmp_path) -> None:
    trace_file = tmp_path / "node.jsonl"

    write_trace_event(
        str(trace_file),
        "backend_start",
        node="backend-8091",
        request_id="request-00001",
        estimated_service_s=68.602,
    )

    record = json.loads(trace_file.read_text(encoding="utf-8"))
    assert record["event"] == "backend_start"
    assert record["node"] == "backend-8091"
    assert record["request_id"] == "request-00001"
    assert record["estimated_service_s"] == 68.602
    assert isinstance(record["ts_ns"], int)


@pytest.mark.parametrize("failure_point", ["json.dumps", "os.open", "os.write"])
def test_write_trace_event_is_best_effort(tmp_path, monkeypatch, failure_point: str) -> None:
    def fail(*args, **kwargs):
        raise OSError("injected trace failure")

    owner_name, attribute = failure_point.split(".")
    monkeypatch.setattr(getattr(trace_logging, owner_name), attribute, fail)

    write_trace_event(
        str(tmp_path / "node.jsonl"),
        "backend_start",
        request_id="request-00001",
    )
