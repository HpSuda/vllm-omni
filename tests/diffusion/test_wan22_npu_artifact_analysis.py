# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import math
import tarfile

import pytest

from benchmarks.diffusion.wan22_npu_artifact_analysis import (
    ArtifactValidationError,
    _clean_normal_calibration,
    analyze_artifact,
    render_markdown,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _write_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _make_artifact(tmp_path, *, corrupt_ids: bool = False):
    rows = []
    events = []
    latencies = []
    for index in range(10):
        request_id = f"request-{index:05d}"
        if corrupt_ids and index == 9:
            request_id = "request-00010"
        workload_class = "short" if index < 5 else "long"
        queue_class = "tail" if index == 9 else "normal"
        preempts = 2 if queue_class == "tail" else 0
        selects = preempts + 1
        backend_s = float(100 + index * 10)
        scheduler_queue_s = 2.0
        latency_s = float(200 + index * 20)
        latencies.append(latency_s)
        video_id = f"video-{index}"
        rows.append(
            {
                "request_id": request_id,
                "workload_class": workload_class,
                "queue_class": queue_class,
                "backend": f"backend-{index % 2}",
                "arrival_counter": index + 2,
                "scheduler_select_count": selects,
                "scheduler_preempt_count": preempts,
                "scheduler_elapsed_s": backend_s - 4.0,
                "scheduler_queue_wait_s": scheduler_queue_s,
                "backend_inference_time_s": backend_s,
                "central_wait_s": float(index),
                "e2e_latency_s": latency_s,
                "success": True,
            }
        )
        direct_events = (
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
            "backend_complete",
        )
        for event_index, event_name in enumerate(direct_events):
            event = {
                "event": event_name,
                "request_id": request_id,
                "video_id": video_id,
                "ts": float(index * 100 + event_index),
                "ts_ns": index * 100 + event_index,
            }
            if event_name == "client_finish":
                event.update(success=True, latency_s=latency_s)
            events.append(event)
        events.append(
            {
                "event": "scheduler_enqueue",
                "request_id": video_id,
                "ts": float(index * 100 + 20),
                "ts_ns": index * 100 + 20,
            }
        )
        for select_index in range(selects):
            events.append(
                {
                    "event": "scheduler_select",
                    "request_id": video_id,
                    "ts": float(index * 100 + 21 + select_index * 2),
                    "ts_ns": index * 100 + 21 + select_index * 2,
                }
            )
            if select_index < preempts:
                events.append(
                    {
                        "event": "scheduler_preempt",
                        "request_id": video_id,
                        "ts": float(index * 100 + 22 + select_index * 2),
                        "ts_ns": index * 100 + 22 + select_index * 2,
                    }
                )
        events.append(
            {
                "event": "scheduler_complete",
                "request_id": video_id,
                "ts": float(index * 100 + 30),
                "ts_ns": index * 100 + 30,
            }
        )

    def percentile(values, quantile):
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    queue_counts = {"normal": 9, "tail": 1}
    workload_counts = {"long": 5, "short": 5}
    backend_counts = {"backend-0": 5, "backend-1": 5}
    result = {
        "duration": 500.0,
        "completed_requests": 10,
        "failed_requests": 0,
        "throughput_qps": 0.02,
        "latency_mean": sum(latencies) / len(latencies),
        "latency_median": 290.0,
        "latency_p95": percentile(latencies, 0.95),
        "latency_p99": percentile(latencies, 0.99),
    }
    request_trace = {
        "summary": {
            "request_count": 10,
            "measured_request_count": 10,
            "successful_request_count": 10,
            "failed_request_count": 0,
            "latency_mean_s": sum(latencies) / len(latencies),
            "latency_p95_s": percentile(latencies, 0.95),
            "latency_p99_s": percentile(latencies, 0.99),
            "queue_class_counts": queue_counts,
            "workload_class_counts": workload_counts,
            "backend_counts": backend_counts,
        },
        "requests": rows,
    }
    path = tmp_path / "artifact.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _write_member(archive, "result.json", json.dumps(result).encode())
        _write_member(archive, "request_trace.json", json.dumps(request_trace).encode())
        jsonl = b"".join(json.dumps(event).encode() + b"\n" for event in events)
        _write_member(archive, "trace/all.jsonl", jsonl)
    return path


def test_analyze_artifact_validates_and_reports_type7_and_calibration(tmp_path) -> None:
    report = analyze_artifact(_make_artifact(tmp_path))

    assert report["validation"]["status"] == "passed"
    assert report["validation"]["raw_trace"]["measured_request_lifecycles_complete"] == 10
    assert report["counts"]["queue_class"] == {"normal": 9, "tail": 1}
    assert report["p95_type7"]["value_s"] == pytest.approx(371.0)
    assert report["p95_type7"]["lower"]["request_id"] == "request-00008"
    assert report["p95_type7"]["lower"]["weight"] == pytest.approx(0.45)
    assert report["p95_type7"]["upper"]["request_id"] == "request-00009"
    assert report["p95_type7"]["upper"]["weight"] == pytest.approx(0.55)
    assert report["tail"]["requests"][0]["request_id"] == "request-00009"
    assert report["tail"]["preemption_count"] == 2
    assert report["per_workload_class"]["short"]["exclusive_service_s"]["mean"] == 118.0

    calibration = report["clean_normal_calibration"]
    assert calibration["request_count_before_trimming"] == 9
    assert calibration["per_workload_class"]["short"]["raw_count"] == 5
    assert calibration["per_workload_class"]["long"]["raw_count"] == 4
    assert calibration["pooled_within_class_log_sigma_mle"] > 0.0
    assert calibration["pooled_within_class_log_sigma_10pct_trimmed_mle"] > 0.0
    markdown = render_markdown(report)
    assert "NumPy Type-7 P95 Boundary" in markdown
    assert "All clean samples" in markdown
    assert "Per-class 10%-trimmed samples" in markdown


def test_clean_normal_calibration_keeps_full_sigma_and_trimmed_robust_center_separate() -> None:
    exclusive_services = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 300.0]
    rows = [
        {
            "request_id": f"request-{index:05d}",
            "workload_class": "short",
            "queue_class": "normal",
            "scheduler_preempt_count": 0,
            "scheduler_select_count": 1,
            "exclusive_service_s": service_s,
        }
        for index, service_s in enumerate(exclusive_services)
    ]

    calibration = _clean_normal_calibration(rows)

    assert calibration["per_workload_class"]["short"]["trim_each_side"] == 1
    assert calibration["per_workload_class"]["short"]["trimmed_mean_s"] == pytest.approx(104.5)
    assert (
        calibration["pooled_within_class_log_sigma_mle"]
        > calibration["pooled_within_class_log_sigma_10pct_trimmed_mle"]
    )


def test_analyze_artifact_rejects_non_contiguous_measured_ids(tmp_path) -> None:
    with pytest.raises(ArtifactValidationError, match="not contiguous"):
        analyze_artifact(_make_artifact(tmp_path, corrupt_ids=True))


def test_analyze_artifact_rejects_nonstandard_json(tmp_path) -> None:
    path = tmp_path / "invalid.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _write_member(archive, "result.json", b'{"completed_requests": NaN}')
        _write_member(archive, "request_trace.json", b"{}")
        _write_member(archive, "trace/client.jsonl", b"{}\n")

    with pytest.raises(ArtifactValidationError, match="Non-standard JSON constant"):
        analyze_artifact(path)
