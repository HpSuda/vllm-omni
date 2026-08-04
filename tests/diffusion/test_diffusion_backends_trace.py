# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import benchmarks.diffusion.backends as backends


def test_chat_client_trace_records_early_failure(monkeypatch, tmp_path) -> None:
    events = []
    monkeypatch.setattr(
        backends,
        "write_trace_event",
        lambda _path, event, **fields: events.append((event, fields)),
    )
    request = backends.RequestFuncInput(
        prompt="trace",
        api_url="http://127.0.0.1:8080/v1/chat/completions",
        model="Qwen/Qwen-Image",
        width=1024,
        height=1024,
        num_inference_steps=25,
        image_paths=[str(tmp_path / "missing.png")],
        request_id="request-00001",
        trace_log_file=str(tmp_path / "client.jsonl"),
        trace_label="client",
    )

    output = asyncio.run(backends.async_request_chat_completions(request, session=None))

    assert output.success is False
    assert [event for event, _fields in events] == ["client_arrive", "client_finish"]
    assert events[0][1]["request_id"] == "request-00001"
    assert events[0][1]["width"] == 1024
    assert events[1][1]["success"] is False
    assert events[1][1]["error"].startswith("Image file not found:")
