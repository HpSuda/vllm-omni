# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_TRACE_LOCK = threading.Lock()


def write_trace_event(
    trace_file: str | None,
    event: str,
    *,
    node: str | None = None,
    request_id: str | None = None,
    **fields: Any,
) -> None:
    """Append one timestamped event to a JSONL trace.

    Trace files are intentionally opt-in. Each managed process writes to its
    own file, so the in-process lock only needs to prevent thread interleaving.
    Tracing is diagnostic and therefore best-effort: serialization or I/O
    failures must never change request handling or scheduler behavior.
    """

    if not trace_file:
        return

    try:
        record: dict[str, Any] = {
            "ts": time.time(),
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "event": event,
        }
        if node:
            record["node"] = node
        if request_id:
            record["request_id"] = request_id
        record.update(fields)

        os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
        line = (json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        with _TRACE_LOCK:
            descriptor = os.open(trace_file, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(descriptor, line)
            finally:
                os.close(descriptor)
    except Exception:
        return
