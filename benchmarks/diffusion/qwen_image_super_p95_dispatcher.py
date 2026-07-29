# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen-Image entry point for named super-P95 scheduling modes."""

from __future__ import annotations

import argparse

import uvicorn

from benchmarks.diffusion.super_p95_dispatcher import (
    apply_release_calendar_tail_pack_backfill_defaults,
    build_app,
    build_arg_parser,
    build_dispatcher_from_args,
)

QWEN_IMAGE_SCHEDULING_MODE_CUSTOM = "custom"
QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL = (
    "release_calendar_tail_pack_backfill"
)
QWEN_IMAGE_SCHEDULING_MODES = (
    QWEN_IMAGE_SCHEDULING_MODE_CUSTOM,
    QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL,
)

_QWEN_IMAGE_MODE_BACKEND_ENV = {
    "VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING": "1",
    "VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION": "1",
    "SUPER_P95_QWEN_SMALL_BATCH2": "1",
    "SUPER_P95_IMAGE_BATCH_SEARCH_WINDOW": "64",
    "SUPER_P95_QWEN_SMALL_BATCH_MIN_PENDING": "1",
}


def _set_default_backend_env(args: argparse.Namespace) -> None:
    configured_keys = {
        item.split("=", 1)[0].strip()
        for item in args.backend_env
        if "=" in item
    }
    for key, value in _QWEN_IMAGE_MODE_BACKEND_ENV.items():
        if key not in configured_keys:
            args.backend_env.append(f"{key}={value}")


def apply_qwen_image_scheduling_mode(args: argparse.Namespace) -> argparse.Namespace:
    if (
        args.qwen_image_scheduling_mode
        == QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL
    ):
        apply_release_calendar_tail_pack_backfill_defaults(args)
        args.backend_scheduler = "super_p95_step"
        _set_default_backend_env(args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser(description="Qwen-Image super-P95 dispatcher")
    parser.add_argument(
        "--qwen-image-scheduling-mode",
        choices=QWEN_IMAGE_SCHEDULING_MODES,
        default=QWEN_IMAGE_SCHEDULING_MODE_CUSTOM,
        help=(
            "Named Qwen-Image scheduling mode. "
            "release_calendar_tail_pack_backfill enables Release Calendar, "
            "Tail Pack, protected drain, idle backfill, and Qwen batch2."
        ),
    )
    return apply_qwen_image_scheduling_mode(parser.parse_args(argv))


def build_qwen_image_dispatcher(args: argparse.Namespace):
    return build_dispatcher_from_args(
        args,
        dispatcher_kwargs={
            # The Qwen backend scheduler preempts a running Tail when a
            # Normal request arrives. Release Calendar can therefore treat a
            # Tail-only backend as immediately available to Normal work.
            "running_tail_preemptible_for_normal": (
                args.qwen_image_scheduling_mode
                == QWEN_IMAGE_SCHEDULING_MODE_RELEASE_CALENDAR_TAIL_PACK_BACKFILL
            ),
        },
    )


def main() -> None:
    args = parse_args()
    dispatcher = build_qwen_image_dispatcher(args)
    uvicorn.run(build_app(dispatcher), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
