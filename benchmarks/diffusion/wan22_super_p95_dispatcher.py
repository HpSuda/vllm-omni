# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan2.2-specific entry point for the reusable super-P95 dispatcher."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import HTTPException

from benchmarks.diffusion.super_p95_dispatcher import (
    SuperP95Dispatcher,
    build_app,
    build_arg_parser,
    build_dispatcher_from_args,
)
from vllm_omni.diffusion.super_p95 import (
    normalize_super_p95_hardware_profile,
    normalize_wan2_2_num_frames,
)

Wan22WorkloadKey = tuple[int, int, int, int]

_SHORT_KEY: Wan22WorkloadKey = (854, 480, 3, 80)
_MEDIUM_KEY: Wan22WorkloadKey = (854, 480, 4, 120)
_LONG_KEY: Wan22WorkloadKey = (1280, 720, 6, 80)

# "production" preserves the estimator already used by super_p95.py.
# Topology-specific profiles come from the four serial E2E measurements recorded
# in wan22_topology_component_profile_20260725.md. They are deliberately kept in
# this Wan2.2-only entry point instead of changing Qwen-Image or the generic
# dispatcher.
WAN22_ESTIMATOR_PROFILES: dict[str, dict[str, dict[Wan22WorkloadKey, float]]] = {
    "production": {
        "910B2": {
            _SHORT_KEY: 38.07,
            _MEDIUM_KEY: 71.34,
            _LONG_KEY: 119.71,
        },
        "910B3": {
            _SHORT_KEY: 38.07,
            _MEDIUM_KEY: 71.34,
            _LONG_KEY: 119.71,
        },
    },
    "2xusp4": {
        "910B3": {
            _SHORT_KEY: 46.461,
            _MEDIUM_KEY: 86.869,
            _LONG_KEY: 188.720,
        },
    },
    "4xusp2": {
        "910B3": {
            _SHORT_KEY: 68.602,
            _MEDIUM_KEY: 132.733,
            _LONG_KEY: 357.126,
        },
    },
    # USP1 has no isolated single-request measurements yet. These anchors are
    # the per-size median of three topology projections (analytic USP scaling,
    # an A+B/USP residual fit, and doubled USP2 residual time), each calibrated
    # with one shared factor against the existing 8xUSP1/50 baseline P95. Keep
    # "inferred" in the profile name so trace output cannot be mistaken for a
    # directly measured service profile.
    "8xusp1_inferred": {
        "910B3": {
            _SHORT_KEY: 110.724,
            _MEDIUM_KEY: 219.548,
            _LONG_KEY: 612.299,
        },
    },
}


@dataclass(frozen=True)
class Wan22ServiceTimeEstimator:
    """Coarse Wan2.2 estimator selected explicitly by deployment topology."""

    profile_name: str

    def __post_init__(self) -> None:
        if self.profile_name not in WAN22_ESTIMATOR_PROFILES:
            choices = ", ".join(sorted(WAN22_ESTIMATOR_PROFILES))
            raise ValueError(f"Unknown Wan2.2 estimator profile {self.profile_name!r}; choose one of: {choices}")

    @property
    def name(self) -> str:
        return f"wan22:{self.profile_name}"

    def estimate(self, path: str, body: dict[str, Any], hardware_profile: str) -> float:
        if path != "/v1/videos":
            raise HTTPException(
                status_code=400,
                detail=f"Wan2.2 dispatcher only supports /v1/videos, got: {path}",
            )

        profile = normalize_super_p95_hardware_profile(hardware_profile)
        profile_anchors = WAN22_ESTIMATOR_PROFILES[self.profile_name]
        anchors = profile_anchors.get(profile)
        if anchors is None:
            supported = ", ".join(sorted(profile_anchors))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Wan2.2 estimator profile {self.profile_name!r} is not calibrated "
                    f"for hardware profile {profile}; supported: {supported}"
                ),
            )

        width, height = _parse_dimensions(body)
        steps = _parse_positive_int(body.get("num_inference_steps"), default=25)
        frames = _parse_num_frames(body)
        key = (width, height, steps, frames)
        exact_match = anchors.get(key)
        if exact_match is not None:
            return exact_match

        # Unknown inputs use a coarse compute-volume scaling from the long
        # anchor. This stays intentionally above kernel-level granularity.
        normalized_frames = normalize_wan2_2_num_frames(frames)
        base_width, base_height, base_steps, base_frames = _LONG_KEY
        work_ratio = width * height * steps * normalized_frames / (base_width * base_height * base_steps * base_frames)
        return anchors[_LONG_KEY] * work_ratio


class Wan22SuperP95Dispatcher(SuperP95Dispatcher):
    """Super-P95 dispatcher with an explicit Wan2.2-only estimator."""

    def __init__(
        self,
        *args: Any,
        wan22_estimator: Wan22ServiceTimeEstimator,
        **kwargs: Any,
    ) -> None:
        kwargs["service_time_estimator_name"] = wan22_estimator.name
        super().__init__(*args, **kwargs)
        self.wan22_estimator = wan22_estimator

    def _estimate_service_s(self, path: str, body: dict[str, Any], hardware_profile: str) -> float:
        return self.wan22_estimator.estimate(path, body, hardware_profile)


def _parse_dimensions(body: dict[str, Any]) -> tuple[int, int]:
    width = _parse_optional_positive_int(body.get("width"))
    height = _parse_optional_positive_int(body.get("height"))
    size = body.get("size")
    if (width is None or height is None) and isinstance(size, str) and "x" in size.lower():
        width_text, height_text = size.lower().split("x", 1)
        width = width or _parse_optional_positive_int(width_text)
        height = height or _parse_optional_positive_int(height_text)
    return width or 854, height or 480


def _parse_num_frames(body: dict[str, Any]) -> int:
    frames = _parse_optional_positive_int(body.get("num_frames"))
    if frames is not None:
        return frames
    seconds = _parse_optional_positive_int(body.get("seconds"))
    if seconds is None:
        return 1
    fps = _parse_positive_int(body.get("fps"), default=24)
    return seconds * fps


def _parse_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_positive_int(value: Any, *, default: int) -> int:
    return _parse_optional_positive_int(value) or default


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(description="Wan2.2-specific super_p95 dispatcher")
    parser.add_argument(
        "--wan22-estimator-profile",
        choices=tuple(WAN22_ESTIMATOR_PROFILES),
        default="production",
        help=(
            "Wan2.2 service-time calibration used by Tail classification, "
            "backend load accounting, and Max Risk ordering."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator = Wan22ServiceTimeEstimator(args.wan22_estimator_profile)
    dispatcher = build_dispatcher_from_args(
        args,
        dispatcher_cls=Wan22SuperP95Dispatcher,
        dispatcher_kwargs={"wan22_estimator": estimator},
    )
    uvicorn.run(build_app(dispatcher), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
