# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.sched.dispatch import estimation as estimation_module
from vllm_omni.diffusion.sched.dispatch.estimation import estimate_service_time_s

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def params(size=512, steps=20):
    return SimpleNamespace(width=size, height=size, num_inference_steps=steps, num_frames=1, num_outputs_per_prompt=1)


@pytest.mark.parametrize("model_class", ["QwenImagePipeline", "WanPipeline", "Wan22Pipeline"])
@pytest.mark.parametrize("profile", ["910B2", "910B3"])
def test_model_profile_exact_and_scaled_estimates(model_class, profile):
    qwen = model_class == "QwenImagePipeline"
    sampling = (
        params(1536, 35) if qwen else SimpleNamespace(width=1280, height=720, num_inference_steps=6, num_frames=80)
    )
    expected = (43.22 if profile == "910B2" else 49.34) if qwen else 119.71
    assert estimate_service_time_s(sampling, model_class, profile) == pytest.approx(expected)
    sampling.num_inference_steps *= 2
    sampling.num_frames = 1 if qwen else 81
    assert estimate_service_time_s(sampling, model_class, profile) == pytest.approx(expected * 2)


@pytest.mark.parametrize("profile,other_profile", [("910B2", "910B3"), ("910B3", "910B2")])
def test_wan_uses_selected_profile_for_exact_and_scaled_estimates(monkeypatch, profile, other_profile):
    sampling = SimpleNamespace(width=1280, height=720, num_inference_steps=6, num_frames=80)
    # Synthetic test data proves profile selection without claiming new calibration.
    anchors = dict(estimation_module._WAN_ANCHORS[profile])
    anchors[(1280, 720, 6, 81)] = 200.0
    monkeypatch.setitem(estimation_module._WAN_ANCHORS, profile, anchors)
    assert estimate_service_time_s(sampling, "Wan22Pipeline", profile) == pytest.approx(200.0)
    assert estimate_service_time_s(sampling, "Wan22Pipeline", other_profile) == pytest.approx(119.71)
    sampling.num_inference_steps = 12
    sampling.num_frames = 81
    assert estimate_service_time_s(sampling, "Wan22Pipeline", profile) == pytest.approx(200.0 * 2)
    assert estimate_service_time_s(sampling, "Wan22Pipeline", other_profile) == pytest.approx(119.71 * 2)


@pytest.mark.parametrize("profile", ["910B2", "910B3"])
@pytest.mark.parametrize(
    "width,steps,frames,expected", [(854, 3, 80, 38.07), (854, 4, 120, 71.34), (1280, 6, 80, 119.71)]
)
def test_wan_equivalent_execution_shapes_have_equal_estimates(profile, width, steps, frames, expected):
    raw = SimpleNamespace(
        width=width, height=480 if width == 854 else 720, num_inference_steps=steps, num_frames=frames
    )
    normalized = SimpleNamespace(**vars(raw))
    normalized.width = width // 16 * 16
    normalized.num_frames = frames // 4 * 4 + 1
    for sampling in (raw, normalized):
        assert estimate_service_time_s(sampling, "Wan22Pipeline", profile) == pytest.approx(expected)
    raw.num_inference_steps += 1
    normalized.num_inference_steps += 1
    assert estimate_service_time_s(raw, "Wan22Pipeline", profile) == pytest.approx(
        estimate_service_time_s(normalized, "Wan22Pipeline", profile)
    )
