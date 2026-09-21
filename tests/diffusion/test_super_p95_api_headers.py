# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Metadata and source-isolated HTTP-boundary tests, not ASGI/runtime tests."""

import ast
import asyncio
import json
import uuid
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S as COST,
)
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_SACRIFICIAL as TAIL,
)
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_TRACE_REQUEST_ID as TRACE_ID,
)
from vllm_omni.diffusion.super_p95 import (
    SuperP95LoadSnapshot,
    apply_super_p95_request_headers,
    apply_super_p95_sampling_headers,
    build_super_p95_response_headers,
    get_super_p95_request_metadata,
    parse_super_p95_load_headers,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"x-super-p95-sacrificial": "true", "X-SUPER-P95-ESTIMATED-SERVICE-S": "12.5", "x-request-id": "client-7"}


def test_headers_are_case_insensitive_and_preserve_model_arguments():
    extra = {"flow_shift": 5.0}
    apply_super_p95_request_headers(extra, HEADERS)
    assert extra == {"flow_shift": 5.0, TAIL: True, COST: 12.5, TRACE_ID: "client-7"}


@pytest.mark.parametrize("value", ["-1", "nan", "NaN", "inf", "+Infinity", "-inf", "invalid", ""])
def test_invalid_service_header_is_ignored(value):
    extra = {}
    apply_super_p95_request_headers(extra, {"X-Super-P95-Estimated-Service-S": value})
    assert COST not in extra
    assert get_super_p95_request_metadata(extra) == (False, None)


@pytest.mark.parametrize("value", [False, "false", "FALSE", "off", "no", "0", 0, None, "invalid"])
def test_false_like_metadata_never_becomes_tail(value):
    assert get_super_p95_request_metadata({TAIL: value, COST: 0.0}) == (False, 0.0)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, False, "12.5"])
def test_invalid_or_nonnumeric_service_metadata_falls_back(value):
    assert get_super_p95_request_metadata({TAIL: True, COST: value}) == (True, None)


def test_valid_false_header_overrides_body_and_copy_does_not_mutate_defaults():
    defaults = {"flow_shift": 5.0, TAIL: True}
    one = SimpleNamespace(extra_args=defaults)
    two = SimpleNamespace(extra_args=defaults)
    apply_super_p95_sampling_headers(one, {"x-super-p95-sacrificial": "false", "x-request-id": "one"})
    assert one.extra_args == {"flow_shift": 5.0, TAIL: False, TRACE_ID: "one"}
    assert defaults == {"flow_shift": 5.0, TAIL: True}
    assert two.extra_args is defaults


@pytest.mark.parametrize("extra", [None, {}])
def test_sampling_headers_accept_unset_extra_args(extra):
    params = SimpleNamespace(extra_args=extra)
    apply_super_p95_sampling_headers(params, HEADERS)
    assert get_super_p95_request_metadata(params.extra_args) == (True, 12.5)


def test_load_headers_round_trip_without_requiring_local_engine_access():
    snapshot = SuperP95LoadSnapshot(normal_load_s=12.5, sacrificial_load_s=3.25)
    headers = build_super_p95_response_headers(snapshot)
    assert parse_super_p95_load_headers({key.lower(): value for key, value in headers.items()}) == snapshot
    assert parse_super_p95_load_headers({}) is None


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "bad"])
def test_invalid_load_snapshot_is_not_accepted(value):
    assert (
        parse_super_p95_load_headers({"x-super-p95-normal-load-s": value, "x-super-p95-sacrificial-load-s": "0"})
        is None
    )


def _isolated_function(relative_path, name, namespace, *, class_name=None, replace_defaults=False):
    """Execute real endpoint source with fake runtime dependencies."""
    tree = ast.parse((ROOT / relative_path).read_text())
    body = tree.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(
        node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    function.decorator_list = []
    if replace_defaults:
        function.args.defaults = [ast.Constant(value=None) for _ in function.args.defaults]
        function.args.kw_defaults = [
            ast.Constant(value=None) if item is not None else None for item in function.args.kw_defaults
        ]
    selected = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(selected), str(ROOT / relative_path), "exec"), scope)
    return scope[name]


def test_generate_diffusion_images_propagates_headers_after_preparation_without_undefined_request():
    captured = {}
    params = SimpleNamespace(extra_args={"normalized_model_option": 1})

    async def generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(images=["image"], stage_durations={}, peak_memory_mb=0, metrics=None)

    engine = SimpleNamespace(generate=generate)
    handler = SimpleNamespace(
        _prepare_diffusion_image_request=lambda **kwargs: (engine, {"prompt": "prompt"}, params, []),
        _truthy_extra_body_flag=lambda *args: False,
        _flatten_diffusion_images=lambda images: images,
    )
    method = _isolated_function(
        "vllm_omni/entrypoints/openai/serving_chat.py",
        "generate_diffusion_images",
        {
            "uuid": uuid,
            "ErrorResponse": type("ErrorResponse", (), {}),
            "AsyncOmni": type("AsyncOmni", (), {}),
            "apply_super_p95_sampling_headers": apply_super_p95_sampling_headers,
        },
        class_name="OmniOpenAIServingChat",
    )
    result = asyncio.run(MethodType(method, handler)(prompt="prompt", raw_request=SimpleNamespace(headers=HEADERS)))
    assert result[0] == ["image"]
    assert captured["request_id"] == "client-7"
    assert captured["sampling_params"].extra_args == {
        "normalized_model_option": 1,
        TAIL: True,
        COST: 12.5,
        TRACE_ID: "client-7",
    }


def test_video_form_injects_private_metadata_after_public_request_construction():
    constructed = {}
    public_extra = {"flow_shift": 5.0}

    class StopBeforeModelLoadingError(Exception):
        pass

    def make_request(**kwargs):
        assert kwargs["extra_params"] == public_extra
        constructed["request"] = SimpleNamespace(**kwargs)
        return constructed["request"]

    def stop(_):
        raise StopBeforeModelLoadingError

    async def read_mask(*args, **kwargs):
        return None

    parse_form = _isolated_function(
        "vllm_omni/entrypoints/openai/video/generation/helpers.py",
        "_parse_video_form",
        {
            "_parse_form_json": lambda value, **kwargs: json.loads(value) if value else None,
            "_read_latent_edit_mask_json": read_mask,
            "VideoGenerationRequest": make_request,
            "Omnivideo": stop,
        },
        replace_defaults=True,
    )
    with pytest.raises(StopBeforeModelLoadingError):
        asyncio.run(
            parse_form(SimpleNamespace(headers=HEADERS), prompt="prompt", extra_params=json.dumps(public_extra))
        )
    assert constructed["request"].extra_params == {"flow_shift": 5.0, TAIL: True, COST: 12.5, TRACE_ID: "client-7"}


def test_multistage_image_generation_injects_only_diffusion_stage_after_default_cloning():
    captured = {}

    class FakeAsyncOmni:
        stage_configs = ["llm", "diffusion"]

        async def generate(self, **kwargs):
            captured.update(kwargs)
            yield SimpleNamespace(images=["image"], stage_durations={}, peak_memory_mb=0)

    llm_params = SimpleNamespace(extra_args={"llm_option": 1})
    cloned_diffusion_params = SimpleNamespace(extra_args={"stage_default": 2})
    initial_params = SimpleNamespace(extra_args={})
    engine = FakeAsyncOmni()
    handler = SimpleNamespace(
        _prepare_diffusion_image_request=lambda **kwargs: (engine, {}, initial_params, []),
        _truthy_extra_body_flag=lambda *args: False,
        _flatten_diffusion_images=lambda images: images,
        _build_multistage_generation_inputs=lambda **kwargs: ({}, [llm_params, cloned_diffusion_params]),
    )
    method = _isolated_function(
        "vllm_omni/entrypoints/openai/serving_chat.py",
        "generate_diffusion_images",
        {
            "uuid": uuid,
            "ErrorResponse": type("ErrorResponse", (), {}),
            "AsyncOmni": FakeAsyncOmni,
            "cast": lambda cls, value: value,
            "get_stage_type": lambda stage: stage,
            "coerce_param_message_types": lambda params, stream: params,
            "apply_super_p95_sampling_headers": apply_super_p95_sampling_headers,
        },
        class_name="OmniOpenAIServingChat",
    )
    asyncio.run(MethodType(method, handler)(prompt="prompt", raw_request=SimpleNamespace(headers=HEADERS)))
    assert captured["sampling_params_list"][0].extra_args == {"llm_option": 1}
    assert captured["sampling_params_list"][1].extra_args == {
        "stage_default": 2,
        TAIL: True,
        COST: 12.5,
        TRACE_ID: "client-7",
    }


def test_scheduler_trace_correlates_external_id_without_changing_native_identity(monkeypatch):
    from vllm_omni.diffusion.sched import super_p95_step_scheduler as policy_module

    events = []
    monkeypatch.setattr(policy_module, "write_trace_event", lambda path, event, **data: events.append(data))
    scheduler = policy_module.SuperP95StepScheduler()
    scheduler._metadata["internal-9"] = SimpleNamespace(is_tail=False, estimated_service_s=12.5)
    scheduler._request_states["internal-9"] = SimpleNamespace(
        req=SimpleNamespace(sampling_params=SimpleNamespace(extra_args={TRACE_ID: "client-7"}))
    )
    scheduler._trace("scheduler_select", "internal-9")
    assert events[0]["request_id"] == "client-7"
    assert events[0]["scheduler_request_id"] == "internal-9"
