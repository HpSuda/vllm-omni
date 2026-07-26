# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from benchmarks.diffusion.simulator.models import (
    BackendView,
    ComponentConfig,
    Priority,
    RequestView,
)


@dataclass(frozen=True)
class ClassificationDecision:
    priority: Priority
    details: dict[str, Any]


@dataclass(frozen=True)
class RoutingDecision:
    backend_name: str
    details: dict[str, Any]


class Classifier(Protocol):
    def classify(
        self,
        request: RequestView,
        minimum_backend_estimate_s: float,
        *,
        now_s: float = 0.0,
        backends: tuple[BackendView, ...] = (),
    ) -> ClassificationDecision: ...


class Router(Protocol):
    load_view: str
    update_latency_ema: bool

    def choose(self, request: RequestView, backends: tuple[BackendView, ...]) -> RoutingDecision: ...


class LocalScheduler(Protocol):
    def choose(
        self,
        *,
        now_s: float,
        backend: BackendView,
        incumbent: RequestView | None,
        pending: tuple[RequestView, ...],
    ) -> str | None: ...


def _reject_unknown(options: dict[str, Any], allowed: set[str], component: str) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown {component} option(s): {', '.join(unknown)}")


def _as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{path} must be a boolean")


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


class AllNormalClassifier:
    def classify(
        self,
        request: RequestView,
        minimum_backend_estimate_s: float,
        *,
        now_s: float = 0.0,
        backends: tuple[BackendView, ...] = (),
    ) -> ClassificationDecision:
        del request, minimum_backend_estimate_s, now_s, backends
        return ClassificationDecision(priority=Priority.NORMAL, details={})


class QuotaTailClassifier:
    """Replica of the dispatcher-side quota classification state machine."""

    def __init__(
        self,
        *,
        quota_every: int,
        quota_amount: int,
        threshold_ratio: float,
        long_request_ratio: float | None,
        initial_arrival_counter: int = 0,
        initial_credits: int = 0,
        initial_min_service_s: float | None = None,
        initial_max_service_s: float = 0.0,
        max_sacrificial: int | None = None,
        eligible_request_types: list[str] | tuple[str, ...] | None = None,
        credit_release_requests: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        quota_every = _as_int(quota_every, "quota_every")
        quota_amount = _as_int(quota_amount, "quota_amount")
        threshold_ratio = _as_float(threshold_ratio, "threshold_ratio")
        initial_arrival_counter = _as_int(initial_arrival_counter, "initial_arrival_counter")
        initial_credits = _as_int(initial_credits, "initial_credits")
        if max_sacrificial is not None:
            max_sacrificial = _as_int(max_sacrificial, "max_sacrificial")
        if long_request_ratio is not None:
            long_request_ratio = _as_float(long_request_ratio, "long_request_ratio")
        if initial_min_service_s is not None:
            initial_min_service_s = _as_float(initial_min_service_s, "initial_min_service_s")
        initial_max_service_s = _as_float(initial_max_service_s, "initial_max_service_s")
        if quota_every <= 0:
            raise ValueError("quota_every must be positive")
        if quota_amount < 0:
            raise ValueError("quota_amount cannot be negative")
        if not math.isfinite(threshold_ratio) or threshold_ratio < 0.0:
            raise ValueError("threshold_ratio cannot be negative")
        if long_request_ratio is not None and (not math.isfinite(long_request_ratio) or long_request_ratio < 1.0):
            raise ValueError("long_request_ratio must be at least 1 or null")
        if initial_arrival_counter < 0 or initial_credits < 0:
            raise ValueError("initial classifier counters cannot be negative")
        if max_sacrificial is not None and max_sacrificial < 0:
            raise ValueError("max_sacrificial cannot be negative")
        if eligible_request_types is not None:
            if isinstance(eligible_request_types, (str, bytes)) or not isinstance(
                eligible_request_types, (list, tuple)
            ):
                raise ValueError("eligible_request_types must be a list or null")
            if any(not isinstance(name, str) or not name for name in eligible_request_types):
                raise ValueError("eligible_request_types entries must be non-empty strings")
        if credit_release_requests is not None:
            if isinstance(credit_release_requests, (str, bytes)) or not isinstance(
                credit_release_requests, (list, tuple)
            ):
                raise ValueError("credit_release_requests must be a list or null")
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index <= 0 for index in credit_release_requests
            ):
                raise ValueError("credit_release_requests entries must be positive integers")
            if len(set(credit_release_requests)) != len(credit_release_requests):
                raise ValueError("credit_release_requests entries must be unique")
        if initial_min_service_s is not None and (
            not math.isfinite(initial_min_service_s) or initial_min_service_s <= 0.0
        ):
            raise ValueError("initial_min_service_s must be positive or null")
        if not math.isfinite(initial_max_service_s) or initial_max_service_s < 0.0:
            raise ValueError("initial_max_service_s cannot be negative")

        self.quota_every = quota_every
        self.quota_amount = quota_amount
        self.threshold_ratio = threshold_ratio
        self.long_request_ratio = long_request_ratio
        self.arrival_counter = initial_arrival_counter
        self.credits = initial_credits
        self.global_min_service_s = initial_min_service_s
        self.global_max_service_s = initial_max_service_s
        self.max_sacrificial = max_sacrificial
        self.eligible_request_types = None if eligible_request_types is None else frozenset(eligible_request_types)
        self.credit_release_requests = None if credit_release_requests is None else frozenset(credit_release_requests)
        self.sacrificial_count = 0
        self.classified_requests = 0

    def classify(
        self,
        request: RequestView,
        minimum_backend_estimate_s: float,
        *,
        now_s: float = 0.0,
        backends: tuple[BackendView, ...] = (),
    ) -> ClassificationDecision:
        del now_s, backends
        self.classified_requests += 1
        self.arrival_counter += 1
        credits_before = self.credits
        quota_added = 0
        should_release_credit = (
            self.classified_requests in self.credit_release_requests
            if self.credit_release_requests is not None
            else self.arrival_counter % self.quota_every == 0
        )
        if should_release_credit:
            quota_added = self.quota_amount
            self.credits += quota_added

        if self.global_min_service_s is None:
            self.global_min_service_s = minimum_backend_estimate_s
        else:
            self.global_min_service_s = min(self.global_min_service_s, minimum_backend_estimate_s)
        self.global_max_service_s = max(self.global_max_service_s, minimum_backend_estimate_s)

        near_maximum = (
            self.global_max_service_s > 0.0
            and minimum_backend_estimate_s >= self.threshold_ratio * self.global_max_service_s
        )
        clearly_long = True
        if self.long_request_ratio is not None:
            clearly_long = (
                self.global_min_service_s > 0.0
                and minimum_backend_estimate_s >= self.long_request_ratio * self.global_min_service_s
            )
        eligible_request_type = (
            self.eligible_request_types is None or request.request_type_name in self.eligible_request_types
        )
        below_sacrificial_cap = self.max_sacrificial is None or self.sacrificial_count < self.max_sacrificial
        sacrificial = (
            self.credits > 0 and near_maximum and clearly_long and eligible_request_type and below_sacrificial_cap
        )
        if sacrificial:
            self.credits -= 1
            self.sacrificial_count += 1

        return ClassificationDecision(
            priority=Priority.SACRIFICIAL if sacrificial else Priority.NORMAL,
            details={
                "arrival_counter": self.arrival_counter,
                "classified_requests": self.classified_requests,
                "credits_before": credits_before,
                "credits_after": self.credits,
                "quota_added": quota_added,
                "global_min_service_s": self.global_min_service_s,
                "global_max_service_s": self.global_max_service_s,
                "near_maximum": near_maximum,
                "clearly_long": clearly_long,
                "eligible_request_type": eligible_request_type,
                "below_sacrificial_cap": below_sacrificial_cap,
                "sacrificial_count": self.sacrificial_count,
            },
        )


class OnlineCreditTailClassifier:
    """Admit Tail requests without knowing a finite workload's size.

    ``prefix_safe`` earns exactly the number of Tail slots that Hyndman-Fan
    type-7 quantile interpolation can exclude at every observed prefix:

        floor((1 - target_quantile) * (arrivals_seen - 1))

    It is appropriate when the classifier state starts with the measured
    stream, but it never needs to know where that stream will end.

    ``smooth_token_bucket`` earns a fractional token per arrival. With
    ``token_rate=0.04`` and ``credit_capacity=1``, Tail admissions are at least
    25 arrivals apart. This is more conservative, but remains bounded across
    arbitrary measurement-window boundaries in a long-running dispatcher.
    """

    _CREDIT_MODES = frozenset({"prefix_safe", "smooth_token_bucket"})

    def __init__(
        self,
        *,
        credit_mode: str,
        target_quantile: float,
        token_rate: float,
        credit_capacity: float | None,
        threshold_ratio: float,
        long_request_ratio: float | None,
        initial_credits: float = 0.0,
        initial_min_service_s: float | None = None,
        initial_max_service_s: float = 0.0,
        eligible_request_types: list[str] | tuple[str, ...] | None = None,
        congestion_threshold: float = 0.0,
        cooldown_requests: int = 0,
    ) -> None:
        if not isinstance(credit_mode, str) or credit_mode not in self._CREDIT_MODES:
            expected = ", ".join(sorted(self._CREDIT_MODES))
            raise ValueError(f"credit_mode must be one of: {expected}")
        target_quantile = _as_float(target_quantile, "target_quantile")
        token_rate = _as_float(token_rate, "token_rate")
        if credit_capacity is not None:
            credit_capacity = _as_float(credit_capacity, "credit_capacity")
        threshold_ratio = _as_float(threshold_ratio, "threshold_ratio")
        if long_request_ratio is not None:
            long_request_ratio = _as_float(long_request_ratio, "long_request_ratio")
        initial_credits = _as_float(initial_credits, "initial_credits")
        if initial_min_service_s is not None:
            initial_min_service_s = _as_float(initial_min_service_s, "initial_min_service_s")
        initial_max_service_s = _as_float(initial_max_service_s, "initial_max_service_s")
        congestion_threshold = _as_float(congestion_threshold, "congestion_threshold")
        cooldown_requests = _as_int(cooldown_requests, "cooldown_requests")

        if not 0.0 < target_quantile < 1.0:
            raise ValueError("target_quantile must be between 0 and 1")
        if not 0.0 < token_rate < 1.0:
            raise ValueError("token_rate must be between 0 and 1")
        if credit_capacity is not None and credit_capacity < 1.0:
            raise ValueError("credit_capacity must be at least 1 or null")
        if threshold_ratio < 0.0:
            raise ValueError("threshold_ratio cannot be negative")
        if long_request_ratio is not None and long_request_ratio < 1.0:
            raise ValueError("long_request_ratio must be at least 1 or null")
        if initial_credits < 0.0:
            raise ValueError("initial_credits cannot be negative")
        if credit_capacity is not None and initial_credits > credit_capacity:
            raise ValueError("initial_credits cannot exceed credit_capacity")
        if initial_min_service_s is not None and initial_min_service_s <= 0.0:
            raise ValueError("initial_min_service_s must be positive or null")
        if initial_max_service_s < 0.0:
            raise ValueError("initial_max_service_s cannot be negative")
        if congestion_threshold < 0.0:
            raise ValueError("congestion_threshold cannot be negative")
        if cooldown_requests < 0:
            raise ValueError("cooldown_requests cannot be negative")
        if eligible_request_types is not None:
            if isinstance(eligible_request_types, (str, bytes)) or not isinstance(
                eligible_request_types, (list, tuple)
            ):
                raise ValueError("eligible_request_types must be a list or null")
            if any(not isinstance(name, str) or not name for name in eligible_request_types):
                raise ValueError("eligible_request_types entries must be non-empty strings")

        self.credit_mode = credit_mode
        self.target_quantile = target_quantile
        self.token_rate = token_rate
        self.credit_capacity = credit_capacity
        self.threshold_ratio = threshold_ratio
        self.long_request_ratio = long_request_ratio
        self.credits = initial_credits
        self.global_min_service_s = initial_min_service_s
        self.global_max_service_s = initial_max_service_s
        self.eligible_request_types = None if eligible_request_types is None else frozenset(eligible_request_types)
        self.congestion_threshold = congestion_threshold
        self.cooldown_requests = cooldown_requests

        self.classified_requests = 0
        self.sacrificial_count = 0
        self._earned_prefix_allowance = 0
        self._last_sacrificial_request: int | None = None

    def _add_credit(self) -> tuple[float, int | None]:
        safe_allowance: int | None = None
        if self.credit_mode == "prefix_safe":
            safe_allowance = math.floor((1.0 - self.target_quantile) * (self.classified_requests - 1) + 1e-12)
            credit_added = float(max(safe_allowance - self._earned_prefix_allowance, 0))
            self._earned_prefix_allowance = safe_allowance
        else:
            credit_added = self.token_rate

        self.credits += credit_added
        if self.credit_capacity is not None:
            self.credits = min(self.credits, self.credit_capacity)
        return credit_added, safe_allowance

    def classify(
        self,
        request: RequestView,
        minimum_backend_estimate_s: float,
        *,
        now_s: float = 0.0,
        backends: tuple[BackendView, ...] = (),
    ) -> ClassificationDecision:
        del now_s
        self.classified_requests += 1
        credits_before = self.credits
        credit_added, safe_allowance = self._add_credit()

        if self.global_min_service_s is None:
            self.global_min_service_s = minimum_backend_estimate_s
        else:
            self.global_min_service_s = min(self.global_min_service_s, minimum_backend_estimate_s)
        self.global_max_service_s = max(self.global_max_service_s, minimum_backend_estimate_s)

        near_maximum = (
            self.global_max_service_s > 0.0
            and minimum_backend_estimate_s >= self.threshold_ratio * self.global_max_service_s
        )
        clearly_long = True
        if self.long_request_ratio is not None:
            clearly_long = (
                self.global_min_service_s > 0.0
                and minimum_backend_estimate_s >= self.long_request_ratio * self.global_min_service_s
            )
        eligible_request_type = (
            self.eligible_request_types is None or request.request_type_name in self.eligible_request_types
        )

        if backends:
            congestion_ratio = min(
                backend.normal_load_s / max(minimum_backend_estimate_s / backend.speed, 1e-12) for backend in backends
            )
        else:
            congestion_ratio = 0.0
        congestion_eligible = congestion_ratio >= self.congestion_threshold

        cooldown_eligible = (
            self._last_sacrificial_request is None
            or self.classified_requests - self._last_sacrificial_request >= self.cooldown_requests
        )
        has_credit = self.credits >= 1.0 - 1e-12
        sacrificial = (
            has_credit
            and near_maximum
            and clearly_long
            and eligible_request_type
            and congestion_eligible
            and cooldown_eligible
        )
        if sacrificial:
            self.credits = max(self.credits - 1.0, 0.0)
            self.sacrificial_count += 1
            self._last_sacrificial_request = self.classified_requests

        if sacrificial:
            reason = "eligible_credit_consumed"
        elif not has_credit:
            reason = "no_credit"
        elif not near_maximum or not clearly_long or not eligible_request_type:
            reason = "not_long_eligible"
        elif not congestion_eligible:
            reason = "below_congestion_threshold"
        else:
            reason = "cooldown_active"

        return ClassificationDecision(
            priority=Priority.SACRIFICIAL if sacrificial else Priority.NORMAL,
            details={
                "credit_mode": self.credit_mode,
                "classified_requests": self.classified_requests,
                "credits_before": credits_before,
                "credits_after": self.credits,
                "credit_added": credit_added,
                "safe_prefix_allowance": safe_allowance,
                "global_min_service_s": self.global_min_service_s,
                "global_max_service_s": self.global_max_service_s,
                "near_maximum": near_maximum,
                "clearly_long": clearly_long,
                "eligible_request_type": eligible_request_type,
                "congestion_ratio": congestion_ratio,
                "congestion_eligible": congestion_eligible,
                "cooldown_eligible": cooldown_eligible,
                "sacrificial_count": self.sacrificial_count,
                "reason": reason,
            },
        )


def _linear_quantile(values: list[float], quantile: float) -> float:
    """Return a type-7 quantile without depending on the simulation engine."""

    if not values:
        raise ValueError("cannot calculate a quantile from an empty history")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


class QuantileRiskTailClassifier:
    """Admit high-risk arrivals to a bounded Tail queue using online state only.

    The protected-completion risk is the minimum estimated E2E latency if the
    request remains Normal and is sent to the best currently visible backend:

        now - arrival + normal load + tail weight * tail load + service estimate

    The admission threshold is calculated from risks observed *before* the
    current arrival. Configured release positions and the admission end are
    cohort coordinates, not observations of future request contents.
    """

    def __init__(
        self,
        *,
        tail_budget: int,
        risk_quantile: float,
        min_observations: int,
        history_window: int | None = None,
        tail_load_weight: float = 0.0,
        risk_threshold_multiplier: float = 1.0,
        risk_margin_s: float = 0.0,
        admission_start_request: int = 1,
        admission_end_request: int | None = None,
        credit_release_requests: list[int] | tuple[int, ...] | None = None,
        credits_per_release: int = 1,
        eligible_request_types: list[str] | tuple[str, ...] | None = None,
        history_eligible_only: bool = True,
        force_consume_at_end: bool = False,
    ) -> None:
        tail_budget = _as_int(tail_budget, "tail_budget")
        risk_quantile = _as_float(risk_quantile, "risk_quantile")
        min_observations = _as_int(min_observations, "min_observations")
        if history_window is not None:
            history_window = _as_int(history_window, "history_window")
        tail_load_weight = _as_float(tail_load_weight, "tail_load_weight")
        risk_threshold_multiplier = _as_float(risk_threshold_multiplier, "risk_threshold_multiplier")
        risk_margin_s = _as_float(risk_margin_s, "risk_margin_s")
        admission_start_request = _as_int(admission_start_request, "admission_start_request")
        if admission_end_request is not None:
            admission_end_request = _as_int(admission_end_request, "admission_end_request")
        credits_per_release = _as_int(credits_per_release, "credits_per_release")
        history_eligible_only = _as_bool(history_eligible_only, "history_eligible_only")
        force_consume_at_end = _as_bool(force_consume_at_end, "force_consume_at_end")

        if tail_budget < 0:
            raise ValueError("tail_budget cannot be negative")
        if not 0.0 <= risk_quantile <= 1.0:
            raise ValueError("risk_quantile must be between 0 and 1")
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if history_window is not None and history_window <= 0:
            raise ValueError("history_window must be positive or null")
        if history_window is not None and history_window < min_observations:
            raise ValueError("history_window cannot be smaller than min_observations")
        if tail_load_weight < 0.0:
            raise ValueError("tail_load_weight cannot be negative")
        if risk_threshold_multiplier < 0.0:
            raise ValueError("risk_threshold_multiplier cannot be negative")
        if risk_margin_s < 0.0:
            raise ValueError("risk_margin_s cannot be negative")
        if admission_start_request <= 0:
            raise ValueError("admission_start_request must be positive")
        if admission_end_request is not None and admission_end_request < admission_start_request:
            raise ValueError("admission_end_request cannot precede admission_start_request")
        if credits_per_release <= 0:
            raise ValueError("credits_per_release must be positive")

        if eligible_request_types is not None:
            if isinstance(eligible_request_types, (str, bytes)) or not isinstance(
                eligible_request_types, (list, tuple)
            ):
                raise ValueError("eligible_request_types must be a list or null")
            if any(not isinstance(name, str) or not name for name in eligible_request_types):
                raise ValueError("eligible_request_types entries must be non-empty strings")
        if credit_release_requests is not None:
            if isinstance(credit_release_requests, (str, bytes)) or not isinstance(
                credit_release_requests, (list, tuple)
            ):
                raise ValueError("credit_release_requests must be a list or null")
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index <= 0 for index in credit_release_requests
            ):
                raise ValueError("credit_release_requests entries must be positive integers")
            if len(set(credit_release_requests)) != len(credit_release_requests):
                raise ValueError("credit_release_requests entries must be unique")
            if any(index < admission_start_request for index in credit_release_requests):
                raise ValueError("credit_release_requests cannot precede admission_start_request")
            if admission_end_request is not None and any(
                index > admission_end_request for index in credit_release_requests
            ):
                raise ValueError("credit_release_requests cannot follow admission_end_request")

        self.tail_budget = tail_budget
        self.risk_quantile = risk_quantile
        self.min_observations = min_observations
        self.history_window = history_window
        self.tail_load_weight = tail_load_weight
        self.risk_threshold_multiplier = risk_threshold_multiplier
        self.risk_margin_s = risk_margin_s
        self.admission_start_request = admission_start_request
        self.admission_end_request = admission_end_request
        self.credit_release_requests = None if credit_release_requests is None else frozenset(credit_release_requests)
        self.credits_per_release = credits_per_release
        self.eligible_request_types = None if eligible_request_types is None else frozenset(eligible_request_types)
        self.history_eligible_only = history_eligible_only
        self.force_consume_at_end = force_consume_at_end

        self.classified_requests = 0
        self.sacrificial_count = 0
        self.credits = tail_budget if self.credit_release_requests is None else 0
        self._risk_history: list[float] = []

    def _protected_risk_s(
        self,
        *,
        request: RequestView,
        minimum_backend_estimate_s: float,
        now_s: float,
        backends: tuple[BackendView, ...],
    ) -> float:
        if now_s < request.arrival_time_s:
            raise ValueError("classifier now_s cannot precede request arrival_time_s")
        if backends:
            projected_backend_s = min(
                backend.normal_load_s
                + self.tail_load_weight * backend.sacrificial_load_s
                + request.estimated_total_s / backend.speed
                for backend in backends
            )
        else:
            # This fallback keeps direct policy calls deterministic and online;
            # the engine always supplies backend views.
            projected_backend_s = minimum_backend_estimate_s
        return now_s - request.arrival_time_s + projected_backend_s

    def classify(
        self,
        request: RequestView,
        minimum_backend_estimate_s: float,
        *,
        now_s: float = 0.0,
        backends: tuple[BackendView, ...] = (),
    ) -> ClassificationDecision:
        self.classified_requests += 1
        request_position = self.classified_requests

        quota_added = 0
        if self.credit_release_requests is not None and request_position in self.credit_release_requests:
            remaining_budget = max(self.tail_budget - self.sacrificial_count - self.credits, 0)
            quota_added = min(self.credits_per_release, remaining_budget)
            self.credits += quota_added
        credits_before = self.credits

        history = self._risk_history
        history_size_before = len(history)
        threshold_base_s = (
            _linear_quantile(history, self.risk_quantile) if len(history) >= self.min_observations else None
        )
        risk_threshold_s = (
            None if threshold_base_s is None else threshold_base_s * self.risk_threshold_multiplier + self.risk_margin_s
        )
        protected_risk_s = self._protected_risk_s(
            request=request,
            minimum_backend_estimate_s=minimum_backend_estimate_s,
            now_s=now_s,
            backends=backends,
        )
        empirical_percentile = (
            None if not history else sum(observed <= protected_risk_s for observed in history) / len(history)
        )

        inside_window = request_position >= self.admission_start_request and (
            self.admission_end_request is None or request_position <= self.admission_end_request
        )
        eligible_request_type = (
            self.eligible_request_types is None or request.request_type_name in self.eligible_request_types
        )
        below_tail_budget = self.sacrificial_count < self.tail_budget
        has_credit = self.credits > 0
        risk_eligible = risk_threshold_s is not None and protected_risk_s >= risk_threshold_s

        remaining_positions = (
            None
            if self.admission_end_request is None or request_position > self.admission_end_request
            else self.admission_end_request - request_position + 1
        )
        forced_by_end = (
            self.force_consume_at_end and remaining_positions is not None and remaining_positions <= self.credits
        )

        sacrificial = (
            inside_window
            and eligible_request_type
            and below_tail_budget
            and has_credit
            and (risk_eligible or forced_by_end)
        )
        if sacrificial:
            self.credits -= 1
            self.sacrificial_count += 1

        if sacrificial:
            reason = "forced_end_budget" if forced_by_end and not risk_eligible else "risk_threshold_met"
        elif not inside_window:
            reason = "outside_admission_window"
        elif not eligible_request_type:
            reason = "ineligible_request_type"
        elif not below_tail_budget:
            reason = "tail_budget_exhausted"
        elif not has_credit:
            reason = "no_released_credit"
        elif risk_threshold_s is None:
            reason = "insufficient_risk_history"
        else:
            reason = "below_risk_threshold"

        risk_observed = eligible_request_type or not self.history_eligible_only
        if risk_observed:
            # This remains a shadow Normal risk even when the request is
            # admitted to Tail, so admission does not bias later thresholds.
            history.append(protected_risk_s)
            if self.history_window is not None and len(history) > self.history_window:
                del history[: len(history) - self.history_window]

        return ClassificationDecision(
            priority=Priority.SACRIFICIAL if sacrificial else Priority.NORMAL,
            details={
                "classified_requests": request_position,
                "protected_risk_s": protected_risk_s,
                "risk_quantile": self.risk_quantile,
                "risk_history_size_before": history_size_before,
                "risk_threshold_base_s": threshold_base_s,
                "risk_threshold_s": risk_threshold_s,
                "empirical_risk_percentile": empirical_percentile,
                "risk_eligible": risk_eligible,
                "inside_admission_window": inside_window,
                "eligible_request_type": eligible_request_type,
                "risk_observed": risk_observed,
                "below_tail_budget": below_tail_budget,
                "remaining_admission_positions": remaining_positions,
                "forced_by_end": forced_by_end,
                "credits_before": credits_before,
                "credits_after": self.credits,
                "quota_added": quota_added,
                "tail_budget": self.tail_budget,
                "sacrificial_count": self.sacrificial_count,
                "reason": reason,
            },
        )


class WeightedLeastLoadRouter:
    def __init__(
        self,
        *,
        sacrificial_load_factor: float,
        load_view: str,
        update_latency_ema: bool,
    ) -> None:
        sacrificial_load_factor = _as_float(sacrificial_load_factor, "sacrificial_load_factor")
        update_latency_ema = _as_bool(update_latency_ema, "update_latency_ema")
        if not math.isfinite(sacrificial_load_factor) or sacrificial_load_factor < 0.0:
            raise ValueError("sacrificial_load_factor cannot be negative")
        if load_view not in {"assigned", "remaining"}:
            raise ValueError("router.load_view must be 'assigned' or 'remaining'")
        self.sacrificial_load_factor = sacrificial_load_factor
        self.load_view = load_view
        self.update_latency_ema = update_latency_ema

    def choose(self, request: RequestView, backends: tuple[BackendView, ...]) -> RoutingDecision:
        del request

        def score(backend: BackendView) -> tuple[float, float, float, float, str]:
            weighted_load = backend.normal_load_s + self.sacrificial_load_factor * backend.sacrificial_load_s
            weighted_inflight = backend.inflight_normal + self.sacrificial_load_factor * backend.inflight_sacrificial
            return (
                weighted_load,
                weighted_inflight,
                backend.latency_ema_s,
                backend.normal_load_s,
                backend.name,
            )

        selected = min(backends, key=score)
        return RoutingDecision(
            backend_name=selected.name,
            details={"score": list(score(selected)), "load_view": self.load_view},
        )


class ProjectedCompletionRouter:
    """Route by estimated completion time while controlling tail isolation.

    Normal requests treat only a configurable fraction of sacrificial work as
    non-preemptible. Sacrificial requests can either spread by their own
    projected completion time or pack onto an already-tail-loaded backend.
    """

    def __init__(
        self,
        *,
        nonpreemptible_weight: float,
        tail_mode: str,
        load_view: str,
        update_latency_ema: bool,
    ) -> None:
        nonpreemptible_weight = _as_float(nonpreemptible_weight, "nonpreemptible_weight")
        update_latency_ema = _as_bool(update_latency_ema, "update_latency_ema")
        if nonpreemptible_weight < 0.0:
            raise ValueError("nonpreemptible_weight cannot be negative")
        if tail_mode not in {"spread", "pack", "sink"}:
            raise ValueError("router.tail_mode must be 'spread', 'pack', or 'sink'")
        if load_view not in {"assigned", "remaining"}:
            raise ValueError("router.load_view must be 'assigned' or 'remaining'")
        self.nonpreemptible_weight = nonpreemptible_weight
        self.tail_mode = tail_mode
        self.load_view = load_view
        self.update_latency_ema = update_latency_ema

    @staticmethod
    def _request_service_s(request: RequestView, backend: BackendView) -> float:
        return request.estimated_total_s / backend.speed

    def _normal_score(self, request: RequestView, backend: BackendView) -> tuple[float, float, int, str]:
        projected_completion_s = (
            backend.normal_load_s
            + self.nonpreemptible_weight * backend.sacrificial_load_s
            + self._request_service_s(request, backend)
        )
        return (
            projected_completion_s,
            backend.latency_ema_s,
            backend.inflight_normal + backend.inflight_sacrificial,
            backend.name,
        )

    def _spread_tail_score(self, request: RequestView, backend: BackendView) -> tuple[float, float, int, str]:
        projected_completion_s = (
            backend.normal_load_s + backend.sacrificial_load_s + self._request_service_s(request, backend)
        )
        return (
            projected_completion_s,
            backend.latency_ema_s,
            backend.inflight_normal + backend.inflight_sacrificial,
            backend.name,
        )

    def choose(self, request: RequestView, backends: tuple[BackendView, ...]) -> RoutingDecision:
        if request.priority == Priority.NORMAL:
            selected = min(
                backends,
                key=lambda backend: self._normal_score(request, backend),
            )
            selected_score = self._normal_score(request, selected)
            mode = "normal"
        elif self.tail_mode == "spread":
            selected = min(
                backends,
                key=lambda backend: self._spread_tail_score(request, backend),
            )
            selected_score = self._spread_tail_score(request, selected)
            mode = "spread"
        elif self.tail_mode == "pack":
            tail_loaded = tuple(backend for backend in backends if backend.sacrificial_load_s > 0.0)
            if tail_loaded:
                # Once a tail pack exists, isolate later tail work there. Prefer
                # the fullest pack, then the least normal work and lower EMA.
                selected = min(
                    tail_loaded,
                    key=lambda backend: (
                        -backend.sacrificial_load_s,
                        backend.normal_load_s,
                        backend.latency_ema_s,
                        backend.name,
                    ),
                )
                selected_score = (
                    -selected.sacrificial_load_s,
                    selected.normal_load_s,
                    selected.latency_ema_s,
                    selected.name,
                )
            else:
                # Seed the pack on the backend with the earliest projected
                # completion when no sacrificial work exists yet.
                selected = min(
                    backends,
                    key=lambda backend: self._spread_tail_score(request, backend),
                )
                selected_score = self._spread_tail_score(request, selected)
            mode = "pack"
        else:
            # Pure-P95 tail sink: put sacrificial work behind the largest
            # protected backlog, then reinforce an existing tail sink.
            selected = min(
                backends,
                key=lambda backend: (
                    -backend.normal_load_s,
                    -backend.sacrificial_load_s,
                    backend.latency_ema_s,
                    backend.name,
                ),
            )
            selected_score = (
                -selected.normal_load_s,
                -selected.sacrificial_load_s,
                selected.latency_ema_s,
                selected.name,
            )
            mode = "sink"

        return RoutingDecision(
            backend_name=selected.name,
            details={
                "score": list(selected_score),
                "mode": mode,
                "load_view": self.load_view,
                "request_service_s": self._request_service_s(request, selected),
            },
        )


class RoundRobinRouter:
    load_view = "assigned"
    update_latency_ema = False

    def __init__(self) -> None:
        self._next_index = 0

    def choose(self, request: RequestView, backends: tuple[BackendView, ...]) -> RoutingDecision:
        del request
        selected = backends[self._next_index % len(backends)]
        self._next_index += 1
        return RoutingDecision(backend_name=selected.name, details={"round_robin_index": self._next_index - 1})


class LeastInflightRouter:
    load_view = "assigned"
    update_latency_ema = False

    def choose(self, request: RequestView, backends: tuple[BackendView, ...]) -> RoutingDecision:
        del request
        selected = min(
            backends,
            key=lambda backend: (backend.inflight_normal + backend.inflight_sacrificial, backend.name),
        )
        return RoutingDecision(
            backend_name=selected.name, details={"inflight": selected.inflight_normal + selected.inflight_sacrificial}
        )


class FifoScheduler:
    def choose(
        self,
        *,
        now_s: float,
        backend: BackendView,
        incumbent: RequestView | None,
        pending: tuple[RequestView, ...],
    ) -> str | None:
        del now_s, backend
        if incumbent is not None:
            return incumbent.request_id
        if not pending:
            return None
        return min(pending, key=lambda request: request.arrival_seq).request_id


class TwoQueueScheduler:
    def __init__(
        self,
        *,
        normal_order: str,
        sacrificial_order: str,
        preempt_normal_over_sacrificial: bool,
        size_class_order: list[str] | tuple[str, ...] = ("short", "medium", "long"),
        max_bypass: int | None = 3,
        aging_s: float | None = None,
        preemption_hysteresis_s: float = 0.0,
        global_work_stealing: bool = False,
        steal_order: str = "max_risk",
        steal_hysteresis_s: float = 0.0,
        steal_cost_s: float = 0.0,
        global_protected_pull: bool = False,
        protected_pull_order: str = "fifo",
        protected_pull_risk_beta: float = 0.5,
        protected_pull_risk_slack_s: float = 0.0,
        protected_pull_guard_fraction: float = 0.05,
        protected_pull_guard_max: int | None = 1,
        protected_pull_guard_min_pending: int = 20,
        protected_pull_tail_head_start: bool = False,
        protected_pull_cost_s: float = 0.0,
    ) -> None:
        preempt_normal_over_sacrificial = _as_bool(preempt_normal_over_sacrificial, "preempt_normal_over_sacrificial")
        global_work_stealing = _as_bool(global_work_stealing, "global_work_stealing")
        global_protected_pull = _as_bool(global_protected_pull, "global_protected_pull")
        if max_bypass is not None:
            max_bypass = _as_int(max_bypass, "max_bypass")
        if aging_s is not None:
            aging_s = _as_float(aging_s, "aging_s")
        preemption_hysteresis_s = _as_float(preemption_hysteresis_s, "preemption_hysteresis_s")
        steal_hysteresis_s = _as_float(steal_hysteresis_s, "steal_hysteresis_s")
        steal_cost_s = _as_float(steal_cost_s, "steal_cost_s")
        protected_pull_risk_beta = _as_float(
            protected_pull_risk_beta,
            "protected_pull_risk_beta",
        )
        protected_pull_risk_slack_s = _as_float(
            protected_pull_risk_slack_s,
            "protected_pull_risk_slack_s",
        )
        protected_pull_guard_fraction = _as_float(
            protected_pull_guard_fraction,
            "protected_pull_guard_fraction",
        )
        if protected_pull_guard_max is not None:
            protected_pull_guard_max = _as_int(
                protected_pull_guard_max,
                "protected_pull_guard_max",
            )
        protected_pull_guard_min_pending = _as_int(
            protected_pull_guard_min_pending,
            "protected_pull_guard_min_pending",
        )
        protected_pull_tail_head_start = _as_bool(
            protected_pull_tail_head_start,
            "protected_pull_tail_head_start",
        )
        protected_pull_cost_s = _as_float(protected_pull_cost_s, "protected_pull_cost_s")
        valid_normal_orders = {
            "fifo",
            "lifo",
            "srpt",
            "bounded_srpt",
            "least_laxity",
            "arrival_plus_cost",
            "size_class_fifo",
            "bounded_size_class_fifo",
        }
        if normal_order not in valid_normal_orders:
            raise ValueError(
                "scheduler.normal_order must be 'fifo', 'lifo', 'srpt', "
                "'bounded_srpt', 'least_laxity', 'arrival_plus_cost', "
                "'size_class_fifo', or 'bounded_size_class_fifo'"
            )
        if sacrificial_order not in {"fifo", "lifo"}:
            raise ValueError("scheduler.sacrificial_order must be 'fifo' or 'lifo'")
        if steal_order not in {"oldest", "max_risk"}:
            raise ValueError("scheduler.steal_order must be 'oldest' or 'max_risk'")
        if protected_pull_order not in {
            "fifo",
            "max_risk",
            "cost_damped_risk",
            "risk_slack_srpt",
            "guarded_max_risk",
            "arrival_plus_cost",
            "highest_response_ratio",
        }:
            raise ValueError(
                "scheduler.protected_pull_order must be 'fifo', 'max_risk', "
                "'cost_damped_risk', 'risk_slack_srpt', "
                "'guarded_max_risk', 'arrival_plus_cost', or "
                "'highest_response_ratio'"
            )
        if max_bypass is not None and max_bypass < 0:
            raise ValueError("max_bypass cannot be negative")
        if aging_s is not None and aging_s < 0.0:
            raise ValueError("aging_s cannot be negative")
        if preemption_hysteresis_s < 0.0:
            raise ValueError("preemption_hysteresis_s cannot be negative")
        if steal_hysteresis_s < 0.0:
            raise ValueError("steal_hysteresis_s cannot be negative")
        if steal_cost_s < 0.0:
            raise ValueError("steal_cost_s cannot be negative")
        if protected_pull_risk_beta < 0.0:
            raise ValueError("protected_pull_risk_beta cannot be negative")
        if protected_pull_risk_slack_s < 0.0:
            raise ValueError("protected_pull_risk_slack_s cannot be negative")
        if not 0.0 <= protected_pull_guard_fraction < 1.0:
            raise ValueError("protected_pull_guard_fraction must be in [0, 1)")
        if protected_pull_guard_max is not None and protected_pull_guard_max < 0:
            raise ValueError("protected_pull_guard_max cannot be negative")
        if protected_pull_guard_min_pending < 1:
            raise ValueError("protected_pull_guard_min_pending must be positive")
        if protected_pull_cost_s < 0.0:
            raise ValueError("protected_pull_cost_s cannot be negative")
        if global_work_stealing and global_protected_pull:
            raise ValueError("global_work_stealing and global_protected_pull are mutually exclusive")
        if not isinstance(size_class_order, (list, tuple)) or not size_class_order:
            raise ValueError("scheduler.size_class_order must be a non-empty list")
        normalized_size_class_order: list[str] = []
        for index, raw_name in enumerate(size_class_order):
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(f"scheduler.size_class_order[{index}] must be a non-empty string")
            name = raw_name.strip()
            if name in normalized_size_class_order:
                raise ValueError(f"scheduler.size_class_order contains duplicate request type {name!r}")
            normalized_size_class_order.append(name)
        if normal_order in {"bounded_srpt", "bounded_size_class_fifo"} and max_bypass is None and aging_s is None:
            raise ValueError(f"{normal_order} requires max_bypass or aging_s")
        self.normal_order = normal_order
        self.sacrificial_order = sacrificial_order
        self.preempt_normal_over_sacrificial = preempt_normal_over_sacrificial
        self.size_class_order = tuple(normalized_size_class_order)
        self._size_class_ranks = {
            request_type_name: rank for rank, request_type_name in enumerate(self.size_class_order)
        }
        self.max_bypass = max_bypass
        self.aging_s = aging_s
        self.preemption_hysteresis_s = preemption_hysteresis_s
        self.global_work_stealing = global_work_stealing
        self.steal_order = steal_order
        self.steal_hysteresis_s = steal_hysteresis_s
        self.steal_cost_s = steal_cost_s
        self.global_protected_pull = global_protected_pull
        self.protected_pull_order = protected_pull_order
        self.protected_pull_risk_beta = protected_pull_risk_beta
        self.protected_pull_risk_slack_s = protected_pull_risk_slack_s
        self.protected_pull_guard_fraction = protected_pull_guard_fraction
        self.protected_pull_guard_max = protected_pull_guard_max
        self.protected_pull_guard_min_pending = protected_pull_guard_min_pending
        self.protected_pull_tail_head_start = protected_pull_tail_head_start
        self.protected_pull_cost_s = protected_pull_cost_s
        self._bypass_counts: dict[str, int] = {}

    @staticmethod
    def _arrival_ordered(requests: list[RequestView], order: str) -> RequestView:
        reverse = order == "lifo"
        return sorted(requests, key=lambda request: request.arrival_seq, reverse=reverse)[0]

    @staticmethod
    def _shortest_remaining(requests: list[RequestView]) -> RequestView:
        return min(
            requests,
            key=lambda request: (
                request.estimated_remaining_s,
                request.arrival_seq,
                request.request_id,
            ),
        )

    @staticmethod
    def _quantile_risk(request: RequestView, now_s: float) -> float:
        """Estimate latency risk as current age plus remaining service."""

        return now_s - request.arrival_time_s + request.estimated_remaining_s

    def _least_laxity(
        self,
        requests: list[RequestView],
        now_s: float,
    ) -> RequestView:
        return min(
            requests,
            key=lambda request: (
                -self._quantile_risk(request, now_s),
                request.arrival_seq,
                request.request_id,
            ),
        )

    def _cost_damped_risk(
        self,
        requests: list[RequestView],
        now_s: float,
    ) -> RequestView:
        """Prioritize request age while damping estimated service disparity."""

        return min(
            requests,
            key=lambda request: (
                -(now_s - request.arrival_time_s + self.protected_pull_risk_beta * request.estimated_total_s),
                request.arrival_seq,
                request.request_id,
            ),
        )

    def _risk_slack_srpt(
        self,
        requests: list[RequestView],
        now_s: float,
    ) -> RequestView:
        """Use SRPT only inside a configurable band below the maximum risk."""

        risk_by_id = {
            request.request_id: self._quantile_risk(request, now_s)
            for request in requests
        }
        maximum_risk_s = max(risk_by_id.values())
        urgent = [
            request
            for request in requests
            if risk_by_id[request.request_id]
            >= maximum_risk_s - self.protected_pull_risk_slack_s
        ]
        return min(
            urgent,
            key=lambda request: (
                request.estimated_total_s,
                -risk_by_id[request.request_id],
                request.arrival_seq,
                request.request_id,
            ),
        )

    def _guarded_max_risk(
        self,
        requests: list[RequestView],
        now_s: float,
    ) -> RequestView:
        """Leave a bounded highest-risk prefix as online P95 sink requests."""

        ordered = sorted(
            requests,
            key=lambda request: (
                -self._quantile_risk(request, now_s),
                request.arrival_seq,
                request.request_id,
            ),
        )
        guard_count = 0
        if len(ordered) >= self.protected_pull_guard_min_pending:
            guard_count = math.floor(
                len(ordered) * self.protected_pull_guard_fraction + 1e-12
            )
            if self.protected_pull_guard_max is not None:
                guard_count = min(guard_count, self.protected_pull_guard_max)
            guard_count = min(guard_count, len(ordered) - 1)
        return ordered[guard_count]

    def _size_class_fifo(self, requests: list[RequestView]) -> RequestView:
        unknown_rank = len(self._size_class_ranks)
        return min(
            requests,
            key=lambda request: (
                self._size_class_ranks.get(request.request_type_name, unknown_rank),
                request.arrival_seq,
                request.request_id,
            ),
        )

    @staticmethod
    def _arrival_plus_cost(requests: list[RequestView]) -> RequestView:
        return min(
            requests,
            key=lambda request: (
                request.arrival_time_s + request.estimated_total_s,
                request.arrival_seq,
                request.request_id,
            ),
        )

    @staticmethod
    def _highest_response_ratio(
        requests: list[RequestView],
        now_s: float,
    ) -> RequestView:
        """Prefer accumulated wait relative to the request's estimated cost."""

        return min(
            requests,
            key=lambda request: (
                -((now_s - request.arrival_time_s + request.estimated_total_s) / max(request.estimated_total_s, 1e-12)),
                request.arrival_seq,
                request.request_id,
            ),
        )

    def _bounded_choice(
        self,
        requests: list[RequestView],
        now_s: float,
        preferred: RequestView,
    ) -> RequestView:
        forced = [
            request
            for request in requests
            if (self.max_bypass is not None and self._bypass_counts.get(request.request_id, 0) >= self.max_bypass)
            or (self.aging_s is not None and now_s - request.arrival_time_s >= self.aging_s)
        ]
        selected = min(forced, key=lambda request: (request.arrival_seq, request.request_id)) if forced else preferred

        self._bypass_counts.pop(selected.request_id, None)
        pending_ids = {request.request_id for request in requests}
        self._bypass_counts = {
            request_id: count for request_id, count in self._bypass_counts.items() if request_id in pending_ids
        }
        for request in requests:
            if request.arrival_seq < selected.arrival_seq:
                self._bypass_counts[request.request_id] = self._bypass_counts.get(request.request_id, 0) + 1
        return selected

    def _normal_ordered(
        self,
        requests: list[RequestView],
        now_s: float,
    ) -> RequestView:
        if self.normal_order in {"fifo", "lifo"}:
            return self._arrival_ordered(requests, self.normal_order)
        if self.normal_order == "srpt":
            return self._shortest_remaining(requests)
        if self.normal_order == "least_laxity":
            return self._least_laxity(requests, now_s)
        if self.normal_order == "arrival_plus_cost":
            return self._arrival_plus_cost(requests)
        if self.normal_order == "size_class_fifo":
            return self._size_class_fifo(requests)
        if self.normal_order == "bounded_size_class_fifo":
            return self._bounded_choice(
                requests,
                now_s,
                self._size_class_fifo(requests),
            )
        return self._bounded_choice(
            requests,
            now_s,
            self._shortest_remaining(requests),
        )

    def choose_protected_pull(
        self,
        *,
        now_s: float,
        pending: tuple[RequestView, ...],
    ) -> str | None:
        """Select from the never-started online global Normal pool."""

        normal = [request for request in pending if request.priority == Priority.NORMAL]
        if not normal:
            return None
        if self.protected_pull_order == "fifo":
            return self._arrival_ordered(normal, "fifo").request_id
        if self.protected_pull_order == "max_risk":
            return self._least_laxity(normal, now_s).request_id
        if self.protected_pull_order == "cost_damped_risk":
            return self._cost_damped_risk(normal, now_s).request_id
        if self.protected_pull_order == "risk_slack_srpt":
            return self._risk_slack_srpt(normal, now_s).request_id
        if self.protected_pull_order == "guarded_max_risk":
            return self._guarded_max_risk(normal, now_s).request_id
        if self.protected_pull_order == "arrival_plus_cost":
            return self._arrival_plus_cost(normal).request_id
        return self._highest_response_ratio(normal, now_s).request_id

    def choose(
        self,
        *,
        now_s: float,
        backend: BackendView,
        incumbent: RequestView | None,
        pending: tuple[RequestView, ...],
    ) -> str | None:
        del backend
        normal = [request for request in pending if request.priority == Priority.NORMAL]
        sacrificial = [request for request in pending if request.priority == Priority.SACRIFICIAL]

        if incumbent is not None:
            if incumbent.priority == Priority.NORMAL:
                if self.normal_order in {"srpt", "bounded_srpt", "least_laxity"}:
                    selected = self._normal_ordered([*normal, incumbent], now_s)
                    if (
                        self.normal_order == "least_laxity"
                        and selected.request_id != incumbent.request_id
                        and self._quantile_risk(selected, now_s)
                        <= self._quantile_risk(incumbent, now_s) + self.preemption_hysteresis_s
                    ):
                        return incumbent.request_id
                    return selected.request_id
                return incumbent.request_id
            should_preempt = self.preempt_normal_over_sacrificial and bool(normal)
            if not should_preempt:
                return incumbent.request_id

        if normal:
            return self._normal_ordered(normal, now_s).request_id
        if sacrificial:
            return self._arrival_ordered(sacrificial, self.sacrificial_order).request_id
        if incumbent is not None:
            return incumbent.request_id
        return None


def build_classifier(config: ComponentConfig) -> Classifier:
    options = dict(config.options)
    if config.kind == "all_normal":
        _reject_unknown(options, set(), "classifier")
        return AllNormalClassifier()
    if config.kind == "quota_tail":
        allowed = {
            "quota_every",
            "quota_amount",
            "threshold_ratio",
            "long_request_ratio",
            "initial_arrival_counter",
            "initial_credits",
            "initial_min_service_s",
            "initial_max_service_s",
            "max_sacrificial",
            "eligible_request_types",
            "credit_release_requests",
        }
        _reject_unknown(options, allowed, "classifier")
        long_ratio = options.get("long_request_ratio", 1.5)
        return QuotaTailClassifier(
            quota_every=_as_int(options.get("quota_every", 20), "classifier.quota_every"),
            quota_amount=_as_int(options.get("quota_amount", 1), "classifier.quota_amount"),
            threshold_ratio=_as_float(options.get("threshold_ratio", 0.8), "classifier.threshold_ratio"),
            long_request_ratio=(None if long_ratio is None else _as_float(long_ratio, "classifier.long_request_ratio")),
            initial_arrival_counter=_as_int(
                options.get("initial_arrival_counter", 0), "classifier.initial_arrival_counter"
            ),
            initial_credits=_as_int(options.get("initial_credits", 0), "classifier.initial_credits"),
            initial_min_service_s=(
                None
                if options.get("initial_min_service_s") is None
                else _as_float(options["initial_min_service_s"], "classifier.initial_min_service_s")
            ),
            initial_max_service_s=_as_float(
                options.get("initial_max_service_s", 0.0), "classifier.initial_max_service_s"
            ),
            max_sacrificial=(
                None
                if options.get("max_sacrificial") is None
                else _as_int(options["max_sacrificial"], "classifier.max_sacrificial")
            ),
            eligible_request_types=options.get("eligible_request_types"),
            credit_release_requests=options.get("credit_release_requests"),
        )
    if config.kind == "online_credit_tail":
        allowed = {
            "credit_mode",
            "target_quantile",
            "token_rate",
            "credit_capacity",
            "threshold_ratio",
            "long_request_ratio",
            "initial_credits",
            "initial_min_service_s",
            "initial_max_service_s",
            "eligible_request_types",
            "congestion_threshold",
            "cooldown_requests",
        }
        _reject_unknown(options, allowed, "classifier")
        long_ratio = options.get("long_request_ratio", 1.5)
        credit_capacity = options.get("credit_capacity")
        return OnlineCreditTailClassifier(
            credit_mode=str(options.get("credit_mode", "prefix_safe")),
            target_quantile=_as_float(
                options.get("target_quantile", 0.95),
                "classifier.target_quantile",
            ),
            token_rate=_as_float(options.get("token_rate", 0.04), "classifier.token_rate"),
            credit_capacity=(
                None if credit_capacity is None else _as_float(credit_capacity, "classifier.credit_capacity")
            ),
            threshold_ratio=_as_float(
                options.get("threshold_ratio", 0.8),
                "classifier.threshold_ratio",
            ),
            long_request_ratio=(None if long_ratio is None else _as_float(long_ratio, "classifier.long_request_ratio")),
            initial_credits=_as_float(
                options.get("initial_credits", 0.0),
                "classifier.initial_credits",
            ),
            initial_min_service_s=(
                None
                if options.get("initial_min_service_s") is None
                else _as_float(
                    options["initial_min_service_s"],
                    "classifier.initial_min_service_s",
                )
            ),
            initial_max_service_s=_as_float(
                options.get("initial_max_service_s", 0.0),
                "classifier.initial_max_service_s",
            ),
            eligible_request_types=options.get("eligible_request_types"),
            congestion_threshold=_as_float(
                options.get("congestion_threshold", 0.0),
                "classifier.congestion_threshold",
            ),
            cooldown_requests=_as_int(
                options.get("cooldown_requests", 0),
                "classifier.cooldown_requests",
            ),
        )
    if config.kind == "quantile_risk_tail":
        allowed = {
            "tail_budget",
            "risk_quantile",
            "min_observations",
            "history_window",
            "tail_load_weight",
            "risk_threshold_multiplier",
            "risk_margin_s",
            "admission_start_request",
            "admission_end_request",
            "credit_release_requests",
            "credits_per_release",
            "eligible_request_types",
            "history_eligible_only",
            "force_consume_at_end",
        }
        _reject_unknown(options, allowed, "classifier")
        history_window = options.get("history_window")
        admission_end_request = options.get("admission_end_request")
        return QuantileRiskTailClassifier(
            tail_budget=_as_int(options.get("tail_budget", 0), "classifier.tail_budget"),
            risk_quantile=_as_float(options.get("risk_quantile", 0.95), "classifier.risk_quantile"),
            min_observations=_as_int(options.get("min_observations", 10), "classifier.min_observations"),
            history_window=(None if history_window is None else _as_int(history_window, "classifier.history_window")),
            tail_load_weight=_as_float(options.get("tail_load_weight", 0.0), "classifier.tail_load_weight"),
            risk_threshold_multiplier=_as_float(
                options.get("risk_threshold_multiplier", 1.0),
                "classifier.risk_threshold_multiplier",
            ),
            risk_margin_s=_as_float(options.get("risk_margin_s", 0.0), "classifier.risk_margin_s"),
            admission_start_request=_as_int(
                options.get("admission_start_request", 1),
                "classifier.admission_start_request",
            ),
            admission_end_request=(
                None
                if admission_end_request is None
                else _as_int(admission_end_request, "classifier.admission_end_request")
            ),
            credit_release_requests=options.get("credit_release_requests"),
            credits_per_release=_as_int(
                options.get("credits_per_release", 1),
                "classifier.credits_per_release",
            ),
            eligible_request_types=options.get("eligible_request_types"),
            history_eligible_only=_as_bool(
                options.get("history_eligible_only", True),
                "classifier.history_eligible_only",
            ),
            force_consume_at_end=_as_bool(
                options.get("force_consume_at_end", False),
                "classifier.force_consume_at_end",
            ),
        )
    raise ValueError(f"unknown classifier type: {config.kind!r}")


def build_router(config: ComponentConfig) -> Router:
    options = dict(config.options)
    if config.kind == "weighted_least_load":
        _reject_unknown(options, {"sacrificial_load_factor", "load_view", "update_latency_ema"}, "router")
        return WeightedLeastLoadRouter(
            sacrificial_load_factor=_as_float(
                options.get("sacrificial_load_factor", 0.1), "router.sacrificial_load_factor"
            ),
            load_view=str(options.get("load_view", "assigned")),
            update_latency_ema=_as_bool(options.get("update_latency_ema", False), "router.update_latency_ema"),
        )
    if config.kind == "projected_completion":
        _reject_unknown(
            options,
            {
                "nonpreemptible_weight",
                "tail_mode",
                "load_view",
                "update_latency_ema",
            },
            "router",
        )
        return ProjectedCompletionRouter(
            nonpreemptible_weight=_as_float(
                options.get("nonpreemptible_weight", 0.1),
                "router.nonpreemptible_weight",
            ),
            tail_mode=str(options.get("tail_mode", "spread")),
            load_view=str(options.get("load_view", "assigned")),
            update_latency_ema=_as_bool(
                options.get("update_latency_ema", False),
                "router.update_latency_ema",
            ),
        )
    if config.kind == "round_robin":
        _reject_unknown(options, set(), "router")
        return RoundRobinRouter()
    if config.kind == "least_inflight":
        _reject_unknown(options, set(), "router")
        return LeastInflightRouter()
    raise ValueError(f"unknown router type: {config.kind!r}")


def build_scheduler(config: ComponentConfig) -> LocalScheduler:
    options = dict(config.options)
    if config.kind == "fifo":
        _reject_unknown(options, set(), "scheduler")
        return FifoScheduler()
    if config.kind == "two_queue":
        _reject_unknown(
            options,
            {
                "normal_order",
                "sacrificial_order",
                "preempt_normal_over_sacrificial",
                "size_class_order",
                "max_bypass",
                "aging_s",
                "preemption_hysteresis_s",
                "global_work_stealing",
                "steal_order",
                "steal_hysteresis_s",
                "steal_cost_s",
                "global_protected_pull",
                "protected_pull_order",
                "protected_pull_risk_beta",
                "protected_pull_risk_slack_s",
                "protected_pull_guard_fraction",
                "protected_pull_guard_max",
                "protected_pull_guard_min_pending",
                "protected_pull_tail_head_start",
                "protected_pull_cost_s",
            },
            "scheduler",
        )
        return TwoQueueScheduler(
            normal_order=str(options.get("normal_order", "fifo")),
            sacrificial_order=str(options.get("sacrificial_order", "lifo")),
            preempt_normal_over_sacrificial=_as_bool(
                options.get("preempt_normal_over_sacrificial", True),
                "scheduler.preempt_normal_over_sacrificial",
            ),
            size_class_order=options.get("size_class_order", ["short", "medium", "long"]),
            max_bypass=(
                None
                if options.get("max_bypass", 3) is None
                else _as_int(options.get("max_bypass", 3), "scheduler.max_bypass")
            ),
            aging_s=(None if options.get("aging_s") is None else _as_float(options["aging_s"], "scheduler.aging_s")),
            preemption_hysteresis_s=_as_float(
                options.get("preemption_hysteresis_s", 0.0),
                "scheduler.preemption_hysteresis_s",
            ),
            global_work_stealing=_as_bool(
                options.get("global_work_stealing", False),
                "scheduler.global_work_stealing",
            ),
            steal_order=str(options.get("steal_order", "max_risk")),
            steal_hysteresis_s=_as_float(
                options.get("steal_hysteresis_s", 0.0),
                "scheduler.steal_hysteresis_s",
            ),
            steal_cost_s=_as_float(
                options.get("steal_cost_s", 0.0),
                "scheduler.steal_cost_s",
            ),
            global_protected_pull=_as_bool(
                options.get("global_protected_pull", False),
                "scheduler.global_protected_pull",
            ),
            protected_pull_order=str(options.get("protected_pull_order", "fifo")),
            protected_pull_risk_beta=_as_float(
                options.get("protected_pull_risk_beta", 0.5),
                "scheduler.protected_pull_risk_beta",
            ),
            protected_pull_risk_slack_s=_as_float(
                options.get("protected_pull_risk_slack_s", 0.0),
                "scheduler.protected_pull_risk_slack_s",
            ),
            protected_pull_guard_fraction=_as_float(
                options.get("protected_pull_guard_fraction", 0.05),
                "scheduler.protected_pull_guard_fraction",
            ),
            protected_pull_guard_max=(
                None
                if options.get("protected_pull_guard_max", 1) is None
                else _as_int(
                    options.get("protected_pull_guard_max", 1),
                    "scheduler.protected_pull_guard_max",
                )
            ),
            protected_pull_guard_min_pending=_as_int(
                options.get("protected_pull_guard_min_pending", 20),
                "scheduler.protected_pull_guard_min_pending",
            ),
            protected_pull_tail_head_start=_as_bool(
                options.get("protected_pull_tail_head_start", False),
                "scheduler.protected_pull_tail_head_start",
            ),
            protected_pull_cost_s=_as_float(
                options.get("protected_pull_cost_s", 0.0),
                "scheduler.protected_pull_cost_s",
            ),
        )
    raise ValueError(f"unknown scheduler type: {config.kind!r}")
