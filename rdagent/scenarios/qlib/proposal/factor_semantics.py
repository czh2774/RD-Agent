from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rdagent.scenarios.qlib.ashare_semantics import (
    QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES,
    QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS,
)

_FACTOR_SETUP_LANGUAGE_PATTERNS = (
    re.compile(r"\bsetup\b", re.IGNORECASE),
    re.compile(r"\bconfiguration\b", re.IGNORECASE),
    re.compile(r"\bcalibration\b", re.IGNORECASE),
    re.compile(r"\benvironment(?:al)?\b", re.IGNORECASE),
    re.compile(r"\bdebug(?:ging)?\b", re.IGNORECASE),
)
_FACTOR_GENERIC_SCREENING_PATTERNS = (
    re.compile(r"\bfirst-?order controllable factors?\b", re.IGNORECASE),
    re.compile(r"\beasiest-to-change variables?\b", re.IGNORECASE),
    re.compile(r"\bscreen(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bmain effects?\b", re.IGNORECASE),
)
_QLIB_ASHARE_DAILY_FIELD_TERMS = tuple(field.lstrip("$") for field in QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS)
_FACTOR_QLIB_DAILY_DATA_HINT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in _QLIB_ASHARE_DAILY_FIELD_TERMS) + r")\b",
    re.IGNORECASE,
)
_FACTOR_DAILY_POINT_IN_TIME_DATA_HINT_PATTERN = re.compile(
    r"\b(daily point[- ]in[- ]time|point[- ]in[- ]time)\b",
    re.IGNORECASE,
)
_FACTOR_MARKET_DATA_HINT_PATTERNS = (
    _FACTOR_QLIB_DAILY_DATA_HINT_PATTERN,
    _FACTOR_DAILY_POINT_IN_TIME_DATA_HINT_PATTERN,
    re.compile(r"\b(price-volume|price volume|momentum|reversal|volatility|price)\b", re.IGNORECASE),
    re.compile(r"\b(alpha|factor|signal|ts_code)\b", re.IGNORECASE),
)
_FACTOR_FORBIDDEN_DEFAULT_SOURCE_PATTERNS = (
    re.compile(r"\b(minute|intraday|high[- ]frequency|tick)\b", re.IGNORECASE),
    re.compile(r"\b(analyst|consensus|expectation|estimate)\b", re.IGNORECASE),
    re.compile(r"\b(unregistered|vendor)\b", re.IGNORECASE),
    re.compile(r"\bturnover\b", re.IGNORECASE),
)


def _text_has_qlib_daily_data_hint(text: str) -> bool:
    return any(pattern.search(text) for pattern in _FACTOR_MARKET_DATA_HINT_PATTERNS)


def _text_has_concrete_qlib_source_hint(text: str) -> bool:
    return bool(
        _FACTOR_QLIB_DAILY_DATA_HINT_PATTERN.search(text) or _FACTOR_DAILY_POINT_IN_TIME_DATA_HINT_PATTERN.search(text)
    )


def _raise_forbidden_default_source_error(text: str, *, subject: str) -> None:
    if any(pattern.search(text) for pattern in _FACTOR_FORBIDDEN_DEFAULT_SOURCE_PATTERNS):
        forbidden = ", ".join(QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES)
        raise ValueError(
            f"{subject} must stay within the Qlib daily A-share research data boundary; "
            f"forbidden default sources include {forbidden}."
        )


def build_qlib_ashare_factor_task_source_boundary() -> str:
    allowed_fields = ", ".join(QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS)
    forbidden_sources = ", ".join(QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES)
    return (
        "Qlib daily A-share research data boundary: use only registered daily Qlib fields "
        f"({allowed_fields}) or explicitly supplied daily point-in-time fields. "
        f"Forbidden default sources: {forbidden_sources}. "
        "Do not infer unregistered fields during factor implementation or code review."
    )


def _default_hypothesis_response_normalizer(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Qlib factor hypothesis response must be a JSON object.")

    normalized_payload = dict(payload)
    primary_hypothesis = normalized_payload.get("primary_hypothesis")
    if isinstance(primary_hypothesis, dict):
        hypothesis_statement = primary_hypothesis.get("statement")
        hypothesis_reason = primary_hypothesis.get("why_this_first")
        if hypothesis_statement is not None:
            normalized_payload["hypothesis"] = hypothesis_statement
        if hypothesis_reason is not None:
            normalized_payload["reason"] = hypothesis_reason

    allowed_keys = {
        "hypothesis",
        "reason",
        "concise_reason",
        "concise_observation",
        "concise_justification",
        "concise_knowledge",
    }
    filtered_payload = {key: value for key, value in normalized_payload.items() if key in allowed_keys}
    if not isinstance(filtered_payload.get("hypothesis"), str) or not isinstance(filtered_payload.get("reason"), str):
        raise ValueError("Qlib factor hypothesis response must include string hypothesis and reason fields.")
    return filtered_payload


def validate_qlib_factor_hypothesis_response(
    payload: dict[str, Any],
    *,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response_normalizer = normalizer or _default_hypothesis_response_normalizer
    normalized = response_normalizer(payload)
    hypothesis = str(normalized.get("hypothesis") or "")
    reason = str(normalized.get("reason") or "")
    combined_text = f"{hypothesis}\n{reason}"

    if any(pattern.search(combined_text) for pattern in _FACTOR_SETUP_LANGUAGE_PATTERNS):
        raise ValueError(
            "Qlib factor hypotheses must stay focused on market-data alpha directions, not setup, configuration, calibration, or debugging plans."
        )
    if any(pattern.search(combined_text) for pattern in _FACTOR_GENERIC_SCREENING_PATTERNS):
        raise ValueError(
            "Qlib factor hypotheses must include a concrete market-data factor idea rather than generic screening language."
        )
    _raise_forbidden_default_source_error(combined_text, subject="Qlib factor hypotheses")
    if not _text_has_qlib_daily_data_hint(hypothesis):
        allowed_fields = ", ".join(QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS)
        raise ValueError(
            "Qlib factor hypotheses must include concrete market-data alpha directions grounded in "
            f"registered daily Qlib A-share fields ({allowed_fields}) or explicitly supplied daily point-in-time fields."
        )
    return normalized


def validate_qlib_factor_experiment_response(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Qlib factor experiment response must be a non-empty JSON object.")

    normalized_response: dict[str, dict[str, Any]] = {}
    for factor_name, factor_payload in payload.items():
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError("Qlib factor experiment response must use non-empty string factor names.")
        if not isinstance(factor_payload, dict):
            raise ValueError(f"Qlib factor experiment output for {factor_name!r} must be a JSON object.")

        description = factor_payload.get("description")
        formulation = factor_payload.get("formulation")
        variables = factor_payload.get("variables")
        if not isinstance(description, str) or not isinstance(formulation, str) or not isinstance(variables, dict):
            raise ValueError(
                f"Qlib factor experiment output for {factor_name!r} must include string description, "
                "string formulation, and object variables fields."
            )
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in variables.items()):
            raise ValueError(f"Qlib factor experiment variables for {factor_name!r} must be string-to-string entries.")

        combined_text = "\n".join(
            [factor_name, description, formulation] + [f"{key}: {value}" for key, value in variables.items()]
        )
        _raise_forbidden_default_source_error(combined_text, subject="Qlib factor experiment outputs")
        if not _text_has_concrete_qlib_source_hint(combined_text):
            allowed_fields = ", ".join(QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS)
            raise ValueError(
                f"Qlib factor experiment output for {factor_name!r} must bind its description, formulation, "
                f"and variables to registered daily Qlib A-share fields ({allowed_fields}) or explicitly supplied "
                "daily point-in-time fields."
            )

        normalized_response[factor_name] = {
            "description": description,
            "formulation": formulation,
            "variables": dict(variables),
        }
    return normalized_response
