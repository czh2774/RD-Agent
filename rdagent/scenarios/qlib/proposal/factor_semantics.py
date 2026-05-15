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
_FACTOR_MARKET_DATA_HINT_PATTERNS = (
    _FACTOR_QLIB_DAILY_DATA_HINT_PATTERN,
    re.compile(r"\b(price-volume|price volume|momentum|reversal|volatility|price)\b", re.IGNORECASE),
    re.compile(r"\b(alpha|factor|signal|ts_code)\b", re.IGNORECASE),
)
_FACTOR_FORBIDDEN_DEFAULT_SOURCE_PATTERNS = (
    re.compile(r"\b(minute|intraday|high[- ]frequency|tick)\b", re.IGNORECASE),
    re.compile(r"\b(analyst|consensus|expectation|estimate)\b", re.IGNORECASE),
    re.compile(r"\b(unregistered|vendor)\b", re.IGNORECASE),
    re.compile(r"\bturnover\b", re.IGNORECASE),
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
    if any(pattern.search(combined_text) for pattern in _FACTOR_FORBIDDEN_DEFAULT_SOURCE_PATTERNS):
        forbidden = ", ".join(QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES) + ", turnover"
        raise ValueError(
            "Qlib factor hypotheses must stay within the Qlib daily A-share research data boundary; "
            f"forbidden default sources include {forbidden}."
        )
    if not any(pattern.search(hypothesis) for pattern in _FACTOR_MARKET_DATA_HINT_PATTERNS):
        allowed_fields = ", ".join(QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS)
        raise ValueError(
            "Qlib factor hypotheses must include concrete market-data alpha directions grounded in "
            f"registered daily Qlib A-share fields ({allowed_fields}) or explicitly supplied daily point-in-time fields."
        )
    return normalized
