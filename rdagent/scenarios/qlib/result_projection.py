from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

from rdagent.scenarios.qlib.ashare_semantics import (
    QLIB_ASHARE_MODEL_FEEDBACK_PROMPT_INVALID_FAILURE,
    QLIB_ASHARE_MODEL_FEEDBACK_PROMPT_METRIC_PATHS,
    QLIB_ASHARE_MODEL_FEEDBACK_PROMPT_MISSING_FAILURE,
    QLIB_ASHARE_TRACE_PROMPT_INVALID_FAILURE,
    QLIB_ASHARE_TRACE_PROMPT_METRIC_PATHS,
    QLIB_ASHARE_TRACE_PROMPT_MISSING_FAILURE,
    QlibAshareSemanticContractError,
)


def require_single_metric_value(
    result: Any,
    metric_name: str,
    *,
    missing_failure: str,
    invalid_failure: str,
) -> float:
    if result is None:
        raise QlibAshareSemanticContractError(missing_failure)
    try:
        frame = pd.DataFrame(result)
    except Exception as exc:  # noqa: BLE001
        raise QlibAshareSemanticContractError(invalid_failure) from exc
    if metric_name not in frame.index:
        raise QlibAshareSemanticContractError(missing_failure)
    metric_row = frame.loc[metric_name]
    if isinstance(metric_row, pd.DataFrame):
        raise QlibAshareSemanticContractError(invalid_failure)
    candidates = metric_row.tolist() if isinstance(metric_row, pd.Series) else [metric_row]
    values: list[float] = []
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError) as exc:
            raise QlibAshareSemanticContractError(invalid_failure) from exc
        if not math.isfinite(value):
            raise QlibAshareSemanticContractError(invalid_failure)
        values.append(value)
    if len(values) != 1:
        raise QlibAshareSemanticContractError(invalid_failure)
    return values[0]


def format_metric_result_for_prompt(
    result: Any,
    metric_paths: Iterable[str],
    *,
    missing_failure: str,
    invalid_failure: str,
    missing_result_value: str | None = None,
) -> str:
    if result is None and missing_result_value is not None:
        return missing_result_value
    results = []
    for metric in metric_paths:
        value = require_single_metric_value(
            result,
            metric,
            missing_failure=missing_failure,
            invalid_failure=invalid_failure,
        )
        results.append(f"{metric}: {value:.6f}")
    return "; ".join(results)


def format_trace_result_for_prompt(result: Any, *, missing_result_value: str | None = None) -> str:
    return format_metric_result_for_prompt(
        result,
        QLIB_ASHARE_TRACE_PROMPT_METRIC_PATHS,
        missing_failure=QLIB_ASHARE_TRACE_PROMPT_MISSING_FAILURE,
        invalid_failure=QLIB_ASHARE_TRACE_PROMPT_INVALID_FAILURE,
        missing_result_value=missing_result_value,
    )


def format_model_feedback_result_for_prompt(result: Any, *, missing_result_value: str | None = None) -> str:
    return format_metric_result_for_prompt(
        result,
        QLIB_ASHARE_MODEL_FEEDBACK_PROMPT_METRIC_PATHS,
        missing_failure=QLIB_ASHARE_MODEL_FEEDBACK_PROMPT_MISSING_FAILURE,
        invalid_failure=QLIB_ASHARE_MODEL_FEEDBACK_PROMPT_INVALID_FAILURE,
        missing_result_value=missing_result_value,
    )
