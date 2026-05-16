from __future__ import annotations

from typing import Any

from rdagent.core.proposal import Trace
from rdagent.scenarios.qlib.result_projection import format_trace_result_for_prompt
from rdagent.utils.agent.tpl import T


def _trace_prompt_result(experiment: Any) -> str | None:
    if experiment is None or getattr(experiment, "result", None) is None:
        return None
    return format_trace_result_for_prompt(experiment.result)


def render_hypothesis_and_feedback(trace: Trace) -> str:
    return T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
        trace=trace,
        qlib_ashare_trace_prompt_results=[_trace_prompt_result(experiment) for experiment, _ in trace.hist],
    )


def render_last_hypothesis_and_feedback(experiment: Any, feedback: Any) -> str:
    return T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
        experiment=experiment,
        feedback=feedback,
        qlib_ashare_trace_prompt_result=_trace_prompt_result(experiment),
    )


def render_sota_hypothesis_and_feedback(experiment: Any, feedback: Any) -> str:
    return T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
        experiment=experiment,
        feedback=feedback,
        qlib_ashare_trace_prompt_result=_trace_prompt_result(experiment),
    )
