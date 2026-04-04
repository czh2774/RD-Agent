import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def _render_feedback_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        rendered = [_render_feedback_value(item) for item in value]
        return "; ".join(item for item in rendered if item)
    if isinstance(value, dict):
        rendered_pairs = []
        for key, nested_value in value.items():
            rendered_value = _render_feedback_value(nested_value)
            if rendered_value:
                rendered_pairs.append(f"{key}: {rendered_value}")
        return "; ".join(rendered_pairs)
    return str(value)


def _first_nonempty_feedback_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in payload:
            continue
        rendered = _render_feedback_value(payload[key])
        if rendered:
            return rendered
    return ""


def _find_nested_feedback_value(payload: Any, target_key: str) -> str:
    if isinstance(payload, dict):
        if target_key in payload:
            rendered = _render_feedback_value(payload[target_key])
            if rendered:
                return rendered
        for nested_value in payload.values():
            rendered = _find_nested_feedback_value(nested_value, target_key)
            if rendered:
                return rendered
    elif isinstance(payload, list):
        for nested_value in payload:
            rendered = _find_nested_feedback_value(nested_value, target_key)
            if rendered:
                return rendered
    return ""


def _first_nonempty_feedback_value_recursive(payload: dict[str, Any], *keys: str) -> str:
    direct = _first_nonempty_feedback_value(payload, *keys)
    if direct:
        return direct
    for key in keys:
        rendered = _find_nested_feedback_value(payload, key)
        if rendered:
            return rendered
    return ""


def _extract_metric_value(result: Any, metric_name: str) -> float | None:
    if result is None:
        return None
    try:
        frame = pd.DataFrame(result)
    except Exception:  # noqa: BLE001
        return None
    if metric_name not in frame.index:
        return None
    metric_row = frame.loc[metric_name]
    if isinstance(metric_row, pd.Series):
        for candidate in metric_row.tolist():
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return None
    try:
        return float(metric_row)
    except (TypeError, ValueError):
        return None


def infer_replace_best_result(
    current_result: dict[str, Any] | None,
    sota_result: dict[str, Any] | None,
) -> bool:
    metric_name = "1day.excess_return_with_cost.annualized_return"
    current_metric = _extract_metric_value(current_result, metric_name)
    sota_metric = _extract_metric_value(sota_result, metric_name)
    if current_metric is None:
        return False
    if sota_metric is None:
        return True
    return current_metric > sota_metric


def normalize_feedback_response(
    payload: dict[str, Any],
    *,
    current_result: dict[str, Any] | None,
    sota_result: dict[str, Any] | None,
) -> dict[str, Any]:
    observations = _first_nonempty_feedback_value_recursive(
        payload,
        "Observations",
        "observations",
        "comparison_to_sota",
        "sota_comparison",
        "conclusion",
    )
    if not observations:
        observation_parts: list[str] = []
        sota_comparison = payload.get("sota_comparison")
        if isinstance(sota_comparison, dict) and sota_comparison.get("summary"):
            observation_parts.append(str(sota_comparison["summary"]))
        limitations = None
        hypothesis_assessment = payload.get("hypothesis_assessment")
        if isinstance(hypothesis_assessment, dict):
            limitations = hypothesis_assessment.get("limitations")
        if isinstance(limitations, list) and limitations:
            observation_parts.append("Limitations: " + "; ".join(str(item) for item in limitations))
        observations = " ".join(observation_parts) or "No observations provided"

    hypothesis_evaluation = _first_nonempty_feedback_value_recursive(
        payload,
        "Feedback for Hypothesis",
        "hypothesis_evaluation",
        "hypothesis_assessment",
    )
    if not hypothesis_evaluation:
        hypothesis_assessment = payload.get("hypothesis_assessment")
        if isinstance(hypothesis_assessment, dict) and hypothesis_assessment:
            hypothesis_evaluation = "; ".join(
                f"{key}={value}" for key, value in hypothesis_assessment.items()
            )
        else:
            hypothesis_evaluation = "No feedback provided"

    new_hypothesis = _first_nonempty_feedback_value_recursive(
        payload,
        "New Hypothesis",
        "new_hypothesis",
        "recommended_next_step",
        "recommendation",
        "next_hypothesis",
        "next_step",
    )
    if not new_hypothesis:
        hypothesis_assessment = payload.get("hypothesis_assessment")
        limitations = (
            hypothesis_assessment.get("limitations")
            if isinstance(hypothesis_assessment, dict)
            else None
        )
        if isinstance(limitations, list) and limitations:
            new_hypothesis = "Address the current limitations: " + "; ".join(
                str(item) for item in limitations
            )
        else:
            new_hypothesis = "No new hypothesis provided"

    reasoning = _first_nonempty_feedback_value_recursive(
        payload,
        "Reasoning",
        "reasoning",
        "conclusion",
        "comparison_to_sota",
        "sota_comparison",
    )
    if not reasoning:
        conclusion = payload.get("conclusion")
        if isinstance(conclusion, dict) and conclusion:
            reasoning = "; ".join(f"{key}={value}" for key, value in conclusion.items())
        else:
            reasoning = "No reasoning provided"

    decision_value = None
    for key in ("Replace Best Result", "replace_best_result", "Decision", "decision"):
        if key in payload:
            decision_value = payload[key]
            break
    if decision_value is None:
        decision = infer_replace_best_result(current_result, sota_result)
        logger.warning(
            "Feedback JSON did not include an explicit replace/decision flag; "
            "inferring it from annualized return improvement."
        )
    else:
        decision = convert2bool(_render_feedback_value(decision_value))

    return {
        "Observations": str(observations),
        "Feedback for Hypothesis": str(hypothesis_evaluation),
        "New Hypothesis": str(new_hypothesis),
        "Reasoning": str(reasoning),
        "Decision": decision,
    }


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = normalize_feedback_response(
            json.loads(response),
            current_result=current_result,
            sota_result=sota_result,
        )

        return HypothesisFeedback(
            observations=response_json["Observations"],
            hypothesis_evaluation=response_json["Feedback for Hypothesis"],
            new_hypothesis=response_json["New Hypothesis"],
            reason=response_json["Reasoning"],
            decision=response_json["Decision"],
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response)

        # Call the APIBackend to generate the response for hypothesis feedback
        response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response_hypothesis)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
