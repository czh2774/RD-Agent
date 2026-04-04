from abc import abstractmethod
import json
from typing import Any, Tuple

from pydantic import BaseModel, ConfigDict

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import (
    ExperimentPlan,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


class HypothesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    reason: str
    concise_reason: str | None = None
    concise_observation: str | None = None
    concise_justification: str | None = None
    concise_knowledge: str | None = None


class QuantHypothesisResponse(HypothesisResponse):
    action: str


def ensure_hypothesis_response_dict(
    payload: dict[str, Any],
    *,
    require_action: bool = False,
    default_action: str | None = None,
) -> dict[str, Any]:
    normalized_payload = dict(payload)

    primary_hypothesis = normalized_payload.get("primary_hypothesis")
    if isinstance(primary_hypothesis, dict):
        hypothesis_statement = primary_hypothesis.get("statement")
        hypothesis_reason = primary_hypothesis.get("why_this_first")
        if hypothesis_statement is not None:
            normalized_payload["hypothesis"] = hypothesis_statement
        if hypothesis_reason is not None:
            normalized_payload["reason"] = hypothesis_reason

    if require_action and not normalized_payload.get("action") and default_action:
        normalized_payload["action"] = default_action

    allowed_keys = {
        "hypothesis",
        "reason",
        "concise_reason",
        "concise_observation",
        "concise_justification",
        "concise_knowledge",
    }
    if require_action:
        allowed_keys.add("action")

    filtered_payload = {
        key: value for key, value in normalized_payload.items() if key in allowed_keys
    }

    response_type = QuantHypothesisResponse if require_action else HypothesisResponse
    return response_type.model_validate(filtered_payload).model_dump()


class LLMHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...

    def hypothesis_response_type(self, context_dict: dict[str, Any]) -> type[BaseModel]:
        require_action = '"action"' in str(context_dict.get("hypothesis_output_format", ""))
        return QuantHypothesisResponse if require_action else HypothesisResponse

    def gen(
        self,
        trace: Trace,
        plan: ExperimentPlan | None = None,
    ) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=(
                self.scen.get_scenario_all_desc(filtered_tag=self.targets)
                if self.targets in ["factor", "model"]
                else self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
            ),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
            user_instruction=plan.get("user_instruction", None) if plan is not None else None,
        )
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=(
                context_dict["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context_dict else ""
            ),
            sota_hypothesis_and_feedback=(
                context_dict["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context_dict else ""
            ),
            RAG=context_dict["RAG"],
        )

        api = APIBackend()
        response_type = self.hypothesis_response_type(context_dict)
        resp = api.build_messages_and_create_chat_completion(
            user_prompt,
            system_prompt,
            json_mode=json_flag,
            response_format=response_type if api.supports_response_schema() else None,
            json_target_type=None if api.supports_response_schema() else response_type,
        )

        hypothesis = self.convert_response(resp)

        return hypothesis


class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"


class ModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"


class FactorAndModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> Experiment: ...

    @wait_retry(retry_n=5)
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = T(".prompts:hypothesis2experiment.system_prompt").r(
            targets=self.targets,
            scenario=trace.scen.get_scenario_all_desc(filtered_tag=self.targets),
            experiment_output_format=context["experiment_output_format"],
        )
        user_prompt = T(".prompts:hypothesis2experiment.user_prompt").r(
            targets=self.targets,
            target_hypothesis=context["target_hypothesis"],
            hypothesis_and_feedback=(
                context["hypothesis_and_feedback"] if "hypothesis_and_feedback" in context else ""
            ),
            last_hypothesis_and_feedback=(
                context["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context else ""
            ),
            sota_hypothesis_and_feedback=(
                context["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context else ""
            ),
            target_list=context["target_list"],
            RAG=context["RAG"],
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, dict[str, str | dict]]
        )

        return self.convert_response(resp, hypothesis, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
