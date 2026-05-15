from __future__ import annotations

from typing import Any

from rdagent.scenarios.qlib.ashare_semantics import (
    QLIB_ASHARE_FORBIDDEN_MODEL_TYPES,
    QLIB_ASHARE_MODEL_TYPE_BOUNDARY_RULE,
    QLIB_ASHARE_SUPPORTED_MODEL_TYPES,
)


def validate_qlib_model_experiment_response(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Qlib model experiment response must be a non-empty JSON object.")

    normalized_response: dict[str, dict[str, Any]] = {}
    for model_name, model_payload in payload.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Qlib model experiment response must use non-empty string model names.")
        if not isinstance(model_payload, dict):
            raise ValueError(f"Qlib model experiment output for {model_name!r} must be a JSON object.")

        description = model_payload.get("description")
        formulation = model_payload.get("formulation")
        architecture = model_payload.get("architecture")
        variables = model_payload.get("variables")
        hyperparameters = model_payload.get("hyperparameters")
        training_hyperparameters = model_payload.get("training_hyperparameters")
        model_type = model_payload.get("model_type")

        if not isinstance(description, str):
            raise ValueError(f"Qlib model experiment output for {model_name!r} must include string description.")
        if not isinstance(formulation, str):
            raise ValueError(f"Qlib model experiment output for {model_name!r} must include string formulation.")
        if not isinstance(architecture, str):
            raise ValueError(f"Qlib model experiment output for {model_name!r} must include string architecture.")
        if not isinstance(variables, dict):
            raise ValueError(f"Qlib model experiment output for {model_name!r} must include object variables.")
        if not isinstance(hyperparameters, dict):
            raise ValueError(f"Qlib model experiment output for {model_name!r} must include object hyperparameters.")
        if not isinstance(training_hyperparameters, dict):
            raise ValueError(
                f"Qlib model experiment output for {model_name!r} must include object training_hyperparameters."
            )
        if model_type not in QLIB_ASHARE_SUPPORTED_MODEL_TYPES:
            supported = ", ".join(QLIB_ASHARE_SUPPORTED_MODEL_TYPES)
            forbidden = ", ".join(QLIB_ASHARE_FORBIDDEN_MODEL_TYPES)
            raise ValueError(
                f"Qlib A-share model_type for {model_name!r} must be one of {supported}; "
                f"forbidden model types include {forbidden}. Boundary: {QLIB_ASHARE_MODEL_TYPE_BOUNDARY_RULE}."
            )

        normalized_response[model_name] = {
            "description": description,
            "formulation": formulation,
            "architecture": architecture,
            "variables": dict(variables),
            "hyperparameters": dict(hyperparameters),
            "training_hyperparameters": dict(training_hyperparameters),
            "model_type": model_type,
        }
    return normalized_response
