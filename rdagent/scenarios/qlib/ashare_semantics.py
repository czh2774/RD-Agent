from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from typing import Any, Mapping


REQUIRED_QLIB_CONTRACT_ID = "rdagent_qlib_joinquant_ashare_semantic_contract_v1"
QLIB_ASHARE_AUTHORITY_COMPONENT = "qlib.backtest.ashare_semantics"
RDAGENT_ASHARE_CONSUMER_COMPONENT = "rdagent.scenarios.qlib.ashare_semantics"


class QlibAshareSemanticContractError(RuntimeError):
    """Raised when the pyqlib A-share semantic contract is absent or malformed."""


def load_qlib_ashare_contract(*, strict_price_limit: bool = True) -> dict[str, Any]:
    """Load and validate the Qlib-owned A-share semantic contract."""

    try:
        qlib_ashare = import_module(QLIB_ASHARE_AUTHORITY_COMPONENT)
    except Exception as exc:
        raise QlibAshareSemanticContractError(
            "pyqlib must expose qlib.backtest.ashare_semantics before RD-Agent "
            "can claim A-share Qlib semantic alignment"
        ) from exc

    contract_builder = getattr(qlib_ashare, "rdagent_ashare_semantic_contract", None)
    if not callable(contract_builder):
        raise QlibAshareSemanticContractError(
            "pyqlib qlib.backtest.ashare_semantics must expose " "rdagent_ashare_semantic_contract()"
        )

    contract = contract_builder(strict_price_limit=strict_price_limit)
    if not isinstance(contract, dict):
        raise QlibAshareSemanticContractError("rdagent_ashare_semantic_contract() must return a dict")
    return _validate_qlib_ashare_contract(contract)


def build_rd_agent_ashare_semantic_context(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build RD-Agent's consumer view without redefining Qlib semantics."""

    qlib_contract = _validate_qlib_ashare_contract(
        dict(contract) if contract is not None else load_qlib_ashare_contract()
    )
    relationship = _mapping(qlib_contract.get("relationship"))
    return {
        "schema_version": "rdagent_ashare_semantic_context.v1",
        "context_id": "rdagent_consumes_qlib_joinquant_ashare_semantics_v1",
        "status": "active",
        "rdagent_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "qlib_contract_id": qlib_contract["contract_id"],
        "qlib_source_component": qlib_contract["source_component"],
        "relationship_boundary": {
            "qlib_role": relationship["qlib_role"],
            "rdagent_role": "research_generation_and_evaluation_context_consumer",
            "semantic_authority": "pyqlib_contract",
            "failure_semantics": "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract",
            "rdagent_may": [
                "render A-share constraints into Qlib research prompts",
                "validate generated factors and models against Qlib contract fields",
                "pass Qlib-owned backtest kwargs to runtime execution surfaces",
            ],
            "rdagent_must_not_redefine": list(qlib_contract["rdagent_must_not_redefine"]),
        },
        "prompt_rules": [
            "Use A-share market semantics only from qlib.backtest.ashare_semantics.",
            "Do not invent trade unit, position, price-limit, or cost values in prompts.",
            "Treat generated factors and models as candidates until Qlib execution and downstream governance evidence exist.",
        ],
        "qlib_market_semantics": deepcopy(qlib_contract["market_semantics"]),
        "runtime_surfaces": deepcopy(qlib_contract["runtime_surfaces"]),
    }


def format_rd_agent_ashare_semantic_context(
    context: Mapping[str, Any] | None = None,
) -> str:
    if context is None:
        payload = build_rd_agent_ashare_semantic_context()
    elif context.get("context_id") == "rdagent_consumes_qlib_joinquant_ashare_semantics_v1":
        payload = dict(context)
    else:
        payload = build_rd_agent_ashare_semantic_context(context)
    boundary = _mapping(payload.get("relationship_boundary"))
    market = _mapping(payload.get("qlib_market_semantics"))
    return "\n".join(
        [
            "A-share Qlib semantic relationship:",
            f"- status: {payload['status']}",
            f"- qlib_contract_id: {payload['qlib_contract_id']}",
            f"- qlib_source_component: {payload['qlib_source_component']}",
            f"- rd-agent role: {boundary['rdagent_role']}",
            f"- qlib role: {boundary['qlib_role']}",
            f"- market: {market.get('market')} / region={market.get('region')}",
            f"- trade_unit authority: pyqlib ({market.get('trade_unit')})",
            f"- position authority: pyqlib ({market.get('position_type')})",
            "- RD-Agent must not redefine: " + ", ".join(str(item) for item in boundary["rdagent_must_not_redefine"]),
            f"- failure_semantics: {boundary['failure_semantics']}",
        ]
    )


def optional_rd_agent_ashare_semantic_context_block() -> str:
    try:
        return format_rd_agent_ashare_semantic_context()
    except QlibAshareSemanticContractError as exc:
        return "\n".join(
            [
                "A-share Qlib semantic relationship:",
                "- status: unavailable",
                "- failure_semantics: fail_closed_on_missing_or_malformed_pyqlib_ashare_contract",
                f"- reason: {exc}",
            ]
        )


def append_ashare_semantic_context(runtime_environment: str) -> str:
    base = runtime_environment.strip()
    context = optional_rd_agent_ashare_semantic_context_block()
    if not base:
        return context
    return f"{base}\n\n{context}"


def _validate_qlib_ashare_contract(contract: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "contract_id": REQUIRED_QLIB_CONTRACT_ID,
        "source_component": QLIB_ASHARE_AUTHORITY_COMPONENT,
        "consumer_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "status": "active",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract {key} must be {value!r}; got {contract.get(key)!r}"
            )

    relationship = _mapping(contract.get("relationship"))
    if relationship.get("qlib_role") != "executable_backtest_semantic_authority":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract must declare Qlib as executable backtest semantic authority"
        )
    if relationship.get("rdagent_role") != "research_candidate_generation_context_consumer":
        raise QlibAshareSemanticContractError("pyqlib A-share contract must declare RD-Agent as context consumer")
    if relationship.get("fail_closed_on_missing_contract") is not True:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract must fail closed when the relationship contract is missing"
        )

    market = _mapping(contract.get("market_semantics"))
    for key in (
        "trade_unit",
        "position_type",
        "limit_threshold",
        "price_limit_modes",
        "authoritative_limit_fields",
        "cost_model",
    ):
        if key not in market:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract market_semantics must include {key}")

    runtime_surfaces = _mapping(contract.get("runtime_surfaces"))
    for key in ("exchange_kwargs", "backtest_kwargs"):
        if key not in runtime_surfaces:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract runtime_surfaces must include {key}")

    must_not_redefine = contract.get("rdagent_must_not_redefine")
    if not isinstance(must_not_redefine, list):
        raise QlibAshareSemanticContractError("pyqlib A-share contract must list rdagent_must_not_redefine")
    for key in ("trade_unit", "position_type", "cost_model"):
        if key not in must_not_redefine:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract must forbid RD-Agent redefining {key}")

    return contract


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
