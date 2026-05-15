from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from typing import Any, Mapping

REQUIRED_QLIB_CONTRACT_ID = "rdagent_qlib_joinquant_ashare_semantic_contract_v1"
REQUIRED_QLIB_RUNTIME_HANDOFF_ID = "qlib_joinquant_ashare_runtime_handoff_v1"
REQUIRED_QLIB_PROMPT_PROJECTION_ID = "qlib_joinquant_ashare_prompt_projection_v1"
REQUIRED_QLIB_PROMPT_PROJECTION_SCHEMA_VERSION = "qlib_ashare_prompt_projection.v1"
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
    semantic_boundary = _mapping(qlib_contract.get("semantic_boundary"))
    failure_semantics = _mapping(qlib_contract.get("failure_semantics"))
    evidence_contract = _mapping(qlib_contract.get("evidence_contract"))
    projection_contract = _mapping(qlib_contract.get("projection_contract"))
    prompt_projection_payload = _mapping(qlib_contract.get("prompt_projection_payload"))
    return {
        "schema_version": "rdagent_ashare_semantic_context.v1",
        "context_id": "rdagent_consumes_qlib_joinquant_ashare_semantics_v1",
        "status": "active",
        "rdagent_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "qlib_contract_id": qlib_contract["contract_id"],
        "qlib_contract_schema_version": qlib_contract["schema_version"],
        "qlib_contract_fingerprint": evidence_contract["semantic_fingerprint"],
        "qlib_source_component": qlib_contract["source_component"],
        "prompt_projection_schema_version": prompt_projection_payload["projection_schema_version"],
        "prompt_projection_kind": prompt_projection_payload["projection_kind"],
        "relationship_boundary": {
            "qlib_role": relationship["qlib_role"],
            "rdagent_role": "research_generation_and_evaluation_context_consumer",
            "semantic_authority": "pyqlib_contract",
            "failure_semantics": "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract",
            "authority_rule": semantic_boundary["authority_rule"],
            "consumer_rule": semantic_boundary["consumer_rule"],
            "rdagent_may": list(semantic_boundary["rdagent_allowed_actions"]),
            "rdagent_forbidden_actions": list(semantic_boundary["rdagent_forbidden_actions"]),
            "rdagent_must_not_redefine": list(qlib_contract["rdagent_must_not_redefine"]),
        },
        "failure_contract": deepcopy(failure_semantics),
        "qlib_evidence_contract": deepcopy(evidence_contract),
        "prompt_projection": deepcopy(projection_contract),
        "prompt_projection_payload": deepcopy(prompt_projection_payload),
        "prompt_rules": [
            "Use A-share market semantics only from qlib.backtest.ashare_semantics.",
            "Render only the Qlib-declared prompt projection; do not expose raw runtime kwargs as prompt authority.",
            "Treat generated factors and models as candidates until Qlib execution and downstream governance evidence exist.",
        ],
    }


def build_rd_agent_ashare_runtime_handoff(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the execution-only Qlib runtime handoff consumed by RD-Agent."""

    qlib_contract = _validate_qlib_ashare_contract(
        dict(contract) if contract is not None else load_qlib_ashare_contract()
    )
    evidence_contract = _mapping(qlib_contract.get("evidence_contract"))
    handoff_contract = _mapping(qlib_contract.get("runtime_handoff_contract"))
    runtime_surfaces = _mapping(qlib_contract.get("runtime_surfaces"))
    return {
        "schema_version": "rdagent_ashare_runtime_handoff.v1",
        "handoff_id": handoff_contract["handoff_id"],
        "status": "active",
        "rdagent_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "qlib_contract_id": qlib_contract["contract_id"],
        "qlib_contract_schema_version": qlib_contract["schema_version"],
        "qlib_contract_fingerprint": evidence_contract["semantic_fingerprint"],
        "qlib_source_component": qlib_contract["source_component"],
        "semantic_authority": QLIB_ASHARE_AUTHORITY_COMPONENT,
        "mutation_policy": handoff_contract["mutation_policy"],
        "consumer_obligations": list(handoff_contract["consumer_obligations"]),
        "runtime_payload": {
            "exchange_kwargs": deepcopy(runtime_surfaces["exchange_kwargs"]),
            "backtest_kwargs": deepcopy(runtime_surfaces["backtest_kwargs"]),
        },
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
    prompt_payload = _mapping(payload.get("prompt_projection_payload"))
    market = _mapping(prompt_payload.get("market_semantics"))
    instrument_identity = _mapping(prompt_payload.get("instrument_identity_semantics"))
    transaction_cost = _mapping(prompt_payload.get("transaction_cost_semantics"))
    suspension_tradability = _mapping(prompt_payload.get("suspension_tradability_semantics"))
    execution_price = _mapping(prompt_payload.get("execution_price_semantics"))
    price_limit = _mapping(prompt_payload.get("price_limit_semantics"))
    settlement = _mapping(prompt_payload.get("settlement_semantics"))
    order_unit = _mapping(prompt_payload.get("order_unit_semantics"))
    projection = _mapping(payload.get("prompt_projection"))
    forbidden_prompt_fields = projection.get("rdagent_prompt_forbidden_fields", [])
    return "\n".join(
        [
            "A-share Qlib semantic relationship:",
            f"- status: {payload['status']}",
            f"- qlib_contract_id: {payload['qlib_contract_id']}",
            f"- qlib_contract_schema_version: {payload['qlib_contract_schema_version']}",
            f"- qlib_contract_fingerprint: {payload['qlib_contract_fingerprint']}",
            f"- qlib_source_component: {payload['qlib_source_component']}",
            f"- prompt_projection_schema_version: {payload['prompt_projection_schema_version']}",
            f"- prompt_projection_kind: {payload['prompt_projection_kind']}",
            f"- rd-agent role: {boundary['rdagent_role']}",
            f"- qlib role: {boundary['qlib_role']}",
            f"- authority rule: {boundary['authority_rule']}",
            f"- consumer rule: {boundary['consumer_rule']}",
            f"- market: {market.get('market')} / region={market.get('region')}",
            f"- instrument identity authority: pyqlib ({instrument_identity.get('canonical_code_format')})",
            "- instrument provider suffixes: "
            + ", ".join(
                f"{suffix}->{prefix}"
                for suffix, prefix in sorted(_mapping(instrument_identity.get("accepted_provider_suffixes")).items())
            ),
            f"- board identity authority: pyqlib ({instrument_identity.get('board_classification_authority')})",
            f"- transaction-cost authority: pyqlib ({transaction_cost.get('runtime_authority')})",
            "- transaction-cost buy components: "
            + ", ".join(str(item) for item in transaction_cost.get("buy_cost_components", [])),
            "- transaction-cost sell components: "
            + ", ".join(str(item) for item in transaction_cost.get("sell_cost_components", [])),
            f"- transaction-cost values: {transaction_cost.get('numeric_values_exposure')}",
            f"- suspension authority: pyqlib ({suspension_tradability.get('runtime_authority')})",
            f"- suspension indicator: {suspension_tradability.get('suspension_indicator_rule')}",
            f"- suspension tradability: {suspension_tradability.get('non_tradable_rule')}",
            f"- suspension limit flags: {suspension_tradability.get('limit_flag_projection')}",
            f"- execution-price authority: pyqlib ({execution_price.get('runtime_authority')})",
            f"- execution-price field: {execution_price.get('execution_price_field')}",
            f"- execution frequency: {execution_price.get('execution_frequency')}",
            f"- intraday execution rule: {execution_price.get('intraday_execution_rule')}",
            f"- trade_unit authority: pyqlib ({market.get('trade_unit')})",
            f"- position authority: pyqlib ({market.get('position_type')})",
            f"- price-limit authority: pyqlib ({price_limit.get('field_authority')})",
            f"- price-limit mode: {price_limit.get('price_limit_mode')}",
            f"- price-limit fallback: {price_limit.get('board_fallback_policy')}",
            f"- settlement authority: pyqlib ({settlement.get('settlement_rule')})",
            f"- same-day sell policy: {settlement.get('same_day_sell_policy')}",
            f"- round-lot authority: pyqlib ({order_unit.get('trade_unit')} {order_unit.get('amount_unit')})",
            f"- round-lot buy rule: {order_unit.get('buy_rounding_rule')}",
            f"- round-lot sell rule: {order_unit.get('sell_rounding_rule')}",
            f"- round-lot full liquidation: {order_unit.get('full_liquidation_rule')}",
            "- RD-Agent must not redefine: " + ", ".join(str(item) for item in boundary["rdagent_must_not_redefine"]),
            "- prompt projection forbids: " + ", ".join(str(item) for item in forbidden_prompt_fields),
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
        "schema_version": "qlib_ashare_semantic_contract.v1",
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

    semantic_boundary = _mapping(contract.get("semantic_boundary"))
    if semantic_boundary.get("authority_component") != QLIB_ASHARE_AUTHORITY_COMPONENT:
        raise QlibAshareSemanticContractError("pyqlib A-share contract semantic_boundary must name Qlib authority")
    if semantic_boundary.get("consumer_component") != RDAGENT_ASHARE_CONSUMER_COMPONENT:
        raise QlibAshareSemanticContractError("pyqlib A-share contract semantic_boundary must name RD-Agent consumer")
    for key in ("authority_rule", "consumer_rule"):
        if not semantic_boundary.get(key):
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract semantic_boundary must include {key}")
    allowed_actions = _string_list(semantic_boundary.get("rdagent_allowed_actions"))
    forbidden_actions = _string_list(semantic_boundary.get("rdagent_forbidden_actions"))
    for action in (
        "render_contract_projection_in_research_context",
        "carry_contract_id_schema_version_and_fingerprint_into_generated_evidence",
        "fail_closed_when_contract_is_missing_malformed_or_unsupported",
    ):
        if action not in allowed_actions:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract must allow RD-Agent action {action}")
    for action in (
        "redefine_instrument_identity_or_board_mapping",
        "redefine_transaction_cost_model",
        "redefine_suspension_or_tradability_rules",
        "redefine_execution_price_or_frequency",
        "redefine_trade_unit_or_position_type",
        "redefine_cost_model_or_exchange_kwargs",
        "treat_research_prompt_projection_as_backtest_authority",
        "claim_a_share_alignment_without_qlib_contract_fingerprint",
    ):
        if action not in forbidden_actions:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract must forbid RD-Agent action {action}")

    failure_semantics = _mapping(contract.get("failure_semantics"))
    for key in (
        "missing_contract",
        "unsupported_schema_version",
        "missing_required_field",
        "malformed_contract",
        "runtime_projection_drift",
        "claim_without_evidence_fingerprint",
    ):
        if failure_semantics.get(key) != "fail_closed":
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract failure_semantics must set {key} to fail_closed"
            )

    evidence_contract = _mapping(contract.get("evidence_contract"))
    fingerprint = evidence_contract.get("semantic_fingerprint")
    if not _is_sha256_hex(fingerprint):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract evidence_contract must include a sha256 semantic_fingerprint"
        )
    if evidence_contract.get("fingerprint_algorithm") != "sha256_json_canonical_v1":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract evidence_contract must declare sha256_json_canonical_v1"
        )
    required_evidence_fields = _string_list(evidence_contract.get("rdagent_required_evidence_fields"))
    for key in (
        "qlib_contract_id",
        "qlib_contract_schema_version",
        "qlib_contract_fingerprint",
        "qlib_source_component",
        "qlib_semantic_authority",
    ):
        if key not in required_evidence_fields:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract evidence_contract must require {key}")

    projection_contract = _mapping(contract.get("projection_contract"))
    prompt_projection_fields = _string_list(projection_contract.get("rdagent_prompt_projection_fields"))
    prompt_forbidden_fields = _string_list(projection_contract.get("rdagent_prompt_forbidden_fields"))
    if "evidence_contract.semantic_fingerprint" not in prompt_projection_fields:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract projection_contract must project the semantic fingerprint"
        )
    for key in (
        "instrument_identity_semantics",
        "transaction_cost_semantics",
        "suspension_tradability_semantics",
        "execution_price_semantics",
        "price_limit_semantics",
        "market_semantics.settlement_rule",
        "settlement_semantics",
        "order_unit_semantics",
    ):
        if key not in prompt_projection_fields:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract projection_contract must project A-share prompt field {key}"
            )
    for key in (
        "runtime_surfaces.policy_defaults",
        "runtime_surfaces.exchange_kwargs",
        "runtime_surfaces.backtest_kwargs",
        "market_semantics.cost_model",
    ):
        if key not in prompt_forbidden_fields:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract projection_contract must forbid prompt field {key}"
            )

    prompt_payload = _mapping(contract.get("prompt_projection_payload"))
    if prompt_payload.get("projection_id") != REQUIRED_QLIB_PROMPT_PROJECTION_ID:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must declare the Qlib prompt projection id"
        )
    if prompt_payload.get("projection_schema_version") != REQUIRED_QLIB_PROMPT_PROJECTION_SCHEMA_VERSION:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must declare the prompt projection schema version"
        )
    if prompt_payload.get("projection_kind") != "research_prompt_context_only":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must remain research-prompt-only"
        )
    if prompt_payload.get("contract_id") != REQUIRED_QLIB_CONTRACT_ID:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must preserve the contract id"
        )
    if prompt_payload.get("contract_schema_version") != "qlib_ashare_semantic_contract.v1":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must preserve the contract schema version"
        )
    if prompt_payload.get("schema_version") != "qlib_ashare_semantic_contract.v1":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must preserve the schema version"
        )
    if prompt_payload.get("source_component") != QLIB_ASHARE_AUTHORITY_COMPONENT:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must name Qlib authority"
        )
    if prompt_payload.get("consumer_component") != RDAGENT_ASHARE_CONSUMER_COMPONENT:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must name RD-Agent consumer"
        )
    if prompt_payload.get("semantic_fingerprint") != evidence_contract["semantic_fingerprint"]:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload must preserve the semantic fingerprint"
        )
    prompt_market = _mapping(prompt_payload.get("market_semantics"))
    for key in (
        "market",
        "region",
        "trade_unit",
        "position_type",
        "settlement_rule",
        "limit_threshold",
        "authoritative_limit_fields",
    ):
        if key not in prompt_market:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload market_semantics must include {key}"
            )
    instrument_identity = _mapping(prompt_payload.get("instrument_identity_semantics"))
    for key in (
        "semantic_name",
        "canonical_code_format",
        "canonical_exchange_prefixes",
        "accepted_provider_suffixes",
        "normalization_examples",
        "board_identity_rules",
        "price_limit_dependency",
        "runtime_authority",
        "board_classification_authority",
        "rdagent_rule",
    ):
        if key not in instrument_identity:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must include {key}"
            )
    if instrument_identity.get("semantic_name") != "a_share_instrument_identity":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must describe A-share identity"
        )
    if instrument_identity.get("canonical_code_format") != "exchange_prefix_plus_six_digit_code":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must declare canonical code format"
        )
    prefixes = _string_list(instrument_identity.get("canonical_exchange_prefixes"))
    for prefix in ("SH", "SZ", "BJ"):
        if prefix not in prefixes:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must include {prefix}"
            )
    suffixes = _mapping(instrument_identity.get("accepted_provider_suffixes"))
    for suffix, prefix in {"XSHG": "SH", "XSHE": "SZ", "XBJ": "BJ"}.items():
        if suffixes.get(suffix) != prefix:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must map "
                f"{suffix} to {prefix}"
            )
    board_rules = instrument_identity.get("board_identity_rules")
    if not isinstance(board_rules, list) or len(board_rules) < 4:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must include board rules"
        )
    board_names = {_mapping(rule).get("board") for rule in board_rules}
    for board in ("star_market", "chinext_registration_sensitive", "beijing_stock_exchange", "main_board"):
        if board not in board_names:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must include {board}"
            )
    if instrument_identity.get("runtime_authority") != "qlib.backtest.ashare_semantics.normalize_ashare_instrument":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must name Qlib runtime authority"
        )
    if instrument_identity.get("board_classification_authority") != (
        "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.limit_threshold_for_instrument"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must name board authority"
        )
    if instrument_identity.get("rdagent_rule") != "describe_only_do_not_redefine_instrument_or_board_identity":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload instrument_identity_semantics must forbid RD-Agent redefinition"
        )
    transaction_cost = _mapping(prompt_payload.get("transaction_cost_semantics"))
    for key in (
        "semantic_name",
        "cost_model_scope",
        "buy_cost_components",
        "sell_cost_components",
        "minimum_fee_rule",
        "zero_trade_rule",
        "market_impact_rule",
        "numeric_values_exposure",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in transaction_cost:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must include {key}"
            )
    if transaction_cost.get("semantic_name") != "a_share_transaction_cost_structure":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must describe A-share costs"
        )
    if transaction_cost.get("cost_model_scope") != "qlib_runtime_execution_only":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must stay runtime scoped"
        )
    buy_components = _string_list(transaction_cost.get("buy_cost_components"))
    sell_components = _string_list(transaction_cost.get("sell_cost_components"))
    for component in ("commission", "minimum_commission_floor"):
        if component not in buy_components:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must include buy {component}"
            )
    for component in ("commission", "stamp_tax", "minimum_commission_floor"):
        if component not in sell_components:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must include sell {component}"
            )
    if transaction_cost.get("numeric_values_exposure") != "runtime_handoff_only_not_prompt_projection":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must not expose numeric values"
        )
    if transaction_cost.get("runtime_authority") != (
        "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.calculate_trade_cost"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must name Qlib runtime authority"
        )
    if transaction_cost.get("rdagent_rule") != "describe_only_do_not_redefine_transaction_cost_model":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload transaction_cost_semantics must forbid RD-Agent redefinition"
        )
    suspension_tradability = _mapping(prompt_payload.get("suspension_tradability_semantics"))
    for key in (
        "semantic_name",
        "suspension_indicator_field",
        "suspension_indicator_rule",
        "non_tradable_rule",
        "limit_flag_projection",
        "authoritative_limit_interaction",
        "missing_limit_bounds_rule",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in suspension_tradability:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"suspension_tradability_semantics must include {key}"
            )
    if suspension_tradability.get("semantic_name") != "a_share_suspension_tradability":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload "
            "suspension_tradability_semantics must describe A-share suspension tradability"
        )
    if suspension_tradability.get("suspension_indicator_field") != "$close":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload "
            "suspension_tradability_semantics must use Qlib close-price suspension indicator"
        )
    if suspension_tradability.get("suspension_indicator_rule") != "missing_close_price_marks_suspended":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload "
            "suspension_tradability_semantics must declare missing close as suspended"
        )
    if suspension_tradability.get("non_tradable_rule") != "suspended_rows_are_not_buyable_or_sellable":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload "
            "suspension_tradability_semantics must make suspended rows non-tradable"
        )
    if suspension_tradability.get("runtime_authority") != (
        "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload "
            "suspension_tradability_semantics must name Qlib runtime authority"
        )
    if suspension_tradability.get("rdagent_rule") != "describe_only_do_not_redefine_suspension_or_tradability":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload "
            "suspension_tradability_semantics must forbid RD-Agent redefinition"
        )
    execution_price = _mapping(prompt_payload.get("execution_price_semantics"))
    for key in (
        "semantic_name",
        "qlib_parameter",
        "execution_price_field",
        "execution_frequency",
        "price_source_authority",
        "intraday_execution_rule",
        "candidate_research_rule",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in execution_price:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload execution_price_semantics must include {key}"
            )
    if execution_price.get("semantic_name") != "a_share_daily_close_execution_price":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must describe A-share close-price execution"
        )
    if execution_price.get("qlib_parameter") != "deal_price":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must bind Qlib deal_price"
        )
    if execution_price.get("execution_price_field") != "$close":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must use Qlib close field"
        )
    if execution_price.get("execution_frequency") != "daily_bar_backtest":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must stay daily-bar scoped"
        )
    if execution_price.get("intraday_execution_rule") != "not_intraday_or_auction_simulation":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must not imply intraday fills"
        )
    if execution_price.get("runtime_authority") != "qlib.backtest.ashare_semantics.joinquant_ashare_exchange_kwargs":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must name Qlib runtime authority"
        )
    if execution_price.get("rdagent_rule") != "describe_only_do_not_redefine_execution_price_or_frequency":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload execution_price_semantics must forbid RD-Agent redefinition"
        )
    price_limit = _mapping(prompt_payload.get("price_limit_semantics"))
    for key in (
        "limit_threshold",
        "price_limit_mode",
        "authoritative_limit_fields",
        "field_authority",
        "missing_authoritative_fields",
        "board_fallback_policy",
        "board_limit_thresholds",
        "rdagent_rule",
    ):
        if key not in price_limit:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload price_limit_semantics must include {key}"
            )
    if price_limit.get("limit_threshold") != prompt_market.get("limit_threshold"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must match market limit"
        )
    if price_limit.get("authoritative_limit_fields") != prompt_market.get("authoritative_limit_fields"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must match market fields"
        )
    if price_limit.get("price_limit_mode") not in {"strict", "auto"}:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare strict or auto mode"
        )
    if price_limit.get("rdagent_rule") != "describe_only_do_not_redefine_price_limit_thresholds_or_fields":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must forbid RD-Agent redefinition"
        )
    board_thresholds = _mapping(price_limit.get("board_limit_thresholds"))
    for key in (
        "main_board",
        "star_chinext",
        "bse",
        "chinext_registration_start_date",
    ):
        if key not in board_thresholds:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload board_limit_thresholds must include {key}"
            )
    settlement = _mapping(prompt_payload.get("settlement_semantics"))
    for key in (
        "settlement_rule",
        "same_day_sell_policy",
        "position_type",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in settlement:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload settlement_semantics must include {key}"
            )
    if settlement.get("settlement_rule") != prompt_market.get("settlement_rule"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must match market settlement"
        )
    if settlement.get("position_type") != prompt_market.get("position_type"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must match market position"
        )
    if settlement.get("rdagent_rule") != "describe_only_do_not_redefine_position_or_settlement":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must forbid RD-Agent redefinition"
        )
    order_unit = _mapping(prompt_payload.get("order_unit_semantics"))
    for key in (
        "semantic_name",
        "qlib_parameter",
        "trade_unit",
        "amount_unit",
        "buy_rounding_rule",
        "sell_rounding_rule",
        "full_liquidation_rule",
        "factor_adjustment_rule",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in order_unit:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload order_unit_semantics must include {key}"
            )
    if order_unit.get("semantic_name") != "a_share_round_lot":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload order_unit_semantics must describe A-share round lot"
        )
    if order_unit.get("qlib_parameter") != "trade_unit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload order_unit_semantics must bind Qlib trade_unit"
        )
    if order_unit.get("trade_unit") != prompt_market.get("trade_unit"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload order_unit_semantics must match market trade_unit"
        )
    if order_unit.get("runtime_authority") != "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload order_unit_semantics must name Qlib runtime authority"
        )
    if order_unit.get("rdagent_rule") != "describe_only_do_not_redefine_trade_unit_or_round_lot_policy":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload order_unit_semantics must forbid RD-Agent redefinition"
        )
    _assert_no_forbidden_prompt_projection_payload(prompt_payload)

    market = _mapping(contract.get("market_semantics"))
    for key in (
        "trade_unit",
        "position_type",
        "settlement_rule",
        "same_day_sell_policy",
        "limit_threshold",
        "price_limit_modes",
        "authoritative_limit_fields",
        "board_threshold_fields",
        "cost_model",
    ):
        if key not in market:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract market_semantics must include {key}")

    runtime_surfaces = _mapping(contract.get("runtime_surfaces"))
    for key in ("exchange_kwargs", "backtest_kwargs"):
        if key not in runtime_surfaces:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract runtime_surfaces must include {key}")

    runtime_handoff = _mapping(contract.get("runtime_handoff_contract"))
    if runtime_handoff.get("handoff_id") != REQUIRED_QLIB_RUNTIME_HANDOFF_ID:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract runtime_handoff_contract must declare the Qlib runtime handoff id"
        )
    if runtime_handoff.get("handoff_kind") != "qlib_owned_execution_kwargs":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract runtime_handoff_contract must describe Qlib-owned execution kwargs"
        )
    if runtime_handoff.get("authority_component") != QLIB_ASHARE_AUTHORITY_COMPONENT:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract runtime_handoff_contract must name Qlib authority"
        )
    if runtime_handoff.get("consumer_component") != RDAGENT_ASHARE_CONSUMER_COMPONENT:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract runtime_handoff_contract must name RD-Agent consumer"
        )
    if runtime_handoff.get("source_fingerprint") != evidence_contract["semantic_fingerprint"]:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract runtime_handoff_contract must preserve the semantic fingerprint"
        )
    if runtime_handoff.get("mutation_policy") != "pass_through_only":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract runtime_handoff_contract must be pass_through_only"
        )
    handoff_payload_paths = _string_list(runtime_handoff.get("payload_paths"))
    for key in ("runtime_surfaces.exchange_kwargs", "runtime_surfaces.backtest_kwargs"):
        if key not in handoff_payload_paths:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract runtime_handoff_contract must expose payload path {key}"
            )
    handoff_forbidden_prompt_paths = _string_list(runtime_handoff.get("forbidden_prompt_paths"))
    for key in ("runtime_surfaces.policy_defaults", "market_semantics.cost_model"):
        if key not in handoff_forbidden_prompt_paths:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract runtime_handoff_contract must forbid prompt path {key}"
            )
    handoff_obligations = _string_list(runtime_handoff.get("consumer_obligations"))
    for key in (
        "preserve_contract_id_schema_version_and_fingerprint",
        "preserve_qlib_source_component",
        "do_not_mutate_runtime_payload_values",
        "fail_closed_on_missing_payload_or_fingerprint",
    ):
        if key not in handoff_obligations:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract runtime_handoff_contract must require obligation {key}"
            )

    must_not_redefine = contract.get("rdagent_must_not_redefine")
    if not isinstance(must_not_redefine, list):
        raise QlibAshareSemanticContractError("pyqlib A-share contract must list rdagent_must_not_redefine")
    for key in (
        "trade_unit",
        "position_type",
        "settlement_rule",
        "same_day_sell_policy",
        "suspension_tradability_semantics",
        "execution_price_semantics",
        "price_limit_modes",
        "authoritative_limit_fields",
        "board_threshold_fields",
        "cost_model",
    ):
        if key not in must_not_redefine:
            raise QlibAshareSemanticContractError(f"pyqlib A-share contract must forbid RD-Agent redefining {key}")

    return contract


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QlibAshareSemanticContractError("pyqlib A-share contract must use string-list semantic fields")
    return list(value)


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _assert_no_forbidden_prompt_projection_payload(value: Any) -> None:
    forbidden_keys = {
        "runtime_surfaces",
        "policy_defaults",
        "exchange_kwargs",
        "backtest_kwargs",
        "cost_model",
        "open_cost",
        "close_cost",
        "close_commission",
        "close_tax",
        "min_cost",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden_keys:
                raise QlibAshareSemanticContractError(
                    f"pyqlib A-share contract prompt_projection_payload must not expose {key}"
                )
            _assert_no_forbidden_prompt_projection_payload(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_prompt_projection_payload(item)
