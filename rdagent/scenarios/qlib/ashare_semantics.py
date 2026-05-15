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
    universe_membership = _mapping(prompt_payload.get("universe_membership_semantics"))
    trading_calendar = _mapping(prompt_payload.get("trading_calendar_semantics"))
    transaction_cost = _mapping(prompt_payload.get("transaction_cost_semantics"))
    market_impact = _mapping(prompt_payload.get("market_impact_semantics"))
    account_update = _mapping(prompt_payload.get("account_update_semantics"))
    account_valuation = _mapping(prompt_payload.get("account_valuation_semantics"))
    trade_indicator = _mapping(prompt_payload.get("trade_indicator_semantics"))
    executor_decision = _mapping(prompt_payload.get("executor_decision_semantics"))
    suspension_tradability = _mapping(prompt_payload.get("suspension_tradability_semantics"))
    execution_price = _mapping(prompt_payload.get("execution_price_semantics"))
    price_adjustment = _mapping(prompt_payload.get("price_adjustment_semantics"))
    price_limit = _mapping(prompt_payload.get("price_limit_semantics"))
    order_tradability = _mapping(prompt_payload.get("order_tradability_semantics"))
    order_fill_amount = _mapping(prompt_payload.get("order_fill_amount_semantics"))
    settlement = _mapping(prompt_payload.get("settlement_semantics"))
    cash_constraint = _mapping(prompt_payload.get("cash_constraint_semantics"))
    cash_settlement = _mapping(prompt_payload.get("cash_settlement_semantics"))
    liquidity_capacity = _mapping(prompt_payload.get("liquidity_capacity_semantics"))
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
            f"- universe membership authority: pyqlib ({universe_membership.get('instrument_provider_authority')})",
            f"- universe market rule: {universe_membership.get('market_universe_rule')}",
            f"- universe membership window rule: {universe_membership.get('membership_window_rule')}",
            f"- universe filter pipe rule: {universe_membership.get('filter_pipe_rule')}",
            f"- universe survivorship rule: {universe_membership.get('survivorship_rule')}",
            f"- trading-calendar authority: pyqlib ({trading_calendar.get('calendar_provider_authority')})",
            f"- trading-calendar locator: pyqlib ({trading_calendar.get('calendar_locator_authority')})",
            f"- trading-calendar frequency: {trading_calendar.get('calendar_frequency')}",
            f"- trading-calendar non-trading day rule: {trading_calendar.get('non_trading_day_rule')}",
            f"- trading-calendar synthetic session rule: {trading_calendar.get('synthetic_session_rule')}",
            f"- transaction-cost authority: pyqlib ({transaction_cost.get('runtime_authority')})",
            "- transaction-cost buy components: "
            + ", ".join(str(item) for item in transaction_cost.get("buy_cost_components", [])),
            "- transaction-cost sell components: "
            + ", ".join(str(item) for item in transaction_cost.get("sell_cost_components", [])),
            f"- transaction-cost values: {transaction_cost.get('numeric_values_exposure')}",
            f"- market-impact authority: pyqlib ({market_impact.get('runtime_authority')})",
            f"- market-impact parameter: {market_impact.get('configuration_parameter')}",
            f"- market-impact ratio rule: {market_impact.get('impact_cost_ratio_rule')}",
            f"- market-impact missing-volume rule: {market_impact.get('missing_volume_rule')}",
            f"- market-impact final cost rule: {market_impact.get('final_cost_rule')}",
            f"- account-update authority: pyqlib ({account_update.get('account_update_authority')})",
            f"- account-update trigger: {account_update.get('trade_update_trigger')}",
            f"- account-update handoff rule: {account_update.get('handoff_rule')}",
            f"- account-update buy rule: {account_update.get('buy_cash_rule')}",
            f"- account-update sell rule: {account_update.get('sell_cash_rule')}",
            f"- account-update sellable rule: {account_update.get('sellable_amount_rule')}",
            f"- account-valuation authority: pyqlib ({account_valuation.get('bar_end_authority')})",
            "- account-valuation bar-end sequence: "
            + " -> ".join(str(item) for item in account_valuation.get("bar_end_sequence", [])),
            f"- account-valuation mark price rule: {account_valuation.get('mark_price_rule')}",
            f"- account-valuation suspension price rule: {account_valuation.get('suspension_price_rule')}",
            f"- account-valuation account value rule: {account_valuation.get('account_value_rule')}",
            f"- account-valuation daily sellable release rule: {account_valuation.get('daily_sellable_release_rule')}",
            f"- trade-indicator authority: pyqlib ({trade_indicator.get('indicator_authority')})",
            f"- trade-indicator account hook: pyqlib ({trade_indicator.get('account_indicator_authority')})",
            "- trade-indicator metrics: "
            + ", ".join(str(item) for item in trade_indicator.get("trade_metric_fields", [])),
            f"- trade-indicator fulfill rate rule: {trade_indicator.get('fulfill_rate_rule')}",
            f"- trade-indicator price advantage rule: {trade_indicator.get('price_advantage_rule')}",
            f"- trade-indicator portfolio boundary: {trade_indicator.get('portfolio_boundary_rule')}",
            f"- executor-decision authority: pyqlib ({executor_decision.get('base_executor_authority')})",
            f"- executor-decision simulator authority: pyqlib ({executor_decision.get('simulator_executor_authority')})",
            f"- executor-decision nested authority: pyqlib ({executor_decision.get('nested_executor_authority')})",
            f"- executor-decision atomicity rule: {executor_decision.get('atomicity_rule')}",
            f"- executor-decision bar-end rule: {executor_decision.get('bar_end_sequence_rule')}",
            f"- executor-decision nested range rule: {executor_decision.get('nested_range_rule')}",
            f"- executor-decision inner decision rule: {executor_decision.get('inner_decision_rule')}",
            f"- suspension authority: pyqlib ({suspension_tradability.get('runtime_authority')})",
            f"- suspension indicator: {suspension_tradability.get('suspension_indicator_rule')}",
            f"- suspension tradability: {suspension_tradability.get('non_tradable_rule')}",
            f"- suspension limit flags: {suspension_tradability.get('limit_flag_projection')}",
            f"- execution-price authority: pyqlib ({execution_price.get('runtime_authority')})",
            f"- execution-price field: {execution_price.get('execution_price_field')}",
            f"- execution frequency: {execution_price.get('execution_frequency')}",
            f"- intraday execution rule: {execution_price.get('intraday_execution_rule')}",
            f"- price-adjustment authority: pyqlib ({price_adjustment.get('runtime_authority')})",
            f"- price-adjustment factor field: {price_adjustment.get('factor_field')}",
            f"- price-adjustment missing factor: {price_adjustment.get('missing_factor_rule')}",
            f"- price-adjustment adjusted-price mode: {price_adjustment.get('adjusted_price_mode_rule')}",
            f"- trade_unit authority: pyqlib ({market.get('trade_unit')})",
            f"- position authority: pyqlib ({market.get('position_type')})",
            f"- price-limit authority: pyqlib ({price_limit.get('field_authority')})",
            f"- price-limit runtime authority: pyqlib ({price_limit.get('runtime_authority')})",
            f"- price-limit mode: {price_limit.get('price_limit_mode')}",
            "- price-limit flag fields: " + ", ".join(str(item) for item in price_limit.get("limit_flag_fields", [])),
            f"- price-limit buy rule: {price_limit.get('buy_limit_rule')}",
            f"- price-limit sell rule: {price_limit.get('sell_limit_rule')}",
            f"- price-limit fallback: {price_limit.get('board_fallback_policy')}",
            f"- price-limit fallback authority: {price_limit.get('fallback_authority_rule')}",
            f"- order-tradability authority: pyqlib ({order_tradability.get('runtime_authority')})",
            f"- order-tradability decision rule: {order_tradability.get('decision_rule')}",
            f"- order-tradability suspension rule: {order_tradability.get('suspension_rule')}",
            f"- order-tradability directional limit rule: {order_tradability.get('directional_limit_rule')}",
            f"- order-tradability failure result: {order_tradability.get('failure_result')}",
            f"- order-fill authority: pyqlib ({order_fill_amount.get('runtime_authority')})",
            f"- order-fill state field: {order_fill_amount.get('fill_state_field')}",
            "- order-fill clip sequence: "
            + " -> ".join(str(item) for item in order_fill_amount.get("clip_sequence", [])),
            f"- order-fill trade value rule: {order_fill_amount.get('trade_value_rule')}",
            f"- order-fill cost rule: {order_fill_amount.get('cost_rule')}",
            f"- settlement authority: pyqlib ({settlement.get('settlement_rule')})",
            f"- settlement runtime authority: pyqlib ({settlement.get('runtime_authority')})",
            f"- same-day sell policy: {settlement.get('same_day_sell_policy')}",
            f"- settlement sellable state: {settlement.get('sellable_state_field')}",
            f"- settlement intraday buy rule: {settlement.get('intraday_buy_rule')}",
            f"- settlement day commit rule: {settlement.get('day_commit_rule')}",
            f"- settlement sell clip: {settlement.get('sell_order_clip_rule')}",
            f"- cash constraint authority: pyqlib ({cash_constraint.get('runtime_authority')})",
            f"- cash state: {cash_constraint.get('cash_state_field')}",
            f"- cash buy rule: {cash_constraint.get('buy_cash_rule')}",
            f"- cash-settlement authority: pyqlib ({cash_settlement.get('settlement_authority')})",
            f"- cash-settlement delayed mode: {cash_settlement.get('delayed_cash_mode')}",
            f"- cash-settlement sell proceeds rule: {cash_settlement.get('sell_proceeds_rule')}",
            f"- cash-settlement available cash rule: {cash_settlement.get('available_cash_rule')}",
            f"- cash-settlement commit rule: {cash_settlement.get('commit_rule')}",
            f"- shorting policy: {cash_constraint.get('shorting_policy')}",
            f"- liquidity capacity authority: pyqlib ({liquidity_capacity.get('runtime_authority')})",
            f"- liquidity capacity parameter: {liquidity_capacity.get('capacity_parameter')}",
            f"- liquidity volume field: {liquidity_capacity.get('volume_field')}",
            f"- liquidity capacity rule: {liquidity_capacity.get('capacity_clip_rule')}",
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
        "redefine_universe_membership_or_instrument_filtering",
        "redefine_trading_calendar_or_data_frequency",
        "redefine_transaction_cost_model",
        "redefine_suspension_or_tradability_rules",
        "redefine_execution_price_or_frequency",
        "redefine_price_adjustment_or_order_factor",
        "redefine_trade_unit_or_position_type",
        "redefine_price_limit_thresholds_or_authoritative_fields",
        "treat_board_fallback_as_primary_price_limit_authority",
        "redefine_order_tradability_or_limit_checks",
        "redefine_order_fill_amount_or_clip_sequence",
        "redefine_market_impact_or_cost_ratio",
        "redefine_account_position_or_cash_mutation_order",
        "redefine_account_valuation_or_bar_end_refresh",
        "redefine_trade_execution_indicators_or_quality_metrics",
        "redefine_executor_decision_lifecycle_or_nested_execution_order",
        "redefine_settlement_or_sellable_position_state",
        "redefine_cash_settlement_or_sell_proceeds_availability",
        "redefine_cash_buying_power_or_shorting_policy",
        "redefine_liquidity_or_volume_capacity_policy",
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
    fingerprint_scope = _string_list(evidence_contract.get("fingerprint_scope"))
    for key in (
        "schema_version",
        "market_semantics",
        "runtime_surfaces",
        "universe_membership_semantics",
        "cash_settlement_semantics",
        "order_tradability_semantics",
        "order_fill_amount_semantics",
        "market_impact_semantics",
        "account_update_semantics",
        "account_valuation_semantics",
        "trade_indicator_semantics",
        "executor_decision_semantics",
        "rdagent_must_not_redefine",
    ):
        if key not in fingerprint_scope:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract evidence_contract fingerprint_scope must include {key}"
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
        "universe_membership_semantics",
        "trading_calendar_semantics",
        "transaction_cost_semantics",
        "market_impact_semantics",
        "account_update_semantics",
        "account_valuation_semantics",
        "trade_indicator_semantics",
        "executor_decision_semantics",
        "suspension_tradability_semantics",
        "execution_price_semantics",
        "price_adjustment_semantics",
        "price_limit_semantics",
        "order_tradability_semantics",
        "order_fill_amount_semantics",
        "market_semantics.data_frequency",
        "market_semantics.settlement_rule",
        "settlement_semantics",
        "cash_settlement_semantics",
        "cash_constraint_semantics",
        "liquidity_capacity_semantics",
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
        "data_frequency",
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
    universe_membership = _mapping(prompt_payload.get("universe_membership_semantics"))
    for key in (
        "semantic_name",
        "membership_input",
        "instrument_provider_authority",
        "local_provider_authority",
        "exchange_codes_authority",
        "market_universe_rule",
        "membership_window_rule",
        "calendar_boundary_rule",
        "filter_pipe_rule",
        "as_list_rule",
        "static_universe_rule",
        "survivorship_rule",
        "rdagent_rule",
    ):
        if key not in universe_membership:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload universe_membership_semantics must include {key}"
            )
    expected_universe_values = {
        "semantic_name": "a_share_universe_membership",
        "membership_input": "Exchange.codes_or_D.instruments_market",
        "instrument_provider_authority": "qlib.data.data.InstrumentProvider.list_instruments",
        "local_provider_authority": "qlib.data.data.LocalInstrumentProvider.list_instruments",
        "exchange_codes_authority": "qlib.backtest.exchange.Exchange.__init__",
        "market_universe_rule": "string_codes_are_resolved_by_qlib_D_instruments",
        "membership_window_rule": "instrument_start_end_spans_are_clipped_to_requested_calendar_window",
        "calendar_boundary_rule": "start_end_defaults_and_membership_filtering_use_qlib_calendar_boundaries",
        "filter_pipe_rule": "qlib_instrument_filter_pipe_is_applied_after_calendar_window_clipping",
        "as_list_rule": "as_list_returns_only_instruments_with_nonempty_effective_spans",
        "static_universe_rule": "rdagent_must_not_treat_all_a_or_index_universe_as_static_without_qlib_membership_spans",
        "survivorship_rule": "membership_must_remain_point_in_time_by_qlib_instrument_spans_and_filters",
        "rdagent_rule": "describe_only_do_not_redefine_universe_membership_or_filters",
    }
    for key, expected_value in expected_universe_values.items():
        if universe_membership.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"universe_membership_semantics must preserve {key}"
            )
    trading_calendar = _mapping(prompt_payload.get("trading_calendar_semantics"))
    for key in (
        "semantic_name",
        "calendar_frequency",
        "calendar_provider_authority",
        "calendar_locator_authority",
        "exchange_frequency_parameter",
        "exchange_default_frequency",
        "index_level",
        "instrument_window_rule",
        "non_trading_day_rule",
        "future_calendar_rule",
        "synthetic_session_rule",
        "rdagent_rule",
    ):
        if key not in trading_calendar:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must include {key}"
            )
    if trading_calendar.get("semantic_name") != "a_share_daily_trading_calendar":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must describe A-share calendar"
        )
    if trading_calendar.get("calendar_frequency") != "day":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must stay day-frequency"
        )
    if trading_calendar.get("calendar_frequency") != prompt_market.get("data_frequency"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must match market data frequency"
        )
    if trading_calendar.get("calendar_provider_authority") != "qlib.data.data.CalendarProvider.calendar":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must name Qlib calendar authority"
        )
    if trading_calendar.get("calendar_locator_authority") != "qlib.data.data.CalendarProvider.locate_index":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must name Qlib locator authority"
        )
    if trading_calendar.get("exchange_frequency_parameter") != "freq":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must bind exchange freq"
        )
    if trading_calendar.get("exchange_default_frequency") != "day":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must bind day exchange default"
        )
    if trading_calendar.get("index_level") != "datetime":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must bind datetime index"
        )
    if trading_calendar.get("instrument_window_rule") != (
        "instrument_membership_is_filtered_against_calendar_boundaries"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must declare instrument calendar filtering"
        )
    if trading_calendar.get("non_trading_day_rule") != (
        "calendar_locate_index_maps_start_forward_and_end_backward_to_real_trading_days"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must declare non-trading day behavior"
        )
    if trading_calendar.get("future_calendar_rule") != (
        "future_trading_days_require_qlib_future_calendar_support_not_prompt_invention"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must forbid prompt future calendars"
        )
    if trading_calendar.get("synthetic_session_rule") != "rdagent_must_not_invent_non_qlib_calendar_sessions":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must forbid synthetic sessions"
        )
    if trading_calendar.get("rdagent_rule") != "describe_only_do_not_redefine_trading_calendar_or_data_frequency":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload trading_calendar_semantics must forbid RD-Agent calendar redefinition"
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
    market_impact = _mapping(prompt_payload.get("market_impact_semantics"))
    for key in (
        "semantic_name",
        "runtime_authority",
        "cost_authority",
        "volume_authority",
        "capacity_authority",
        "configuration_parameter",
        "volume_field",
        "total_trade_value_rule",
        "impact_cost_ratio_rule",
        "missing_volume_rule",
        "cost_ratio_rule",
        "final_cost_rule",
        "joinquant_cost_rule",
        "numeric_value_exposure",
        "rdagent_rule",
    ):
        if key not in market_impact:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload market_impact_semantics must include {key}"
            )
    expected_market_impact_values = {
        "semantic_name": "a_share_market_impact_cost_adjustment",
        "runtime_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
        "cost_authority": "qlib.backtest.exchange.Exchange._calculate_trade_cost",
        "volume_authority": "qlib.backtest.exchange.Exchange.get_volume",
        "capacity_authority": "qlib.backtest.exchange.Exchange._clip_amount_by_volume",
        "configuration_parameter": "impact_cost",
        "volume_field": "$volume",
        "total_trade_value_rule": "total_trade_value_equals_quote_volume_times_trade_price",
        "impact_cost_ratio_rule": "impact_cost_times_post_volume_clip_trade_value_over_total_trade_value_squared",
        "missing_volume_rule": "missing_zero_or_nan_total_trade_value_uses_raw_impact_cost_ratio",
        "cost_ratio_rule": "adjusted_cost_ratio_is_added_to_buy_or_sell_cost_ratio_before_cash_guards",
        "final_cost_rule": "trade_cost_is_recomputed_after_final_deal_amount_with_adjusted_cost_ratio",
        "joinquant_cost_rule": "joinquant_ashare_policy_receives_adjusted_cost_ratio_as_impact_cost",
        "numeric_value_exposure": "runtime_handoff_only_not_prompt_projection",
        "rdagent_rule": "describe_only_do_not_redefine_market_impact_or_cost_ratio",
    }
    for key, expected_value in expected_market_impact_values.items():
        if market_impact.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"market_impact_semantics must preserve {key}"
            )
    account_update = _mapping(prompt_payload.get("account_update_semantics"))
    for key in (
        "semantic_name",
        "execution_authority",
        "account_update_authority",
        "account_metrics_authority",
        "position_update_authority",
        "ashare_sellable_update_authority",
        "trade_update_trigger",
        "failed_or_zero_trade_rule",
        "handoff_rule",
        "trade_amount_rule",
        "buy_mutation_order",
        "sell_mutation_order",
        "buy_cash_rule",
        "sell_cash_rule",
        "sellable_amount_rule",
        "infinite_position_rule",
        "rdagent_rule",
    ):
        if key not in account_update:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload account_update_semantics must include {key}"
            )
    expected_account_update_values = {
        "semantic_name": "a_share_account_position_cash_mutation",
        "execution_authority": "qlib.backtest.exchange.Exchange.deal_order",
        "account_update_authority": "qlib.backtest.account.Account.update_order",
        "account_metrics_authority": "qlib.backtest.account.Account._update_state_from_order",
        "position_update_authority": "qlib.backtest.position.Position.update_order",
        "ashare_sellable_update_authority": "qlib.backtest.position.AsharePosition._sell_stock",
        "trade_update_trigger": "only_trade_value_greater_than_one_e_minus_five_mutates_account_or_position",
        "failed_or_zero_trade_rule": "failed_order_or_zero_trade_value_does_not_update_position_or_account",
        "handoff_rule": "exchange_passes_final_trade_value_cost_and_price_to_account_or_position_update",
        "trade_amount_rule": "mutated_amount_equals_trade_value_divided_by_trade_price",
        "buy_mutation_order": "position_updates_before_account_metrics",
        "sell_mutation_order": "account_metrics_update_before_position_update",
        "buy_cash_rule": "buy_subtracts_trade_value_plus_cost_from_cash",
        "sell_cash_rule": "sell_routes_trade_value_minus_cost_to_cash_or_cash_delay_by_settle_type",
        "sellable_amount_rule": "ashare_sells_reduce_sellable_amount_and_day_bar_count_refresh_releases_total_amount",
        "infinite_position_rule": "skip_update_position_does_not_mutate_account_or_position",
        "rdagent_rule": "describe_only_do_not_redefine_account_position_or_cash_mutation_order",
    }
    for key, expected_value in expected_account_update_values.items():
        if account_update.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"account_update_semantics must preserve {key}"
            )
    account_valuation = _mapping(prompt_payload.get("account_valuation_semantics"))
    for key in (
        "semantic_name",
        "bar_end_authority",
        "position_refresh_authority",
        "portfolio_metrics_authority",
        "history_position_authority",
        "price_update_authority",
        "value_authority",
        "stock_value_authority",
        "holding_count_authority",
        "ashare_sellable_release_authority",
        "close_price_authority",
        "bar_end_sequence",
        "mark_price_rule",
        "suspension_price_rule",
        "account_value_rule",
        "stock_value_rule",
        "portfolio_return_rule",
        "history_snapshot_rule",
        "holding_count_rule",
        "daily_sellable_release_rule",
        "infinite_position_rule",
        "rdagent_rule",
    ):
        if key not in account_valuation:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload account_valuation_semantics must include {key}"
            )
    expected_account_valuation_values = {
        "semantic_name": "a_share_account_bar_end_valuation",
        "bar_end_authority": "qlib.backtest.account.Account.update_bar_end",
        "position_refresh_authority": "qlib.backtest.account.Account.update_current_position",
        "portfolio_metrics_authority": "qlib.backtest.account.Account.update_portfolio_metrics",
        "history_position_authority": "qlib.backtest.account.Account.update_hist_positions",
        "price_update_authority": "qlib.backtest.position.Position.update_stock_price",
        "value_authority": "qlib.backtest.position.Position.calculate_value",
        "stock_value_authority": "qlib.backtest.position.Position.calculate_stock_value",
        "holding_count_authority": "qlib.backtest.position.Position.add_count_all",
        "ashare_sellable_release_authority": "qlib.backtest.position.AsharePosition.add_count_all",
        "close_price_authority": "qlib.backtest.exchange.Exchange.get_close",
        "bar_end_sequence": [
            "refresh_current_position_prices_and_holding_counts",
            "update_portfolio_metrics_when_enabled",
            "snapshot_history_positions_when_enabled",
            "update_trade_indicators",
        ],
        "mark_price_rule": "non_suspended_positions_mark_to_bar_close_at_bar_end",
        "suspension_price_rule": "suspended_positions_keep_previous_price_during_bar_end_refresh",
        "account_value_rule": "account_value_equals_stock_value_plus_cash_plus_cash_delay",
        "stock_value_rule": "stock_value_equals_position_amount_times_current_position_price",
        "portfolio_return_rule": "return_rate_uses_account_earning_plus_current_cost_over_last_account_value",
        "history_snapshot_rule": "history_positions_store_deepcopy_after_now_account_value_and_weights_refresh",
        "holding_count_rule": "bar_end_refresh_increments_position_count_for_account_frequency",
        "daily_sellable_release_rule": "ashare_day_bar_count_refresh_releases_total_amount_to_sellable_amount",
        "infinite_position_rule": "skip_update_position_does_not_refresh_prices_counts_metrics_or_history",
        "rdagent_rule": "describe_only_do_not_redefine_account_valuation_or_bar_end_refresh",
    }
    for key, expected_value in expected_account_valuation_values.items():
        if account_valuation.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"account_valuation_semantics must preserve {key}"
            )
    trade_indicator = _mapping(prompt_payload.get("trade_indicator_semantics"))
    for key in (
        "semantic_name",
        "account_indicator_authority",
        "indicator_authority",
        "atomic_order_update_authority",
        "nested_order_aggregation_authority",
        "trade_indicator_authority",
        "record_authority",
        "order_indicator_state",
        "trade_indicator_state",
        "history_state",
        "order_metric_fields",
        "trade_metric_fields",
        "bar_end_rule",
        "atomic_rule",
        "nested_rule",
        "fulfill_rate_rule",
        "price_advantage_rule",
        "positive_rate_rule",
        "deal_amount_metric_rule",
        "trade_value_metric_rule",
        "order_count_rule",
        "weighting_rule",
        "base_price_rule",
        "unsupported_base_price_rule",
        "record_rule",
        "portfolio_boundary_rule",
        "rdagent_rule",
    ):
        if key not in trade_indicator:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload trade_indicator_semantics must include {key}"
            )
    expected_trade_indicator_values = {
        "semantic_name": "a_share_trade_execution_indicator",
        "account_indicator_authority": "qlib.backtest.account.Account.update_indicator",
        "indicator_authority": "qlib.backtest.report.Indicator",
        "atomic_order_update_authority": "qlib.backtest.report.Indicator.update_order_indicators",
        "nested_order_aggregation_authority": "qlib.backtest.report.Indicator.agg_order_indicators",
        "trade_indicator_authority": "qlib.backtest.report.Indicator.cal_trade_indicators",
        "record_authority": "qlib.backtest.report.Indicator.record",
        "order_indicator_state": "Indicator.order_indicator",
        "trade_indicator_state": "Indicator.trade_indicator",
        "history_state": ["Indicator.order_indicator_his", "Indicator.trade_indicator_his"],
        "order_metric_fields": [
            "amount",
            "inner_amount",
            "deal_amount",
            "trade_price",
            "trade_value",
            "trade_cost",
            "trade_dir",
            "pa",
            "ffr",
            "base_price",
            "base_volume",
        ],
        "trade_metric_fields": ["ffr", "pa", "pos", "deal_amount", "value", "count"],
        "bar_end_rule": "account_update_indicator_runs_after_current_position_valuation_and_portfolio_metrics",
        "atomic_rule": "atomic_executor_uses_trade_info_to_update_order_indicators",
        "nested_rule": "non_atomic_executor_aggregates_inner_order_indicators_and_outer_decision",
        "fulfill_rate_rule": "ffr_equals_deal_amount_reindexed_zero_for_missing_over_order_amount",
        "price_advantage_rule": "pa_equals_directional_trade_price_over_base_price_minus_one",
        "positive_rate_rule": "pos_equals_fraction_of_positive_pa",
        "deal_amount_metric_rule": "deal_amount_metric_sums_absolute_deal_amount",
        "trade_value_metric_rule": "value_metric_sums_absolute_trade_value",
        "order_count_rule": "count_metric_counts_order_amount_entries",
        "weighting_rule": "ffr_and_pa_support_mean_amount_weighted_value_weighted",
        "base_price_rule": "base_price_uses_exchange_deal_price_with_twap_or_vwap_aggregation",
        "unsupported_base_price_rule": "non_deal_price_base_price_is_not_supported",
        "record_rule": "bar_end_records_order_indicator_and_trade_indicator_by_trade_start_time",
        "portfolio_boundary_rule": "trade_indicators_are_execution_quality_metrics_not_portfolio_return_metrics",
        "rdagent_rule": "describe_only_do_not_redefine_trade_execution_indicators_or_quality_metrics",
    }
    for key, expected_value in expected_trade_indicator_values.items():
        if trade_indicator.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"trade_indicator_semantics must preserve {key}"
            )
    executor_decision = _mapping(prompt_payload.get("executor_decision_semantics"))
    for key in (
        "semantic_name",
        "base_executor_authority",
        "simulator_executor_authority",
        "nested_executor_authority",
        "decision_authority",
        "decision_update_authority",
        "range_limit_authority",
        "data_calendar_range_authority",
        "inner_decision_modification_authority",
        "calendar_authority",
        "level_infra_authority",
        "atomicity_rule",
        "settle_sequence_rule",
        "bar_end_sequence_rule",
        "track_data_rule",
        "simulator_order_rule",
        "simulator_trade_type_rule",
        "daily_dealt_amount_rule",
        "nested_init_rule",
        "nested_update_rule",
        "nested_range_rule",
        "inner_decision_rule",
        "empty_decision_rule",
        "inner_result_rule",
        "rdagent_rule",
    ):
        if key not in executor_decision:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload executor_decision_semantics must include {key}"
            )
    expected_executor_decision_values = {
        "semantic_name": "a_share_executor_trade_decision_lifecycle",
        "base_executor_authority": "qlib.backtest.executor.BaseExecutor.collect_data",
        "simulator_executor_authority": "qlib.backtest.executor.SimulatorExecutor._collect_data",
        "nested_executor_authority": "qlib.backtest.executor.NestedExecutor._collect_data",
        "decision_authority": "qlib.backtest.decision.BaseTradeDecision",
        "decision_update_authority": "qlib.backtest.decision.BaseTradeDecision.update",
        "range_limit_authority": "qlib.backtest.decision.BaseTradeDecision.get_range_limit",
        "data_calendar_range_authority": "qlib.backtest.decision.BaseTradeDecision.get_data_cal_range_limit",
        "inner_decision_modification_authority": "qlib.backtest.decision.BaseTradeDecision.mod_inner_decision",
        "calendar_authority": "qlib.backtest.utils.TradeCalendarManager",
        "level_infra_authority": "qlib.backtest.utils.LevelInfrastructure",
        "atomicity_rule": "atomic_executor_rejects_trade_decision_range_limit",
        "settle_sequence_rule": "settle_start_runs_before_collection_and_settle_commit_after_bar_end_when_enabled",
        "bar_end_sequence_rule": "executor_updates_account_bar_end_before_trade_calendar_step",
        "track_data_rule": "track_data_yields_outer_trade_decision_for_training_data_only",
        "simulator_order_rule": "simulator_executor_retrieves_order_decisions_and_deals_each_order_through_exchange",
        "simulator_trade_type_rule": (
            "serial_preserves_order_sequence_parallel_sorts_buys_before_sells_to_surface_cash_conflicts"
        ),
        "daily_dealt_amount_rule": "simulator_resets_dealt_order_amount_when_trade_day_advances",
        "nested_init_rule": (
            "nested_executor_resets_inner_executor_to_outer_step_window_and_passes_outer_decision_to_inner_strategy"
        ),
        "nested_update_rule": "nested_executor_updates_outer_decision_with_inner_calendar_before_range_limit_alignment",
        "nested_range_rule": "nested_executor_skips_inner_steps_outside_range_limit_when_align_range_limit_is_enabled",
        "inner_decision_rule": (
            "outer_trade_decision_may_propagate_trade_range_into_inner_trade_decision_only_when_inner_range_absent"
        ),
        "empty_decision_rule": "empty_decision_can_skip_inner_loop_when_skip_empty_decision_is_enabled",
        "inner_result_rule": "nested_executor_collects_inner_execute_results_order_indicators_and_decision_time_windows",
        "rdagent_rule": "describe_only_do_not_redefine_executor_decision_lifecycle_or_nested_execution_order",
    }
    for key, expected_value in expected_executor_decision_values.items():
        if executor_decision.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"executor_decision_semantics must preserve {key}"
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
    price_adjustment = _mapping(prompt_payload.get("price_adjustment_semantics"))
    for key in (
        "semantic_name",
        "factor_field",
        "factor_usage",
        "missing_factor_rule",
        "adjusted_price_mode_rule",
        "extra_quote_factor_rule",
        "suspension_interaction",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in price_adjustment:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must include {key}"
            )
    if price_adjustment.get("semantic_name") != "a_share_price_adjustment_order_factor":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must describe A-share order factor semantics"
        )
    if price_adjustment.get("factor_field") != "$factor":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must bind Qlib $factor"
        )
    if price_adjustment.get("factor_usage") != (
        "convert_adjusted_amounts_to_trade_unit_amounts_when_unadjusted_prices_are_used"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must declare factor usage"
        )
    if price_adjustment.get("missing_factor_rule") != (
        "non_suspended_rows_with_missing_factor_use_adjusted_price_mode_and_disable_trade_unit_rounding"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must fail closed on missing factor rule drift"
        )
    if price_adjustment.get("adjusted_price_mode_rule") != (
        "trade_unit_rounding_is_not_supported_when_adjusted_price_mode_is_active"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must declare adjusted-price mode"
        )
    if price_adjustment.get("runtime_authority") != "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must name Qlib runtime authority"
        )
    if price_adjustment.get("rdagent_rule") != "describe_only_do_not_redefine_price_adjustment_or_order_factor":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_adjustment_semantics must forbid RD-Agent redefinition"
        )
    price_limit = _mapping(prompt_payload.get("price_limit_semantics"))
    for key in (
        "limit_threshold",
        "price_limit_mode",
        "authoritative_limit_fields",
        "field_authority",
        "semantic_name",
        "limit_flag_fields",
        "limit_flag_meaning",
        "buy_limit_rule",
        "sell_limit_rule",
        "missing_authoritative_fields",
        "strict_mode_missing_fields_rule",
        "board_fallback_policy",
        "fallback_authority_rule",
        "board_limit_thresholds",
        "runtime_authority",
        "rdagent_rule",
    ):
        if key not in price_limit:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload price_limit_semantics must include {key}"
            )
    if price_limit.get("semantic_name") != "a_share_price_limit_authority":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must describe A-share price-limit authority"
        )
    if price_limit.get("limit_threshold") != prompt_market.get("limit_threshold"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must match market limit"
        )
    if price_limit.get("field_authority") != "provider_up_down_limit_fields":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must use provider field authority"
        )
    if price_limit.get("authoritative_limit_fields") != prompt_market.get("authoritative_limit_fields"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must match market fields"
        )
    if price_limit.get("authoritative_limit_fields") != ["$up_limit", "$down_limit"]:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must bind provider up/down fields"
        )
    if price_limit.get("limit_flag_fields") != ["limit_buy", "limit_sell"]:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must expose limit flag fields"
        )
    if price_limit.get("limit_flag_meaning") != "true_flags_mark_direction_not_tradable":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare limit flag meaning"
        )
    if price_limit.get("buy_limit_rule") != "buy_price_at_or_above_up_limit_or_suspended_sets_limit_buy":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare buy limit rule"
        )
    if price_limit.get("sell_limit_rule") != "sell_price_at_or_below_down_limit_or_suspended_sets_limit_sell":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare sell limit rule"
        )
    if price_limit.get("price_limit_mode") not in {"strict", "auto"}:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare strict or auto mode"
        )
    if price_limit.get("missing_authoritative_fields") != (
        "fail_closed_in_strict_mode_else_qlib_board_fallback_for_legacy_datasets"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare missing field behavior"
        )
    if price_limit.get("strict_mode_missing_fields_rule") != (
        "missing_authoritative_fields_or_non_suspended_bounds_fail_closed"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must declare strict missing-field rule"
        )
    if price_limit.get("board_fallback_policy") != "runtime_compatibility_only_when_authoritative_fields_are_absent":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must keep board fallback bounded"
        )
    if price_limit.get("fallback_authority_rule") != (
        "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must not promote board fallback"
        )
    if price_limit.get("runtime_authority") != (
        "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload price_limit_semantics must name Qlib runtime authority"
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
    order_tradability = _mapping(prompt_payload.get("order_tradability_semantics"))
    for key in (
        "semantic_name",
        "runtime_authority",
        "tradability_authority",
        "suspension_authority",
        "price_limit_authority",
        "failure_result",
        "failed_order_state_field",
        "directional_limit_rule",
        "all_direction_limit_rule",
        "suspension_rule",
        "limit_rule",
        "decision_rule",
        "rdagent_rule",
    ):
        if key not in order_tradability:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload order_tradability_semantics must include {key}"
            )
    expected_order_tradability_values = {
        "semantic_name": "a_share_order_tradability_gate",
        "runtime_authority": "qlib.backtest.exchange.Exchange.check_order",
        "tradability_authority": "qlib.backtest.exchange.Exchange.is_stock_tradable",
        "suspension_authority": "qlib.backtest.exchange.Exchange.check_stock_suspended",
        "price_limit_authority": "qlib.backtest.exchange.Exchange.check_stock_limit",
        "failure_result": "deal_amount_zero_trade_value_zero_cost_nan_price",
        "failed_order_state_field": "Order.deal_amount",
        "directional_limit_rule": "buy_orders_check_limit_buy_and_sell_orders_check_limit_sell",
        "all_direction_limit_rule": "missing_direction_checks_any_buy_or_sell_limit",
        "suspension_rule": "missing_close_or_unknown_stock_is_not_tradable",
        "limit_rule": "limit_flags_true_mark_direction_not_tradable",
        "decision_rule": "check_order_delegates_to_is_stock_tradable_before_deal_execution",
        "rdagent_rule": "describe_only_do_not_redefine_order_tradability_or_limit_checks",
    }
    for key, expected_value in expected_order_tradability_values.items():
        if order_tradability.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"order_tradability_semantics must preserve {key}"
            )
    order_fill_amount = _mapping(prompt_payload.get("order_fill_amount_semantics"))
    for key in (
        "semantic_name",
        "runtime_authority",
        "fill_state_field",
        "initial_fill_rule",
        "clip_sequence",
        "volume_clip_authority",
        "sellable_position_authority",
        "cash_authority",
        "cash_limit_authority",
        "round_lot_authority",
        "factor_authority",
        "unknown_position_rule",
        "sell_full_liquidation_rule",
        "trade_value_rule",
        "cost_rule",
        "rdagent_rule",
    ):
        if key not in order_fill_amount:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload order_fill_amount_semantics must include {key}"
            )
    expected_order_fill_values = {
        "semantic_name": "a_share_order_fill_amount_gate",
        "runtime_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
        "fill_state_field": "Order.deal_amount",
        "initial_fill_rule": "deal_amount_starts_as_order_amount_before_runtime_clips",
        "clip_sequence": [
            "volume_capacity_clip",
            "sellable_position_clip",
            "sell_cash_cost_guard",
            "buy_cash_cost_guard",
            "round_lot_or_full_liquidation_clip",
        ],
        "volume_clip_authority": "qlib.backtest.exchange.Exchange._clip_amount_by_volume",
        "sellable_position_authority": "qlib.backtest.position.Position.get_sellable_amount",
        "cash_authority": "qlib.backtest.position.Position.get_cash",
        "cash_limit_authority": "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit",
        "round_lot_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
        "factor_authority": "qlib.backtest.exchange.Exchange.get_factor",
        "unknown_position_rule": "unknown_position_uses_round_lot_without_cash_or_sellable_clips",
        "sell_full_liquidation_rule": (
            "sells_equal_to_current_sellable_amount_keep_full_liquidation_without_round_lot_residual"
        ),
        "trade_value_rule": "trade_value_is_final_deal_amount_times_trade_price",
        "cost_rule": "trade_cost_recomputed_after_final_deal_amount",
        "rdagent_rule": "describe_only_do_not_redefine_order_fill_amount_or_clip_sequence",
    }
    for key, expected_value in expected_order_fill_values.items():
        if order_fill_amount.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"order_fill_amount_semantics must preserve {key}"
            )
    settlement = _mapping(prompt_payload.get("settlement_semantics"))
    for key in (
        "semantic_name",
        "settlement_rule",
        "same_day_sell_policy",
        "position_type",
        "sellable_state_field",
        "initial_sellable_rule",
        "intraday_buy_rule",
        "intraday_bar_rule",
        "day_commit_rule",
        "sell_order_clip_rule",
        "sell_overdraft_rule",
        "runtime_authority",
        "exchange_clip_authority",
        "rdagent_rule",
    ):
        if key not in settlement:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload settlement_semantics must include {key}"
            )
    if settlement.get("semantic_name") != "a_share_t_plus_1_stock_settlement":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must describe A-share T+1 settlement"
        )
    if settlement.get("settlement_rule") != prompt_market.get("settlement_rule"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must match market settlement"
        )
    if settlement.get("settlement_rule") != "t_plus_1_stock":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must bind T+1 stock settlement"
        )
    if settlement.get("same_day_sell_policy") != "shares_bought_today_are_unsellable_until_day_commit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must bind same-day sell policy"
        )
    if settlement.get("position_type") != prompt_market.get("position_type"):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must match market position"
        )
    if settlement.get("position_type") != "AsharePosition":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must use AsharePosition"
        )
    if settlement.get("sellable_state_field") != "sellable_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must expose sellable state"
        )
    if settlement.get("initial_sellable_rule") != "existing_or_settled_holdings_are_sellable":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must declare initial sellability"
        )
    if settlement.get("intraday_buy_rule") != "same_day_buys_increase_total_amount_but_not_sellable_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must declare intraday buy rule"
        )
    if settlement.get("intraday_bar_rule") != "non_day_bars_do_not_release_same_day_buys":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must declare intraday bar rule"
        )
    if settlement.get("day_commit_rule") != "day_bar_commit_sets_sellable_amount_to_total_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must declare day commit rule"
        )
    if settlement.get("sell_order_clip_rule") != "sell_orders_are_clipped_by_position_get_sellable_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must declare sell clipping"
        )
    if settlement.get("sell_overdraft_rule") != "AsharePosition_rejects_sells_above_sellable_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must declare sell overdraft rule"
        )
    if settlement.get("runtime_authority") != "qlib.backtest.position.AsharePosition":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must name Qlib position authority"
        )
    if settlement.get("exchange_clip_authority") != "qlib.backtest.exchange.Exchange._calc_trade_info_by_order":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must name exchange clip authority"
        )
    if settlement.get("rdagent_rule") != "describe_only_do_not_redefine_position_or_settlement":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload settlement_semantics must forbid RD-Agent redefinition"
        )
    cash_constraint = _mapping(prompt_payload.get("cash_constraint_semantics"))
    for key in (
        "semantic_name",
        "cash_state_field",
        "cash_query_rule",
        "buy_cash_rule",
        "minimum_cost_rule",
        "partial_buy_rule",
        "shorting_policy",
        "sell_position_rule",
        "sell_cash_rule",
        "runtime_authority",
        "cash_limit_authority",
        "position_cash_authority",
        "rdagent_rule",
    ):
        if key not in cash_constraint:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must include {key}"
            )
    if cash_constraint.get("semantic_name") != "a_share_cash_buying_power_and_shorting_policy":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must describe A-share cash and shorting policy"
        )
    if cash_constraint.get("cash_state_field") != "cash":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must expose cash state"
        )
    if cash_constraint.get("cash_query_rule") != "buying_power_uses_position_get_cash_without_unsettled_cash":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must declare cash query rule"
        )
    if cash_constraint.get("buy_cash_rule") != "buy_orders_are_clipped_by_available_cash_and_transaction_cost":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must declare buy cash clipping"
        )
    if cash_constraint.get("minimum_cost_rule") != "orders_without_cash_for_minimum_cost_are_zeroed":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must declare minimum cost cash rule"
        )
    if cash_constraint.get("partial_buy_rule") != (
        "cash_insufficient_orders_are_reduced_by_exchange_cash_limit_then_round_lot"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must declare partial buy rule"
        )
    if cash_constraint.get("shorting_policy") != "equity_short_selling_is_not_enabled":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must forbid implicit shorting"
        )
    if cash_constraint.get("sell_position_rule") != "sell_orders_are_clipped_by_position_get_sellable_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must declare sell position clipping"
        )
    if cash_constraint.get("sell_cash_rule") != "sell_orders_zero_when_cash_plus_trade_value_cannot_cover_sell_cost":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must declare sell cash guard"
        )
    if cash_constraint.get("runtime_authority") != "qlib.backtest.exchange.Exchange._calc_trade_info_by_order":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must name exchange runtime authority"
        )
    if cash_constraint.get("cash_limit_authority") != "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must name cash limit authority"
        )
    if cash_constraint.get("position_cash_authority") != "qlib.backtest.position.Position.get_cash":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must name position cash authority"
        )
    if cash_constraint.get("rdagent_rule") != "describe_only_do_not_redefine_cash_or_shorting_policy":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload cash_constraint_semantics must forbid RD-Agent cash redefinition"
        )
    cash_settlement = _mapping(prompt_payload.get("cash_settlement_semantics"))
    for key in (
        "semantic_name",
        "settlement_authority",
        "settle_start_authority",
        "settle_commit_authority",
        "available_cash_authority",
        "delayed_cash_state_field",
        "delayed_cash_mode",
        "no_delay_cash_mode",
        "sell_proceeds_rule",
        "default_sell_proceeds_rule",
        "available_cash_rule",
        "account_value_rule",
        "commit_rule",
        "rdagent_rule",
    ):
        if key not in cash_settlement:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload cash_settlement_semantics must include {key}"
            )
    expected_cash_settlement_values = {
        "semantic_name": "a_share_sell_proceeds_cash_settlement",
        "settlement_authority": "qlib.backtest.position.Position",
        "settle_start_authority": "qlib.backtest.position.Position.settle_start",
        "settle_commit_authority": "qlib.backtest.position.Position.settle_commit",
        "available_cash_authority": "qlib.backtest.position.Position.get_cash",
        "delayed_cash_state_field": "cash_delay",
        "delayed_cash_mode": "Position.ST_CASH",
        "no_delay_cash_mode": "Position.ST_NO",
        "sell_proceeds_rule": "sell_proceeds_enter_cash_delay_when_settle_type_is_cash",
        "default_sell_proceeds_rule": "sell_proceeds_enter_cash_immediately_when_settle_type_is_none",
        "available_cash_rule": "get_cash_excludes_cash_delay_unless_include_settle_is_true",
        "account_value_rule": "calculate_value_includes_cash_delay",
        "commit_rule": "settle_commit_moves_cash_delay_into_cash_and_clears_delay_state",
        "rdagent_rule": "describe_only_do_not_redefine_cash_settlement_or_sell_proceeds_availability",
    }
    for key, expected_value in expected_cash_settlement_values.items():
        if cash_settlement.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"cash_settlement_semantics must preserve {key}"
            )
    liquidity_capacity = _mapping(prompt_payload.get("liquidity_capacity_semantics"))
    for key in (
        "semantic_name",
        "volume_field",
        "capacity_parameter",
        "capacity_scope",
        "default_capacity_rule",
        "volume_limit_aggregation_rule",
        "cumulative_limit_rule",
        "current_limit_rule",
        "dealt_order_state",
        "capacity_clip_rule",
        "runtime_authority",
        "threshold_parser_authority",
        "rdagent_rule",
    ):
        if key not in liquidity_capacity:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must include {key}"
            )
    if liquidity_capacity.get("semantic_name") != "a_share_volume_capacity_limit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must describe A-share capacity"
        )
    if liquidity_capacity.get("volume_field") != "$volume":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must bind Qlib volume field"
        )
    if liquidity_capacity.get("capacity_parameter") != "volume_threshold":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must bind volume_threshold"
        )
    if liquidity_capacity.get("capacity_scope") != "runtime_handoff_only_when_volume_threshold_is_configured":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must keep capacity runtime scoped"
        )
    if (
        liquidity_capacity.get("default_capacity_rule")
        != "no_prompt_defined_capacity_limit_in_default_joinquant_ashare_contract"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must forbid prompt capacity defaults"
        )
    if liquidity_capacity.get("volume_limit_aggregation_rule") != "multiple_volume_limits_are_aggregated_by_min":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must declare min aggregation"
        )
    if liquidity_capacity.get("cumulative_limit_rule") != "cum_volume_limits_subtract_dealt_order_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must declare cumulative volume behavior"
        )
    if liquidity_capacity.get("current_limit_rule") != "current_volume_limits_use_current_quote_value":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must declare current volume behavior"
        )
    if liquidity_capacity.get("dealt_order_state") != "dealt_order_amount":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must declare dealt order state"
        )
    if (
        liquidity_capacity.get("capacity_clip_rule")
        != "order_deal_amount_is_clipped_to_nonnegative_configured_volume_capacity"
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must declare capacity clipping"
        )
    if liquidity_capacity.get("runtime_authority") != "qlib.backtest.exchange.Exchange._clip_amount_by_volume":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must name Qlib capacity authority"
        )
    if liquidity_capacity.get("threshold_parser_authority") != "qlib.backtest.exchange.Exchange._get_vol_limit":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must name threshold parser authority"
        )
    if liquidity_capacity.get("rdagent_rule") != "describe_only_do_not_redefine_liquidity_or_volume_capacity":
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract prompt_projection_payload liquidity_capacity_semantics must forbid RD-Agent capacity redefinition"
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
        "data_frequency",
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
        "data_frequency",
        "instrument_identity_semantics",
        "universe_membership_semantics",
        "trading_calendar_semantics",
        "transaction_cost_semantics",
        "market_impact_semantics",
        "account_update_semantics",
        "account_valuation_semantics",
        "trade_indicator_semantics",
        "executor_decision_semantics",
        "suspension_tradability_semantics",
        "execution_price_semantics",
        "price_adjustment_semantics",
        "price_limit_semantics",
        "order_tradability_semantics",
        "order_fill_amount_semantics",
        "settlement_semantics",
        "cash_settlement_semantics",
        "cash_constraint_semantics",
        "liquidity_capacity_semantics",
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
