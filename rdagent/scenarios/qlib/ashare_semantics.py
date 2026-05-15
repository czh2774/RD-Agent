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
QLIB_ASHARE_LABEL_COLUMN = "LABEL0"
QLIB_ASHARE_LABEL_EXPRESSION = "Ref($close, -2)/Ref($close, -1) - 1"
QLIB_ASHARE_LABEL_TEMPLATE_PATHS = (
    "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
    "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
    "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
    "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
    "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
)
QLIB_ASHARE_TEMPLATE_MARKET = "csi300"
QLIB_ASHARE_TEMPLATE_BENCHMARK = "SH000300"
QLIB_ASHARE_UNIVERSE_BENCHMARK_TEMPLATE_PATHS = QLIB_ASHARE_LABEL_TEMPLATE_PATHS
QLIB_ASHARE_RUNTIME_TEMPLATE_PATHS = QLIB_ASHARE_LABEL_TEMPLATE_PATHS
QLIB_ASHARE_RUNTIME_EXCHANGE_KWARGS = {
    "limit_threshold": "joinquant_ashare",
    "ashare_price_limit_mode": "strict",
    "ashare_limit_options": {
        "open_cost": 0.0003,
        "close_commission": 0.0003,
        "close_tax": 0.001,
        "min_cost": 5.0,
    },
    "trade_unit": 100,
    "deal_price": "close",
    "open_cost": 0.0003,
    "close_cost": 0.0013,
    "min_cost": 5.0,
}
QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS = {
    "pos_type": "AsharePosition",
    "exchange_kwargs": QLIB_ASHARE_RUNTIME_EXCHANGE_KWARGS,
}
QLIB_ASHARE_FORBIDDEN_LEGACY_EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "open_cost": 0.0005,
    "close_cost": 0.0015,
}
QLIB_ASHARE_RESEARCH_DATA_SOURCE_PROMPT_PATHS = ("rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml",)
QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS = ("$open", "$close", "$high", "$low", "$vwap", "$volume")
QLIB_ASHARE_POINT_IN_TIME_REGISTRATION_RULE = "user_or_provider_supplied_non_price_volume_fields_must_name_source_owner_field_identity_and_daily_point_in_time_validity"
QLIB_ASHARE_TURNOVER_INPUT_BOUNDARY_RULE = (
    "turnover_is_not_a_default_factor_input_field_even_when_qlib_reports_portfolio_turnover"
)
QLIB_ASHARE_TURNOVER_REPORT_METRIC_RULE = (
    "report_turnover_is_post_backtest_portfolio_metric_not_default_factor_input_field"
)
QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES = (
    "turnover",
    "minute_level_high_frequency_data",
    "analyst_consensus_expectation_factor",
    "unregistered_external_vendor_fields",
)
QLIB_ASHARE_LABEL_PROMPT_PATHS = ("rdagent/scenarios/qlib/experiment/prompts.yaml",)
QLIB_ASHARE_PREDICTION_SIGNAL_PROMPT_PATHS = (
    "rdagent/scenarios/qlib/experiment/prompts.yaml",
    "rdagent/scenarios/qlib/prompts.yaml",
)
QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS = ("IC", "ICIR", "Rank IC", "Rank ICIR")
QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS = (
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.max_drawdown",
)
QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITHOUT_COST = QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS[0]
QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS = (
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
)
QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITH_COST = QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS[0]
QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS = (
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
)
QLIB_ASHARE_PORTFOLIO_UI_METRIC_PATHS = QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS
QLIB_ASHARE_PROMPT_METRIC_PATHS = ("IC", *QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS)
QLIB_ASHARE_FEEDBACK_METRIC_PATHS = ("IC", *QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS)
QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC = QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS[0]
QLIB_ASHARE_BANDIT_METRIC_PATHS = (
    *QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS,
    *QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS,
)
QLIB_ASHARE_UI_SELECTED_METRICS = ("IC", *QLIB_ASHARE_PORTFOLIO_UI_METRIC_PATHS)
QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME = "drawdown_adjusted_return"
QLIB_ASHARE_FEEDBACK_METRIC_PROMPT_PATHS = (
    "rdagent/scenarios/qlib/experiment/prompts.yaml",
    "rdagent/scenarios/qlib/prompts.yaml",
)
QLIB_ASHARE_FEEDBACK_METRIC_SOURCE_PATHS = (
    "rdagent/scenarios/qlib/developer/feedback.py",
    "rdagent/scenarios/qlib/proposal/bandit.py",
    "rdagent/scenarios/qlib/experiment/prompts.yaml",
    "rdagent/scenarios/qlib/prompts.yaml",
    "rdagent/log/ui/app.py",
)
QLIB_ASHARE_EXCESS_RETURN_FORBIDDEN_SUBSTITUTIONS = (
    "raw_return_as_excess_return",
    "market_universe_as_benchmark_return",
    "with_cost_metric_without_report_cost_column",
    "prompt_defined_cost_or_benchmark_formula",
)


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
        "template_runtime_binding": deepcopy(handoff_contract["template_runtime_binding"]),
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
    strategy_order = _mapping(prompt_payload.get("strategy_order_semantics"))
    supervised_label = _mapping(prompt_payload.get("supervised_label_semantics"))
    prediction_signal = _mapping(prompt_payload.get("prediction_signal_semantics"))
    signal_ic = _mapping(prompt_payload.get("signal_ic_semantics"))
    portfolio_risk = _mapping(prompt_payload.get("portfolio_risk_semantics"))
    excess_return = _mapping(prompt_payload.get("excess_return_semantics"))
    feedback_metric = _mapping(prompt_payload.get("feedback_metric_semantics"))
    benchmark_return = _mapping(prompt_payload.get("benchmark_return_semantics"))
    universe_benchmark_binding = _mapping(prompt_payload.get("universe_benchmark_binding_semantics"))
    runtime_template_binding = _mapping(prompt_payload.get("runtime_handoff_template_binding_semantics"))
    research_data_source = _mapping(prompt_payload.get("research_data_source_semantics"))
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
            f"- strategy-order authority: pyqlib ({strategy_order.get('topk_strategy_authority')})",
            f"- strategy-order template binding: {strategy_order.get('template_strategy_binding')}",
            f"- strategy-order prediction window: {strategy_order.get('prediction_window_rule')}",
            f"- strategy-order dropout rule: {strategy_order.get('dropout_rule')}",
            f"- strategy-order order return rule: {strategy_order.get('target_order_return_rule')}",
            f"- supervised-label authority: pyqlib ({supervised_label.get('handler_authority')})",
            f"- supervised-label column: {supervised_label.get('label_column')}",
            f"- supervised-label expression: {supervised_label.get('label_expression')}",
            f"- supervised-label horizon: {supervised_label.get('label_horizon_rule')}",
            f"- supervised-label prompt wording: {supervised_label.get('prompt_wording_rule')}",
            f"- prediction-signal authority: pyqlib ({prediction_signal.get('model_signal_authority')})",
            f"- prediction-signal artifact: {prediction_signal.get('prediction_artifact')}",
            f"- prediction-signal column: {prediction_signal.get('prediction_column')}",
            f"- prediction-signal model rule: {prediction_signal.get('model_predict_rule')}",
            f"- prediction-signal ranking rule: {prediction_signal.get('strategy_ranking_rule')}",
            f"- prediction-signal prompt wording: {prediction_signal.get('prompt_wording_rule')}",
            f"- signal-ic authority: pyqlib ({signal_ic.get('signal_analysis_authority')})",
            f"- signal-ic calculation: pyqlib ({signal_ic.get('ic_calculation_authority')})",
            "- signal-ic metrics: " + ", ".join(str(item) for item in signal_ic.get("metric_fields", [])),
            f"- signal-ic groupby: {signal_ic.get('groupby_level')}",
            f"- signal-ic IC rule: {signal_ic.get('ic_rule')}",
            f"- signal-ic Rank IC rule: {signal_ic.get('rank_ic_rule')}",
            f"- signal-ic portfolio boundary: {signal_ic.get('portfolio_boundary_rule')}",
            f"- portfolio-risk authority: pyqlib ({portfolio_risk.get('risk_analysis_authority')})",
            "- portfolio-risk metrics: "
            + ", ".join(str(item) for item in portfolio_risk.get("risk_metric_fields", [])),
            "- portfolio-risk consumed paths: "
            + ", ".join(str(item) for item in portfolio_risk.get("rdagent_consumed_metric_paths", [])),
            f"- portfolio-risk metric path format: {portfolio_risk.get('metric_path_format')}",
            f"- portfolio-risk metric path whitespace rule: {portfolio_risk.get('metric_path_whitespace_rule')}",
            f"- portfolio-risk turnover metric rule: {portfolio_risk.get('turnover_report_metric_rule')}",
            "- portfolio-risk prompt paths: "
            + ", ".join(str(item) for item in portfolio_risk.get("rdagent_prompt_metric_paths", [])),
            "- portfolio-risk feedback paths: "
            + ", ".join(str(item) for item in portfolio_risk.get("rdagent_feedback_metric_paths", [])),
            "- portfolio-risk bandit paths: "
            + ", ".join(str(item) for item in portfolio_risk.get("rdagent_bandit_metric_paths", [])),
            "- portfolio-risk UI paths: "
            + ", ".join(str(item) for item in portfolio_risk.get("rdagent_ui_metric_paths", [])),
            f"- portfolio-risk annualization scaler: {portfolio_risk.get('day_annualization_scaler')}",
            f"- portfolio-risk max drawdown rule: {portfolio_risk.get('max_drawdown_rule')}",
            f"- excess-return authority: pyqlib ({excess_return.get('report_column_authority')})",
            f"- excess-return without-cost formula: {excess_return.get('without_cost_formula')}",
            f"- excess-return with-cost formula: {excess_return.get('with_cost_formula')}",
            "- excess-return metric paths: "
            + ", ".join(
                str(item)
                for item in (
                    excess_return.get("metric_path_without_cost"),
                    excess_return.get("metric_path_with_cost"),
                )
                if item
            ),
            "- excess-return forbidden substitutions: "
            + ", ".join(str(item) for item in excess_return.get("forbidden_substitutions", [])),
            f"- excess-return prompt rule: {excess_return.get('rdagent_prompt_rule')}",
            f"- feedback-metric authority: pyqlib ({feedback_metric.get('portfolio_metric_authority')})",
            f"- feedback-metric primary: {feedback_metric.get('feedback_primary_metric')}",
            "- feedback-metric paths: "
            + ", ".join(str(item) for item in feedback_metric.get("feedback_metric_paths", [])),
            f"- feedback-metric bandit utility: {feedback_metric.get('derived_bandit_utility_name')}",
            f"- feedback-metric utility rule: {feedback_metric.get('derived_bandit_utility_rule')}",
            "- feedback-metric forbidden aliases: "
            + ", ".join(str(item) for item in feedback_metric.get("forbidden_metric_aliases", [])),
            f"- benchmark-return authority: pyqlib ({benchmark_return.get('benchmark_calculation_authority')})",
            f"- benchmark-return default: {benchmark_return.get('default_benchmark')}",
            f"- benchmark-return field: {benchmark_return.get('benchmark_field_expression')}",
            f"- benchmark-return sample rule: {benchmark_return.get('sample_rule')}",
            f"- benchmark-return report column: {benchmark_return.get('report_column')}",
            f"- universe-benchmark market value: {universe_benchmark_binding.get('template_market_value')}",
            f"- universe-benchmark benchmark value: {universe_benchmark_binding.get('template_benchmark_value')}",
            f"- universe-benchmark market rule: {universe_benchmark_binding.get('market_universe_rule')}",
            f"- universe-benchmark benchmark rule: {universe_benchmark_binding.get('benchmark_rule')}",
            f"- universe-benchmark separation rule: {universe_benchmark_binding.get('separation_rule')}",
            f"- universe-benchmark template rule: {universe_benchmark_binding.get('rdagent_rule')}",
            f"- runtime-handoff template binding: {runtime_template_binding.get('binding_kind')}",
            f"- runtime-handoff template rule: {runtime_template_binding.get('runtime_rule')}",
            f"- runtime-handoff prompt boundary: {runtime_template_binding.get('prompt_boundary_rule')}",
            f"- research data-source frequency: {research_data_source.get('data_frequency')}",
            "- research data-source fields: "
            + ", ".join(str(item) for item in research_data_source.get("primary_price_volume_fields", [])),
            "- research data-source forbidden defaults: "
            + ", ".join(str(item) for item in research_data_source.get("forbidden_default_prompt_sources", [])),
            f"- research data-source PIT registration: {research_data_source.get('point_in_time_registration_rule')}",
            f"- research data-source turnover input boundary: {research_data_source.get('turnover_input_boundary_rule')}",
            f"- research data-source rule: {research_data_source.get('rdagent_rule')}",
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
        "redefine_strategy_signal_to_order_generation",
        "redefine_supervised_label_expression_or_horizon",
        "redefine_prediction_signal_score_or_return_realization",
        "redefine_signal_ic_or_rank_ic_metrics",
        "redefine_portfolio_risk_analysis_metrics",
        "redefine_benchmark_relative_excess_return_or_cost_treatment",
        "redefine_feedback_metric_paths_or_label_derived_utility_as_qlib_metric",
        "redefine_benchmark_return_series_or_default_benchmark",
        "redefine_universe_benchmark_template_binding_or_cross_alias_market_and_benchmark",
        "redefine_runtime_handoff_or_template_execution_kwargs",
        "redefine_research_data_source_availability_or_imply_unregistered_sources",
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
        "strategy_order_semantics",
        "supervised_label_semantics",
        "prediction_signal_semantics",
        "signal_ic_semantics",
        "portfolio_risk_semantics",
        "excess_return_semantics",
        "feedback_metric_semantics",
        "benchmark_return_semantics",
        "universe_benchmark_binding_semantics",
        "runtime_handoff_template_binding_semantics",
        "research_data_source_semantics",
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
        "strategy_order_semantics",
        "supervised_label_semantics",
        "prediction_signal_semantics",
        "signal_ic_semantics",
        "portfolio_risk_semantics",
        "excess_return_semantics",
        "feedback_metric_semantics",
        "benchmark_return_semantics",
        "universe_benchmark_binding_semantics",
        "runtime_handoff_template_binding_semantics",
        "research_data_source_semantics",
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
    strategy_order = _mapping(prompt_payload.get("strategy_order_semantics"))
    for key in (
        "semantic_name",
        "base_strategy_authority",
        "topk_strategy_authority",
        "weight_strategy_authority",
        "order_generator_authority",
        "interacting_order_generator_authority",
        "non_interacting_order_generator_authority",
        "target_amount_order_authority",
        "trade_decision_type",
        "signal_authority",
        "template_strategy_binding",
        "prediction_window_rule",
        "dataframe_signal_rule",
        "missing_signal_rule",
        "topk_selection_rule",
        "dropout_rule",
        "sell_order_rule",
        "buy_budget_rule",
        "hold_threshold_rule",
        "only_tradable_rule",
        "forbid_all_trade_at_limit_rule",
        "buy_round_lot_rule",
        "weight_strategy_rule",
        "interacting_generator_rule",
        "non_interacting_generator_rule",
        "target_order_rule",
        "target_order_return_rule",
        "rdagent_rule",
    ):
        if key not in strategy_order:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload strategy_order_semantics must include {key}"
            )
    expected_strategy_order_values = {
        "semantic_name": "a_share_strategy_signal_to_order_generation",
        "base_strategy_authority": "qlib.strategy.base.BaseStrategy.generate_trade_decision",
        "topk_strategy_authority": "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision",
        "weight_strategy_authority": "qlib.contrib.strategy.signal_strategy.WeightStrategyBase.generate_trade_decision",
        "order_generator_authority": (
            "qlib.contrib.strategy.order_generator.OrderGenerator.generate_order_list_from_target_weight_position"
        ),
        "interacting_order_generator_authority": (
            "qlib.contrib.strategy.order_generator.OrderGenWInteract.generate_order_list_from_target_weight_position"
        ),
        "non_interacting_order_generator_authority": (
            "qlib.contrib.strategy.order_generator.OrderGenWOInteract.generate_order_list_from_target_weight_position"
        ),
        "target_amount_order_authority": "qlib.backtest.exchange.Exchange.generate_order_for_target_amount_position",
        "trade_decision_type": "qlib.backtest.decision.TradeDecisionWO",
        "signal_authority": "qlib.backtest.signal.Signal.get_signal",
        "template_strategy_binding": "qlib.contrib.strategy.TopkDropoutStrategy",
        "prediction_window_rule": "strategy_reads_signal_from_previous_calendar_step_shift_one",
        "dataframe_signal_rule": "topk_dropout_uses_first_signal_column_when_prediction_is_dataframe",
        "missing_signal_rule": "missing_signal_returns_empty_TradeDecisionWO",
        "topk_selection_rule": "topk_dropout_ranks_current_holdings_and_new_candidates_by_pred_score_descending",
        "dropout_rule": "combined_last_and_today_scores_prevent_dropping_higher_score_stock_for_lower_score_buy",
        "sell_order_rule": "sell_orders_are_generated_before_buy_orders_and_simulated_on_temp_position_for_cash",
        "buy_budget_rule": "buy_budget_equals_temp_cash_times_risk_degree_divided_by_buy_count",
        "hold_threshold_rule": "sell_requires_current_holding_count_at_least_hold_thresh",
        "only_tradable_rule": "only_tradable_filters_selection_candidates_by_exchange_tradability",
        "forbid_all_trade_at_limit_rule": (
            "forbid_all_trade_at_limit_checks_any_limit_direction_else_directional_limit"
        ),
        "buy_round_lot_rule": "buy_amount_uses_deal_price_factor_and_exchange_round_amount_by_trade_unit",
        "weight_strategy_rule": "weight_strategy_delegates_target_weight_to_order_generator_after_signal_lookup",
        "interacting_generator_rule": "interacting_order_generator_uses_trade_date_tradability_and_prices",
        "non_interacting_generator_rule": (
            "non_interacting_order_generator_uses_pred_date_close_or_current_holding_price"
        ),
        "target_order_rule": "exchange_generates_target_amount_orders_with_deterministic_shuffled_stock_order",
        "target_order_return_rule": "exchange_returns_sell_orders_before_buy_orders",
        "rdagent_rule": "describe_only_do_not_redefine_strategy_signal_to_order_generation",
    }
    for key, expected_value in expected_strategy_order_values.items():
        if strategy_order.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"strategy_order_semantics must preserve {key}"
            )
    supervised_label = _mapping(prompt_payload.get("supervised_label_semantics"))
    for key in (
        "semantic_name",
        "handler_authority",
        "handler360_authority",
        "loader_authority",
        "processor_authority",
        "label_group",
        "label_column",
        "label_expression",
        "label_expression_source",
        "label_horizon_rule",
        "prediction_execution_alignment_rule",
        "dropna_processor_rule",
        "template_binding_rule",
        "prompt_wording_rule",
        "rdagent_template_paths",
        "rdagent_prompt_paths",
        "rdagent_rule",
    ):
        if key not in supervised_label:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload supervised_label_semantics must include {key}"
            )
    expected_supervised_label_values = {
        "semantic_name": "a_share_supervised_forward_return_label",
        "handler_authority": "qlib.contrib.data.handler.Alpha158",
        "handler360_authority": "qlib.contrib.data.handler.Alpha360",
        "loader_authority": "qlib.contrib.data.loader.Alpha158DL",
        "processor_authority": "qlib.data.dataset.processor.DropnaLabel",
        "label_group": "label",
        "label_column": QLIB_ASHARE_LABEL_COLUMN,
        "label_expression": QLIB_ASHARE_LABEL_EXPRESSION,
        "label_expression_source": "Alpha158.get_label_config_and_Alpha360.get_label_config",
        "label_horizon_rule": "label_at_datetime_t_is_close_t_plus_2_over_close_t_plus_1_minus_one",
        "prediction_execution_alignment_rule": (
            "label_horizon_matches_strategy_previous_step_signal_execution_without_same_day_fill_assumption"
        ),
        "dropna_processor_rule": "DropnaLabel_removes_missing_LABEL0_rows_before_training_or_evaluation",
        "template_binding_rule": "rdagent_templates_must_use_LABEL0_and_the_qlib_owned_label_expression",
        "prompt_wording_rule": (
            "describe_as_qlib_contract_defined_forward_return_label_not_undefined_next_several_days_return"
        ),
        "rdagent_template_paths": list(QLIB_ASHARE_LABEL_TEMPLATE_PATHS),
        "rdagent_prompt_paths": list(QLIB_ASHARE_LABEL_PROMPT_PATHS),
        "rdagent_rule": "describe_only_do_not_redefine_supervised_label_expression_or_horizon",
    }
    for key, expected_value in expected_supervised_label_values.items():
        if supervised_label.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"supervised_label_semantics must preserve {key}"
            )
    prediction_signal = _mapping(prompt_payload.get("prediction_signal_semantics"))
    for key in (
        "semantic_name",
        "model_signal_authority",
        "signal_cache_authority",
        "signal_interface_authority",
        "signal_record_authority",
        "strategy_consumption_authority",
        "prediction_artifact",
        "prediction_column",
        "model_predict_rule",
        "series_prediction_rule",
        "dataframe_prediction_rule",
        "resample_rule",
        "strategy_ranking_rule",
        "missing_signal_rule",
        "label_alignment_rule",
        "prompt_wording_rule",
        "rdagent_prompt_paths",
        "rdagent_rule",
    ):
        if key not in prediction_signal:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload prediction_signal_semantics must include {key}"
            )
    expected_prediction_signal_values = {
        "semantic_name": "a_share_prediction_signal_score",
        "model_signal_authority": "qlib.backtest.signal.ModelSignal",
        "signal_cache_authority": "qlib.backtest.signal.SignalWCache",
        "signal_interface_authority": "qlib.backtest.signal.Signal.get_signal",
        "signal_record_authority": "qlib.workflow.record_temp.SignalRecord",
        "strategy_consumption_authority": (
            "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision"
        ),
        "prediction_artifact": "pred.pkl",
        "prediction_column": "score",
        "model_predict_rule": "model_predict_output_is_prediction_score_not_realized_or_executable_return",
        "series_prediction_rule": "series_prediction_is_saved_as_score_column",
        "dataframe_prediction_rule": "first_prediction_column_is_used_when_prediction_is_dataframe",
        "resample_rule": "SignalWCache_uses_last_signal_between_decision_start_and_end",
        "strategy_ranking_rule": "TopkDropoutStrategy_sorts_prediction_scores_descending_for_candidate_selection",
        "missing_signal_rule": "missing_signal_returns_empty_TradeDecisionWO",
        "label_alignment_rule": "prediction_score_is_trained_against_qlib_owned_LABEL0_without_redefining_return_horizon",
        "prompt_wording_rule": (
            "describe_as_prediction_signal_score_for_LABEL0_not_realized_future_return_or_guaranteed_portfolio_return"
        ),
        "rdagent_prompt_paths": list(QLIB_ASHARE_PREDICTION_SIGNAL_PROMPT_PATHS),
        "rdagent_rule": "describe_only_do_not_redefine_prediction_signal_score_or_return_realization",
    }
    for key, expected_value in expected_prediction_signal_values.items():
        if prediction_signal.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"prediction_signal_semantics must preserve {key}"
            )
    signal_ic = _mapping(prompt_payload.get("signal_ic_semantics"))
    for key in (
        "semantic_name",
        "signal_record_authority",
        "signal_analysis_authority",
        "high_frequency_signal_analysis_authority",
        "ic_calculation_authority",
        "prediction_artifact",
        "label_artifact",
        "ic_artifact",
        "rank_ic_artifact",
        "prediction_column_rule",
        "label_source_rule",
        "missing_label_rule",
        "label_column_rule",
        "groupby_level",
        "ic_rule",
        "rank_ic_rule",
        "dropna_rule",
        "metric_fields",
        "metric_aggregation_rule",
        "icir_rule",
        "rank_icir_rule",
        "recorder_metric_rule",
        "rdagent_consumed_metric_paths",
        "portfolio_boundary_rule",
        "rdagent_rule",
    ):
        if key not in signal_ic:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload signal_ic_semantics must include {key}"
            )
    expected_signal_ic_values = {
        "semantic_name": "a_share_signal_information_coefficient",
        "signal_record_authority": "qlib.workflow.record_temp.SignalRecord",
        "signal_analysis_authority": "qlib.workflow.record_temp.SigAnaRecord",
        "high_frequency_signal_analysis_authority": "qlib.workflow.record_temp.HFSignalRecord",
        "ic_calculation_authority": "qlib.contrib.eva.alpha.calc_ic",
        "prediction_artifact": "pred.pkl",
        "label_artifact": "label.pkl",
        "ic_artifact": "ic.pkl",
        "rank_ic_artifact": "ric.pkl",
        "prediction_column_rule": "series_prediction_is_converted_to_score_dataframe_else_first_prediction_column_is_used",
        "label_source_rule": "dataset_prepare_test_label_uses_DataHandlerLP_DK_R_when_supported_else_handler_default",
        "missing_label_rule": "missing_or_empty_label_skips_signal_analysis_generation",
        "label_column_rule": "SigAnaRecord_uses_configured_label_col_default_zero",
        "groupby_level": "datetime",
        "ic_rule": "IC_is_per_datetime_pearson_correlation_between_pred_and_label",
        "rank_ic_rule": "Rank_IC_is_per_datetime_spearman_correlation_between_pred_and_label",
        "dropna_rule": "calc_ic_preserves_nan_by_default_and_drops_nan_only_when_dropna_true",
        "metric_fields": ["IC", "ICIR", "Rank IC", "Rank ICIR"],
        "metric_aggregation_rule": "IC_and_Rank_IC_metrics_are_series_means",
        "icir_rule": "ICIR_is_IC_mean_divided_by_IC_sample_std",
        "rank_icir_rule": "Rank_ICIR_is_Rank_IC_mean_divided_by_Rank_IC_sample_std",
        "recorder_metric_rule": "SigAnaRecord_and_HFSignalRecord_log_metrics_with_exact_metric_names",
        "rdagent_consumed_metric_paths": ["IC", "ICIR", "Rank IC", "Rank ICIR"],
        "portfolio_boundary_rule": "signal_ic_metrics_are_prediction_label_quality_metrics_not_portfolio_return_metrics",
        "rdagent_rule": "describe_only_do_not_redefine_signal_ic_or_rank_ic_metrics",
    }
    for key, expected_value in expected_signal_ic_values.items():
        if signal_ic.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"signal_ic_semantics must preserve {key}"
            )
    portfolio_risk = _mapping(prompt_payload.get("portfolio_risk_semantics"))
    for key in (
        "semantic_name",
        "record_authority",
        "risk_analysis_authority",
        "freq_authority",
        "backtest_source_rule",
        "report_artifact_rule",
        "risk_artifact_rule",
        "recorder_metric_rule",
        "default_frequency_rule",
        "required_report_columns",
        "turnover_report_metric_rule",
        "report_type_fields",
        "excess_without_cost_rule",
        "excess_with_cost_rule",
        "risk_metric_fields",
        "default_accumulation_mode",
        "supported_accumulation_modes",
        "sum_mode_rule",
        "day_annualization_scaler",
        "annualization_scaler_rule",
        "mean_rule",
        "std_rule",
        "annualized_return_rule",
        "information_ratio_rule",
        "max_drawdown_rule",
        "metric_path_format",
        "metric_path_frequency",
        "metric_path_whitespace_rule",
        "metric_path_report_type_rule",
        "rdagent_prompt_metric_paths",
        "rdagent_feedback_metric_paths",
        "rdagent_bandit_metric_paths",
        "rdagent_ui_metric_paths",
        "rdagent_consumed_metric_paths",
        "rdagent_rule",
    ):
        if key not in portfolio_risk:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload portfolio_risk_semantics must include {key}"
            )
    expected_portfolio_risk_values = {
        "semantic_name": "a_share_portfolio_risk_analysis",
        "record_authority": "qlib.workflow.record_temp.PortAnaRecord",
        "risk_analysis_authority": "qlib.contrib.evaluate.risk_analysis",
        "freq_authority": "qlib.utils.resam.Freq.parse",
        "backtest_source_rule": "PortAnaRecord_runs_normal_backtest_and_reads_portfolio_metric_dict_by_freq",
        "report_artifact_rule": "report_normal_dataframe_saved_as_portfolio_analysis_report_normal_{freq}_pkl",
        "risk_artifact_rule": "risk_analysis_dataframe_saved_as_portfolio_analysis_port_analysis_{freq}_pkl",
        "recorder_metric_rule": "risk_metrics_are_logged_as_{freq}.{report_type}.{risk_metric}",
        "default_frequency_rule": "missing_risk_analysis_freq_uses_first_executor_portfolio_metric_frequency",
        "required_report_columns": ["return", "bench", "cost", "turnover"],
        "turnover_report_metric_rule": QLIB_ASHARE_TURNOVER_REPORT_METRIC_RULE,
        "report_type_fields": ["excess_return_without_cost", "excess_return_with_cost"],
        "excess_without_cost_rule": "report_return_minus_benchmark",
        "excess_with_cost_rule": "report_return_minus_benchmark_minus_cost",
        "risk_metric_fields": ["mean", "std", "annualized_return", "information_ratio", "max_drawdown"],
        "default_accumulation_mode": "sum",
        "supported_accumulation_modes": ["sum", "product"],
        "sum_mode_rule": "qlib_sum_mode_uses_arithmetic_cumulative_return_not_geometric_compounding",
        "day_annualization_scaler": 238,
        "annualization_scaler_rule": "risk_analysis_parses_freq_when_N_is_absent_and_N_overrides_freq_when_present",
        "mean_rule": "sum_mode_mean_equals_return_series_mean",
        "std_rule": "sum_mode_std_uses_sample_standard_deviation_ddof_one",
        "annualized_return_rule": "sum_mode_annualized_return_equals_mean_times_annualization_scaler",
        "information_ratio_rule": "information_ratio_equals_mean_over_std_times_square_root_annualization_scaler",
        "max_drawdown_rule": "sum_mode_max_drawdown_equals_min_of_cumulative_return_minus_running_cumulative_max",
        "metric_path_format": "{freq}.{report_type}.{risk_metric}",
        "metric_path_frequency": "1day",
        "metric_path_whitespace_rule": "metric_paths_are_exact_without_leading_or_trailing_whitespace",
        "metric_path_report_type_rule": "prompt_context_uses_without_cost_and_feedback_bandit_ui_use_with_cost",
        "rdagent_prompt_metric_paths": list(QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS),
        "rdagent_feedback_metric_paths": list(QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS),
        "rdagent_bandit_metric_paths": list(QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS),
        "rdagent_ui_metric_paths": list(QLIB_ASHARE_PORTFOLIO_UI_METRIC_PATHS),
        "rdagent_consumed_metric_paths": [
            *QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS,
            *QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS,
        ],
        "rdagent_rule": "describe_only_do_not_redefine_portfolio_risk_analysis_metrics",
    }
    for key, expected_value in expected_portfolio_risk_values.items():
        if portfolio_risk.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"portfolio_risk_semantics must preserve {key}"
            )
    excess_return = _mapping(prompt_payload.get("excess_return_semantics"))
    for key in (
        "semantic_name",
        "benchmark_dependency",
        "portfolio_risk_dependency",
        "report_column_authority",
        "risk_record_authority",
        "report_graph_authority",
        "risk_graph_authority",
        "online_analysis_authority",
        "user_analysis_authority",
        "required_report_columns",
        "without_cost_field",
        "with_cost_field",
        "without_cost_formula",
        "with_cost_formula",
        "cumulative_without_cost_field",
        "cumulative_with_cost_field",
        "cost_source",
        "benchmark_source",
        "metric_path_without_cost",
        "metric_path_with_cost",
        "rdagent_prompt_rule",
        "forbidden_substitutions",
        "rdagent_rule",
    ):
        if key not in excess_return:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload excess_return_semantics must include {key}"
            )
    expected_excess_return_values = {
        "semantic_name": "a_share_benchmark_relative_excess_return",
        "benchmark_dependency": "benchmark_return_semantics",
        "portfolio_risk_dependency": "portfolio_risk_semantics",
        "report_column_authority": "qlib.backtest.report.PortfolioMetrics",
        "risk_record_authority": "qlib.workflow.record_temp.PortAnaRecord",
        "report_graph_authority": "qlib.contrib.report.analysis_position.report._calculate_report_data",
        "risk_graph_authority": "qlib.contrib.report.analysis_position.risk_analysis._get_risk_analysis_data_with_report",
        "online_analysis_authority": "qlib.contrib.online.operator",
        "user_analysis_authority": "qlib.contrib.online.user",
        "required_report_columns": ["return", "bench", "cost"],
        "without_cost_field": "excess_return_without_cost",
        "with_cost_field": "excess_return_with_cost",
        "without_cost_formula": "return - bench",
        "with_cost_formula": "return - bench - cost",
        "cumulative_without_cost_field": "cum_ex_return_wo_cost",
        "cumulative_with_cost_field": "cum_ex_return_w_cost",
        "cost_source": "reported_cost_column_from_trade_indicator_semantics",
        "benchmark_source": "reported_bench_column_from_benchmark_return_semantics",
        "metric_path_without_cost": QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITHOUT_COST,
        "metric_path_with_cost": QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITH_COST,
        "rdagent_prompt_rule": "generated_research_must_report_benchmark_relative_excess_return_not_raw_return",
        "forbidden_substitutions": list(QLIB_ASHARE_EXCESS_RETURN_FORBIDDEN_SUBSTITUTIONS),
        "rdagent_rule": "describe_only_do_not_redefine_benchmark_relative_excess_return",
    }
    for key, expected_value in expected_excess_return_values.items():
        if excess_return.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"excess_return_semantics must preserve {key}"
            )
    feedback_metric = _mapping(prompt_payload.get("feedback_metric_semantics"))
    for key in (
        "semantic_name",
        "signal_metric_authority",
        "portfolio_metric_authority",
        "risk_metric_authority",
        "prompt_metric_paths",
        "feedback_metric_paths",
        "bandit_metric_paths",
        "feedback_primary_metric",
        "sota_fallback_rule",
        "derived_bandit_utility_name",
        "derived_bandit_utility_rule",
        "forbidden_metric_aliases",
        "prompt_metric_wording_rule",
        "rdagent_source_paths",
        "rdagent_rule",
    ):
        if key not in feedback_metric:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload feedback_metric_semantics must include {key}"
            )
    expected_feedback_metric_values = {
        "semantic_name": "a_share_rd_agent_feedback_metric_consumption",
        "signal_metric_authority": "qlib.workflow.record_temp.SigAnaRecord",
        "portfolio_metric_authority": "qlib.workflow.record_temp.PortAnaRecord",
        "risk_metric_authority": "qlib.contrib.evaluate.risk_analysis",
        "prompt_metric_paths": list(QLIB_ASHARE_PROMPT_METRIC_PATHS),
        "feedback_metric_paths": list(QLIB_ASHARE_FEEDBACK_METRIC_PATHS),
        "bandit_metric_paths": list(QLIB_ASHARE_BANDIT_METRIC_PATHS),
        "feedback_primary_metric": QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC,
        "sota_fallback_rule": "missing_explicit_feedback_decision_uses_feedback_primary_metric_improvement",
        "derived_bandit_utility_name": QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME,
        "derived_bandit_utility_rule": "rdagent_may_compute_arr_over_abs_max_drawdown_as_derived_utility_not_qlib_metric",
        "forbidden_metric_aliases": ["sharpe", "Sharpe"],
        "prompt_metric_wording_rule": "describe_exact_qlib_metric_paths_not_generic_return_sharpe_or_and_so_on",
        "rdagent_source_paths": list(QLIB_ASHARE_FEEDBACK_METRIC_SOURCE_PATHS),
        "rdagent_rule": "consume_exact_qlib_metric_paths_and_label_derived_bandit_utility_as_non_qlib_metric",
    }
    for key, expected_value in expected_feedback_metric_values.items():
        if feedback_metric.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"feedback_metric_semantics must preserve {key}"
            )
    benchmark_return = _mapping(prompt_payload.get("benchmark_return_semantics"))
    for key in (
        "semantic_name",
        "default_benchmark",
        "benchmark_constant_authority",
        "backtest_entry_authority",
        "account_config_authority",
        "portfolio_metric_authority",
        "benchmark_calculation_authority",
        "benchmark_sampling_authority",
        "feature_query_authority",
        "resample_authority",
        "accepted_benchmark_inputs",
        "default_rule",
        "none_rule",
        "series_rule",
        "code_rule",
        "basket_rule",
        "benchmark_field_expression",
        "missing_frequency_rule",
        "missing_benchmark_rule",
        "fillna_rule",
        "sample_rule",
        "direct_bench_value_rule",
        "unusable_benchmark_rule",
        "report_column",
        "portfolio_risk_dependency",
        "rdagent_rule",
    ):
        if key not in benchmark_return:
            raise QlibAshareSemanticContractError(
                f"pyqlib A-share contract prompt_projection_payload benchmark_return_semantics must include {key}"
            )
    expected_benchmark_return_values = {
        "semantic_name": "a_share_benchmark_return_series",
        "default_benchmark": "SH000300",
        "benchmark_constant_authority": "qlib.tests.config.CSI300_BENCH",
        "backtest_entry_authority": "qlib.backtest.backtest",
        "account_config_authority": "qlib.backtest.create_account_instance",
        "portfolio_metric_authority": "qlib.backtest.report.PortfolioMetrics",
        "benchmark_calculation_authority": "qlib.backtest.report.PortfolioMetrics._cal_benchmark",
        "benchmark_sampling_authority": "qlib.backtest.report.PortfolioMetrics._sample_benchmark",
        "feature_query_authority": "qlib.utils.resam.get_higher_eq_freq_feature",
        "resample_authority": "qlib.utils.resam.resam_ts_data",
        "accepted_benchmark_inputs": ["str", "list", "dict", "pd.Series", "None"],
        "default_rule": "missing_benchmark_key_uses_CSI300_BENCH_SH000300",
        "none_rule": "benchmark_config_none_or_benchmark_none_disables_benchmark_series",
        "series_rule": "pd_series_benchmark_is_used_directly_as_per_period_return_series",
        "code_rule": "str_benchmark_is_queried_as_single_code_close_over_ref_close_minus_one",
        "basket_rule": "list_or_dict_benchmark_is_queried_as_codes_and_averaged_by_datetime",
        "benchmark_field_expression": "$close/Ref($close,1)-1",
        "missing_frequency_rule": "non_series_benchmark_requires_freq_else_ValueError",
        "missing_benchmark_rule": "empty_feature_result_raises_ValueError",
        "fillna_rule": "queried_benchmark_returns_fillna_zero_after_datetime_average",
        "sample_rule": "bar_benchmark_return_equals_product_of_one_plus_period_returns_minus_one",
        "direct_bench_value_rule": "provided_bench_value_overrides_sampling",
        "unusable_benchmark_rule": "trade_end_time_and_bench_value_both_none_raise_ValueError",
        "report_column": "bench",
        "portfolio_risk_dependency": "portfolio_risk_excess_returns_use_report_normal_bench_column",
        "rdagent_rule": "describe_only_do_not_redefine_benchmark_return_series_or_default_benchmark",
    }
    for key, expected_value in expected_benchmark_return_values.items():
        if benchmark_return.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload " f"benchmark_return_semantics must preserve {key}"
            )
    universe_benchmark_binding = _mapping(prompt_payload.get("universe_benchmark_binding_semantics"))
    for key in (
        "semantic_name",
        "market_universe_authority",
        "benchmark_authority",
        "template_market_value",
        "template_benchmark_value",
        "template_market_anchor",
        "template_instruments_binding",
        "template_benchmark_anchor",
        "template_backtest_benchmark_binding",
        "market_universe_rule",
        "benchmark_rule",
        "separation_rule",
        "forbidden_template_values",
        "rdagent_template_paths",
        "rdagent_rule",
    ):
        if key not in universe_benchmark_binding:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"universe_benchmark_binding_semantics must include {key}"
            )
    expected_universe_benchmark_binding_values = {
        "semantic_name": "a_share_rd_agent_universe_benchmark_binding",
        "market_universe_authority": "qlib.tests.config.CSI300_MARKET",
        "benchmark_authority": "qlib.tests.config.CSI300_BENCH",
        "template_market_value": QLIB_ASHARE_TEMPLATE_MARKET,
        "template_benchmark_value": QLIB_ASHARE_TEMPLATE_BENCHMARK,
        "template_market_anchor": f"market: &market {QLIB_ASHARE_TEMPLATE_MARKET}",
        "template_instruments_binding": "instruments: *market",
        "template_benchmark_anchor": f"benchmark: &benchmark {QLIB_ASHARE_TEMPLATE_BENCHMARK}",
        "template_backtest_benchmark_binding": "benchmark: *benchmark",
        "market_universe_rule": "csi300_template_market_selects_instruments_only",
        "benchmark_rule": "SH000300_template_benchmark_is_portfolio_excess_return_baseline_only",
        "separation_rule": "market_universe_membership_and_benchmark_return_series_are_not_substitutable",
        "forbidden_template_values": ["all_a", "all", "SH000300_as_market", "csi300_as_benchmark"],
        "rdagent_template_paths": list(QLIB_ASHARE_UNIVERSE_BENCHMARK_TEMPLATE_PATHS),
        "rdagent_rule": "bind_market_to_instruments_and_benchmark_to_backtest_without_cross_aliasing",
    }
    for key, expected_value in expected_universe_benchmark_binding_values.items():
        if universe_benchmark_binding.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"universe_benchmark_binding_semantics must preserve {key}"
            )
    if universe_benchmark_binding["template_market_value"] == universe_benchmark_binding["template_benchmark_value"]:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract universe_benchmark_binding_semantics must not alias market and benchmark"
        )
    if universe_benchmark_binding["template_market_value"] in _string_list(
        universe_benchmark_binding["forbidden_template_values"]
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract universe_benchmark_binding_semantics must not use a forbidden market value"
        )
    if universe_benchmark_binding["template_benchmark_value"] in _string_list(
        universe_benchmark_binding["forbidden_template_values"]
    ):
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract universe_benchmark_binding_semantics must not use a forbidden benchmark value"
        )
    runtime_template_binding = _mapping(prompt_payload.get("runtime_handoff_template_binding_semantics"))
    expected_runtime_template_binding_values = {
        "semantic_name": "a_share_rd_agent_runtime_handoff_template_binding",
        "handoff_id": REQUIRED_QLIB_RUNTIME_HANDOFF_ID,
        "binding_kind": "rdagent_qlib_template_backtest_runtime_kwargs",
        "rdagent_template_paths": list(QLIB_ASHARE_RUNTIME_TEMPLATE_PATHS),
        "runtime_rule": "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values",
        "prompt_boundary_rule": "execution_kwargs_remain_runtime_handoff_not_prompt_authority",
        "rdagent_rule": "consume_qlib_runtime_handoff_values_without_redefining_a_share_execution_kwargs",
    }
    for key, expected_value in expected_runtime_template_binding_values.items():
        if runtime_template_binding.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"runtime_handoff_template_binding_semantics must preserve {key}"
            )
    for forbidden_key in ("required_backtest_kwargs", "forbidden_legacy_exchange_kwargs"):
        if forbidden_key in runtime_template_binding:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"runtime_handoff_template_binding_semantics must not expose {forbidden_key}"
            )
    research_data_source = _mapping(prompt_payload.get("research_data_source_semantics"))
    expected_research_data_source_values = {
        "semantic_name": "a_share_research_data_source_boundary",
        "data_frequency": "day",
        "source_scope": "qlib_daily_research_and_backtest_inputs",
        "index_contract": ["datetime", "instrument"],
        "primary_price_volume_fields": list(QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS),
        "handler_authority": "qlib.contrib.data.handler.Alpha158",
        "handler360_authority": "qlib.contrib.data.handler.Alpha360",
        "loader_authority": "qlib.contrib.data.loader.Alpha158DL",
        "allowed_prompt_source_tables": [
            "daily_stock_trade_data",
            "daily_price_volume_derived_features",
            "provider_supplied_point_in_time_fundamental_or_industry_fields",
        ],
        "point_in_time_rule": (
            "non_price_volume_fields_are_allowed_only_when_user_or_provider_supplies_daily_point_in_time_data"
        ),
        "point_in_time_registration_rule": QLIB_ASHARE_POINT_IN_TIME_REGISTRATION_RULE,
        "forbidden_default_prompt_sources": list(QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES),
        "turnover_input_boundary_rule": QLIB_ASHARE_TURNOVER_INPUT_BOUNDARY_RULE,
        "frequency_rule": "rdagent_factor_extraction_prompts_must_not_advertise_minute_or_intraday_data_as_default",
        "rdagent_prompt_paths": list(QLIB_ASHARE_RESEARCH_DATA_SOURCE_PROMPT_PATHS),
        "rdagent_rule": "describe_only_use_qlib_registered_daily_or_user_supplied_point_in_time_sources",
    }
    for key, expected_value in expected_research_data_source_values.items():
        if research_data_source.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract prompt_projection_payload "
                f"research_data_source_semantics must preserve {key}"
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
    template_runtime_binding = _mapping(runtime_handoff.get("template_runtime_binding"))
    expected_template_runtime_binding = {
        "semantic_name": "a_share_rd_agent_runtime_handoff_template_binding",
        "handoff_id": REQUIRED_QLIB_RUNTIME_HANDOFF_ID,
        "binding_kind": "rdagent_qlib_template_backtest_runtime_kwargs",
        "rdagent_template_paths": list(QLIB_ASHARE_RUNTIME_TEMPLATE_PATHS),
        "required_backtest_kwargs": QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS,
        "forbidden_legacy_exchange_kwargs": QLIB_ASHARE_FORBIDDEN_LEGACY_EXCHANGE_KWARGS,
        "runtime_rule": "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values",
        "prompt_boundary_rule": "execution_kwargs_remain_runtime_handoff_not_prompt_authority",
        "rdagent_rule": "consume_qlib_runtime_handoff_values_without_redefining_a_share_execution_kwargs",
    }
    for key, expected_value in expected_template_runtime_binding.items():
        if template_runtime_binding.get(key) != expected_value:
            raise QlibAshareSemanticContractError(
                "pyqlib A-share contract runtime_handoff_contract " f"template_runtime_binding must preserve {key}"
            )
    if template_runtime_binding["required_backtest_kwargs"] != runtime_surfaces["backtest_kwargs"]:
        raise QlibAshareSemanticContractError(
            "pyqlib A-share contract template_runtime_binding must match runtime_surfaces.backtest_kwargs"
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
        "strategy_order_semantics",
        "supervised_label_semantics",
        "signal_ic_semantics",
        "portfolio_risk_semantics",
        "excess_return_semantics",
        "feedback_metric_semantics",
        "benchmark_return_semantics",
        "universe_benchmark_binding_semantics",
        "research_data_source_semantics",
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
