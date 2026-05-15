from __future__ import annotations

import sys
import types
from copy import deepcopy
from typing import Any

import pytest

from rdagent.scenarios.qlib.ashare_semantics import (
    QlibAshareSemanticContractError,
    append_ashare_semantic_context,
    build_rd_agent_ashare_runtime_handoff,
    build_rd_agent_ashare_semantic_context,
    format_rd_agent_ashare_semantic_context,
    load_qlib_ashare_contract,
)


def _valid_contract() -> dict[str, Any]:
    return {
        "schema_version": "qlib_ashare_semantic_contract.v1",
        "contract_id": "rdagent_qlib_joinquant_ashare_semantic_contract_v1",
        "status": "active",
        "source_component": "qlib.backtest.ashare_semantics",
        "consumer_component": "rdagent.scenarios.qlib.ashare_semantics",
        "relationship": {
            "qlib_role": "executable_backtest_semantic_authority",
            "rdagent_role": "research_candidate_generation_context_consumer",
            "relationship_rule": (
                "RD-Agent may consume Qlib's A-share contract for research generation and evaluation context, "
                "but it must not redefine trading-calendar/data-frequency, trade unit, position, execution-price, "
                "price-adjustment, "
                "suspension/tradability, price-limit, settlement, cash/shorting, liquidity/capacity, or cost semantics."
            ),
            "fail_closed_on_missing_contract": True,
        },
        "semantic_boundary": {
            "authority_component": "qlib.backtest.ashare_semantics",
            "consumer_component": "rdagent.scenarios.qlib.ashare_semantics",
            "authority_rule": "Qlib owns executable JoinQuant-compatible A-share backtest semantics.",
            "consumer_rule": "RD-Agent may consume a bounded research-generation projection of this contract only.",
            "rdagent_allowed_actions": [
                "render_contract_projection_in_research_context",
                "carry_contract_id_schema_version_and_fingerprint_into_generated_evidence",
                "pass_qlib_owned_runtime_kwargs_to_execution_surfaces",
                "fail_closed_when_contract_is_missing_malformed_or_unsupported",
            ],
            "rdagent_forbidden_actions": [
                "redefine_instrument_identity_or_board_mapping",
                "redefine_trading_calendar_or_data_frequency",
                "redefine_transaction_cost_model",
                "redefine_suspension_or_tradability_rules",
                "redefine_execution_price_or_frequency",
                "redefine_price_adjustment_or_order_factor",
                "redefine_trade_unit_or_position_type",
                "redefine_price_limit_thresholds_or_authoritative_fields",
                "treat_board_fallback_as_primary_price_limit_authority",
                "redefine_settlement_or_sellable_position_state",
                "redefine_cash_buying_power_or_shorting_policy",
                "redefine_liquidity_or_volume_capacity_policy",
                "redefine_cost_model_or_exchange_kwargs",
                "treat_research_prompt_projection_as_backtest_authority",
                "claim_a_share_alignment_without_qlib_contract_fingerprint",
            ],
        },
        "failure_semantics": {
            "missing_contract": "fail_closed",
            "unsupported_schema_version": "fail_closed",
            "missing_required_field": "fail_closed",
            "malformed_contract": "fail_closed",
            "runtime_projection_drift": "fail_closed",
            "claim_without_evidence_fingerprint": "fail_closed",
        },
        "evidence_contract": {
            "semantic_fingerprint": "a" * 64,
            "fingerprint_algorithm": "sha256_json_canonical_v1",
            "fingerprint_scope": [
                "schema_version",
                "market_semantics",
                "runtime_surfaces",
                "rdagent_must_not_redefine",
            ],
            "rdagent_required_evidence_fields": [
                "qlib_contract_id",
                "qlib_contract_schema_version",
                "qlib_contract_fingerprint",
                "qlib_source_component",
                "qlib_semantic_authority",
            ],
        },
        "projection_contract": {
            "rdagent_prompt_projection_fields": [
                "contract_id",
                "schema_version",
                "source_component",
                "consumer_component",
                "semantic_boundary",
                "failure_semantics",
                "evidence_contract.semantic_fingerprint",
                "market_semantics.market",
                "market_semantics.region",
                "market_semantics.data_frequency",
                "market_semantics.trade_unit",
                "market_semantics.position_type",
                "market_semantics.settlement_rule",
                "market_semantics.limit_threshold",
                "market_semantics.authoritative_limit_fields",
                "instrument_identity_semantics",
                "trading_calendar_semantics",
                "transaction_cost_semantics",
                "suspension_tradability_semantics",
                "execution_price_semantics",
                "price_adjustment_semantics",
                "price_limit_semantics",
                "settlement_semantics",
                "cash_constraint_semantics",
                "liquidity_capacity_semantics",
                "order_unit_semantics",
            ],
            "rdagent_prompt_forbidden_fields": [
                "runtime_surfaces.policy_defaults",
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
                "market_semantics.cost_model",
            ],
        },
        "prompt_projection_payload": {
            "projection_id": "qlib_joinquant_ashare_prompt_projection_v1",
            "projection_schema_version": "qlib_ashare_prompt_projection.v1",
            "projection_kind": "research_prompt_context_only",
            "contract_id": "rdagent_qlib_joinquant_ashare_semantic_contract_v1",
            "contract_schema_version": "qlib_ashare_semantic_contract.v1",
            "schema_version": "qlib_ashare_semantic_contract.v1",
            "source_component": "qlib.backtest.ashare_semantics",
            "consumer_component": "rdagent.scenarios.qlib.ashare_semantics",
            "semantic_fingerprint": "a" * 64,
            "semantic_boundary": {
                "authority_component": "qlib.backtest.ashare_semantics",
                "consumer_component": "rdagent.scenarios.qlib.ashare_semantics",
                "authority_rule": "Qlib owns executable JoinQuant-compatible A-share backtest semantics.",
                "consumer_rule": "RD-Agent may consume a bounded research-generation projection of this contract only.",
            },
            "failure_semantics": {
                "missing_contract": "fail_closed",
                "unsupported_schema_version": "fail_closed",
                "missing_required_field": "fail_closed",
                "malformed_contract": "fail_closed",
                "runtime_projection_drift": "fail_closed",
                "claim_without_evidence_fingerprint": "fail_closed",
            },
            "market_semantics": {
                "market": "china_a_share",
                "region": "cn",
                "data_frequency": "day",
                "trade_unit": 100,
                "position_type": "AsharePosition",
                "settlement_rule": "t_plus_1_stock",
                "limit_threshold": "joinquant_ashare",
                "authoritative_limit_fields": ["$up_limit", "$down_limit"],
            },
            "instrument_identity_semantics": {
                "semantic_name": "a_share_instrument_identity",
                "canonical_code_format": "exchange_prefix_plus_six_digit_code",
                "canonical_exchange_prefixes": ["SH", "SZ", "BJ"],
                "accepted_provider_suffixes": {
                    "XSHG": "SH",
                    "SH": "SH",
                    "XSHE": "SZ",
                    "SZ": "SZ",
                    "XBJ": "BJ",
                    "BJ": "BJ",
                },
                "normalization_examples": {
                    "600000.XSHG": "SH600000",
                    "000001.XSHE": "SZ000001",
                    "430047.XBJ": "BJ430047",
                },
                "board_identity_rules": [
                    {"match": "SH688*", "board": "star_market"},
                    {
                        "match": "SZ300*",
                        "board": "chinext_registration_sensitive",
                        "effective_start": "2020-08-24",
                    },
                    {"match": "BJ*|SH8*|SH4*|SH9*|SZ8*|SZ4*|SZ9*", "board": "beijing_stock_exchange"},
                    {"match": "fallback", "board": "main_board"},
                ],
                "price_limit_dependency": "board_identity_is_runtime_fallback_only_when_authoritative_limit_fields_absent",
                "runtime_authority": "qlib.backtest.ashare_semantics.normalize_ashare_instrument",
                "board_classification_authority": (
                    "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.limit_threshold_for_instrument"
                ),
                "rdagent_rule": "describe_only_do_not_redefine_instrument_or_board_identity",
            },
            "trading_calendar_semantics": {
                "semantic_name": "a_share_daily_trading_calendar",
                "calendar_frequency": "day",
                "calendar_provider_authority": "qlib.data.data.CalendarProvider.calendar",
                "calendar_locator_authority": "qlib.data.data.CalendarProvider.locate_index",
                "exchange_frequency_parameter": "freq",
                "exchange_default_frequency": "day",
                "index_level": "datetime",
                "instrument_window_rule": "instrument_membership_is_filtered_against_calendar_boundaries",
                "non_trading_day_rule": (
                    "calendar_locate_index_maps_start_forward_and_end_backward_to_real_trading_days"
                ),
                "future_calendar_rule": "future_trading_days_require_qlib_future_calendar_support_not_prompt_invention",
                "synthetic_session_rule": "rdagent_must_not_invent_non_qlib_calendar_sessions",
                "rdagent_rule": "describe_only_do_not_redefine_trading_calendar_or_data_frequency",
            },
            "transaction_cost_semantics": {
                "semantic_name": "a_share_transaction_cost_structure",
                "cost_model_scope": "qlib_runtime_execution_only",
                "buy_cost_components": ["commission", "minimum_commission_floor"],
                "sell_cost_components": ["commission", "stamp_tax", "minimum_commission_floor"],
                "minimum_fee_rule": "commission_floor_applies_to_nonzero_trade_value",
                "zero_trade_rule": "zero_trade_value_has_zero_cost",
                "market_impact_rule": "optional_impact_cost_is_added_by_runtime_execution",
                "numeric_values_exposure": "runtime_handoff_only_not_prompt_projection",
                "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.calculate_trade_cost",
                "rdagent_rule": "describe_only_do_not_redefine_transaction_cost_model",
            },
            "suspension_tradability_semantics": {
                "semantic_name": "a_share_suspension_tradability",
                "suspension_indicator_field": "$close",
                "suspension_indicator_rule": "missing_close_price_marks_suspended",
                "non_tradable_rule": "suspended_rows_are_not_buyable_or_sellable",
                "limit_flag_projection": "qlib_sets_limit_buy_and_limit_sell_true_for_suspended_rows",
                "authoritative_limit_interaction": "suspension_takes_precedence_over_up_down_limit_fields",
                "missing_limit_bounds_rule": "missing_limit_bounds_are_tolerated_only_when_close_is_missing",
                "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits",
                "rdagent_rule": "describe_only_do_not_redefine_suspension_or_tradability",
            },
            "execution_price_semantics": {
                "semantic_name": "a_share_daily_close_execution_price",
                "qlib_parameter": "deal_price",
                "execution_price_field": "$close",
                "execution_frequency": "daily_bar_backtest",
                "price_source_authority": "qlib_exchange_deal_price",
                "intraday_execution_rule": "not_intraday_or_auction_simulation",
                "candidate_research_rule": "generated_factors_must_not_assume_intraday_fill_prices",
                "runtime_authority": "qlib.backtest.ashare_semantics.joinquant_ashare_exchange_kwargs",
                "rdagent_rule": "describe_only_do_not_redefine_execution_price_or_frequency",
            },
            "price_adjustment_semantics": {
                "semantic_name": "a_share_price_adjustment_order_factor",
                "factor_field": "$factor",
                "factor_usage": "convert_adjusted_amounts_to_trade_unit_amounts_when_unadjusted_prices_are_used",
                "missing_factor_rule": (
                    "non_suspended_rows_with_missing_factor_use_adjusted_price_mode_and_disable_trade_unit_rounding"
                ),
                "adjusted_price_mode_rule": "trade_unit_rounding_is_not_supported_when_adjusted_price_mode_is_active",
                "extra_quote_factor_rule": "missing_extra_quote_factor_defaults_to_one",
                "suspension_interaction": "missing_factor_is_tolerated_when_close_is_missing",
                "runtime_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
                "rdagent_rule": "describe_only_do_not_redefine_price_adjustment_or_order_factor",
            },
            "price_limit_semantics": {
                "semantic_name": "a_share_price_limit_authority",
                "limit_threshold": "joinquant_ashare",
                "price_limit_mode": "strict",
                "authoritative_limit_fields": ["$up_limit", "$down_limit"],
                "field_authority": "provider_up_down_limit_fields",
                "limit_flag_fields": ["limit_buy", "limit_sell"],
                "limit_flag_meaning": "true_flags_mark_direction_not_tradable",
                "buy_limit_rule": "buy_price_at_or_above_up_limit_or_suspended_sets_limit_buy",
                "sell_limit_rule": "sell_price_at_or_below_down_limit_or_suspended_sets_limit_sell",
                "missing_authoritative_fields": (
                    "fail_closed_in_strict_mode_else_qlib_board_fallback_for_legacy_datasets"
                ),
                "strict_mode_missing_fields_rule": "missing_authoritative_fields_or_non_suspended_bounds_fail_closed",
                "board_fallback_policy": "runtime_compatibility_only_when_authoritative_fields_are_absent",
                "fallback_authority_rule": "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority",
                "board_limit_thresholds": {
                    "main_board": 0.095,
                    "star_chinext": 0.195,
                    "bse": 0.295,
                    "chinext_registration_start_date": "2020-08-24",
                },
                "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits",
                "rdagent_rule": "describe_only_do_not_redefine_price_limit_thresholds_or_fields",
            },
            "settlement_semantics": {
                "semantic_name": "a_share_t_plus_1_stock_settlement",
                "settlement_rule": "t_plus_1_stock",
                "same_day_sell_policy": "shares_bought_today_are_unsellable_until_day_commit",
                "position_type": "AsharePosition",
                "sellable_state_field": "sellable_amount",
                "initial_sellable_rule": "existing_or_settled_holdings_are_sellable",
                "intraday_buy_rule": "same_day_buys_increase_total_amount_but_not_sellable_amount",
                "intraday_bar_rule": "non_day_bars_do_not_release_same_day_buys",
                "day_commit_rule": "day_bar_commit_sets_sellable_amount_to_total_amount",
                "sell_order_clip_rule": "sell_orders_are_clipped_by_position_get_sellable_amount",
                "sell_overdraft_rule": "AsharePosition_rejects_sells_above_sellable_amount",
                "runtime_authority": "qlib.backtest.position.AsharePosition",
                "exchange_clip_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
                "rdagent_rule": "describe_only_do_not_redefine_position_or_settlement",
            },
            "cash_constraint_semantics": {
                "semantic_name": "a_share_cash_buying_power_and_shorting_policy",
                "cash_state_field": "cash",
                "cash_query_rule": "buying_power_uses_position_get_cash_without_unsettled_cash",
                "buy_cash_rule": "buy_orders_are_clipped_by_available_cash_and_transaction_cost",
                "minimum_cost_rule": "orders_without_cash_for_minimum_cost_are_zeroed",
                "partial_buy_rule": "cash_insufficient_orders_are_reduced_by_exchange_cash_limit_then_round_lot",
                "shorting_policy": "equity_short_selling_is_not_enabled",
                "sell_position_rule": "sell_orders_are_clipped_by_position_get_sellable_amount",
                "sell_cash_rule": "sell_orders_zero_when_cash_plus_trade_value_cannot_cover_sell_cost",
                "runtime_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
                "cash_limit_authority": "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit",
                "position_cash_authority": "qlib.backtest.position.Position.get_cash",
                "rdagent_rule": "describe_only_do_not_redefine_cash_or_shorting_policy",
            },
            "liquidity_capacity_semantics": {
                "semantic_name": "a_share_volume_capacity_limit",
                "volume_field": "$volume",
                "capacity_parameter": "volume_threshold",
                "capacity_scope": "runtime_handoff_only_when_volume_threshold_is_configured",
                "default_capacity_rule": "no_prompt_defined_capacity_limit_in_default_joinquant_ashare_contract",
                "volume_limit_aggregation_rule": "multiple_volume_limits_are_aggregated_by_min",
                "cumulative_limit_rule": "cum_volume_limits_subtract_dealt_order_amount",
                "current_limit_rule": "current_volume_limits_use_current_quote_value",
                "dealt_order_state": "dealt_order_amount",
                "capacity_clip_rule": "order_deal_amount_is_clipped_to_nonnegative_configured_volume_capacity",
                "runtime_authority": "qlib.backtest.exchange.Exchange._clip_amount_by_volume",
                "threshold_parser_authority": "qlib.backtest.exchange.Exchange._get_vol_limit",
                "rdagent_rule": "describe_only_do_not_redefine_liquidity_or_volume_capacity",
            },
            "order_unit_semantics": {
                "semantic_name": "a_share_round_lot",
                "qlib_parameter": "trade_unit",
                "trade_unit": 100,
                "amount_unit": "share",
                "buy_rounding_rule": "round_buy_amount_down_to_trade_unit_after_cash_and_volume_limits",
                "sell_rounding_rule": "round_sell_amount_down_to_trade_unit_except_full_liquidation",
                "full_liquidation_rule": "sell_all_remaining_position_without_round_lot_residual",
                "factor_adjustment_rule": "apply_order_factor_when_trade_uses_unadjusted_prices",
                "runtime_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
                "rdagent_rule": "describe_only_do_not_redefine_trade_unit_or_round_lot_policy",
            },
        },
        "runtime_handoff_contract": {
            "handoff_id": "qlib_joinquant_ashare_runtime_handoff_v1",
            "handoff_kind": "qlib_owned_execution_kwargs",
            "authority_component": "qlib.backtest.ashare_semantics",
            "consumer_component": "rdagent.scenarios.qlib.ashare_semantics",
            "source_fingerprint": "a" * 64,
            "payload_paths": [
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
            ],
            "forbidden_prompt_paths": [
                "runtime_surfaces.policy_defaults",
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
                "market_semantics.cost_model",
            ],
            "mutation_policy": "pass_through_only",
            "consumer_obligations": [
                "preserve_contract_id_schema_version_and_fingerprint",
                "preserve_qlib_source_component",
                "do_not_mutate_runtime_payload_values",
                "fail_closed_on_missing_payload_or_fingerprint",
            ],
        },
        "market_semantics": {
            "market": "china_a_share",
            "region": "cn",
            "data_frequency": "day",
            "trade_unit": 100,
            "position_type": "AsharePosition",
            "settlement_rule": "t_plus_1_stock",
            "same_day_sell_policy": "shares_bought_today_are_unsellable_until_day_commit",
            "deal_price": "close",
            "limit_threshold": "joinquant_ashare",
            "limit_threshold_aliases": [
                "ashare_joinquant",
                "cn_ashare_joinquant",
                "joinquant_ashare",
            ],
            "price_limit_modes": ["auto", "strict", "board_fallback"],
            "authoritative_limit_fields": ["$up_limit", "$down_limit"],
            "board_threshold_fields": {
                "main_board_threshold": 0.095,
                "star_chinext_threshold": 0.195,
                "bse_threshold": 0.295,
                "chinext_registration_start_date": "2020-08-24",
            },
            "cost_model": {
                "open_cost": 0.0003,
                "close_cost": 0.0013,
                "close_commission": 0.0003,
                "close_tax": 0.001,
                "min_cost": 5.0,
            },
        },
        "runtime_surfaces": {
            "exchange_kwargs": {
                "limit_threshold": "joinquant_ashare",
                "trade_unit": 100,
            },
            "backtest_kwargs": {
                "pos_type": "AsharePosition",
                "exchange_kwargs": {"limit_threshold": "joinquant_ashare"},
            },
        },
        "rdagent_must_not_redefine": [
            "instrument_identity_semantics",
            "trading_calendar_semantics",
            "transaction_cost_semantics",
            "suspension_tradability_semantics",
            "execution_price_semantics",
            "price_adjustment_semantics",
            "price_limit_semantics",
            "settlement_semantics",
            "cash_constraint_semantics",
            "liquidity_capacity_semantics",
            "trade_unit",
            "position_type",
            "settlement_rule",
            "same_day_sell_policy",
            "data_frequency",
            "price_limit_modes",
            "authoritative_limit_fields",
            "board_threshold_fields",
            "cost_model",
        ],
    }


def _install_qlib_contract_stub(monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]) -> None:
    qlib_module = types.ModuleType("qlib")
    backtest_module = types.ModuleType("qlib.backtest")
    ashare_module = types.ModuleType("qlib.backtest.ashare_semantics")
    ashare_module.rdagent_ashare_semantic_contract = lambda strict_price_limit=True: deepcopy(contract)
    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.backtest", backtest_module)
    monkeypatch.setitem(sys.modules, "qlib.backtest.ashare_semantics", ashare_module)


def test_load_qlib_ashare_contract_consumes_pyqlib_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_qlib_contract_stub(monkeypatch, _valid_contract())

    contract = load_qlib_ashare_contract()

    assert contract["source_component"] == "qlib.backtest.ashare_semantics"
    assert contract["relationship"]["qlib_role"] == "executable_backtest_semantic_authority"
    assert contract["relationship"]["rdagent_role"] == "research_candidate_generation_context_consumer"


def test_rd_agent_context_does_not_redefine_qlib_ashare_runtime_semantics() -> None:
    context = build_rd_agent_ashare_semantic_context(_valid_contract())
    boundary = context["relationship_boundary"]

    assert context["qlib_contract_id"] == "rdagent_qlib_joinquant_ashare_semantic_contract_v1"
    assert context["qlib_contract_schema_version"] == "qlib_ashare_semantic_contract.v1"
    assert context["qlib_contract_fingerprint"] == "a" * 64
    assert context["prompt_projection_schema_version"] == "qlib_ashare_prompt_projection.v1"
    assert context["prompt_projection_kind"] == "research_prompt_context_only"
    assert boundary["semantic_authority"] == "pyqlib_contract"
    assert boundary["failure_semantics"] == "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract"
    assert boundary["authority_rule"] == "Qlib owns executable JoinQuant-compatible A-share backtest semantics."
    assert "render_contract_projection_in_research_context" in boundary["rdagent_may"]
    assert "redefine_instrument_identity_or_board_mapping" in boundary["rdagent_forbidden_actions"]
    assert "redefine_trading_calendar_or_data_frequency" in boundary["rdagent_forbidden_actions"]
    assert "redefine_transaction_cost_model" in boundary["rdagent_forbidden_actions"]
    assert "redefine_suspension_or_tradability_rules" in boundary["rdagent_forbidden_actions"]
    assert "redefine_execution_price_or_frequency" in boundary["rdagent_forbidden_actions"]
    assert "redefine_price_adjustment_or_order_factor" in boundary["rdagent_forbidden_actions"]
    assert "redefine_price_limit_thresholds_or_authoritative_fields" in boundary["rdagent_forbidden_actions"]
    assert "treat_board_fallback_as_primary_price_limit_authority" in boundary["rdagent_forbidden_actions"]
    assert "redefine_settlement_or_sellable_position_state" in boundary["rdagent_forbidden_actions"]
    assert "redefine_cash_buying_power_or_shorting_policy" in boundary["rdagent_forbidden_actions"]
    assert "redefine_liquidity_or_volume_capacity_policy" in boundary["rdagent_forbidden_actions"]
    assert "treat_research_prompt_projection_as_backtest_authority" in boundary["rdagent_forbidden_actions"]
    assert boundary["rdagent_must_not_redefine"] == [
        "instrument_identity_semantics",
        "trading_calendar_semantics",
        "transaction_cost_semantics",
        "suspension_tradability_semantics",
        "execution_price_semantics",
        "price_adjustment_semantics",
        "price_limit_semantics",
        "settlement_semantics",
        "cash_constraint_semantics",
        "liquidity_capacity_semantics",
        "trade_unit",
        "position_type",
        "settlement_rule",
        "same_day_sell_policy",
        "data_frequency",
        "price_limit_modes",
        "authoritative_limit_fields",
        "board_threshold_fields",
        "cost_model",
    ]
    assert context["failure_contract"]["runtime_projection_drift"] == "fail_closed"
    assert "runtime_surfaces.backtest_kwargs" in context["prompt_projection"]["rdagent_prompt_forbidden_fields"]
    assert context["prompt_projection_payload"]["market_semantics"]["trade_unit"] == 100
    assert context["prompt_projection_payload"]["market_semantics"]["data_frequency"] == "day"
    assert (
        context["prompt_projection_payload"]["instrument_identity_semantics"]["runtime_authority"]
        == "qlib.backtest.ashare_semantics.normalize_ashare_instrument"
    )
    assert (
        context["prompt_projection_payload"]["instrument_identity_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_instrument_or_board_identity"
    )
    assert context["prompt_projection_payload"]["trading_calendar_semantics"]["calendar_frequency"] == "day"
    assert (
        context["prompt_projection_payload"]["trading_calendar_semantics"]["calendar_provider_authority"]
        == "qlib.data.data.CalendarProvider.calendar"
    )
    assert (
        context["prompt_projection_payload"]["trading_calendar_semantics"]["non_trading_day_rule"]
        == "calendar_locate_index_maps_start_forward_and_end_backward_to_real_trading_days"
    )
    assert (
        context["prompt_projection_payload"]["trading_calendar_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_trading_calendar_or_data_frequency"
    )
    assert (
        context["prompt_projection_payload"]["transaction_cost_semantics"]["numeric_values_exposure"]
        == "runtime_handoff_only_not_prompt_projection"
    )
    assert (
        context["prompt_projection_payload"]["transaction_cost_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_transaction_cost_model"
    )
    assert (
        context["prompt_projection_payload"]["suspension_tradability_semantics"]["non_tradable_rule"]
        == "suspended_rows_are_not_buyable_or_sellable"
    )
    assert (
        context["prompt_projection_payload"]["suspension_tradability_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_suspension_or_tradability"
    )
    assert (
        context["prompt_projection_payload"]["execution_price_semantics"]["execution_frequency"] == "daily_bar_backtest"
    )
    assert (
        context["prompt_projection_payload"]["execution_price_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_execution_price_or_frequency"
    )
    assert (
        context["prompt_projection_payload"]["price_adjustment_semantics"]["missing_factor_rule"]
        == "non_suspended_rows_with_missing_factor_use_adjusted_price_mode_and_disable_trade_unit_rounding"
    )
    assert (
        context["prompt_projection_payload"]["price_adjustment_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_price_adjustment_or_order_factor"
    )
    assert context["prompt_projection_payload"]["price_limit_semantics"]["price_limit_mode"] == "strict"
    assert context["prompt_projection_payload"]["price_limit_semantics"]["limit_flag_fields"] == [
        "limit_buy",
        "limit_sell",
    ]
    assert (
        context["prompt_projection_payload"]["price_limit_semantics"]["fallback_authority_rule"]
        == "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority"
    )
    assert (
        context["prompt_projection_payload"]["price_limit_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_price_limit_thresholds_or_fields"
    )
    assert context["prompt_projection_payload"]["settlement_semantics"]["settlement_rule"] == "t_plus_1_stock"
    assert (
        context["prompt_projection_payload"]["settlement_semantics"]["same_day_sell_policy"]
        == "shares_bought_today_are_unsellable_until_day_commit"
    )
    assert context["prompt_projection_payload"]["settlement_semantics"]["sellable_state_field"] == "sellable_amount"
    assert (
        context["prompt_projection_payload"]["settlement_semantics"]["day_commit_rule"]
        == "day_bar_commit_sets_sellable_amount_to_total_amount"
    )
    assert (
        context["prompt_projection_payload"]["cash_constraint_semantics"]["buy_cash_rule"]
        == "buy_orders_are_clipped_by_available_cash_and_transaction_cost"
    )
    assert (
        context["prompt_projection_payload"]["cash_constraint_semantics"]["shorting_policy"]
        == "equity_short_selling_is_not_enabled"
    )
    assert (
        context["prompt_projection_payload"]["liquidity_capacity_semantics"]["capacity_parameter"] == "volume_threshold"
    )
    assert (
        context["prompt_projection_payload"]["liquidity_capacity_semantics"]["runtime_authority"]
        == "qlib.backtest.exchange.Exchange._clip_amount_by_volume"
    )
    assert context["prompt_projection_payload"]["order_unit_semantics"]["trade_unit"] == 100
    assert (
        context["prompt_projection_payload"]["order_unit_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_trade_unit_or_round_lot_policy"
    )
    assert "qlib_market_semantics" not in context
    assert "runtime_surfaces" not in context


def test_rd_agent_runtime_handoff_keeps_execution_payload_separate_from_prompt_context() -> None:
    handoff = build_rd_agent_ashare_runtime_handoff(_valid_contract())

    assert handoff["schema_version"] == "rdagent_ashare_runtime_handoff.v1"
    assert handoff["handoff_id"] == "qlib_joinquant_ashare_runtime_handoff_v1"
    assert handoff["qlib_contract_id"] == "rdagent_qlib_joinquant_ashare_semantic_contract_v1"
    assert handoff["qlib_contract_fingerprint"] == "a" * 64
    assert handoff["semantic_authority"] == "qlib.backtest.ashare_semantics"
    assert handoff["mutation_policy"] == "pass_through_only"
    assert "do_not_mutate_runtime_payload_values" in handoff["consumer_obligations"]
    assert handoff["runtime_payload"]["exchange_kwargs"]["trade_unit"] == 100
    assert handoff["runtime_payload"]["backtest_kwargs"]["pos_type"] == "AsharePosition"


def test_legacy_qlib_without_contract_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "qlib", types.ModuleType("qlib"))
    monkeypatch.setitem(sys.modules, "qlib.backtest", types.ModuleType("qlib.backtest"))
    monkeypatch.setitem(
        sys.modules,
        "qlib.backtest.ashare_semantics",
        types.ModuleType("qlib.backtest.ashare_semantics"),
    )

    with pytest.raises(QlibAshareSemanticContractError, match="rdagent_ashare_semantic_contract"):
        load_qlib_ashare_contract()


def test_malformed_qlib_contract_without_boundary_contract_fails_closed() -> None:
    contract = _valid_contract()
    del contract["semantic_boundary"]

    with pytest.raises(QlibAshareSemanticContractError, match="semantic_boundary"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_contract_without_fingerprint_fails_closed() -> None:
    contract = _valid_contract()
    contract["evidence_contract"]["semantic_fingerprint"] = "not-a-sha"

    with pytest.raises(QlibAshareSemanticContractError, match="semantic_fingerprint"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_runtime_handoff_without_matching_fingerprint_fails_closed() -> None:
    contract = _valid_contract()
    contract["runtime_handoff_contract"]["source_fingerprint"] = "b" * 64

    with pytest.raises(QlibAshareSemanticContractError, match="runtime_handoff_contract"):
        build_rd_agent_ashare_runtime_handoff(contract)


def test_malformed_qlib_prompt_projection_with_raw_runtime_payload_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["runtime_surfaces"] = contract["runtime_surfaces"]

    with pytest.raises(QlibAshareSemanticContractError, match="prompt_projection_payload"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_matching_fingerprint_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["semantic_fingerprint"] = "b" * 64

    with pytest.raises(QlibAshareSemanticContractError, match="prompt_projection_payload"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_projection_schema_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["projection_schema_version"]

    with pytest.raises(QlibAshareSemanticContractError, match="projection schema"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_t1_settlement_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["settlement_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_cash_constraint_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["cash_constraint_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="cash_constraint_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_liquidity_capacity_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["liquidity_capacity_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="liquidity_capacity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_price_limit_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["price_limit_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="price_limit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_instrument_identity_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["instrument_identity_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="instrument_identity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_trading_calendar_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["trading_calendar_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="trading_calendar_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_transaction_cost_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["transaction_cost_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="transaction_cost_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_suspension_tradability_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["suspension_tradability_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="suspension_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_execution_price_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["execution_price_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="execution_price_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_price_adjustment_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["price_adjustment_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="price_adjustment_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_order_unit_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["order_unit_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="order_unit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_price_limit_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_limit_semantics"]["rdagent_rule"] = "rdagent_may_override_price_limits"

    with pytest.raises(QlibAshareSemanticContractError, match="price_limit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_price_limit_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_limit_semantics"]["field_authority"] = "rdagent_board_thresholds"

    with pytest.raises(QlibAshareSemanticContractError, match="price_limit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_price_limit_flags_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_limit_semantics"]["limit_flag_fields"] = ["limit_up", "limit_down"]

    with pytest.raises(QlibAshareSemanticContractError, match="price_limit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_price_limit_fallback_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_limit_semantics"][
        "fallback_authority_rule"
    ] = "board_thresholds_are_primary_authority"

    with pytest.raises(QlibAshareSemanticContractError, match="price_limit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_price_limit_runtime_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_limit_semantics"][
        "runtime_authority"
    ] = "rdagent.scenarios.qlib.price_limits"

    with pytest.raises(QlibAshareSemanticContractError, match="price_limit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_instrument_identity_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["instrument_identity_semantics"][
        "rdagent_rule"
    ] = "rdagent_may_override_instrument_identity"

    with pytest.raises(QlibAshareSemanticContractError, match="instrument_identity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_calendar_frequency_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["trading_calendar_semantics"]["calendar_frequency"] = "1min"

    with pytest.raises(QlibAshareSemanticContractError, match="trading_calendar_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_calendar_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["trading_calendar_semantics"][
        "calendar_provider_authority"
    ] = "rdagent.calendar.Provider"

    with pytest.raises(QlibAshareSemanticContractError, match="trading_calendar_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_calendar_session_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["trading_calendar_semantics"][
        "synthetic_session_rule"
    ] = "rdagent_may_invent_missing_sessions"

    with pytest.raises(QlibAshareSemanticContractError, match="trading_calendar_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_transaction_cost_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["transaction_cost_semantics"][
        "rdagent_rule"
    ] = "rdagent_may_override_transaction_cost"

    with pytest.raises(QlibAshareSemanticContractError, match="transaction_cost_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_suspension_tradability_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["suspension_tradability_semantics"][
        "rdagent_rule"
    ] = "rdagent_may_override_suspension"

    with pytest.raises(QlibAshareSemanticContractError, match="suspension_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_suspension_indicator_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["suspension_tradability_semantics"][
        "suspension_indicator_rule"
    ] = "rdagent_can_infer_suspension_from_prompt"

    with pytest.raises(QlibAshareSemanticContractError, match="suspension_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_execution_price_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["execution_price_semantics"][
        "rdagent_rule"
    ] = "rdagent_may_override_execution_price"

    with pytest.raises(QlibAshareSemanticContractError, match="execution_price_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_price_adjustment_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_adjustment_semantics"][
        "rdagent_rule"
    ] = "rdagent_may_override_price_adjustment"

    with pytest.raises(QlibAshareSemanticContractError, match="price_adjustment_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_factor_field_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_adjustment_semantics"]["factor_field"] = "$adj_factor"

    with pytest.raises(QlibAshareSemanticContractError, match="price_adjustment_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_missing_factor_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["price_adjustment_semantics"][
        "missing_factor_rule"
    ] = "missing_factor_can_be_inferred_by_prompt"

    with pytest.raises(QlibAshareSemanticContractError, match="price_adjustment_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_intraday_execution_price_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["execution_price_semantics"]["execution_frequency"] = "intraday_tick"

    with pytest.raises(QlibAshareSemanticContractError, match="execution_price_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_numeric_transaction_cost_values_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["transaction_cost_semantics"]["numeric_values_exposure"] = "prompt_projection"

    with pytest.raises(QlibAshareSemanticContractError, match="transaction_cost_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_settlement_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["settlement_semantics"]["rdagent_rule"] = "rdagent_may_override_settlement"

    with pytest.raises(QlibAshareSemanticContractError, match="settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_same_day_sell_policy_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["settlement_semantics"]["same_day_sell_policy"] = "same_day_buys_are_sellable"

    with pytest.raises(QlibAshareSemanticContractError, match="settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_sellable_state_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["settlement_semantics"]["sellable_state_field"] = "amount"

    with pytest.raises(QlibAshareSemanticContractError, match="settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_day_commit_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["settlement_semantics"][
        "day_commit_rule"
    ] = "intraday_buys_are_released_on_minute_bar"

    with pytest.raises(QlibAshareSemanticContractError, match="settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_settlement_runtime_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["settlement_semantics"]["runtime_authority"] = "rdagent.position.Ashare"

    with pytest.raises(QlibAshareSemanticContractError, match="settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_cash_buy_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["cash_constraint_semantics"]["buy_cash_rule"] = "rdagent_can_overbuy_cash"

    with pytest.raises(QlibAshareSemanticContractError, match="cash_constraint_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_shorting_policy_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["cash_constraint_semantics"]["shorting_policy"] = "shorting_enabled"

    with pytest.raises(QlibAshareSemanticContractError, match="cash_constraint_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_cash_runtime_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["cash_constraint_semantics"][
        "runtime_authority"
    ] = "rdagent.cash.ExecutionPolicy"

    with pytest.raises(QlibAshareSemanticContractError, match="cash_constraint_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_capacity_parameter_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["liquidity_capacity_semantics"]["capacity_parameter"] = "prompt_capacity"

    with pytest.raises(QlibAshareSemanticContractError, match="liquidity_capacity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_capacity_clip_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["liquidity_capacity_semantics"][
        "capacity_clip_rule"
    ] = "rdagent_can_assume_full_fill"

    with pytest.raises(QlibAshareSemanticContractError, match="liquidity_capacity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_capacity_runtime_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["liquidity_capacity_semantics"][
        "runtime_authority"
    ] = "rdagent.liquidity.CapacityModel"

    with pytest.raises(QlibAshareSemanticContractError, match="liquidity_capacity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_order_unit_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_unit_semantics"]["rdagent_rule"] = "rdagent_may_override_round_lots"

    with pytest.raises(QlibAshareSemanticContractError, match="order_unit_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_optional_prompt_context_reports_unavailable_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "qlib.backtest.ashare_semantics", raising=False)

    text = append_ashare_semantic_context("runtime-ok")

    assert "runtime-ok" in text
    assert "- status: unavailable" in text
    assert "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract" in text


def test_formatted_context_is_operator_readable_without_raw_cost_redefinition() -> None:
    text = format_rd_agent_ashare_semantic_context(build_rd_agent_ashare_semantic_context(_valid_contract()))

    assert "qlib_contract_id: rdagent_qlib_joinquant_ashare_semantic_contract_v1" in text
    assert "qlib_contract_schema_version: qlib_ashare_semantic_contract.v1" in text
    assert "qlib_contract_fingerprint: " + ("a" * 64) in text
    assert "qlib_source_component: qlib.backtest.ashare_semantics" in text
    assert "prompt_projection_schema_version: qlib_ashare_prompt_projection.v1" in text
    assert "prompt_projection_kind: research_prompt_context_only" in text
    assert "instrument identity authority: pyqlib (exchange_prefix_plus_six_digit_code)" in text
    assert "XSHG->SH" in text
    assert "XSHE->SZ" in text
    assert "XBJ->BJ" in text
    assert (
        "board identity authority: pyqlib "
        "(qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.limit_threshold_for_instrument)"
    ) in text
    assert "trading-calendar authority: pyqlib (qlib.data.data.CalendarProvider.calendar)" in text
    assert "trading-calendar locator: pyqlib (qlib.data.data.CalendarProvider.locate_index)" in text
    assert "trading-calendar frequency: day" in text
    assert (
        "trading-calendar non-trading day rule: "
        "calendar_locate_index_maps_start_forward_and_end_backward_to_real_trading_days"
    ) in text
    assert "trading-calendar synthetic session rule: rdagent_must_not_invent_non_qlib_calendar_sessions" in text
    assert (
        "transaction-cost authority: pyqlib "
        "(qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.calculate_trade_cost)"
    ) in text
    assert "transaction-cost buy components: commission, minimum_commission_floor" in text
    assert "transaction-cost sell components: commission, stamp_tax, minimum_commission_floor" in text
    assert "transaction-cost values: runtime_handoff_only_not_prompt_projection" in text
    assert (
        "suspension authority: pyqlib "
        "(qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits)"
    ) in text
    assert "suspension indicator: missing_close_price_marks_suspended" in text
    assert "suspension tradability: suspended_rows_are_not_buyable_or_sellable" in text
    assert "suspension limit flags: qlib_sets_limit_buy_and_limit_sell_true_for_suspended_rows" in text
    assert (
        "execution-price authority: pyqlib " "(qlib.backtest.ashare_semantics.joinquant_ashare_exchange_kwargs)"
    ) in text
    assert "execution-price field: $close" in text
    assert "execution frequency: daily_bar_backtest" in text
    assert "intraday execution rule: not_intraday_or_auction_simulation" in text
    assert (
        "price-adjustment authority: pyqlib " "(qlib.backtest.exchange.Exchange.round_amount_by_trade_unit)"
    ) in text
    assert "price-adjustment factor field: $factor" in text
    assert (
        "price-adjustment missing factor: "
        "non_suspended_rows_with_missing_factor_use_adjusted_price_mode_and_disable_trade_unit_rounding"
    ) in text
    assert (
        "price-adjustment adjusted-price mode: "
        "trade_unit_rounding_is_not_supported_when_adjusted_price_mode_is_active"
    ) in text
    assert "price-limit authority: pyqlib (provider_up_down_limit_fields)" in text
    assert (
        "price-limit runtime authority: pyqlib "
        "(qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits)"
    ) in text
    assert "price-limit mode: strict" in text
    assert "price-limit flag fields: limit_buy, limit_sell" in text
    assert "price-limit buy rule: buy_price_at_or_above_up_limit_or_suspended_sets_limit_buy" in text
    assert "price-limit sell rule: sell_price_at_or_below_down_limit_or_suspended_sets_limit_sell" in text
    assert "price-limit fallback: runtime_compatibility_only_when_authoritative_fields_are_absent" in text
    assert (
        "price-limit fallback authority: "
        "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority"
    ) in text
    assert "settlement authority: pyqlib (t_plus_1_stock)" in text
    assert "settlement runtime authority: pyqlib (qlib.backtest.position.AsharePosition)" in text
    assert "same-day sell policy: shares_bought_today_are_unsellable_until_day_commit" in text
    assert "settlement sellable state: sellable_amount" in text
    assert ("settlement intraday buy rule: " "same_day_buys_increase_total_amount_but_not_sellable_amount") in text
    assert "settlement day commit rule: day_bar_commit_sets_sellable_amount_to_total_amount" in text
    assert "settlement sell clip: sell_orders_are_clipped_by_position_get_sellable_amount" in text
    assert "cash constraint authority: pyqlib (qlib.backtest.exchange.Exchange._calc_trade_info_by_order)" in text
    assert "cash state: cash" in text
    assert "cash buy rule: buy_orders_are_clipped_by_available_cash_and_transaction_cost" in text
    assert "shorting policy: equity_short_selling_is_not_enabled" in text
    assert "liquidity capacity authority: pyqlib (qlib.backtest.exchange.Exchange._clip_amount_by_volume)" in text
    assert "liquidity capacity parameter: volume_threshold" in text
    assert "liquidity volume field: $volume" in text
    assert (
        "liquidity capacity rule: " "order_deal_amount_is_clipped_to_nonnegative_configured_volume_capacity"
    ) in text
    assert "round-lot authority: pyqlib (100 share)" in text
    assert "round-lot buy rule: round_buy_amount_down_to_trade_unit_after_cash_and_volume_limits" in text
    assert "round-lot sell rule: round_sell_amount_down_to_trade_unit_except_full_liquidation" in text
    assert "round-lot full liquidation: sell_all_remaining_position_without_round_lot_residual" in text
    assert (
        "RD-Agent must not redefine: instrument_identity_semantics, "
        "trading_calendar_semantics, transaction_cost_semantics, suspension_tradability_semantics, "
        "execution_price_semantics, price_adjustment_semantics, price_limit_semantics, settlement_semantics, "
        "cash_constraint_semantics, liquidity_capacity_semantics, trade_unit, position_type, settlement_rule, "
        "same_day_sell_policy, data_frequency, price_limit_modes, authoritative_limit_fields, "
        "board_threshold_fields, cost_model"
    ) in text
    assert "prompt projection forbids: runtime_surfaces.policy_defaults" in text
    assert "runtime_surfaces.backtest_kwargs" in text
    assert "AsharePosition" in text
    assert "open_cost" not in text
    assert "close_tax" not in text
