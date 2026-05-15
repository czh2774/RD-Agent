from __future__ import annotations

import re
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from rdagent.scenarios.qlib.ashare_semantics import (
    QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME,
    QLIB_ASHARE_BANDIT_METRIC_PATHS,
    QLIB_ASHARE_DERIVED_FEATURE_SOURCE_RULE,
    QLIB_ASHARE_EXCESS_RETURN_FORBIDDEN_SUBSTITUTIONS,
    QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITH_COST,
    QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITHOUT_COST,
    QLIB_ASHARE_FEEDBACK_METRIC_PATHS,
    QLIB_ASHARE_FEEDBACK_METRIC_PROMPT_PATHS,
    QLIB_ASHARE_FEEDBACK_METRIC_SOURCE_PATHS,
    QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC,
    QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES,
    QLIB_ASHARE_FORBIDDEN_LEGACY_EXCHANGE_KWARGS,
    QLIB_ASHARE_LABEL_COLUMN,
    QLIB_ASHARE_LABEL_EXPRESSION,
    QLIB_ASHARE_LABEL_PROMPT_PATHS,
    QLIB_ASHARE_LABEL_TEMPLATE_PATHS,
    QLIB_ASHARE_MODEL_IMPLEMENTATION_PROMPT_PATHS,
    QLIB_ASHARE_MODEL_OUTPUT_FORMAT_RULE,
    QLIB_ASHARE_MODEL_TASK_BOUNDARY_RULE,
    QLIB_ASHARE_POINT_IN_TIME_REGISTRATION_RULE,
    QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS,
    QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS,
    QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS,
    QLIB_ASHARE_PORTFOLIO_UI_METRIC_PATHS,
    QLIB_ASHARE_PREDICTION_SIGNAL_PROMPT_PATHS,
    QLIB_ASHARE_PROMPT_METRIC_PATHS,
    QLIB_ASHARE_PROMPT_OBLIGATION_RULE,
    QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS,
    QLIB_ASHARE_RESEARCH_DATA_SOURCE_PROMPT_PATHS,
    QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS,
    QLIB_ASHARE_RUNTIME_EXCHANGE_KWARGS,
    QLIB_ASHARE_RUNTIME_TEMPLATE_PATHS,
    QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS,
    QLIB_ASHARE_TEMPLATE_BENCHMARK,
    QLIB_ASHARE_TEMPLATE_MARKET,
    QLIB_ASHARE_TURNOVER_INPUT_BOUNDARY_RULE,
    QLIB_ASHARE_TURNOVER_REPORT_METRIC_RULE,
    QLIB_ASHARE_UI_SELECTED_METRICS,
    QLIB_ASHARE_UNIVERSE_BENCHMARK_TEMPLATE_PATHS,
    QlibAshareSemanticContractError,
    append_ashare_semantic_context,
    build_qlib_ashare_model_task_output_boundary,
    build_rd_agent_ashare_runtime_handoff,
    build_rd_agent_ashare_semantic_context,
    format_rd_agent_ashare_semantic_context,
    load_qlib_ashare_contract,
)
from rdagent.scenarios.qlib.proposal.factor_semantics import (
    build_qlib_ashare_factor_task_source_boundary,
    validate_qlib_factor_experiment_response,
    validate_qlib_factor_hypothesis_response,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_prompt_block(prompt_text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}: \|-\n(?P<body>.*?)(?=^\S|\Z)", prompt_text, flags=re.M | re.S)
    if match is None:
        raise AssertionError(f"Missing prompt block: {key}")
    return match.group("body")


def _market_impact_semantics() -> dict[str, str]:
    return {
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


def _account_update_semantics() -> dict[str, str]:
    return {
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


def _account_valuation_semantics() -> dict[str, Any]:
    return {
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


def _trade_indicator_semantics() -> dict[str, Any]:
    return {
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


def _executor_decision_semantics() -> dict[str, str]:
    return {
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


def _strategy_order_semantics() -> dict[str, str]:
    return {
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


def _supervised_label_semantics() -> dict[str, Any]:
    return {
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


def _prediction_signal_semantics() -> dict[str, Any]:
    return {
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
        "rdagent_model_output_format_rule": QLIB_ASHARE_MODEL_OUTPUT_FORMAT_RULE,
        "rdagent_model_task_boundary_rule": QLIB_ASHARE_MODEL_TASK_BOUNDARY_RULE,
        "rdagent_implementation_prompt_paths": list(QLIB_ASHARE_MODEL_IMPLEMENTATION_PROMPT_PATHS),
        "rdagent_prompt_paths": list(QLIB_ASHARE_PREDICTION_SIGNAL_PROMPT_PATHS),
        "rdagent_rule": "describe_only_do_not_redefine_prediction_signal_score_or_return_realization",
    }


def _signal_ic_semantics() -> dict[str, Any]:
    return {
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


def _portfolio_risk_semantics() -> dict[str, Any]:
    return {
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


def _feedback_metric_semantics() -> dict[str, Any]:
    return {
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


def _excess_return_semantics() -> dict[str, Any]:
    return {
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


def _benchmark_return_semantics() -> dict[str, Any]:
    return {
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


def _universe_benchmark_binding_semantics() -> dict[str, Any]:
    return {
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


def _runtime_handoff_template_binding_semantics() -> dict[str, Any]:
    return {
        "semantic_name": "a_share_rd_agent_runtime_handoff_template_binding",
        "handoff_id": "qlib_joinquant_ashare_runtime_handoff_v1",
        "binding_kind": "rdagent_qlib_template_backtest_runtime_kwargs",
        "rdagent_template_paths": list(QLIB_ASHARE_RUNTIME_TEMPLATE_PATHS),
        "required_backtest_kwargs": deepcopy(QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS),
        "forbidden_legacy_exchange_kwargs": deepcopy(QLIB_ASHARE_FORBIDDEN_LEGACY_EXCHANGE_KWARGS),
        "runtime_rule": "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values",
        "prompt_boundary_rule": "execution_kwargs_remain_runtime_handoff_not_prompt_authority",
        "rdagent_rule": "consume_qlib_runtime_handoff_values_without_redefining_a_share_execution_kwargs",
    }


def _runtime_handoff_template_binding_prompt_semantics() -> dict[str, Any]:
    projection = _runtime_handoff_template_binding_semantics()
    del projection["required_backtest_kwargs"]
    del projection["forbidden_legacy_exchange_kwargs"]
    return projection


def _research_data_source_semantics() -> dict[str, Any]:
    return {
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
        "derived_feature_source_rule": QLIB_ASHARE_DERIVED_FEATURE_SOURCE_RULE,
        "point_in_time_rule": (
            "non_price_volume_fields_are_allowed_only_when_user_or_provider_supplies_daily_point_in_time_data"
        ),
        "point_in_time_registration_rule": QLIB_ASHARE_POINT_IN_TIME_REGISTRATION_RULE,
        "forbidden_default_prompt_sources": list(QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES),
        "turnover_input_boundary_rule": QLIB_ASHARE_TURNOVER_INPUT_BOUNDARY_RULE,
        "frequency_rule": "rdagent_factor_extraction_prompts_must_not_advertise_minute_or_intraday_data_as_default",
        "rdagent_prompt_obligation_rule": QLIB_ASHARE_PROMPT_OBLIGATION_RULE,
        "rdagent_prompt_paths": list(QLIB_ASHARE_RESEARCH_DATA_SOURCE_PROMPT_PATHS),
        "rdagent_rule": "describe_only_use_qlib_registered_daily_or_user_supplied_point_in_time_sources",
    }


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
                "but it must not redefine universe-membership, trading-calendar/data-frequency, trade unit, position, execution-price, "
                "price-adjustment, "
                "suspension/tradability, price-limit, order-tradability, order-fill, account-position update, account valuation, trade indicator/execution-quality, executor/trade-decision lifecycle, strategy signal-to-order generation, supervised label, prediction signal, signal IC, portfolio risk analysis, benchmark-relative excess return, feedback metric consumption, benchmark return, universe/benchmark binding, runtime handoff template binding, research data-source, settlement, cash-settlement, cash/shorting, liquidity/capacity, market-impact, or cost semantics."
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
                "settlement_semantics",
                "cash_settlement_semantics",
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
            "universe_membership_semantics": {
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
                "static_universe_rule": (
                    "rdagent_must_not_treat_all_a_or_index_universe_as_static_without_qlib_membership_spans"
                ),
                "survivorship_rule": "membership_must_remain_point_in_time_by_qlib_instrument_spans_and_filters",
                "rdagent_rule": "describe_only_do_not_redefine_universe_membership_or_filters",
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
            "order_tradability_semantics": {
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
            },
            "order_fill_amount_semantics": {
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
            },
            "market_impact_semantics": _market_impact_semantics(),
            "account_update_semantics": _account_update_semantics(),
            "account_valuation_semantics": _account_valuation_semantics(),
            "trade_indicator_semantics": _trade_indicator_semantics(),
            "executor_decision_semantics": _executor_decision_semantics(),
            "strategy_order_semantics": _strategy_order_semantics(),
            "supervised_label_semantics": _supervised_label_semantics(),
            "prediction_signal_semantics": _prediction_signal_semantics(),
            "signal_ic_semantics": _signal_ic_semantics(),
            "portfolio_risk_semantics": _portfolio_risk_semantics(),
            "excess_return_semantics": _excess_return_semantics(),
            "feedback_metric_semantics": _feedback_metric_semantics(),
            "benchmark_return_semantics": _benchmark_return_semantics(),
            "universe_benchmark_binding_semantics": _universe_benchmark_binding_semantics(),
            "runtime_handoff_template_binding_semantics": _runtime_handoff_template_binding_prompt_semantics(),
            "research_data_source_semantics": _research_data_source_semantics(),
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
            "cash_settlement_semantics": {
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
            "template_runtime_binding": _runtime_handoff_template_binding_semantics(),
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
            "exchange_kwargs": deepcopy(QLIB_ASHARE_RUNTIME_EXCHANGE_KWARGS),
            "backtest_kwargs": deepcopy(QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS),
        },
        "rdagent_must_not_redefine": [
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
            "settlement_semantics",
            "cash_settlement_semantics",
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
    assert "redefine_universe_membership_or_instrument_filtering" in boundary["rdagent_forbidden_actions"]
    assert "redefine_trading_calendar_or_data_frequency" in boundary["rdagent_forbidden_actions"]
    assert "redefine_transaction_cost_model" in boundary["rdagent_forbidden_actions"]
    assert "redefine_suspension_or_tradability_rules" in boundary["rdagent_forbidden_actions"]
    assert "redefine_execution_price_or_frequency" in boundary["rdagent_forbidden_actions"]
    assert "redefine_price_adjustment_or_order_factor" in boundary["rdagent_forbidden_actions"]
    assert "redefine_price_limit_thresholds_or_authoritative_fields" in boundary["rdagent_forbidden_actions"]
    assert "treat_board_fallback_as_primary_price_limit_authority" in boundary["rdagent_forbidden_actions"]
    assert "redefine_order_tradability_or_limit_checks" in boundary["rdagent_forbidden_actions"]
    assert "redefine_order_fill_amount_or_clip_sequence" in boundary["rdagent_forbidden_actions"]
    assert "redefine_market_impact_or_cost_ratio" in boundary["rdagent_forbidden_actions"]
    assert "redefine_account_position_or_cash_mutation_order" in boundary["rdagent_forbidden_actions"]
    assert "redefine_account_valuation_or_bar_end_refresh" in boundary["rdagent_forbidden_actions"]
    assert "redefine_trade_execution_indicators_or_quality_metrics" in boundary["rdagent_forbidden_actions"]
    assert "redefine_executor_decision_lifecycle_or_nested_execution_order" in boundary["rdagent_forbidden_actions"]
    assert "redefine_strategy_signal_to_order_generation" in boundary["rdagent_forbidden_actions"]
    assert "redefine_supervised_label_expression_or_horizon" in boundary["rdagent_forbidden_actions"]
    assert "redefine_prediction_signal_score_or_return_realization" in boundary["rdagent_forbidden_actions"]
    assert "redefine_signal_ic_or_rank_ic_metrics" in boundary["rdagent_forbidden_actions"]
    assert "redefine_portfolio_risk_analysis_metrics" in boundary["rdagent_forbidden_actions"]
    assert "redefine_benchmark_relative_excess_return_or_cost_treatment" in boundary["rdagent_forbidden_actions"]
    assert (
        "redefine_feedback_metric_paths_or_label_derived_utility_as_qlib_metric"
        in boundary["rdagent_forbidden_actions"]
    )
    assert "redefine_benchmark_return_series_or_default_benchmark" in boundary["rdagent_forbidden_actions"]
    assert (
        "redefine_universe_benchmark_template_binding_or_cross_alias_market_and_benchmark"
        in boundary["rdagent_forbidden_actions"]
    )
    assert "redefine_runtime_handoff_or_template_execution_kwargs" in boundary["rdagent_forbidden_actions"]
    assert (
        "redefine_research_data_source_availability_or_imply_unregistered_sources"
        in boundary["rdagent_forbidden_actions"]
    )
    assert "redefine_settlement_or_sellable_position_state" in boundary["rdagent_forbidden_actions"]
    assert "redefine_cash_settlement_or_sell_proceeds_availability" in boundary["rdagent_forbidden_actions"]
    assert "redefine_cash_buying_power_or_shorting_policy" in boundary["rdagent_forbidden_actions"]
    assert "redefine_liquidity_or_volume_capacity_policy" in boundary["rdagent_forbidden_actions"]
    assert "treat_research_prompt_projection_as_backtest_authority" in boundary["rdagent_forbidden_actions"]
    assert boundary["rdagent_must_not_redefine"] == [
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
        "settlement_semantics",
        "cash_settlement_semantics",
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
    assert (
        context["prompt_projection_payload"]["universe_membership_semantics"]["instrument_provider_authority"]
        == "qlib.data.data.InstrumentProvider.list_instruments"
    )
    assert (
        context["prompt_projection_payload"]["universe_membership_semantics"]["market_universe_rule"]
        == "string_codes_are_resolved_by_qlib_D_instruments"
    )
    assert (
        context["prompt_projection_payload"]["universe_membership_semantics"]["filter_pipe_rule"]
        == "qlib_instrument_filter_pipe_is_applied_after_calendar_window_clipping"
    )
    assert (
        context["prompt_projection_payload"]["universe_membership_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_universe_membership_or_filters"
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
    assert context["prompt_projection_payload"]["market_impact_semantics"] == _market_impact_semantics()
    assert context["prompt_projection_payload"]["account_update_semantics"] == _account_update_semantics()
    assert context["prompt_projection_payload"]["account_valuation_semantics"] == _account_valuation_semantics()
    assert context["prompt_projection_payload"]["trade_indicator_semantics"] == _trade_indicator_semantics()
    assert context["prompt_projection_payload"]["executor_decision_semantics"] == _executor_decision_semantics()
    assert context["prompt_projection_payload"]["strategy_order_semantics"] == _strategy_order_semantics()
    assert context["prompt_projection_payload"]["supervised_label_semantics"] == _supervised_label_semantics()
    assert context["prompt_projection_payload"]["prediction_signal_semantics"] == _prediction_signal_semantics()
    assert context["prompt_projection_payload"]["signal_ic_semantics"] == _signal_ic_semantics()
    assert context["prompt_projection_payload"]["portfolio_risk_semantics"] == _portfolio_risk_semantics()
    assert context["prompt_projection_payload"]["excess_return_semantics"] == _excess_return_semantics()
    assert context["prompt_projection_payload"]["feedback_metric_semantics"] == _feedback_metric_semantics()
    assert context["prompt_projection_payload"]["benchmark_return_semantics"] == _benchmark_return_semantics()
    assert (
        context["prompt_projection_payload"]["universe_benchmark_binding_semantics"]
        == _universe_benchmark_binding_semantics()
    )
    assert (
        context["prompt_projection_payload"]["runtime_handoff_template_binding_semantics"]
        == _runtime_handoff_template_binding_prompt_semantics()
    )
    assert context["prompt_projection_payload"]["research_data_source_semantics"] == _research_data_source_semantics()
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
    assert (
        context["prompt_projection_payload"]["order_tradability_semantics"]["runtime_authority"]
        == "qlib.backtest.exchange.Exchange.check_order"
    )
    assert (
        context["prompt_projection_payload"]["order_tradability_semantics"]["decision_rule"]
        == "check_order_delegates_to_is_stock_tradable_before_deal_execution"
    )
    assert (
        context["prompt_projection_payload"]["order_tradability_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_order_tradability_or_limit_checks"
    )
    assert (
        context["prompt_projection_payload"]["order_fill_amount_semantics"]["runtime_authority"]
        == "qlib.backtest.exchange.Exchange._calc_trade_info_by_order"
    )
    assert context["prompt_projection_payload"]["order_fill_amount_semantics"]["clip_sequence"] == [
        "volume_capacity_clip",
        "sellable_position_clip",
        "sell_cash_cost_guard",
        "buy_cash_cost_guard",
        "round_lot_or_full_liquidation_clip",
    ]
    assert (
        context["prompt_projection_payload"]["order_fill_amount_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_order_fill_amount_or_clip_sequence"
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
        context["prompt_projection_payload"]["cash_settlement_semantics"]["sell_proceeds_rule"]
        == "sell_proceeds_enter_cash_delay_when_settle_type_is_cash"
    )
    assert (
        context["prompt_projection_payload"]["cash_settlement_semantics"]["available_cash_rule"]
        == "get_cash_excludes_cash_delay_unless_include_settle_is_true"
    )
    assert (
        context["prompt_projection_payload"]["cash_settlement_semantics"]["commit_rule"]
        == "settle_commit_moves_cash_delay_into_cash_and_clears_delay_state"
    )
    assert (
        context["prompt_projection_payload"]["cash_settlement_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_cash_settlement_or_sell_proceeds_availability"
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


def test_rd_agent_metric_path_constants_match_qlib_contract() -> None:
    contract = _valid_contract()
    portfolio = contract["prompt_projection_payload"]["portfolio_risk_semantics"]
    feedback = contract["prompt_projection_payload"]["feedback_metric_semantics"]
    signal = contract["prompt_projection_payload"]["signal_ic_semantics"]
    binding = contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"]

    assert list(QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS) == signal["rdagent_consumed_metric_paths"]
    assert list(QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS) == portfolio["rdagent_prompt_metric_paths"]
    assert list(QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS) == portfolio["rdagent_feedback_metric_paths"]
    assert list(QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS) == portfolio["rdagent_bandit_metric_paths"]
    assert list(QLIB_ASHARE_PORTFOLIO_UI_METRIC_PATHS) == portfolio["rdagent_ui_metric_paths"]
    assert list(QLIB_ASHARE_PROMPT_METRIC_PATHS) == ["IC", *portfolio["rdagent_prompt_metric_paths"]]
    assert list(QLIB_ASHARE_FEEDBACK_METRIC_PATHS) == ["IC", *portfolio["rdagent_feedback_metric_paths"]]
    assert list(QLIB_ASHARE_BANDIT_METRIC_PATHS) == [
        *signal["rdagent_consumed_metric_paths"],
        *portfolio["rdagent_bandit_metric_paths"],
    ]
    assert list(QLIB_ASHARE_PROMPT_METRIC_PATHS) == feedback["prompt_metric_paths"]
    assert list(QLIB_ASHARE_FEEDBACK_METRIC_PATHS) == feedback["feedback_metric_paths"]
    assert list(QLIB_ASHARE_BANDIT_METRIC_PATHS) == feedback["bandit_metric_paths"]
    assert QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC == feedback["feedback_primary_metric"]
    assert QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME == feedback["derived_bandit_utility_name"]
    assert list(QLIB_ASHARE_UI_SELECTED_METRICS) == ["IC", *portfolio["rdagent_ui_metric_paths"]]
    assert QLIB_ASHARE_TEMPLATE_MARKET == binding["template_market_value"]
    assert QLIB_ASHARE_TEMPLATE_BENCHMARK == binding["template_benchmark_value"]
    assert list(QLIB_ASHARE_UNIVERSE_BENCHMARK_TEMPLATE_PATHS) == binding["rdagent_template_paths"]
    assert all(path == path.strip() for path in QLIB_ASHARE_BANDIT_METRIC_PATHS)


def test_rd_agent_metric_consumers_use_qlib_contract_metric_path_constants() -> None:
    bandit_source = (REPO_ROOT / "rdagent/scenarios/qlib/proposal/bandit.py").read_text()
    feedback_source = (REPO_ROOT / "rdagent/scenarios/qlib/developer/feedback.py").read_text()
    ui_source = (REPO_ROOT / "rdagent/log/ui/app.py").read_text()
    prompts_source = (REPO_ROOT / "rdagent/scenarios/qlib/prompts.yaml").read_text()

    assert "annualized_return " not in bandit_source
    assert "QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS" in bandit_source
    assert "QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS" in bandit_source
    assert "IMPORTANT_METRICS = list(QLIB_ASHARE_FEEDBACK_METRIC_PATHS)" in feedback_source
    assert "QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC" in feedback_source
    assert "metric_name = QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC" in feedback_source
    assert "QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME" in bandit_source
    assert "drawdown_adjusted_return" in bandit_source
    assert "sharpe" not in bandit_source.lower()
    assert "Sharpe" not in ui_source
    assert "QLIB_SELECTED_METRICS = list(QLIB_ASHARE_UI_SELECTED_METRICS)" in ui_source
    for path in QLIB_ASHARE_PROMPT_METRIC_PATHS:
        assert path in prompts_source
    assert all(path == path.strip() for path in QLIB_ASHARE_BANDIT_METRIC_PATHS)


def test_rd_agent_templates_bind_qlib_owned_universe_and_benchmark_without_cross_aliasing() -> None:
    contract = _valid_contract()
    binding = contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"]

    for relative_path in QLIB_ASHARE_UNIVERSE_BENCHMARK_TEMPLATE_PATHS:
        template_text = (REPO_ROOT / relative_path).read_text()

        assert binding["template_market_anchor"] in template_text
        assert binding["template_instruments_binding"] in template_text
        assert binding["template_benchmark_anchor"] in template_text
        assert binding["template_backtest_benchmark_binding"] in template_text
        assert "instruments: *benchmark" not in template_text
        assert "benchmark: *market" not in template_text
        for forbidden_value in binding["forbidden_template_values"]:
            assert f"market: &market {forbidden_value}" not in template_text
            assert f"benchmark: &benchmark {forbidden_value}" not in template_text


def test_rd_agent_templates_bind_qlib_owned_runtime_handoff_backtest_kwargs() -> None:
    binding = _runtime_handoff_template_binding_semantics()

    for relative_path in QLIB_ASHARE_RUNTIME_TEMPLATE_PATHS:
        template_text = (REPO_ROOT / relative_path).read_text()

        assert relative_path in binding["rdagent_template_paths"]
        assert "pos_type: AsharePosition" in template_text
        assert "limit_threshold: joinquant_ashare" in template_text
        assert "ashare_price_limit_mode: strict" in template_text
        assert "ashare_limit_options:" in template_text
        assert "close_commission: 0.0003" in template_text
        assert "close_tax: 0.001" in template_text
        assert "trade_unit: 100" in template_text
        assert "deal_price: close" in template_text
        assert "open_cost: 0.0003" in template_text
        assert "close_cost: 0.0013" in template_text
        assert "min_cost: 5.0" in template_text
        assert "limit_threshold: 0.095" not in template_text
        assert "open_cost: 0.0005" not in template_text
        assert "close_cost: 0.0015" not in template_text


def test_rd_agent_factor_extraction_prompts_use_qlib_daily_research_source_boundary() -> None:
    source_boundary = _research_data_source_semantics()

    for relative_path in QLIB_ASHARE_RESEARCH_DATA_SOURCE_PROMPT_PATHS:
        prompt_text = (REPO_ROOT / relative_path).read_text()

        assert "Qlib daily A-share research data boundary" in prompt_text
        assert "`datetime` and `instrument`" in prompt_text
        for field in source_boundary["primary_price_volume_fields"]:
            assert field in prompt_text
        assert "Alpha158/Alpha360" in prompt_text
        assert "derived features must not introduce new source fields" in prompt_text
        assert "daily point-in-time" in prompt_text
        assert "source owner, field identity, and daily point-in-time validity" in prompt_text
        assert "High-Frequency Data:" not in prompt_text
        assert "Consensus Expectations Factor" not in prompt_text
        assert "containing open close high low volume vwap in each minute" not in prompt_text
        assert "Do not assume turnover, minute-level high-frequency data" in prompt_text
        assert "Do not treat turnover, minute-level high-frequency data" in prompt_text
        assert "Turnover may appear as a Qlib post-backtest portfolio report metric" in prompt_text
        assert "analyst consensus expectation factors" in prompt_text


def test_rd_agent_factor_relevance_prompt_applies_qlib_source_boundary_forbidden_defaults() -> None:
    prompt_text = (REPO_ROOT / "rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml").read_text()
    relevance_prompt = _read_prompt_block(prompt_text, "factor_relevance_system")

    assert "Qlib daily A-share research data boundary" in relevance_prompt
    assert "`datetime` and `instrument`" in relevance_prompt
    for field in QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS:
        assert field in relevance_prompt
    assert "Alpha158/Alpha360" in relevance_prompt
    assert "derived features must not introduce new source fields" in relevance_prompt
    assert "source owner, field identity, and daily point-in-time validity" in relevance_prompt
    assert "Do not assume turnover, minute-level high-frequency data" in relevance_prompt
    assert "Do not treat turnover, minute-level high-frequency data" in relevance_prompt
    assert "Turnover may appear as a Qlib post-backtest portfolio report metric" in relevance_prompt
    assert "not a default factor input field" in relevance_prompt
    assert "analyst consensus expectation factors" in relevance_prompt


def test_rd_agent_factor_duplicate_prompt_applies_qlib_source_boundary_forbidden_defaults() -> None:
    prompt_text = (REPO_ROOT / "rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml").read_text()
    duplicate_prompt = _read_prompt_block(prompt_text, "factor_duplicate_system")

    assert "Qlib daily A-share research data boundary" in duplicate_prompt
    assert "`datetime` and `instrument`" in duplicate_prompt
    for field in QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS:
        assert field in duplicate_prompt
    assert "Alpha158/Alpha360" in duplicate_prompt
    assert "derived features must not introduce new source fields" in duplicate_prompt
    assert "source owner, field identity, and daily point-in-time validity" in duplicate_prompt
    assert "Do not assume turnover, minute-level high-frequency data" in duplicate_prompt
    assert "Do not treat turnover, minute-level high-frequency data" in duplicate_prompt
    assert "Turnover may appear as a Qlib post-backtest portfolio report metric" in duplicate_prompt
    assert "not a default factor input field" in duplicate_prompt
    assert "analyst consensus expectation factors" in duplicate_prompt


def test_rd_agent_feedback_metric_prompts_use_exact_qlib_paths_without_sharpe_alias() -> None:
    combined = "\n".join((REPO_ROOT / path).read_text() for path in QLIB_ASHARE_FEEDBACK_METRIC_PROMPT_PATHS)

    assert (
        "Qlib-owned prompt metric paths IC, 1day.excess_return_without_cost.annualized_return, "
        "and 1day.excess_return_without_cost.max_drawdown"
    ) in combined
    assert (
        "Qlib benchmark-relative feedback primary metric " "1day.excess_return_with_cost.annualized_return"
    ) in combined
    assert "sharpe ratio" not in combined.lower()
    assert "return, sharpe ratio, max drawdown, and so on" not in combined


def test_rd_agent_prompts_describe_excess_return_as_benchmark_relative_not_raw_return() -> None:
    combined = "\n".join(
        (REPO_ROOT / path).read_text()
        for path in (
            "rdagent/scenarios/qlib/experiment/prompts.yaml",
            "rdagent/scenarios/qlib/prompts.yaml",
        )
    )

    assert "Qlib benchmark-relative excess-return metrics" in combined
    assert "sources of benchmark-relative excess return" in combined
    assert (
        "These `excess_return_*` paths are benchmark-relative metrics computed from Qlib report columns, "
        "not raw portfolio return aliases."
    ) in combined
    assert "enhance our investment returns" not in combined
    assert "sources of excess returns" not in combined
    assert "raw portfolio return aliases" in combined


def test_rd_agent_factor_proposal_validator_uses_qlib_daily_research_data_boundary() -> None:
    accepted = validate_qlib_factor_hypothesis_response(
        {
            "hypothesis": "Use close-to-vwap reversal and volume persistence on daily Qlib A-share bars.",
            "reason": "The signal is grounded in daily $close, $vwap, and $volume fields from the Qlib boundary.",
        }
    )
    assert accepted["hypothesis"].startswith("Use close-to-vwap")

    registered_pit = validate_qlib_factor_hypothesis_response(
        {
            "hypothesis": "Use provider supplied daily point-in-time industry field to form an industry concentration signal.",
            "reason": (
                "The provider names source owner, field identity, and daily point-in-time validity before RD-Agent "
                "uses the field."
            ),
        }
    )
    assert "provider supplied daily point-in-time" in registered_pit["hypothesis"]

    forbidden_payloads = [
        {
            "hypothesis": "Use analyst consensus revisions to predict close-to-close reversal.",
            "reason": "Consensus expectation data may identify future alpha.",
        },
        {
            "hypothesis": "Use minute-level high-frequency vwap momentum during the morning session.",
            "reason": "Intraday price behavior may improve the daily signal.",
        },
        {
            "hypothesis": "Use turnover acceleration as the primary A-share liquidity factor.",
            "reason": "Turnover is not part of the default Qlib daily research data boundary.",
        },
        {
            "hypothesis": "Use point-in-time industry membership to neutralize daily momentum.",
            "reason": "Industry classification is point-in-time but the source owner and field identity are unspecified.",
        },
    ]
    for payload in forbidden_payloads:
        with pytest.raises(ValueError, match="Qlib daily A-share research data boundary"):
            validate_qlib_factor_hypothesis_response(payload)


def test_rd_agent_factor_experiment_validator_uses_qlib_daily_research_data_boundary() -> None:
    accepted = validate_qlib_factor_experiment_response(
        {
            "close_vwap_reversal": {
                "description": "[Reversal Factor] Daily Qlib A-share close-to-vwap reversal with volume confirmation.",
                "formulation": r"factor_t = -($close_t - $vwap_t) / $vwap_t \times log(1 + $volume_t)",
                "variables": {
                    "$close": "Qlib registered daily close field.",
                    "$vwap": "Qlib registered daily VWAP field.",
                    "$volume": "Qlib registered daily volume field.",
                },
            }
        }
    )
    assert accepted["close_vwap_reversal"]["variables"]["$close"] == "Qlib registered daily close field."

    registered_pit = validate_qlib_factor_experiment_response(
        {
            "provider_industry_concentration": {
                "description": "[Industry Factor] Provider supplied daily point-in-time industry concentration.",
                "formulation": r"factor_t = industry_concentration_t",
                "variables": {
                    "industry_concentration_t": (
                        "Provider supplied daily point-in-time industry field with source owner, field identity, "
                        "and daily point-in-time validity."
                    ),
                },
            }
        }
    )
    assert "provider_industry_concentration" in registered_pit

    forbidden_payloads = [
        {
            "turnover_acceleration": {
                "description": "[Liquidity Factor] Use turnover acceleration.",
                "formulation": r"factor_t = turnover_t / turnover_{t-5} - 1",
                "variables": {"turnover": "Default turnover series."},
            }
        },
        {
            "morning_vwap_momentum": {
                "description": "[Momentum Factor] Use minute-level high-frequency VWAP.",
                "formulation": r"factor_t = vwap_{10:30} / open - 1",
                "variables": {"vwap_{10:30}": "Intraday minute VWAP."},
            }
        },
        {
            "consensus_revision": {
                "description": "[Expectation Factor] Use analyst consensus expectation revisions.",
                "formulation": r"factor_t = consensus_eps_t - consensus_eps_{t-20}",
                "variables": {"consensus_eps": "Analyst consensus EPS expectation."},
            }
        },
        {
            "unregistered_industry_neutral": {
                "description": "[Industry Factor] Use point-in-time industry membership.",
                "formulation": r"factor_t = industry_member_t",
                "variables": {"industry_member_t": "Daily point-in-time industry field."},
            }
        },
    ]
    for payload in forbidden_payloads:
        with pytest.raises(ValueError, match="Qlib daily A-share research data boundary"):
            validate_qlib_factor_experiment_response(payload)

    with pytest.raises(ValueError, match="registered daily Qlib A-share fields"):
        validate_qlib_factor_experiment_response(
            {
                "ambiguous_liquidity": {
                    "description": "[Liquidity Factor] Use broad liquidity pressure.",
                    "formulation": r"factor_t = A_t / B_t",
                    "variables": {
                        "A_t": "Market activity.",
                        "B_t": "Market baseline.",
                    },
                }
            }
        )


def test_rd_agent_factor_task_information_carries_qlib_source_boundary_to_coder() -> None:
    boundary = build_qlib_ashare_factor_task_source_boundary()
    assert "Qlib daily A-share research data boundary" in boundary
    for field in QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS:
        assert field in boundary
    for forbidden_source in QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES:
        assert forbidden_source in boundary
    assert QLIB_ASHARE_DERIVED_FEATURE_SOURCE_RULE in boundary
    assert QLIB_ASHARE_POINT_IN_TIME_REGISTRATION_RULE in boundary
    assert QLIB_ASHARE_TURNOVER_INPUT_BOUNDARY_RULE in boundary
    assert "turnover" in boundary

    factor_task_source = (REPO_ROOT / "rdagent/components/coder/factor_coder/factor.py").read_text()
    assert "source_data_boundary" in factor_task_source
    assert "source_data_boundary: {self.source_data_boundary}" in factor_task_source
    assert '"source_data_boundary"] = self.source_data_boundary' in factor_task_source

    proposal_source = (REPO_ROOT / "rdagent/scenarios/qlib/proposal/factor_proposal.py").read_text()
    assert "build_qlib_ashare_factor_task_source_boundary" in proposal_source
    assert "source_data_boundary=source_data_boundary" in proposal_source

    json_loader_source = (REPO_ROOT / "rdagent/scenarios/qlib/factor_experiment_loader/json_loader.py").read_text()
    assert "build_qlib_ashare_factor_task_source_boundary" in json_loader_source
    assert "source_data_boundary=source_data_boundary" in json_loader_source

    coder_prompt = (REPO_ROOT / "rdagent/components/coder/factor_coder/prompts.yaml").read_text()
    assert "{{ factor_information_str }}" in coder_prompt
    assert "{{ factor_information }}" in coder_prompt

    workflow = (REPO_ROOT / ".github/workflows/internal_ashare_semantics.yml").read_text()
    assert "rdagent/components/coder/factor_coder/factor.py" in workflow
    assert "rdagent/components/coder/factor_coder/prompts.yaml" in workflow
    assert "rdagent/scenarios/qlib/factor_experiment_loader/json_loader.py" in workflow


def test_rd_agent_model_task_information_carries_qlib_prediction_signal_boundary_to_coder() -> None:
    boundary = build_qlib_ashare_model_task_output_boundary(_valid_contract())
    assert "Qlib A-share model output boundary" in boundary
    assert f"Qlib prediction signal score for {QLIB_ASHARE_LABEL_COLUMN}" in boundary
    assert "pred.pkl" in boundary
    assert "`score` column" in boundary
    assert "`datetime` and `instrument`" in boundary
    assert "not_realized_or_executable_return" in boundary
    assert QLIB_ASHARE_MODEL_OUTPUT_FORMAT_RULE in boundary
    assert QLIB_ASHARE_MODEL_TASK_BOUNDARY_RULE in boundary
    assert "not_graph_node_output" in boundary

    model_task_source = (REPO_ROOT / "rdagent/components/coder/model_coder/model.py").read_text()
    assert "model_output_boundary" in model_task_source
    assert "model_output_boundary: {self.model_output_boundary}" in model_task_source

    proposal_source = (REPO_ROOT / "rdagent/scenarios/qlib/proposal/model_proposal.py").read_text()
    assert "build_qlib_ashare_model_task_output_boundary" in proposal_source
    assert "model_output_boundary=model_output_boundary" in proposal_source

    coder_prompt = (REPO_ROOT / "rdagent/components/coder/model_coder/prompts.yaml").read_text()
    assert "{{ model_information_str }}" in coder_prompt
    for relative_path in QLIB_ASHARE_MODEL_IMPLEMENTATION_PROMPT_PATHS:
        assert (REPO_ROOT / relative_path).exists()

    workflow = (REPO_ROOT / ".github/workflows/internal_ashare_semantics.yml").read_text()
    assert "rdagent/components/coder/model_coder/model.py" in workflow
    assert "rdagent/components/coder/model_coder/prompts.yaml" in workflow
    assert "rdagent/scenarios/qlib/proposal/model_proposal.py" in workflow


def test_rd_agent_factor_coder_prompts_enforce_qlib_source_boundary_as_non_bypassable() -> None:
    coder_prompt = (REPO_ROOT / "rdagent/components/coder/factor_coder/prompts.yaml").read_text()
    source_boundary_prompt_keys = [
        "evaluator_code_feedback_v1_system",
        "evolving_strategy_factor_implementation_v1_system",
        "evolving_strategy_error_summary_v2_system",
        "select_implementable_factor_system",
        "evaluator_final_decision_v1_system",
    ]

    for prompt_key in source_boundary_prompt_keys:
        prompt_block = _read_prompt_block(coder_prompt, prompt_key)
        assert "source_data_boundary" in prompt_block
        assert "non-bypassable implementation boundary" in prompt_block
        assert "must not infer fields, vendors, frequencies, or data sources outside that boundary" in prompt_block

    implementation_prompt = _read_prompt_block(coder_prompt, "evolving_strategy_factor_implementation_v1_system")
    assert "intraday or minute data" in implementation_prompt
    assert "turnover as a factor input field" in implementation_prompt
    assert "Qlib portfolio-report turnover is a post-backtest metric" in implementation_prompt
    assert "consensus data" in implementation_prompt

    selector_prompt = _read_prompt_block(coder_prompt, "select_implementable_factor_system")
    assert "not implementable under the current source boundary" in selector_prompt

    final_decision_prompt = _read_prompt_block(coder_prompt, "evaluator_final_decision_v1_system")
    assert "source boundary violation makes final_decision False" in final_decision_prompt


def test_rd_agent_factor_prompt_specification_uses_registered_daily_qlib_fields() -> None:
    prompt_text = (REPO_ROOT / "rdagent/scenarios/qlib/prompts.yaml").read_text()

    assert (
        "Qlib registered daily A-share fields (`$open`, `$close`, `$high`, `$low`, `$vwap`, `$volume`)" in prompt_text
    )
    assert "Qlib Alpha158/Alpha360 derived features computed only from those registered daily price-volume fields" in (
        prompt_text
    )
    assert "source owner, field identity, and daily point-in-time validity" in prompt_text
    assert (
        "Do not treat turnover, minute-level high-frequency data, analyst consensus expectation factors, "
        "or unregistered external vendor fields as default Qlib inputs."
    ) in prompt_text
    assert "Turnover may appear as a Qlib post-backtest portfolio report metric" in prompt_text
    assert (
        "Every factor description, formulation, and variables map must bind the factor to Qlib registered daily "
        "A-share fields (`$open`, `$close`, `$high`, `$low`, `$vwap`, `$volume`)"
    ) in prompt_text
    assert (
        "Do not use turnover, minute-level high-frequency data, analyst consensus expectation factors, "
        "or unregistered external vendor fields as default Qlib inputs."
    ) in prompt_text
    assert "price, volume, turnover, return, or related financial fields" not in prompt_text


def test_rd_agent_bandit_uses_derived_drawdown_adjusted_return_without_sharpe_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeArray(list):
        pass

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = FakeArray
    fake_numpy.array = lambda values: FakeArray(values)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.delitem(sys.modules, "rdagent.scenarios.qlib.proposal.bandit", raising=False)

    from rdagent.scenarios.qlib.proposal.bandit import (
        Metrics,
        extract_metrics_from_experiment,
    )

    experiment = types.SimpleNamespace(
        result={
            QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[0]: 0.1,
            QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[1]: 0.2,
            QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[2]: 0.3,
            QLIB_ASHARE_SIGNAL_IC_METRIC_PATHS[3]: 0.4,
            QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS[0]: 0.2,
            QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS[1]: 1.3,
            QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS[2]: -0.1,
        }
    )

    metrics = extract_metrics_from_experiment(experiment)

    assert isinstance(metrics, Metrics)
    assert abs(metrics.drawdown_adjusted_return - 2.0) < 1e-12
    assert abs(metrics.as_vector()[-1] - 2.0) < 1e-12
    assert not hasattr(metrics, "sharpe")


def test_rd_agent_runtime_handoff_keeps_execution_payload_separate_from_prompt_context() -> None:
    handoff = build_rd_agent_ashare_runtime_handoff(_valid_contract())

    assert handoff["schema_version"] == "rdagent_ashare_runtime_handoff.v1"
    assert handoff["handoff_id"] == "qlib_joinquant_ashare_runtime_handoff_v1"
    assert handoff["qlib_contract_id"] == "rdagent_qlib_joinquant_ashare_semantic_contract_v1"
    assert handoff["qlib_contract_fingerprint"] == "a" * 64
    assert handoff["semantic_authority"] == "qlib.backtest.ashare_semantics"
    assert handoff["mutation_policy"] == "pass_through_only"
    assert "do_not_mutate_runtime_payload_values" in handoff["consumer_obligations"]
    assert handoff["runtime_payload"]["exchange_kwargs"] == QLIB_ASHARE_RUNTIME_EXCHANGE_KWARGS
    assert handoff["runtime_payload"]["backtest_kwargs"] == QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS
    assert handoff["runtime_payload"]["exchange_kwargs"]["trade_unit"] == 100
    assert handoff["runtime_payload"]["backtest_kwargs"]["pos_type"] == "AsharePosition"
    assert handoff["template_runtime_binding"] == _runtime_handoff_template_binding_semantics()


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


def test_malformed_qlib_runtime_handoff_without_template_binding_fails_closed() -> None:
    contract = _valid_contract()
    del contract["runtime_handoff_contract"]["template_runtime_binding"]

    with pytest.raises(QlibAshareSemanticContractError, match="template_runtime_binding"):
        build_rd_agent_ashare_runtime_handoff(contract)


def test_malformed_qlib_runtime_handoff_with_legacy_template_limit_fails_closed() -> None:
    contract = _valid_contract()
    contract["runtime_handoff_contract"]["template_runtime_binding"]["required_backtest_kwargs"]["exchange_kwargs"][
        "limit_threshold"
    ] = 0.095

    with pytest.raises(QlibAshareSemanticContractError, match="template_runtime_binding"):
        build_rd_agent_ashare_runtime_handoff(contract)


def test_malformed_qlib_runtime_handoff_template_binding_drift_from_runtime_surface_fails_closed() -> None:
    contract = _valid_contract()
    contract["runtime_handoff_contract"]["template_runtime_binding"]["required_backtest_kwargs"][
        "pos_type"
    ] = "Position"

    with pytest.raises(QlibAshareSemanticContractError, match="template_runtime_binding"):
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


def test_malformed_qlib_prompt_projection_without_cash_settlement_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["cash_settlement_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="cash_settlement_semantics"):
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


def test_malformed_qlib_prompt_projection_without_order_tradability_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["order_tradability_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="order_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_order_fill_amount_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["order_fill_amount_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="order_fill_amount_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_instrument_identity_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["instrument_identity_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="instrument_identity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_universe_membership_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["universe_membership_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="universe_membership_semantics"):
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


def test_malformed_qlib_prompt_projection_without_market_impact_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["market_impact_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="market_impact_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_account_update_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["account_update_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="account_update_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_account_valuation_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["account_valuation_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="account_valuation_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_trade_indicator_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["trade_indicator_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="trade_indicator_semantics"):
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


def test_malformed_qlib_prompt_projection_with_mutable_order_tradability_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_tradability_semantics"][
        "runtime_authority"
    ] = "rdagent.scenarios.qlib.order_gate"

    with pytest.raises(QlibAshareSemanticContractError, match="order_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_directional_limit_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_tradability_semantics"][
        "directional_limit_rule"
    ] = "rdagent_can_infer_direction_limits"

    with pytest.raises(QlibAshareSemanticContractError, match="order_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_order_failure_result_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_tradability_semantics"]["failure_result"] = "raise_prompt_error"

    with pytest.raises(QlibAshareSemanticContractError, match="order_tradability_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_order_fill_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_fill_amount_semantics"][
        "runtime_authority"
    ] = "rdagent.scenarios.qlib.fill_policy"

    with pytest.raises(QlibAshareSemanticContractError, match="order_fill_amount_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_order_fill_sequence_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_fill_amount_semantics"]["clip_sequence"] = [
        "prompt_round_lot_first",
        "cash_after_fill",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="order_fill_amount_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_order_fill_trade_value_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["order_fill_amount_semantics"][
        "trade_value_rule"
    ] = "trade_value_uses_requested_amount"

    with pytest.raises(QlibAshareSemanticContractError, match="order_fill_amount_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_instrument_identity_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["instrument_identity_semantics"][
        "rdagent_rule"
    ] = "rdagent_may_override_instrument_identity"

    with pytest.raises(QlibAshareSemanticContractError, match="instrument_identity_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_universe_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_membership_semantics"][
        "instrument_provider_authority"
    ] = "rdagent.universe.StaticProvider"

    with pytest.raises(QlibAshareSemanticContractError, match="universe_membership_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_static_universe_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_membership_semantics"][
        "static_universe_rule"
    ] = "rdagent_may_treat_all_a_as_static"

    with pytest.raises(QlibAshareSemanticContractError, match="universe_membership_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_filter_pipe_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_membership_semantics"][
        "filter_pipe_rule"
    ] = "rdagent_may_apply_prompt_filters"

    with pytest.raises(QlibAshareSemanticContractError, match="universe_membership_semantics"):
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


def test_malformed_qlib_prompt_projection_with_mutable_market_impact_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["market_impact_semantics"][
        "runtime_authority"
    ] = "rdagent.market_impact.Model"

    with pytest.raises(QlibAshareSemanticContractError, match="market_impact_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_market_impact_ratio_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["market_impact_semantics"][
        "impact_cost_ratio_rule"
    ] = "rdagent_estimates_impact_cost_from_prompt_capacity"

    with pytest.raises(QlibAshareSemanticContractError, match="market_impact_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_market_impact_cost_exposure_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["market_impact_semantics"]["numeric_value_exposure"] = "prompt_projection"

    with pytest.raises(QlibAshareSemanticContractError, match="market_impact_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_account_update_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["account_update_semantics"][
        "account_update_authority"
    ] = "rdagent.account.UpdatePolicy"

    with pytest.raises(QlibAshareSemanticContractError, match="account_update_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_account_update_trigger_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["account_update_semantics"][
        "trade_update_trigger"
    ] = "prompt_may_update_zero_value_trades"

    with pytest.raises(QlibAshareSemanticContractError, match="account_update_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_account_cash_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["account_update_semantics"][
        "sell_cash_rule"
    ] = "sell_proceeds_are_always_available_immediately"

    with pytest.raises(QlibAshareSemanticContractError, match="account_update_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_account_valuation_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["account_valuation_semantics"][
        "bar_end_authority"
    ] = "rdagent.account.ValuationPolicy"

    with pytest.raises(QlibAshareSemanticContractError, match="account_valuation_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_account_valuation_sequence_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["account_valuation_semantics"]["bar_end_sequence"] = [
        "update_trade_indicators",
        "prompt_mark_to_market",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="account_valuation_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_account_valuation_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["account_valuation_semantics"][
        "account_value_rule"
    ] = "account_value_excludes_cash_delay"

    with pytest.raises(QlibAshareSemanticContractError, match="account_valuation_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_trade_indicator_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["trade_indicator_semantics"][
        "indicator_authority"
    ] = "rdagent.execution.Indicator"

    with pytest.raises(QlibAshareSemanticContractError, match="trade_indicator_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_trade_metric_fields_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["trade_indicator_semantics"]["trade_metric_fields"] = [
        "return",
        "turnover",
        "cost",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="trade_indicator_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_trade_indicator_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["trade_indicator_semantics"][
        "portfolio_boundary_rule"
    ] = "trade_indicators_are_portfolio_return_metrics"

    with pytest.raises(QlibAshareSemanticContractError, match="trade_indicator_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_executor_decision_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["executor_decision_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="executor_decision_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_executor_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["executor_decision_semantics"][
        "base_executor_authority"
    ] = "rdagent.execution.Executor"

    with pytest.raises(QlibAshareSemanticContractError, match="executor_decision_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_executor_range_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["executor_decision_semantics"][
        "nested_range_rule"
    ] = "rdagent_may_ignore_qlib_range_limit_alignment"

    with pytest.raises(QlibAshareSemanticContractError, match="executor_decision_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_executor_inner_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["executor_decision_semantics"][
        "inner_decision_rule"
    ] = "inner_trade_decision_always_overrides_outer_decision"

    with pytest.raises(QlibAshareSemanticContractError, match="executor_decision_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_strategy_order_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["strategy_order_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="strategy_order_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_strategy_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["strategy_order_semantics"][
        "topk_strategy_authority"
    ] = "rdagent.strategy.TopkDropoutStrategy.generate_trade_decision"

    with pytest.raises(QlibAshareSemanticContractError, match="strategy_order_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_strategy_dropout_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["strategy_order_semantics"][
        "dropout_rule"
    ] = "rdagent_may_drop_high_score_stock_for_lower_score_buy"

    with pytest.raises(QlibAshareSemanticContractError, match="strategy_order_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_strategy_return_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["strategy_order_semantics"][
        "target_order_return_rule"
    ] = "exchange_returns_buy_orders_before_sell_orders"

    with pytest.raises(QlibAshareSemanticContractError, match="strategy_order_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_supervised_label_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["supervised_label_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="supervised_label_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_supervised_label_expression_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["supervised_label_semantics"]["label_expression"] = "$close/Ref($close,1)-1"

    with pytest.raises(QlibAshareSemanticContractError, match="supervised_label_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_supervised_label_prompt_paths_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["supervised_label_semantics"]["rdagent_prompt_paths"] = [
        "rdagent/scenarios/qlib/prompts.yaml"
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="supervised_label_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_rd_agent_templates_bind_qlib_owned_supervised_label_expression() -> None:
    for relative_path in QLIB_ASHARE_LABEL_TEMPLATE_PATHS:
        template_text = (REPO_ROOT / relative_path).read_text()

        assert QLIB_ASHARE_LABEL_EXPRESSION in template_text
        assert f'- ["{QLIB_ASHARE_LABEL_COLUMN}"]' in template_text


def test_rd_agent_prompts_describe_qlib_owned_supervised_label_without_vague_horizon() -> None:
    for relative_path in QLIB_ASHARE_LABEL_PROMPT_PATHS:
        prompt_text = (REPO_ROOT / relative_path).read_text()

        assert QLIB_ASHARE_LABEL_EXPRESSION in prompt_text
        assert f"Qlib contract-defined {QLIB_ASHARE_LABEL_COLUMN} forward return" in prompt_text
        assert "next several days return" not in prompt_text
        assert "next several days' returns" not in prompt_text


def test_malformed_qlib_prompt_projection_without_prediction_signal_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["prediction_signal_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="prediction_signal_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_prediction_signal_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["prediction_signal_semantics"][
        "model_signal_authority"
    ] = "rdagent.scenarios.qlib.Signal"

    with pytest.raises(QlibAshareSemanticContractError, match="prediction_signal_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_prediction_signal_wording_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["prediction_signal_semantics"][
        "prompt_wording_rule"
    ] = "describe_as_predicted_future_return"

    with pytest.raises(QlibAshareSemanticContractError, match="prediction_signal_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_model_output_format_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["prediction_signal_semantics"][
        "rdagent_model_output_format_rule"
    ] = "rdagent_model_output_format_may_describe_generic_node_predictions"

    with pytest.raises(QlibAshareSemanticContractError, match="prediction_signal_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_model_task_boundary_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["prediction_signal_semantics"][
        "rdagent_model_task_boundary_rule"
    ] = "rdagent_model_tasks_may_omit_qlib_prediction_signal_boundary"

    with pytest.raises(QlibAshareSemanticContractError, match="prediction_signal_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_missing_model_implementation_prompt_consumer_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["prediction_signal_semantics"]["rdagent_implementation_prompt_paths"] = [
        "rdagent/scenarios/qlib/prompts.yaml",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="prediction_signal_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_rd_agent_prompts_describe_prediction_signal_without_realized_return_claims() -> None:
    combined = "\n".join((REPO_ROOT / path).read_text() for path in QLIB_ASHARE_PREDICTION_SIGNAL_PROMPT_PATHS)
    experiment_prompt = (REPO_ROOT / "rdagent/scenarios/qlib/experiment/prompts.yaml").read_text()
    qlib_prompt = (REPO_ROOT / "rdagent/scenarios/qlib/prompts.yaml").read_text()
    model_output_prompt = _read_prompt_block(qlib_prompt, "model_experiment_output_format")

    assert "Qlib prediction signal score for LABEL0" in combined
    assert "Qlib prediction signal score for LABEL0" in model_output_prompt
    assert "`score` column in `pred.pkl`" in model_output_prompt
    assert "`datetime` and `instrument`" in model_output_prompt
    assert "trained against Qlib-owned LABEL0" in model_output_prompt
    assert "node u" not in model_output_prompt
    assert "predicted output for node" not in model_output_prompt
    assert "\\\\hat{y}_u" not in model_output_prompt
    assert "predicting future returns" not in combined
    assert "predicts the future returns" not in combined
    assert "predicted returns" not in combined
    assert "predicted return based on a strategy" not in combined
    assert "physics value" not in experiment_prompt


def test_malformed_qlib_prompt_projection_without_signal_ic_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["signal_ic_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="signal_ic_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_signal_ic_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["signal_ic_semantics"]["ic_calculation_authority"] = "rdagent.metrics.calc_ic"

    with pytest.raises(QlibAshareSemanticContractError, match="signal_ic_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_signal_ic_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["signal_ic_semantics"][
        "ic_rule"
    ] = "IC_is_prompt_defined_factor_return_correlation"

    with pytest.raises(QlibAshareSemanticContractError, match="signal_ic_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_signal_ic_metric_paths_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["signal_ic_semantics"]["rdagent_consumed_metric_paths"] = [
        "signal_ic",
        "rank_ic",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="signal_ic_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_portfolio_risk_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["portfolio_risk_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_portfolio_risk_authority_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["portfolio_risk_semantics"][
        "risk_analysis_authority"
    ] = "rdagent.analysis.risk_analysis"

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_portfolio_risk_scaler_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["portfolio_risk_semantics"]["day_annualization_scaler"] = 252

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_turnover_report_metric_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["portfolio_risk_semantics"][
        "turnover_report_metric_rule"
    ] = "turnover_is_a_default_factor_input_and_report_metric"

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_portfolio_metric_paths_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["portfolio_risk_semantics"]["rdagent_consumed_metric_paths"] = [
        "annual_return",
        "max_drawdown",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_whitespace_portfolio_metric_path_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["portfolio_risk_semantics"]["rdagent_bandit_metric_paths"] = [
        "1day.excess_return_with_cost.annualized_return ",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_metric_path_format_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["portfolio_risk_semantics"]["metric_path_format"] = "{risk_metric}"

    with pytest.raises(QlibAshareSemanticContractError, match="portfolio_risk_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_feedback_metric_semantics_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["feedback_metric_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="feedback_metric_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_feedback_primary_metric_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["feedback_metric_semantics"]["feedback_primary_metric"] = "annual_return"

    with pytest.raises(QlibAshareSemanticContractError, match="feedback_metric_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_feedback_derived_utility_alias_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["feedback_metric_semantics"]["derived_bandit_utility_name"] = "sharpe"

    with pytest.raises(QlibAshareSemanticContractError, match="feedback_metric_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_excess_return_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["excess_return_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="excess_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_raw_return_as_excess_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["excess_return_semantics"]["without_cost_formula"] = "return"

    with pytest.raises(QlibAshareSemanticContractError, match="excess_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_missing_excess_cost_column_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["excess_return_semantics"]["required_report_columns"] = ["return", "bench"]

    with pytest.raises(QlibAshareSemanticContractError, match="excess_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_prompt_defined_excess_formula_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["excess_return_semantics"]["forbidden_substitutions"] = [
        "raw_return_as_excess_return",
        "market_universe_as_benchmark_return",
        "with_cost_metric_without_report_cost_column",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="excess_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_benchmark_return_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["benchmark_return_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="benchmark_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_benchmark_default_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["benchmark_return_semantics"]["default_benchmark"] = "SH000905"

    with pytest.raises(QlibAshareSemanticContractError, match="benchmark_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_benchmark_field_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["benchmark_return_semantics"]["benchmark_field_expression"] = "$close"

    with pytest.raises(QlibAshareSemanticContractError, match="benchmark_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_benchmark_sample_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["benchmark_return_semantics"][
        "sample_rule"
    ] = "bar_benchmark_return_equals_simple_sum_of_period_returns"

    with pytest.raises(QlibAshareSemanticContractError, match="benchmark_return_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_universe_benchmark_binding_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="universe_benchmark_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_universe_template_value_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"]["template_market_value"] = "all_a"

    with pytest.raises(QlibAshareSemanticContractError, match="universe_benchmark_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_benchmark_template_value_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"][
        "template_benchmark_value"
    ] = "SH000905"

    with pytest.raises(QlibAshareSemanticContractError, match="universe_benchmark_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_cross_aliased_universe_benchmark_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"][
        "template_benchmark_value"
    ] = QLIB_ASHARE_TEMPLATE_MARKET

    with pytest.raises(QlibAshareSemanticContractError, match="universe_benchmark_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_universe_benchmark_separation_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"][
        "separation_rule"
    ] = "market_universe_membership_and_benchmark_return_series_are_substitutable"

    with pytest.raises(QlibAshareSemanticContractError, match="universe_benchmark_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_runtime_template_binding_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["runtime_handoff_template_binding_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="runtime_handoff_template_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_runtime_template_kwargs_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["runtime_handoff_template_binding_semantics"]["required_backtest_kwargs"] = (
        deepcopy(QLIB_ASHARE_RUNTIME_BACKTEST_KWARGS)
    )

    with pytest.raises(QlibAshareSemanticContractError, match="runtime_handoff_template_binding_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_research_data_source_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["research_data_source_semantics"]

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_intraday_research_data_source_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"]["data_frequency"] = "1min"

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_pit_registration_rule_fails_closed() -> None:
    contract = _valid_contract()
    del contract["prompt_projection_payload"]["research_data_source_semantics"]["point_in_time_registration_rule"]

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_pit_registration_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"][
        "point_in_time_registration_rule"
    ] = "rdagent_may_infer_point_in_time_fields_from_factor_text"

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_derived_feature_source_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"][
        "derived_feature_source_rule"
    ] = "alpha158_alpha360_features_may_introduce_unregistered_source_fields"

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_prompt_obligation_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"][
        "rdagent_prompt_obligation_rule"
    ] = "rdagent_relevance_prompts_may_apply_generic_quant_relevance_without_source_boundaries"

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_missing_research_prompt_consumer_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"]["rdagent_prompt_paths"] = [
        "rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml"
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_turnover_input_boundary_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"][
        "turnover_input_boundary_rule"
    ] = "turnover_is_a_default_factor_input_when_portfolio_reports_turnover"

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_consensus_research_default_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"]["forbidden_default_prompt_sources"] = [
        "turnover",
        "minute_level_high_frequency_data",
        "unregistered_external_vendor_fields",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_without_turnover_research_default_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["research_data_source_semantics"]["forbidden_default_prompt_sources"] = [
        "minute_level_high_frequency_data",
        "analyst_consensus_expectation_factor",
        "unregistered_external_vendor_fields",
    ]

    with pytest.raises(QlibAshareSemanticContractError, match="research_data_source_semantics"):
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


def test_malformed_qlib_prompt_projection_with_mutable_cash_settlement_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["cash_settlement_semantics"][
        "sell_proceeds_rule"
    ] = "sell_proceeds_are_always_reusable_same_step"

    with pytest.raises(QlibAshareSemanticContractError, match="cash_settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_available_cash_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["cash_settlement_semantics"][
        "available_cash_rule"
    ] = "get_cash_includes_unsettled_sell_proceeds"

    with pytest.raises(QlibAshareSemanticContractError, match="cash_settlement_semantics"):
        build_rd_agent_ashare_semantic_context(contract)


def test_malformed_qlib_prompt_projection_with_mutable_cash_commit_rule_fails_closed() -> None:
    contract = _valid_contract()
    contract["prompt_projection_payload"]["cash_settlement_semantics"]["commit_rule"] = "rdagent_can_release_cash_delay"

    with pytest.raises(QlibAshareSemanticContractError, match="cash_settlement_semantics"):
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
    consumed_portfolio_paths = ", ".join(
        [*QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS, *QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS]
    )

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
    assert "universe membership authority: pyqlib (qlib.data.data.InstrumentProvider.list_instruments)" in text
    assert "universe market rule: string_codes_are_resolved_by_qlib_D_instruments" in text
    assert (
        "universe membership window rule: " "instrument_start_end_spans_are_clipped_to_requested_calendar_window"
    ) in text
    assert "universe filter pipe rule: qlib_instrument_filter_pipe_is_applied_after_calendar_window_clipping" in text
    assert (
        "universe survivorship rule: " "membership_must_remain_point_in_time_by_qlib_instrument_spans_and_filters"
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
    assert "market-impact authority: pyqlib (qlib.backtest.exchange.Exchange._calc_trade_info_by_order)" in text
    assert "market-impact parameter: impact_cost" in text
    assert (
        "market-impact ratio rule: impact_cost_times_post_volume_clip_trade_value_over_total_trade_value_squared"
        in text
    )
    assert "market-impact missing-volume rule: missing_zero_or_nan_total_trade_value_uses_raw_impact_cost_ratio" in text
    assert (
        "market-impact final cost rule: trade_cost_is_recomputed_after_final_deal_amount_with_adjusted_cost_ratio"
        in text
    )
    assert "account-update authority: pyqlib (qlib.backtest.account.Account.update_order)" in text
    assert "account-update trigger: only_trade_value_greater_than_one_e_minus_five_mutates_account_or_position" in text
    assert (
        "account-update handoff rule: exchange_passes_final_trade_value_cost_and_price_to_account_or_position_update"
        in text
    )
    assert "account-update buy rule: buy_subtracts_trade_value_plus_cost_from_cash" in text
    assert "account-update sell rule: sell_routes_trade_value_minus_cost_to_cash_or_cash_delay_by_settle_type" in text
    assert (
        "account-update sellable rule: "
        "ashare_sells_reduce_sellable_amount_and_day_bar_count_refresh_releases_total_amount"
    ) in text
    assert "account-valuation authority: pyqlib (qlib.backtest.account.Account.update_bar_end)" in text
    assert (
        "account-valuation bar-end sequence: refresh_current_position_prices_and_holding_counts -> "
        "update_portfolio_metrics_when_enabled -> snapshot_history_positions_when_enabled -> update_trade_indicators"
    ) in text
    assert "account-valuation mark price rule: non_suspended_positions_mark_to_bar_close_at_bar_end" in text
    assert (
        "account-valuation suspension price rule: " "suspended_positions_keep_previous_price_during_bar_end_refresh"
    ) in text
    assert "account-valuation account value rule: account_value_equals_stock_value_plus_cash_plus_cash_delay" in text
    assert (
        "account-valuation daily sellable release rule: "
        "ashare_day_bar_count_refresh_releases_total_amount_to_sellable_amount"
    ) in text
    assert "trade-indicator authority: pyqlib (qlib.backtest.report.Indicator)" in text
    assert "trade-indicator account hook: pyqlib (qlib.backtest.account.Account.update_indicator)" in text
    assert "trade-indicator metrics: ffr, pa, pos, deal_amount, value, count" in text
    assert (
        "trade-indicator fulfill rate rule: " "ffr_equals_deal_amount_reindexed_zero_for_missing_over_order_amount"
    ) in text
    assert (
        "trade-indicator price advantage rule: " "pa_equals_directional_trade_price_over_base_price_minus_one"
    ) in text
    assert (
        "trade-indicator portfolio boundary: "
        "trade_indicators_are_execution_quality_metrics_not_portfolio_return_metrics"
    ) in text
    assert "executor-decision authority: pyqlib (qlib.backtest.executor.BaseExecutor.collect_data)" in text
    assert (
        "executor-decision simulator authority: pyqlib " "(qlib.backtest.executor.SimulatorExecutor._collect_data)"
    ) in text
    assert "executor-decision nested authority: pyqlib (qlib.backtest.executor.NestedExecutor._collect_data)" in text
    assert "executor-decision atomicity rule: atomic_executor_rejects_trade_decision_range_limit" in text
    assert "executor-decision bar-end rule: executor_updates_account_bar_end_before_trade_calendar_step" in text
    assert (
        "executor-decision nested range rule: "
        "nested_executor_skips_inner_steps_outside_range_limit_when_align_range_limit_is_enabled"
    ) in text
    assert (
        "executor-decision inner decision rule: "
        "outer_trade_decision_may_propagate_trade_range_into_inner_trade_decision_only_when_inner_range_absent"
    ) in text
    assert (
        "strategy-order authority: pyqlib "
        "(qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision)"
    ) in text
    assert "strategy-order template binding: qlib.contrib.strategy.TopkDropoutStrategy" in text
    assert "strategy-order prediction window: strategy_reads_signal_from_previous_calendar_step_shift_one" in text
    assert (
        "strategy-order dropout rule: "
        "combined_last_and_today_scores_prevent_dropping_higher_score_stock_for_lower_score_buy"
    ) in text
    assert "strategy-order order return rule: exchange_returns_sell_orders_before_buy_orders" in text
    assert "supervised-label authority: pyqlib (qlib.contrib.data.handler.Alpha158)" in text
    assert f"supervised-label column: {QLIB_ASHARE_LABEL_COLUMN}" in text
    assert f"supervised-label expression: {QLIB_ASHARE_LABEL_EXPRESSION}" in text
    assert "supervised-label horizon: label_at_datetime_t_is_close_t_plus_2_over_close_t_plus_1_minus_one" in text
    assert (
        "supervised-label prompt wording: "
        "describe_as_qlib_contract_defined_forward_return_label_not_undefined_next_several_days_return"
    ) in text
    assert "prediction-signal authority: pyqlib (qlib.backtest.signal.ModelSignal)" in text
    assert "prediction-signal artifact: pred.pkl" in text
    assert "prediction-signal column: score" in text
    assert (
        "prediction-signal model rule: " "model_predict_output_is_prediction_score_not_realized_or_executable_return"
    ) in text
    assert (
        "prediction-signal ranking rule: "
        "TopkDropoutStrategy_sorts_prediction_scores_descending_for_candidate_selection"
    ) in text
    assert (
        "prediction-signal prompt wording: "
        "describe_as_prediction_signal_score_for_LABEL0_not_realized_future_return_or_guaranteed_portfolio_return"
    ) in text
    assert f"prediction-signal model output format: {QLIB_ASHARE_MODEL_OUTPUT_FORMAT_RULE}" in text
    assert f"prediction-signal model task boundary: {QLIB_ASHARE_MODEL_TASK_BOUNDARY_RULE}" in text
    assert (
        "prediction-signal implementation prompts: "
        + ", ".join(str(path) for path in QLIB_ASHARE_MODEL_IMPLEMENTATION_PROMPT_PATHS)
    ) in text
    assert "signal-ic authority: pyqlib (qlib.workflow.record_temp.SigAnaRecord)" in text
    assert "signal-ic calculation: pyqlib (qlib.contrib.eva.alpha.calc_ic)" in text
    assert "signal-ic metrics: IC, ICIR, Rank IC, Rank ICIR" in text
    assert "signal-ic groupby: datetime" in text
    assert "signal-ic IC rule: IC_is_per_datetime_pearson_correlation_between_pred_and_label" in text
    assert "signal-ic Rank IC rule: Rank_IC_is_per_datetime_spearman_correlation_between_pred_and_label" in text
    assert (
        "signal-ic portfolio boundary: "
        "signal_ic_metrics_are_prediction_label_quality_metrics_not_portfolio_return_metrics"
    ) in text
    assert "portfolio-risk authority: pyqlib (qlib.contrib.evaluate.risk_analysis)" in text
    assert "portfolio-risk metrics: mean, std, annualized_return, information_ratio, max_drawdown" in text
    assert f"portfolio-risk consumed paths: {consumed_portfolio_paths}" in text
    assert "portfolio-risk metric path format: {freq}.{report_type}.{risk_metric}" in text
    assert (
        "portfolio-risk metric path whitespace rule: " "metric_paths_are_exact_without_leading_or_trailing_whitespace"
    ) in text
    assert f"portfolio-risk prompt paths: {', '.join(QLIB_ASHARE_PORTFOLIO_PROMPT_METRIC_PATHS)}" in text
    assert f"portfolio-risk feedback paths: {', '.join(QLIB_ASHARE_PORTFOLIO_FEEDBACK_METRIC_PATHS)}" in text
    assert f"portfolio-risk bandit paths: {', '.join(QLIB_ASHARE_PORTFOLIO_BANDIT_METRIC_PATHS)}" in text
    assert f"portfolio-risk UI paths: {', '.join(QLIB_ASHARE_PORTFOLIO_UI_METRIC_PATHS)}" in text
    assert "portfolio-risk annualization scaler: 238" in text
    assert (
        "portfolio-risk max drawdown rule: "
        "sum_mode_max_drawdown_equals_min_of_cumulative_return_minus_running_cumulative_max"
    ) in text
    assert "excess-return authority: pyqlib (qlib.backtest.report.PortfolioMetrics)" in text
    assert "excess-return without-cost formula: return - bench" in text
    assert "excess-return with-cost formula: return - bench - cost" in text
    assert (
        "excess-return metric paths: "
        f"{QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITHOUT_COST}, {QLIB_ASHARE_EXCESS_RETURN_METRIC_PATH_WITH_COST}"
    ) in text
    assert (
        "excess-return forbidden substitutions: " f"{', '.join(QLIB_ASHARE_EXCESS_RETURN_FORBIDDEN_SUBSTITUTIONS)}"
    ) in text
    assert (
        "excess-return prompt rule: " "generated_research_must_report_benchmark_relative_excess_return_not_raw_return"
    ) in text
    assert "feedback-metric authority: pyqlib (qlib.workflow.record_temp.PortAnaRecord)" in text
    assert f"feedback-metric primary: {QLIB_ASHARE_FEEDBACK_PRIMARY_METRIC}" in text
    assert f"feedback-metric paths: {', '.join(QLIB_ASHARE_FEEDBACK_METRIC_PATHS)}" in text
    assert f"feedback-metric bandit utility: {QLIB_ASHARE_BANDIT_DERIVED_UTILITY_NAME}" in text
    assert (
        "feedback-metric utility rule: "
        "rdagent_may_compute_arr_over_abs_max_drawdown_as_derived_utility_not_qlib_metric"
    ) in text
    assert "feedback-metric forbidden aliases: sharpe, Sharpe" in text
    assert "benchmark-return authority: pyqlib (qlib.backtest.report.PortfolioMetrics._cal_benchmark)" in text
    assert "benchmark-return default: SH000300" in text
    assert "benchmark-return field: $close/Ref($close,1)-1" in text
    assert "benchmark-return sample rule: bar_benchmark_return_equals_product_of_one_plus_period_returns_minus_one" in (
        text
    )
    assert "benchmark-return report column: bench" in text
    assert f"universe-benchmark market value: {QLIB_ASHARE_TEMPLATE_MARKET}" in text
    assert f"universe-benchmark benchmark value: {QLIB_ASHARE_TEMPLATE_BENCHMARK}" in text
    assert "universe-benchmark market rule: csi300_template_market_selects_instruments_only" in text
    assert (
        "universe-benchmark benchmark rule: " "SH000300_template_benchmark_is_portfolio_excess_return_baseline_only"
    ) in text
    assert (
        "universe-benchmark separation rule: "
        "market_universe_membership_and_benchmark_return_series_are_not_substitutable"
    ) in text
    assert (
        "universe-benchmark template rule: "
        "bind_market_to_instruments_and_benchmark_to_backtest_without_cross_aliasing"
    ) in text
    assert "runtime-handoff template binding: rdagent_qlib_template_backtest_runtime_kwargs" in text
    assert (
        "runtime-handoff template rule: "
        "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values"
    ) in text
    assert ("runtime-handoff prompt boundary: " "execution_kwargs_remain_runtime_handoff_not_prompt_authority") in text
    assert "research data-source frequency: day" in text
    assert f"research data-source fields: {', '.join(QLIB_ASHARE_RESEARCH_DATA_SOURCE_FIELDS)}" in text
    assert f"research data-source derived feature rule: {QLIB_ASHARE_DERIVED_FEATURE_SOURCE_RULE}" in text
    assert f"research data-source prompt paths: {', '.join(QLIB_ASHARE_RESEARCH_DATA_SOURCE_PROMPT_PATHS)}" in text
    assert (
        "research data-source forbidden defaults: " f"{', '.join(QLIB_ASHARE_FORBIDDEN_DEFAULT_RESEARCH_SOURCES)}"
    ) in text
    assert f"research data-source PIT registration: {QLIB_ASHARE_POINT_IN_TIME_REGISTRATION_RULE}" in text
    assert f"research data-source prompt obligation: {QLIB_ASHARE_PROMPT_OBLIGATION_RULE}" in text
    assert f"research data-source turnover input boundary: {QLIB_ASHARE_TURNOVER_INPUT_BOUNDARY_RULE}" in text
    assert f"portfolio-risk turnover metric rule: {QLIB_ASHARE_TURNOVER_REPORT_METRIC_RULE}" in text
    assert (
        "research data-source rule: " "describe_only_use_qlib_registered_daily_or_user_supplied_point_in_time_sources"
    ) in text
    assert "required_backtest_kwargs" not in text
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
    assert "order-tradability authority: pyqlib (qlib.backtest.exchange.Exchange.check_order)" in text
    assert (
        "order-tradability decision rule: " "check_order_delegates_to_is_stock_tradable_before_deal_execution"
    ) in text
    assert "order-tradability suspension rule: missing_close_or_unknown_stock_is_not_tradable" in text
    assert (
        "order-tradability directional limit rule: " "buy_orders_check_limit_buy_and_sell_orders_check_limit_sell"
    ) in text
    assert "order-tradability failure result: deal_amount_zero_trade_value_zero_cost_nan_price" in text
    assert "order-fill authority: pyqlib (qlib.backtest.exchange.Exchange._calc_trade_info_by_order)" in text
    assert "order-fill state field: Order.deal_amount" in text
    assert (
        "order-fill clip sequence: volume_capacity_clip -> sellable_position_clip -> sell_cash_cost_guard -> "
        "buy_cash_cost_guard -> round_lot_or_full_liquidation_clip"
    ) in text
    assert "order-fill trade value rule: trade_value_is_final_deal_amount_times_trade_price" in text
    assert "order-fill cost rule: trade_cost_recomputed_after_final_deal_amount" in text
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
    assert "cash-settlement authority: pyqlib (qlib.backtest.position.Position)" in text
    assert "cash-settlement delayed mode: Position.ST_CASH" in text
    assert "cash-settlement sell proceeds rule: sell_proceeds_enter_cash_delay_when_settle_type_is_cash" in text
    assert (
        "cash-settlement available cash rule: " "get_cash_excludes_cash_delay_unless_include_settle_is_true"
    ) in text
    assert ("cash-settlement commit rule: " "settle_commit_moves_cash_delay_into_cash_and_clears_delay_state") in text
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
        "universe_membership_semantics, trading_calendar_semantics, transaction_cost_semantics, "
        "market_impact_semantics, account_update_semantics, account_valuation_semantics, trade_indicator_semantics, executor_decision_semantics, strategy_order_semantics, supervised_label_semantics, prediction_signal_semantics, signal_ic_semantics, portfolio_risk_semantics, excess_return_semantics, feedback_metric_semantics, benchmark_return_semantics, universe_benchmark_binding_semantics, runtime_handoff_template_binding_semantics, research_data_source_semantics, suspension_tradability_semantics, execution_price_semantics, price_adjustment_semantics, "
        "price_limit_semantics, order_tradability_semantics, order_fill_amount_semantics, settlement_semantics, "
        "cash_settlement_semantics, cash_constraint_semantics, liquidity_capacity_semantics, trade_unit, position_type, "
        "settlement_rule, same_day_sell_policy, "
        "data_frequency, price_limit_modes, authoritative_limit_fields, board_threshold_fields, cost_model"
    ) in text
    assert "prompt projection forbids: runtime_surfaces.policy_defaults" in text
    assert "runtime_surfaces.backtest_kwargs" in text
    assert "AsharePosition" in text
    assert "open_cost" not in text
    assert "close_tax" not in text
