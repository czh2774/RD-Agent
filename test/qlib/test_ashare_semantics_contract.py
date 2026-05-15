from __future__ import annotations

from copy import deepcopy
import sys
import types
from typing import Any

import pytest

from rdagent.scenarios.qlib.ashare_semantics import (
    QlibAshareSemanticContractError,
    append_ashare_semantic_context,
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
                "but it must not redefine trade unit, position, price-limit, or cost semantics."
            ),
            "fail_closed_on_missing_contract": True,
        },
        "market_semantics": {
            "market": "china_a_share",
            "region": "cn",
            "trade_unit": 100,
            "position_type": "AsharePosition",
            "deal_price": "close",
            "limit_threshold": "joinquant_ashare",
            "limit_threshold_aliases": [
                "ashare_joinquant",
                "cn_ashare_joinquant",
                "joinquant_ashare",
            ],
            "price_limit_modes": ["auto", "strict", "board_fallback"],
            "authoritative_limit_fields": ["$up_limit", "$down_limit"],
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
            "trade_unit",
            "position_type",
            "price_limit_modes",
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
    assert boundary["semantic_authority"] == "pyqlib_contract"
    assert boundary["failure_semantics"] == "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract"
    assert boundary["rdagent_must_not_redefine"] == [
        "trade_unit",
        "position_type",
        "price_limit_modes",
        "cost_model",
    ]
    assert context["qlib_market_semantics"]["trade_unit"] == 100
    assert context["runtime_surfaces"]["backtest_kwargs"]["pos_type"] == "AsharePosition"


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


def test_optional_prompt_context_reports_unavailable_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "qlib.backtest.ashare_semantics", raising=False)

    text = append_ashare_semantic_context("runtime-ok")

    assert "runtime-ok" in text
    assert "- status: unavailable" in text
    assert "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract" in text


def test_formatted_context_is_operator_readable_without_raw_cost_redefinition() -> None:
    text = format_rd_agent_ashare_semantic_context(build_rd_agent_ashare_semantic_context(_valid_contract()))

    assert "qlib_contract_id: rdagent_qlib_joinquant_ashare_semantic_contract_v1" in text
    assert "qlib_source_component: qlib.backtest.ashare_semantics" in text
    assert "RD-Agent must not redefine: trade_unit, position_type, price_limit_modes, cost_model" in text
