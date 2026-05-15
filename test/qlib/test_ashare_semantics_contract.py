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
                "but it must not redefine trade unit, position, price-limit, or cost semantics."
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
                "redefine_trade_unit_or_position_type",
                "redefine_price_limit_thresholds_or_authoritative_fields",
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
                "market_semantics.trade_unit",
                "market_semantics.position_type",
                "market_semantics.limit_threshold",
                "market_semantics.authoritative_limit_fields",
            ],
            "rdagent_prompt_forbidden_fields": [
                "runtime_surfaces.policy_defaults",
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
                "market_semantics.cost_model",
            ],
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
    assert context["qlib_contract_schema_version"] == "qlib_ashare_semantic_contract.v1"
    assert context["qlib_contract_fingerprint"] == "a" * 64
    assert boundary["semantic_authority"] == "pyqlib_contract"
    assert boundary["failure_semantics"] == "fail_closed_on_missing_or_malformed_pyqlib_ashare_contract"
    assert boundary["authority_rule"] == "Qlib owns executable JoinQuant-compatible A-share backtest semantics."
    assert "render_contract_projection_in_research_context" in boundary["rdagent_may"]
    assert "treat_research_prompt_projection_as_backtest_authority" in boundary["rdagent_forbidden_actions"]
    assert boundary["rdagent_must_not_redefine"] == [
        "trade_unit",
        "position_type",
        "price_limit_modes",
        "cost_model",
    ]
    assert context["failure_contract"]["runtime_projection_drift"] == "fail_closed"
    assert "runtime_surfaces.backtest_kwargs" in context["prompt_projection"]["rdagent_prompt_forbidden_fields"]
    assert context["qlib_market_semantics"]["trade_unit"] == 100
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
    assert "RD-Agent must not redefine: trade_unit, position_type, price_limit_modes, cost_model" in text
    assert "prompt projection forbids: runtime_surfaces.policy_defaults" in text
    assert "runtime_surfaces.backtest_kwargs" in text
    assert "open_cost" not in text
    assert "close_tax" not in text
