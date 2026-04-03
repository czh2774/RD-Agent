from __future__ import annotations

import json
from typing import Any, Optional, Type, Union, cast

from pydantic import BaseModel

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.deprec import DeprecBackend
from rdagent.oai.llm_conf import (
    LLM_SETTINGS,
    ReasoningEffort,
    SUPPORTED_REASONING_EFFORTS,
)


class OpenAIResponsesAPIBackend(DeprecBackend):
    """
    Primary OpenAI SDK backend for Codex-compatible /responses providers.
    """

    @staticmethod
    def _messages_to_responses_payload(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions_parts: list[str] = []
        input_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role in {"system", "developer"}:
                instructions_parts.append(content)
                continue
            input_messages.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        instructions = "\n\n".join(part for part in instructions_parts if part) or None
        return instructions, input_messages

    @staticmethod
    def _normalize_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(schema))

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                    properties = cast(dict[str, Any], node["properties"])
                    node["required"] = list(properties.keys())
                    node["additionalProperties"] = False
                    for child in properties.values():
                        visit(child)
                if isinstance(node.get("items"), dict):
                    visit(node["items"])
                for key in ("anyOf", "oneOf", "allOf"):
                    if isinstance(node.get(key), list):
                        for child in node[key]:
                            visit(child)
                if isinstance(node.get("$defs"), dict):
                    for child in node["$defs"].values():
                        visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(normalized)
        return normalized

    @classmethod
    def _response_format_to_responses_text_config(
        cls,
        response_format: Optional[Union[dict, Type[BaseModel]]],
    ) -> Optional[dict[str, Any]]:
        if response_format is None or response_format == {"type": "json_object"}:
            return None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return {
                "format": {
                    "type": "json_schema",
                    "name": response_format.__name__.lower(),
                    "schema": cls._normalize_strict_json_schema(response_format.model_json_schema()),
                    "strict": True,
                }
            }
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema = dict(response_format.get("json_schema") or {})
            strict = bool(response_format.get("strict", json_schema.pop("strict", False)))
            schema = cls._normalize_strict_json_schema(cast(dict[str, Any], json_schema.get("schema", {})))
            return {
                "format": {
                    "type": "json_schema",
                    "name": str(json_schema.get("name") or "rdagent_response"),
                    "schema": schema,
                    "strict": strict,
                }
            }
        return None

    @staticmethod
    def _response_to_dict(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return dumped
        return {}

    @classmethod
    def _extract_responses_text(cls, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        response_json = cls._response_to_dict(response)
        output_text = response_json.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        for item in response_json.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue
            chunks: list[str] = []
            for content in content_items:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "".join(chunks)
        raise RuntimeError("Responses API did not include assistant output text.")

    def _get_complete_kwargs(
        self,
    ) -> tuple[str, float | None, int | None, ReasoningEffort | None]:
        model = LLM_SETTINGS.chat_model
        temperature = LLM_SETTINGS.chat_temperature
        max_tokens = LLM_SETTINGS.chat_max_tokens
        reasoning_effort = LLM_SETTINGS.reasoning_effort

        if LLM_SETTINGS.chat_model_map:
            for tag, mapping in LLM_SETTINGS.chat_model_map.items():
                if tag not in logger._tag:
                    continue
                model = mapping.get("model", model)
                if "temperature" in mapping:
                    temperature = float(mapping["temperature"])
                if "max_tokens" in mapping:
                    max_tokens = int(mapping["max_tokens"])
                mapped_effort = mapping.get("reasoning_effort")
                if isinstance(mapped_effort, str) and mapped_effort in SUPPORTED_REASONING_EFFORTS:
                    reasoning_effort = cast(ReasoningEffort, mapped_effort)
                elif "reasoning_effort" in mapping:
                    reasoning_effort = None
                break

        if model.startswith("gpt-5") and reasoning_effort is not None and temperature != 1:
            temperature = None
        return model, temperature, max_tokens, reasoning_effort

    def supports_response_schema(self) -> bool:
        return True

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def]
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        if self.chat_stream:
            logger.warning(
                "OpenAIResponsesAPIBackend stream fallback is using chat.completions compatibility mode."
            )
            return super()._create_chat_completion_inner_function(
                messages=messages,
                response_format=response_format,
                *args,
                **kwargs,
            )

        if LLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        model, temperature, max_tokens, reasoning_effort = self._get_complete_kwargs()
        instructions, input_messages = self._messages_to_responses_payload(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "stream": False,
        }
        if instructions:
            call_kwargs["instructions"] = instructions
        if reasoning_effort is not None:
            call_kwargs["reasoning"] = {"effort": reasoning_effort}
        if max_tokens is not None:
            call_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        text_config = self._response_format_to_responses_text_config(response_format)
        if text_config is not None:
            call_kwargs["text"] = text_config

        response = self.chat_client.responses.create(**call_kwargs)
        content = self._extract_responses_text(response)
        response_json = self._response_to_dict(response)
        status = getattr(response, "status", None) or response_json.get("status")
        if status not in {None, "completed", "in_progress"}:
            raise RuntimeError(f"Responses API did not complete successfully: status={status!r}")

        if LLM_SETTINGS.log_llm_chat_content:
            logger.info(
                f"{LogColors.GREEN}Using chat model{LogColors.END} {model} via OpenAI SDK /responses",
                tag="llm_messages",
            )
            logger.info(f"{LogColors.CYAN}Response:{content}{LogColors.END}", tag="llm_messages")
            usage = getattr(response, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            logger.info(
                json.dumps(
                    {
                        "model": model,
                        "status": status,
                        "usage": usage,
                    }
                ),
                tag="llm_messages",
            )
        return content, "stop"
