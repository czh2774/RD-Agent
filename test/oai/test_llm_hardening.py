from __future__ import annotations

import os
from unittest import TestCase, main
from unittest.mock import patch

from rdagent.app.utils import health_check as health_check_module
from rdagent.oai.llm_conf import LLMSettings, rewrite_loopback_url


class LLMHardeningTest(TestCase):
    def test_rewrite_loopback_url_rewrites_localhost_alias(self) -> None:
        self.assertEqual(
            rewrite_loopback_url("http://127.0.0.1:58008/v1", "host.docker.internal"),
            "http://host.docker.internal:58008/v1",
        )

    def test_llm_settings_rewrite_loopback_alias(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_BASE": "http://localhost:58008/v1",
                "HOST_LOOPBACK_ALIAS": "host.docker.internal",
            },
            clear=False,
        ):
            settings = LLMSettings()

        self.assertEqual(settings.openai_api_base, "http://host.docker.internal:58008/v1")

    def test_env_check_prefers_proxy_embedding_endpoint_and_rewrites_loopback(self) -> None:
        captured: dict[str, dict[str, str | None]] = {}

        def fake_test_chat(
            chat_model: str,
            chat_api_key: str,
            chat_api_base: str | None,
            emit_logs: bool = True,
        ) -> bool:
            captured["chat"] = {
                "chat_model": chat_model,
                "chat_api_key": chat_api_key,
                "chat_api_base": chat_api_base,
                "emit_logs": emit_logs,
            }
            return True

        def fake_test_embedding(
            embedding_model: str,
            embedding_api_key: str,
            embedding_api_base: str | None,
            emit_logs: bool = True,
        ) -> bool:
            captured["embedding"] = {
                "embedding_model": embedding_model,
                "embedding_api_key": embedding_api_key,
                "embedding_api_base": embedding_api_base,
                "emit_logs": emit_logs,
            }
            return True

        with (
            patch.dict(
                os.environ,
                {
                    "BACKEND": "rdagent.oai.backend.LiteLLMAPIBackend",
                    "OPENAI_API_KEY": "chat-key",
                    "OPENAI_API_BASE": "http://127.0.0.1:58008/v1",
                    "CHAT_MODEL": "gpt-5.4",
                    "EMBEDDING_MODEL": "text-embedding-3-large",
                    "LITELLM_PROXY_API_KEY": "embedding-key",
                    "LITELLM_PROXY_API_BASE": "http://localhost:58009/v1",
                    "LITELLM_HOST_LOOPBACK_ALIAS": "host.docker.internal",
                },
                clear=False,
            ),
            patch.object(health_check_module, "test_chat", side_effect=fake_test_chat),
            patch.object(health_check_module, "test_embedding", side_effect=fake_test_embedding),
        ):
            payload = health_check_module.env_check(emit_logs=False)

        self.assertEqual(captured["chat"]["chat_api_base"], "http://host.docker.internal:58008/v1")
        self.assertEqual(captured["embedding"]["embedding_api_key"], "embedding-key")
        self.assertEqual(captured["embedding"]["embedding_api_base"], "http://host.docker.internal:58009/v1")
        self.assertFalse(captured["chat"]["emit_logs"])
        self.assertFalse(captured["embedding"]["emit_logs"])
        self.assertEqual(payload["status"], "ok")

    def test_build_runtime_info_reports_rewritten_endpoints(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BACKEND": "rdagent.oai.backend.LiteLLMAPIBackend",
                "CONDA_DEFAULT_ENV": "rdagent",
                "MODEL_CoSTEER_ENV_TYPE": "conda",
                "CHAT_MODEL": "gpt-5.4",
                "EMBEDDING_MODEL": "text-embedding-3-large",
                "OPENAI_API_BASE": "http://localhost:58008/v1",
                "OPENAI_API_KEY": "chat-key",
                "LITELLM_PROXY_API_BASE": "http://127.0.0.1:58009/v1",
                "LITELLM_HOST_LOOPBACK_ALIAS": "host.docker.internal",
                "QLIB_PROVIDER_URI": "/workspace/app020/var/qlib_data/cn_data",
            },
            clear=False,
        ):
            payload = health_check_module.build_runtime_info()

        self.assertEqual(payload["chat_api_base"], "http://host.docker.internal:58008/v1")
        self.assertEqual(payload["embedding_api_base"], "http://host.docker.internal:58009/v1")
        self.assertEqual(payload["qlib_provider_uri"], "/workspace/app020/var/qlib_data/cn_data")


if __name__ == "__main__":
    main()
