import json
import os
import socket
from typing import Any

import docker
import fire
import litellm
from litellm import completion, embedding
from litellm.utils import ModelResponse

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import rewrite_loopback_url
from rdagent.utils.env import cleanup_container


def build_runtime_info() -> dict[str, Any]:
    host_loopback_alias = os.getenv("LITELLM_HOST_LOOPBACK_ALIAS") or os.getenv("HOST_LOOPBACK_ALIAS") or ""
    chat_api_base = rewrite_loopback_url(os.getenv("OPENAI_API_BASE", ""), host_loopback_alias)
    embedding_api_base = rewrite_loopback_url(
        os.getenv("LITELLM_PROXY_API_BASE") or os.getenv("OPENAI_API_BASE", ""),
        host_loopback_alias,
    )
    return {
        "status": "ok",
        "cwd": os.getcwd(),
        "backend": os.getenv("BACKEND", ""),
        "conda_default_env": os.getenv("CONDA_DEFAULT_ENV", ""),
        "model_costeer_env_type": os.getenv("MODEL_CoSTEER_ENV_TYPE", ""),
        "chat_model": os.getenv("CHAT_MODEL", ""),
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "chat_api_base": chat_api_base,
        "embedding_api_base": embedding_api_base,
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "embedding_api_key_present": bool(os.getenv("LITELLM_PROXY_API_KEY") or os.getenv("OPENAI_API_KEY")),
        "host_loopback_alias": host_loopback_alias,
        "qlib_provider_uri": os.getenv("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"),
    }


def check_docker_status(emit_logs: bool = True) -> dict[str, Any]:
    container = None
    payload: dict[str, Any] = {
        "status": "fail",
        "docker_accessible": False,
        "hello_world_ran": False,
    }
    try:
        client = docker.from_env()
        client.images.pull("hello-world")
        container = client.containers.run("hello-world", detach=True)
        logs = container.logs().decode("utf-8")
        payload.update(
            {
                "status": "ok",
                "docker_accessible": True,
                "hello_world_ran": True,
                "hello_world_logs": logs,
            }
        )
        if emit_logs:
            print(logs)
            logger.info(f"The docker status is normal")
    except docker.errors.DockerException as e:
        payload.update(
            {
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        if emit_logs:
            logger.error(f"An error occurred: {e}")
            logger.warning(
                f"Docker status is exception, please check the docker configuration or reinstall it. Refs: https://docs.docker.com/engine/install/ubuntu/."
            )
    finally:
        cleanup_container(container, "health check")
    return payload


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_and_list_free_ports(start_port=19899, max_ports=10, emit_logs: bool = True) -> dict[str, Any]:
    is_occupied = is_port_in_use(port=start_port)
    if is_occupied:
        free_ports = []
        for port in range(start_port, start_port + max_ports):
            if not is_port_in_use(port):
                free_ports.append(port)
        payload = {
            "status": "fail",
            "start_port": start_port,
            "port_occupied": True,
            "available_ports": free_ports,
        }
        if emit_logs:
            logger.warning(
                f"Port 19899 is occupied, please replace it with an available port when running the `rdagent ui/server_ui` command. Available ports: {free_ports}"
            )
    else:
        payload = {
            "status": "ok",
            "start_port": start_port,
            "port_occupied": False,
            "available_ports": [start_port],
        }
        if emit_logs:
            logger.info(f"Port 19899 is not occupied, you can run the `rdagent ui/server_ui` command")
    return payload


def test_chat(chat_model, chat_api_key, chat_api_base, emit_logs: bool = True):
    if emit_logs:
        logger.info(f"🧪 Testing chat model: {chat_model}")
    try:
        if chat_api_base is None:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                messages=[
                    {"role": "user", "content": "Hello!"},
                ],
            )
        else:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                api_base=chat_api_base,
                messages=[
                    {"role": "user", "content": "Hello!"},
                ],
            )
        if emit_logs:
            logger.info(f"✅ Chat test passed.")
        return True
    except Exception as e:
        if emit_logs:
            logger.error(f"❌ Chat test failed: {e}")
        return False


def test_embedding(embedding_model, embedding_api_key, embedding_api_base, emit_logs: bool = True):
    if emit_logs:
        logger.info(f"🧪 Testing embedding model: {embedding_model}")
    try:
        response = embedding(
            model=embedding_model,
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            input="Hello world!",
        )
        if emit_logs:
            logger.info("✅ Embedding test passed.")
        return True
    except Exception as e:
        if emit_logs:
            logger.error(f"❌ Embedding test failed: {e}")
        return False


def env_check(emit_logs: bool = True) -> dict[str, Any]:
    host_loopback_alias = os.getenv("LITELLM_HOST_LOOPBACK_ALIAS") or os.getenv("HOST_LOOPBACK_ALIAS") or ""
    if "BACKEND" not in os.environ:
        if emit_logs:
            logger.warning(
                f"We did not find BACKEND in your configuration, please add it to your .env file. "
                f"You can run a command like this: `dotenv set BACKEND rdagent.oai.backend.OpenAIResponsesAPIBackend`"
            )

    if "DEEPSEEK_API_KEY" in os.environ:
        chat_api_key = os.getenv("DEEPSEEK_API_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("LITELLM_PROXY_API_KEY")
        embedding_api_base = os.getenv("LITELLM_PROXY_API_BASE")
        if "DEEPSEEK_API_BASE" in os.environ:
            chat_api_base = os.getenv("DEEPSEEK_API_BASE")
        elif "OPENAI_API_BASE" in os.environ:
            chat_api_base = os.getenv("OPENAI_API_BASE")
        else:
            chat_api_base = None
    elif "OPENAI_API_KEY" in os.environ:
        chat_api_key = os.getenv("OPENAI_API_KEY")
        chat_api_base = os.getenv("OPENAI_API_BASE")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("LITELLM_PROXY_API_KEY") or chat_api_key
        embedding_api_base = os.getenv("LITELLM_PROXY_API_BASE") or chat_api_base
    else:
        if emit_logs:
            logger.error("No valid configuration was found, please check your .env file.")
        return {
            "status": "fail",
            "backend_present": "BACKEND" in os.environ,
            "error_type": "MissingCredentials",
            "runtime_info": build_runtime_info(),
        }

    chat_api_base = rewrite_loopback_url(chat_api_base or "", host_loopback_alias) or None
    embedding_api_base = rewrite_loopback_url(embedding_api_base or "", host_loopback_alias) or None

    if emit_logs:
        logger.info("🚀 Starting test...\n")
    result_embedding = test_embedding(
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_api_base=embedding_api_base,
        emit_logs=emit_logs,
    )
    result_chat = test_chat(
        chat_model=chat_model,
        chat_api_key=chat_api_key,
        chat_api_base=chat_api_base,
        emit_logs=emit_logs,
    )

    payload = {
        "status": "ok" if result_chat and result_embedding else "fail",
        "backend_present": "BACKEND" in os.environ,
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "chat_api_base": chat_api_base,
        "embedding_api_base": embedding_api_base,
        "chat_ok": result_chat,
        "embedding_ok": result_embedding,
        "runtime_info": build_runtime_info(),
    }

    if emit_logs:
        if result_chat and result_embedding:
            logger.info("✅ All tests completed.")
        else:
            logger.error(" One or more tests failed. Please check credentials or model support.")
    return payload


def health_report(
    check_env: bool = True,
    check_docker: bool = True,
    check_ports: bool = True,
    emit_logs: bool = True,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    if check_env:
        checks["env"] = env_check(emit_logs=emit_logs)
    if check_docker:
        checks["docker"] = check_docker_status(emit_logs=emit_logs)
    if check_ports:
        checks["ports"] = check_and_list_free_ports(emit_logs=emit_logs)

    if not checks:
        if emit_logs:
            logger.warning("⚠️ All health check items are disabled. Please enable at least one check.")
        return {
            "status": "fail",
            "error_type": "NoChecksEnabled",
            "checks": {},
            "runtime_info": build_runtime_info(),
        }

    overall_ok = all(check.get("status") == "ok" for check in checks.values())
    return {
        "status": "ok" if overall_ok else "fail",
        "checks": checks,
        "runtime_info": build_runtime_info(),
    }


def health_check(
    check_env: bool = True,
    check_docker: bool = True,
    check_ports: bool = True,
    output_json: bool = False,
):
    """
    Run the RD-Agent health check:
    - Check if Docker is available
    - Check that the default ports are not occupied
    - (Optional) Check that the API Key and model are configured correctly.

    Args:
        check_env (bool): Whether to check API Key and model configuration.
        check_docker (bool): Checks if Docker is installed and running.
        check_ports (bool): Whether to check if the default port (19899) is occupied.
    """
    payload = health_report(
        check_env=check_env,
        check_docker=check_docker,
        check_ports=check_ports,
        emit_logs=not output_json,
    )
    if output_json:
        print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    fire.Fire(health_check)
