from __future__ import annotations

from pywork.tui.app import PyWorkApp, get_default_tui_llm_config


def test_default_tui_llm_config_is_qwen() -> None:
    config = get_default_tui_llm_config()

    assert config["default_provider"] == "qwen"
    assert config["fallback_to_mock"] is False
    assert config["providers"]["qwen"]["model"] == "qwen3.6-flash"
    assert config["providers"]["qwen"]["api_key_env"] == "DASHSCOPE_API_KEY"


def test_runtime_config_uses_default_qwen_when_llm_missing() -> None:
    app = PyWorkApp(
        config={
            "permissions": {
                "mode": "default",
            }
        }
    )

    runtime_config = app.get_runtime_config()

    assert runtime_config["llm"]["default_provider"] == "qwen"
    assert runtime_config["llm"]["providers"]["qwen"]["model"] == "qwen3.6-flash"


def test_runtime_config_preserves_user_llm_config() -> None:
    user_llm_config = {
        "default_provider": "openai",
        "fallback_to_mock": False,
        "providers": {
            "openai": {
                "provider": "openai",
                "model": "gpt-5.5",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.1,
            }
        },
    }

    app = PyWorkApp(
        config={
            "llm": user_llm_config,
            "permissions": {
                "mode": "accept_edits",
            },
        }
    )

    runtime_config = app.get_runtime_config()

    assert runtime_config["llm"] == user_llm_config
    assert runtime_config["llm"]["default_provider"] == "openai"
    assert runtime_config["llm"]["providers"]["openai"]["model"] == "gpt-5.5"
    assert runtime_config["permissions"]["mode"] == "accept_edits"


def test_runtime_config_uses_default_qwen_when_llm_is_empty() -> None:
    app = PyWorkApp(
        config={
            "llm": {},
        }
    )

    runtime_config = app.get_runtime_config()

    assert runtime_config["llm"]["default_provider"] == "qwen"


def test_configured_provider_and_model_label_use_user_llm_config() -> None:
    app = PyWorkApp(
        config={
            "llm": {
                "default_provider": "deepseek",
                "fallback_to_mock": False,
                "providers": {
                    "deepseek": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    }
                },
            }
        }
    )

    assert app.get_configured_provider_name() == "deepseek"
    assert app.get_configured_model_label() == "deepseek-v4-flash/deepseek"