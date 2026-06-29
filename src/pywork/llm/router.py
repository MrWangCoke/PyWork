from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pywork.llm.providers import (
    BaseLLMProvider,
    LLMProviderConfig,
    LLMProviderConfigError,
    LLMProviderRequestError,
    LLMRequest,
    LLMResponse,
    create_provider,
    create_provider_config_from_dict,
)
from pywork.schemas.message_schema import AnyMessage


class LLMRouterError(Exception):
    """
    LLMRouter 基础异常。
    """


class LLMRouterConfigError(LLMRouterError):
    """
    Router 配置异常。
    """


class LLMRouterProviderNotFoundError(LLMRouterError):
    """
    找不到指定 Provider。
    """


@dataclass
class LLMRouterConfig:
    """
    Router 配置。

    支持多 Provider：

    {
      "default_provider": "deepseek",
      "providers": {
        "deepseek": {
          "provider": "deepseek",
          "model": "deepseek-chat",
          "api_format": "openai_compatible",
          "base_url": "https://api.deepseek.com",
          "api_key_env": "DEEPSEEK_API_KEY"
        },
        "openai": {
          "provider": "openai",
          "model": "gpt-5.5",
          "api_format": "openai",
          "api_key_env": "OPENAI_API_KEY"
        }
      }
    }
    """

    default_provider: str = "default"
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)
    fallback_providers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "default_provider": self.default_provider,
            "providers": {
                name: config.to_safe_dict()
                for name, config in self.providers.items()
            },
            "fallback_providers": self.fallback_providers,
            "metadata": self.metadata,
        }


@dataclass
class LLMRouterChatOptions:
    """
    一次 chat 调用的临时选项。

    这些选项会覆盖 ProviderConfig 里的默认值。
    """

    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def unwrap_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    兼容两种传法：

    1. 直接传 llm 配置：
        {
          "default_provider": "...",
          "providers": {...}
        }

    2. 传整个项目配置：
        {
          "llm": {
            "default_provider": "...",
            "providers": {...}
          }
        }
    """
    if "llm" in config and isinstance(config["llm"], dict):
        return config["llm"]

    return config


def normalize_provider_name(name: str | None) -> str:
    if not name:
        return "default"

    return name.strip().lower()


def infer_api_format(provider: str, config: dict[str, Any]) -> str:
    """
    根据 provider 名字推断 api_format。
    """
    if "api_format" in config:
        return str(config["api_format"])

    normalized = provider.strip().lower()

    if normalized == "openai":
        return "openai"

    if normalized == "anthropic":
        return "anthropic"

    return "openai_compatible"


def provider_config_from_named_dict(
    name: str,
    data: dict[str, Any],
) -> LLMProviderConfig:
    """
    从 providers.xxx 配置生成 LLMProviderConfig。
    """
    provider = str(data.get("provider", name))
    api_format = infer_api_format(provider, data)

    model = data.get("model")

    if not model:
        raise LLMRouterConfigError(
            f"LLM provider {name!r} missing model"
        )

    return LLMProviderConfig(
        provider=provider,
        model=str(model),
        api_format=api_format,  # type: ignore[arg-type]
        api_key=data.get("api_key"),
        api_key_env=data.get("api_key_env"),
        base_url=data.get("base_url"),
        timeout=float(data.get("timeout", 60.0)),
        max_retries=int(data.get("max_retries", 2)),
        temperature=data.get("temperature"),
        max_tokens=data.get("max_tokens"),
        extra_headers=data.get("extra_headers", {}) or {},
        extra_body=data.get("extra_body", {}) or {},
        metadata=data.get("metadata", {}) or {},
    )


def router_config_from_dict(config: dict[str, Any]) -> LLMRouterConfig:
    """
    从 dict 创建 LLMRouterConfig。

    支持三种格式。

    格式一：推荐，多 Provider：

    {
      "default_provider": "deepseek",
      "providers": {
        "deepseek": {...},
        "openai": {...}
      }
    }

    格式二：只有 default：

    {
      "default": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        ...
      }
    }

    格式三：扁平配置：

    {
      "provider": "deepseek",
      "model": "deepseek-chat",
      ...
    }
    """
    llm_config = unwrap_llm_config(config)

    providers: dict[str, LLMProviderConfig] = {}

    raw_providers = llm_config.get("providers")

    if isinstance(raw_providers, dict) and raw_providers:
        for name, raw_config in raw_providers.items():
            if not isinstance(raw_config, dict):
                raise LLMRouterConfigError(
                    f"LLM provider config {name!r} must be a dict"
                )

            normalized_name = normalize_provider_name(str(name))
            providers[normalized_name] = provider_config_from_named_dict(
                normalized_name,
                raw_config,
            )

        default_provider = normalize_provider_name(
            llm_config.get("default_provider")
            or llm_config.get("default")
            or next(iter(providers.keys()))
        )

    else:
        default_provider = normalize_provider_name(
            llm_config.get("default_provider")
            or llm_config.get("provider")
            or "default"
        )

        provider_config = create_provider_config_from_dict(llm_config)
        providers[default_provider] = provider_config

    if default_provider not in providers:
        if "default" in providers:
            default_provider = "default"
        else:
            raise LLMRouterConfigError(
                f"default_provider {default_provider!r} not found in providers"
            )

    fallback_providers = [
        normalize_provider_name(item)
        for item in llm_config.get("fallback_providers", []) or []
    ]

    return LLMRouterConfig(
        default_provider=default_provider,
        providers=providers,
        fallback_providers=fallback_providers,
        metadata=llm_config.get("metadata", {}) or {},
    )


class LLMRouter:
    """
    LLM 统一入口。

    Runtime 后面只依赖这个类，不直接依赖 OpenAI / Anthropic / DeepSeek 等 SDK。
    """

    def __init__(
        self,
        *,
        config: LLMRouterConfig | None = None,
        providers: dict[str, BaseLLMProvider] | None = None,
    ) -> None:
        self.config = config or LLMRouterConfig()
        self._providers: dict[str, BaseLLMProvider] = {}

        if providers:
            for name, provider in providers.items():
                self.register_provider(name, provider)

        for name, provider_config in self.config.providers.items():
            normalized_name = normalize_provider_name(name)

            if normalized_name not in self._providers:
                self._providers[normalized_name] = create_provider(provider_config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LLMRouter:
        return cls(
            config=router_config_from_dict(config)
        )

    @classmethod
    def from_provider_config(cls, config: LLMProviderConfig) -> LLMRouter:
        name = normalize_provider_name(config.provider)

        router_config = LLMRouterConfig(
            default_provider=name,
            providers={
                name: config,
            },
        )

        return cls(config=router_config)

    @property
    def default_provider_name(self) -> str:
        return normalize_provider_name(self.config.default_provider)

    def has_provider(self, name: str) -> bool:
        return normalize_provider_name(name) in self._providers

    def register_provider(
        self,
        name: str,
        provider: BaseLLMProvider,
        *,
        set_default: bool = False,
    ) -> None:
        normalized_name = normalize_provider_name(name)

        self._providers[normalized_name] = provider

        if set_default:
            self.config.default_provider = normalized_name

    def unregister_provider(self, name: str) -> None:
        normalized_name = normalize_provider_name(name)

        if normalized_name in self._providers:
            del self._providers[normalized_name]

    def list_provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def list_providers(self) -> dict[str, dict[str, Any]]:
        return {
            name: provider.to_safe_dict()
            for name, provider in self._providers.items()
        }

    def require_provider(
        self,
        name: str | None = None,
    ) -> BaseLLMProvider:
        provider_name = normalize_provider_name(name or self.default_provider_name)

        provider = self._providers.get(provider_name)

        if provider is None:
            raise LLMRouterProviderNotFoundError(
                f"LLM provider not found: {provider_name!r}. "
                f"Available providers: {', '.join(self.list_provider_names())}"
            )

        return provider

    def get_default_provider(self) -> BaseLLMProvider:
        return self.require_provider(self.default_provider_name)

    def build_request(
        self,
        messages: list[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: LLMRouterChatOptions | None = None,
    ) -> LLMRequest:
        options = options or LLMRouterChatOptions()

        return LLMRequest(
            messages=messages,
            tools=tools or [],
            model=options.model,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            metadata=options.metadata,
        )

    async def chat(
        self,
        messages: list[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        use_fallbacks: bool = True,
    ) -> LLMResponse:
        """
        统一 Chat 入口。
        """
        options = LLMRouterChatOptions(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {},
        )

        request = self.build_request(
            messages,
            tools=tools,
            options=options,
        )

        provider_names = self._build_provider_order(
            requested_provider=provider,
            use_fallbacks=use_fallbacks,
        )

        last_error: Exception | None = None

        for provider_name in provider_names:
            llm_provider = self.require_provider(provider_name)

            try:
                response = await llm_provider.chat(request)

                response.metadata.update(
                    {
                        "router_provider": provider_name,
                        "fallback_used": provider_name != normalize_provider_name(
                            provider or self.default_provider_name
                        ),
                    }
                )

                return response

            except Exception as exc:
                last_error = exc

                if not use_fallbacks:
                    break

                continue

        if last_error is not None:
            raise LLMProviderRequestError(
                f"All LLM providers failed. Last error: {last_error}"
            ) from last_error

        raise LLMRouterProviderNotFoundError("No LLM provider available")

    def run_chat(
        self,
        messages: list[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        use_fallbacks: bool = True,
    ) -> LLMResponse:
        """
        同步 Chat 入口。

        CLI 可以用这个。
        Textual / Runtime 异步环境里用 await chat()。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.chat(
                    messages,
                    tools=tools,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                    use_fallbacks=use_fallbacks,
                )
            )

        raise LLMRouterError(
            "LLMRouter.run_chat() cannot be used inside a running event loop. "
            "Use await router.chat(...) instead."
        )

    def _build_provider_order(
        self,
        *,
        requested_provider: str | None = None,
        use_fallbacks: bool = True,
    ) -> list[str]:
        primary = normalize_provider_name(
            requested_provider or self.default_provider_name
        )

        provider_order = [primary]

        if use_fallbacks:
            for item in self.config.fallback_providers:
                name = normalize_provider_name(item)

                if name not in provider_order:
                    provider_order.append(name)

        return provider_order

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "default_provider": self.default_provider_name,
            "providers": self.list_providers(),
            "fallback_providers": self.config.fallback_providers,
            "metadata": self.config.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_safe_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


def create_llm_router(
    config: dict[str, Any] | LLMRouterConfig | LLMProviderConfig | None = None,
) -> LLMRouter:
    """
    创建 LLMRouter 的统一函数。
    """
    if config is None:
        return LLMRouter()

    if isinstance(config, LLMRouterConfig):
        return LLMRouter(config=config)

    if isinstance(config, LLMProviderConfig):
        return LLMRouter.from_provider_config(config)

    if isinstance(config, dict):
        return LLMRouter.from_dict(config)

    raise LLMRouterConfigError(
        f"Unsupported router config type: {type(config).__name__}"
    )


def main() -> int:
    from pywork.schemas.message_schema import create_system_message, create_user_message

    demo_config = {
        "llm": {
            "default_provider": "deepseek",
            "fallback_providers": [
                "openai",
            ],
            "providers": {
                "deepseek": {
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "api_format": "openai_compatible",
                    "base_url": "https://api.deepseek.com",
                    "api_key_env": "DEEPSEEK_API_KEY",
                },
                "openai": {
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "api_format": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "anthropic": {
                    "provider": "anthropic",
                    "model": "claude-opus-4-8",
                    "api_format": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            },
        }
    }

    router_config = router_config_from_dict(demo_config)

    print("Router config:")
    print(
        json.dumps(
            router_config.to_safe_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\nCreate router:")
    try:
        router = create_llm_router(demo_config)
        print(router.to_json(indent=2))
    except LLMProviderConfigError as exc:
        print("Provider config error:")
        print(exc)

    messages = [
        create_system_message("You are PyWork, a coding agent."),
        create_user_message("hello"),
    ]

    print("\nMessage count:")
    print(len(messages))

    print(
        "\nNo live request is sent in this demo. "
        "Use await router.chat(messages) after setting API keys."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())