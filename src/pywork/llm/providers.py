from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pywork.llm.messages import (
    anthropic_message_to_pywork,
    get_attr_or_key,
    messages_to_anthropic,
    messages_to_anthropic_system,
    messages_to_openai,
    openai_message_to_pywork,
    tool_definitions_to_anthropic,
    tool_definitions_to_openai,
)
from pywork.schemas.message_schema import (
    AnyMessage,
    AssistantMessage,
    create_assistant_message,
)
from pywork.schemas.tool_schema import ToolCall


ProviderFormat = Literal[
    "openai",
    "anthropic",
    "openai_compatible",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LLMProviderError(Exception):
    """
    LLM Provider 基础异常。
    """


class LLMProviderConfigError(LLMProviderError):
    """
    Provider 配置错误。
    """


class LLMProviderRequestError(LLMProviderError):
    """
    Provider 请求错误。
    """


@dataclass(frozen=True)
class LLMTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMProviderConfig:
    """
    Provider 配置。

    api_key 可以直接传，也可以通过 api_key_env 从环境变量读取。
    """

    provider: str
    model: str

    api_format: ProviderFormat = "openai"

    api_key: str | None = None
    api_key_env: str | None = None

    base_url: str | None = None

    timeout: float = 60.0
    max_retries: int = 2

    temperature: float | None = None
    max_tokens: int | None = None

    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key

        if self.api_key_env:
            return os.getenv(self.api_key_env)

        return None

    def masked_api_key_hint(self) -> str:
        key = self.resolve_api_key()

        if not key:
            return ""

        if len(key) <= 8:
            return "***"

        return f"{key[:4]}...{key[-4:]}"

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_format": self.api_format,
            "api_key_env": self.api_key_env,
            "api_key": self.masked_api_key_hint(),
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "metadata": self.metadata,
        }


@dataclass
class LLMRequest:
    """
    一次 LLM 请求。
    """

    messages: list[AnyMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    统一 LLM 响应。
    """

    message: AssistantMessage

    provider: str
    model: str

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    finish_reason: str | None = None
    token_usage: LLMTokenUsage = field(default_factory=LLMTokenUsage)

    raw_response: Any = None

    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "content": self.content,
            "tool_calls": [
                call.model_dump(mode="json")
                for call in self.tool_calls
            ],
            "finish_reason": self.finish_reason,
            "token_usage": self.token_usage.to_dict(),
            "message": self.message.model_dump(mode="json"),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


class BaseLLMProvider(ABC):
    """
    Provider 适配器抽象基类。
    """

    provider_format: ProviderFormat

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.validate_config()

    def validate_config(self) -> None:
        if not self.config.provider.strip():
            raise LLMProviderConfigError("provider cannot be empty")

        if not self.config.model.strip():
            raise LLMProviderConfigError("model cannot be empty")

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def model_name(self) -> str:
        return self.config.model

    @abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        发送一次非流式 Chat 请求。
        """

    def run_chat(self, request: LLMRequest) -> LLMResponse:
        """
        同步包装。

        注意：如果在 Textual / FastAPI 这种已有事件循环环境中，应该使用 await chat()。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.chat(request))

        raise LLMProviderRequestError(
            "run_chat() cannot be used inside a running event loop. "
            "Use await provider.chat(...) instead."
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "provider_format": self.provider_format,
            "config": self.config.to_safe_dict(),
        }


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI 官方 SDK Provider。
    """

    provider_format: ProviderFormat = "openai"

    def __init__(self, config: LLMProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def validate_config(self) -> None:
        super().validate_config()

        if not self.config.resolve_api_key():
            raise LLMProviderConfigError(
                "OpenAI API key is missing. "
                "Set api_key or api_key_env."
            )

    def get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMProviderConfigError(
                "openai package is not installed."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.config.resolve_api_key(),
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        if self.config.extra_headers:
            kwargs["default_headers"] = self.config.extra_headers

        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        client = self.get_client()

        model = request.model or self.config.model
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens
        )
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.config.temperature
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_to_openai(request.messages),
        }

        if request.tools:
            payload["tools"] = tool_definitions_to_openai(request.tools)

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if temperature is not None:
            payload["temperature"] = temperature

        if self.config.extra_body:
            payload.update(self.config.extra_body)

        try:
            raw_response = await client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMProviderRequestError(str(exc)) from exc

        return self._parse_response(
            raw_response,
            model=model,
            metadata=request.metadata,
        )

    def _parse_response(
        self,
        raw_response: Any,
        *,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        choices = get_attr_or_key(raw_response, "choices", []) or []

        if not choices:
            raise LLMProviderRequestError("OpenAI response has no choices")

        choice = choices[0]
        raw_message = get_attr_or_key(choice, "message", None)

        if raw_message is None:
            raise LLMProviderRequestError("OpenAI choice has no message")

        message = openai_message_to_pywork(raw_message)

        if not isinstance(message, AssistantMessage):
            message = create_assistant_message(
                content=getattr(message, "content", ""),
                metadata={
                    "source": "openai_response_cast",
                },
            )

        finish_reason = get_attr_or_key(choice, "finish_reason", None)

        usage = get_attr_or_key(raw_response, "usage", None)
        input_tokens = int(
            get_attr_or_key(
                usage,
                "prompt_tokens",
                0,
            )
            or 0
        )
        output_tokens = int(
            get_attr_or_key(
                usage,
                "completion_tokens",
                0,
            )
            or 0
        )
        total_tokens = int(
            get_attr_or_key(
                usage,
                "total_tokens",
                input_tokens + output_tokens,
            )
            or input_tokens + output_tokens
        )

        tool_calls = list(message.tool_calls)

        return LLMResponse(
            message=message,
            provider=self.provider_name,
            model=model,
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=str(finish_reason) if finish_reason else None,
            token_usage=LLMTokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            raw_response=raw_response,
            metadata=metadata or {},
        )


class OpenAICompatibleProvider(OpenAIProvider):
    """
    OpenAI-compatible Provider。

    适用于：
    - DeepSeek
    - Qwen OpenAI-compatible endpoint
    - Ollama /v1
    - 其他兼容 OpenAI Chat Completions 的服务
    """

    provider_format: ProviderFormat = "openai_compatible"

    def validate_config(self) -> None:
        BaseLLMProvider.validate_config(self)

        if not self.config.base_url:
            raise LLMProviderConfigError(
                "OpenAI-compatible provider requires base_url."
            )

        if not self.config.resolve_api_key():
            raise LLMProviderConfigError(
                "OpenAI-compatible API key is missing. "
                "Set api_key or api_key_env. "
                "For local endpoints, use a dummy key if the server requires one."
            )


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic SDK Provider。
    """

    provider_format: ProviderFormat = "anthropic"

    def __init__(self, config: LLMProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def validate_config(self) -> None:
        super().validate_config()

        if not self.config.resolve_api_key():
            raise LLMProviderConfigError(
                "Anthropic API key is missing. "
                "Set api_key or api_key_env."
            )

    def get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise LLMProviderConfigError(
                "anthropic package is not installed."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.config.resolve_api_key(),
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        if self.config.extra_headers:
            kwargs["default_headers"] = self.config.extra_headers

        self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        client = self.get_client()

        model = request.model or self.config.model
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens
        )
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.config.temperature
        )

        if max_tokens is None:
            max_tokens = 1024

        tools = (
            tool_definitions_to_anthropic(request.tools)
            if request.tools
            else None
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_to_anthropic(request.messages),
            "max_tokens": max_tokens,
        }

        system = messages_to_anthropic_system(request.messages)

        if system:
            payload["system"] = system

        if temperature is not None:
            payload["temperature"] = temperature

        if tools:
            payload["tools"] = tools

        if self.config.extra_body:
            payload.update(self.config.extra_body)

        try:
            raw_response = await client.messages.create(**payload)
        except Exception as exc:
            raise LLMProviderRequestError(str(exc)) from exc

        return self._parse_response(
            raw_response,
            model=model,
            metadata=request.metadata,
        )

    def _parse_response(
        self,
        raw_response: Any,
        *,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        message = anthropic_message_to_pywork(raw_response)

        stop_reason = get_attr_or_key(
            raw_response,
            "stop_reason",
            None,
        )

        usage = get_attr_or_key(
            raw_response,
            "usage",
            None,
        )

        input_tokens = int(
            get_attr_or_key(
                usage,
                "input_tokens",
                0,
            )
            or 0
        )
        output_tokens = int(
            get_attr_or_key(
                usage,
                "output_tokens",
                0,
            )
            or 0
        )

        tool_calls = list(message.tool_calls)

        return LLMResponse(
            message=message,
            provider=self.provider_name,
            model=model,
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=str(stop_reason) if stop_reason else None,
            token_usage=LLMTokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            raw_response=raw_response,
            metadata=metadata or {},
        )


def create_provider(config: LLMProviderConfig) -> BaseLLMProvider:
    """
    根据 config.api_format 创建 Provider。
    """
    api_format = config.api_format.strip().lower()

    if api_format == "openai":
        return OpenAIProvider(config)

    if api_format == "anthropic":
        return AnthropicProvider(config)

    if api_format == "openai_compatible":
        return OpenAICompatibleProvider(config)

    raise LLMProviderConfigError(f"Unsupported api_format: {config.api_format!r}")


def create_provider_config_from_dict(config: dict[str, Any]) -> LLMProviderConfig:
    """
    从 dict 创建 LLMProviderConfig。

    支持两种格式：

    1. 扁平：
        {
          "provider": "deepseek",
          "model": "deepseek-chat",
          "api_format": "openai_compatible",
          "base_url": "https://api.deepseek.com",
          "api_key_env": "DEEPSEEK_API_KEY"
        }

    2. 项目默认配置：
        {
          "default": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            ...
          }
        }
    """
    default = config.get("default", config)

    provider = str(default.get("provider", "openai"))
    model = str(default.get("model", "gpt-5.5"))

    api_format = str(
        default.get(
            "api_format",
            "openai_compatible" if provider not in {"openai", "anthropic"} else provider,
        )
    )

    return LLMProviderConfig(
        provider=provider,
        model=model,
        api_format=api_format,  # type: ignore[arg-type]
        api_key=default.get("api_key"),
        api_key_env=default.get("api_key_env"),
        base_url=default.get("base_url"),
        timeout=float(default.get("timeout", 60.0)),
        max_retries=int(default.get("max_retries", 2)),
        temperature=default.get("temperature"),
        max_tokens=default.get("max_tokens"),
        extra_headers=default.get("extra_headers", {}) or {},
        extra_body=default.get("extra_body", {}) or {},
        metadata=default.get("metadata", {}) or {},
    )


def main() -> int:
    from pywork.schemas.message_schema import create_system_message, create_user_message

    messages: list[AnyMessage] = [
        create_system_message("You are PyWork, a coding agent."),
        create_user_message("hello"),
    ]

    request = LLMRequest(
        messages=messages,
        tools=[
            {
                "name": "echo",
                "description": "Echo input text.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                        }
                    },
                    "required": ["text"],
                },
            }
        ],
    )

    configs = [
        LLMProviderConfig(
            provider="openai",
            model="gpt-5.5",
            api_format="openai",
            api_key_env="OPENAI_API_KEY",
        ),
        LLMProviderConfig(
            provider="anthropic",
            model="claude-opus-4-8",
            api_format="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        LLMProviderConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_format="openai_compatible",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
        ),
    ]

    print("Provider configs:")
    for item in configs:
        print(json.dumps(item.to_safe_dict(), ensure_ascii=False, indent=2))

    print("\nRequest preview:")
    print(
        json.dumps(
            {
                "message_count": len(request.messages),
                "tool_count": len(request.tools),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nNo live request is sent in this demo. "
        "Provider live calls will be wired through llm/router.py."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())