from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from pywork.schemas.message_schema import (
    AnyMessage,
    MessageRole,
    SystemMessage,
    create_system_message,
    create_user_message,
    create_assistant_message,
)
from pywork.schemas.tool_schema import ToolCall


class TokenBudgetError(Exception):
    """
    TokenBudget 基础异常。
    """


class TokenBudgetOverflowError(TokenBudgetError):
    """
    Token 超出预算。
    """


class TruncationStrategy(str, Enum):
    """
    消息裁剪策略。
    """

    KEEP_RECENT = "keep_recent"
    KEEP_SYSTEM_AND_RECENT = "keep_system_and_recent"


@dataclass
class TokenCounterConfig:
    """
    Token 计数器配置。

    tiktoken 不存在时，会自动降级为字符估算。
    """

    model: str | None = None
    encoding_name: str = "cl100k_base"

    fallback_chars_per_token: float = 3.8

    tokens_per_message: int = 3
    tokens_per_name: int = 1
    assistant_reply_overhead: int = 3

    count_message_metadata: bool = False


@dataclass(frozen=True)
class TokenCountResult:
    token_count: int
    estimated: bool
    encoding_name: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "estimated": self.estimated,
            "encoding_name": self.encoding_name,
            "model": self.model,
        }


@dataclass(frozen=True)
class MessageTokenCount:
    message_id: str
    role: str
    token_count: int
    estimated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "token_count": self.token_count,
            "estimated": self.estimated,
        }


@dataclass
class TokenBudgetConfig:
    """
    上下文预算配置。

    max_context_tokens:
        模型最大上下文。

    reserved_output_tokens:
        预留给模型输出的 token。

    reserved_tool_tokens:
        预留给工具调用、工具结果、结构化输出等额外开销。

    safety_margin:
        安全余量，避免刚好顶满。
    """

    max_context_tokens: int = 128_000
    reserved_output_tokens: int = 4096
    reserved_tool_tokens: int = 1024
    safety_margin: int = 512

    truncation_strategy: TruncationStrategy = TruncationStrategy.KEEP_SYSTEM_AND_RECENT

    def input_budget(self) -> int:
        budget = (
            self.max_context_tokens
            - self.reserved_output_tokens
            - self.reserved_tool_tokens
            - self.safety_margin
        )

        return max(0, budget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "reserved_tool_tokens": self.reserved_tool_tokens,
            "safety_margin": self.safety_margin,
            "input_budget": self.input_budget(),
            "truncation_strategy": self.truncation_strategy.value,
        }


@dataclass(frozen=True)
class TokenBudgetReport:
    """
    一次预算检查结果。
    """

    input_tokens: int
    tool_tokens: int
    total_prompt_tokens: int

    input_budget: int
    max_context_tokens: int
    reserved_output_tokens: int
    reserved_tool_tokens: int
    safety_margin: int

    remaining_input_tokens: int
    available_output_tokens: int

    over_budget: bool
    estimated: bool

    message_counts: list[MessageTokenCount] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "tool_tokens": self.tool_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "input_budget": self.input_budget,
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "reserved_tool_tokens": self.reserved_tool_tokens,
            "safety_margin": self.safety_margin,
            "remaining_input_tokens": self.remaining_input_tokens,
            "available_output_tokens": self.available_output_tokens,
            "over_budget": self.over_budget,
            "estimated": self.estimated,
            "message_counts": [
                item.to_dict()
                for item in self.message_counts
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


@dataclass(frozen=True)
class TrimmedMessagesResult:
    """
    消息裁剪结果。
    """

    messages: list[AnyMessage]
    dropped_messages: list[AnyMessage]
    before_report: TokenBudgetReport
    after_report: TokenBudgetReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_count": len(self.messages),
            "dropped_message_count": len(self.dropped_messages),
            "before_report": self.before_report.to_dict(),
            "after_report": self.after_report.to_dict(),
            "messages": [
                {
                    "message_id": message.message_id,
                    "role": str(message.role),
                    "content_preview": message.content[:80],
                }
                for message in self.messages
            ],
            "dropped_messages": [
                {
                    "message_id": message.message_id,
                    "role": str(message.role),
                    "content_preview": message.content[:80],
                }
                for message in self.dropped_messages
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


def role_value(message: AnyMessage) -> str:
    role = message.role

    if isinstance(role, MessageRole):
        return role.value

    return str(role)


def safe_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


class TokenCounter:
    """
    Token 计数器。

    优先使用 tiktoken。
    如果没有安装 tiktoken，则使用字符估算。
    """

    def __init__(
        self,
        config: TokenCounterConfig | None = None,
    ) -> None:
        self.config = config or TokenCounterConfig()
        self._encoding: Any = None
        self._encoding_loaded = False
        self._estimated = True

    def get_encoding(self) -> Any | None:
        if self._encoding_loaded:
            return self._encoding

        self._encoding_loaded = True

        try:
            import tiktoken
        except ImportError:
            self._encoding = None
            self._estimated = True
            return None

        try:
            if self.config.model:
                self._encoding = tiktoken.encoding_for_model(self.config.model)
            else:
                self._encoding = tiktoken.get_encoding(self.config.encoding_name)

            self._estimated = False
            return self._encoding

        except Exception:
            try:
                self._encoding = tiktoken.get_encoding(self.config.encoding_name)
                self._estimated = False
                return self._encoding
            except Exception:
                self._encoding = None
                self._estimated = True
                return None

    @property
    def is_estimated(self) -> bool:
        self.get_encoding()
        return self._estimated

    def count_text(self, text: str) -> TokenCountResult:
        text = text or ""
        encoding = self.get_encoding()

        if encoding is not None:
            return TokenCountResult(
                token_count=len(encoding.encode(text)),
                estimated=False,
                encoding_name=self.config.encoding_name,
                model=self.config.model,
            )

        if not text:
            count = 0
        else:
            count = max(
                1,
                math.ceil(len(text) / self.config.fallback_chars_per_token),
            )

        return TokenCountResult(
            token_count=count,
            estimated=True,
            encoding_name=None,
            model=self.config.model,
        )

    def count_json(self, value: Any) -> TokenCountResult:
        return self.count_text(
            safe_json_dumps(value)
        )

    def count_tool_call(self, tool_call: ToolCall) -> TokenCountResult:
        return self.count_json(
            tool_call.model_dump(mode="json")
        )

    def count_message(self, message: AnyMessage) -> MessageTokenCount:
        total = self.config.tokens_per_message

        total += self.count_text(role_value(message)).token_count
        total += self.count_text(message.content).token_count

        if message.name:
            total += self.config.tokens_per_name
            total += self.count_text(message.name).token_count

        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for call in tool_calls:
                total += self.count_tool_call(call).token_count

        tool_call_id = getattr(message, "tool_call_id", None)
        tool_name = getattr(message, "tool_name", None)

        if tool_call_id:
            total += self.count_text(str(tool_call_id)).token_count

        if tool_name:
            total += self.count_text(str(tool_name)).token_count

        if self.config.count_message_metadata and message.metadata:
            total += self.count_json(message.metadata).token_count

        return MessageTokenCount(
            message_id=message.message_id,
            role=role_value(message),
            token_count=total,
            estimated=self.is_estimated,
        )

    def count_messages(
        self,
        messages: list[AnyMessage],
        *,
        include_assistant_reply_overhead: bool = True,
    ) -> tuple[int, list[MessageTokenCount]]:
        counts = [
            self.count_message(message)
            for message in messages
        ]

        total = sum(item.token_count for item in counts)

        if include_assistant_reply_overhead:
            total += self.config.assistant_reply_overhead

        return total, counts

    def count_tool_definitions(
        self,
        tool_definitions: Iterable[dict[str, Any]] | None,
    ) -> TokenCountResult:
        if not tool_definitions:
            return TokenCountResult(
                token_count=0,
                estimated=self.is_estimated,
                encoding_name=self.config.encoding_name,
                model=self.config.model,
            )

        return self.count_json(
            list(tool_definitions)
        )


class TokenBudgetManager:
    """
    Token 预算管理器。

    Runtime / LLM Router 后面主要用这个类。
    """

    def __init__(
        self,
        *,
        counter: TokenCounter | None = None,
        counter_config: TokenCounterConfig | None = None,
        budget_config: TokenBudgetConfig | None = None,
    ) -> None:
        self.counter = counter or TokenCounter(counter_config)
        self.budget_config = budget_config or TokenBudgetConfig()

    def count_messages(
        self,
        messages: list[AnyMessage],
    ) -> tuple[int, list[MessageTokenCount]]:
        return self.counter.count_messages(messages)

    def count_tools(
        self,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> TokenCountResult:
        return self.counter.count_tool_definitions(tool_definitions)

    def build_report(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> TokenBudgetReport:
        input_tokens, message_counts = self.counter.count_messages(messages)
        tool_result = self.count_tools(tool_definitions)

        total_prompt_tokens = input_tokens + tool_result.token_count
        input_budget = self.budget_config.input_budget()

        remaining_input_tokens = input_budget - total_prompt_tokens

        available_output_tokens = max(
            0,
            self.budget_config.max_context_tokens
            - total_prompt_tokens
            - self.budget_config.reserved_tool_tokens
            - self.budget_config.safety_margin,
        )

        return TokenBudgetReport(
            input_tokens=input_tokens,
            tool_tokens=tool_result.token_count,
            total_prompt_tokens=total_prompt_tokens,
            input_budget=input_budget,
            max_context_tokens=self.budget_config.max_context_tokens,
            reserved_output_tokens=self.budget_config.reserved_output_tokens,
            reserved_tool_tokens=self.budget_config.reserved_tool_tokens,
            safety_margin=self.budget_config.safety_margin,
            remaining_input_tokens=remaining_input_tokens,
            available_output_tokens=available_output_tokens,
            over_budget=total_prompt_tokens > input_budget,
            estimated=self.counter.is_estimated or tool_result.estimated,
            message_counts=message_counts,
        )

    def assert_within_budget(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> TokenBudgetReport:
        report = self.build_report(
            messages,
            tool_definitions=tool_definitions,
        )

        if report.over_budget:
            raise TokenBudgetOverflowError(
                f"Prompt is over budget: "
                f"{report.total_prompt_tokens} > {report.input_budget}"
            )

        return report

    def trim_messages(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> TrimmedMessagesResult:
        before_report = self.build_report(
            messages,
            tool_definitions=tool_definitions,
        )

        if not before_report.over_budget:
            return TrimmedMessagesResult(
                messages=list(messages),
                dropped_messages=[],
                before_report=before_report,
                after_report=before_report,
            )

        strategy = self.budget_config.truncation_strategy

        if strategy == TruncationStrategy.KEEP_RECENT:
            kept = self._trim_keep_recent(
                messages,
                tool_definitions=tool_definitions,
            )

        elif strategy == TruncationStrategy.KEEP_SYSTEM_AND_RECENT:
            kept = self._trim_keep_system_and_recent(
                messages,
                tool_definitions=tool_definitions,
            )

        else:
            raise TokenBudgetError(
                f"Unsupported truncation strategy: {strategy}"
            )

        kept_ids = {
            message.message_id
            for message in kept
        }

        dropped = [
            message
            for message in messages
            if message.message_id not in kept_ids
        ]

        after_report = self.build_report(
            kept,
            tool_definitions=tool_definitions,
        )

        return TrimmedMessagesResult(
            messages=kept,
            dropped_messages=dropped,
            before_report=before_report,
            after_report=after_report,
        )

    def _fits(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> bool:
        return not self.build_report(
            messages,
            tool_definitions=tool_definitions,
        ).over_budget

    def _trim_keep_recent(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> list[AnyMessage]:
        kept_reversed: list[AnyMessage] = []

        for message in reversed(messages):
            candidate = list(reversed(kept_reversed + [message]))

            if self._fits(candidate, tool_definitions=tool_definitions):
                kept_reversed.append(message)

        return list(reversed(kept_reversed))

    def _trim_keep_system_and_recent(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> list[AnyMessage]:
        system_messages = [
            message
            for message in messages
            if role_value(message) == "system"
        ]

        non_system_messages = [
            message
            for message in messages
            if role_value(message) != "system"
        ]

        kept_system: list[AnyMessage] = []

        for message in system_messages:
            candidate = kept_system + [message]

            if self._fits(candidate, tool_definitions=tool_definitions):
                kept_system.append(message)

        kept_recent_reversed: list[AnyMessage] = []

        for message in reversed(non_system_messages):
            recent = list(reversed(kept_recent_reversed + [message]))
            candidate = kept_system + recent

            if self._fits(candidate, tool_definitions=tool_definitions):
                kept_recent_reversed.append(message)

        return kept_system + list(reversed(kept_recent_reversed))

    def get_available_output_tokens(
        self,
        messages: list[AnyMessage],
        *,
        tool_definitions: Iterable[dict[str, Any]] | None = None,
    ) -> int:
        report = self.build_report(
            messages,
            tool_definitions=tool_definitions,
        )

        return report.available_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "counter_config": {
                "model": self.counter.config.model,
                "encoding_name": self.counter.config.encoding_name,
                "fallback_chars_per_token": self.counter.config.fallback_chars_per_token,
                "tokens_per_message": self.counter.config.tokens_per_message,
                "tokens_per_name": self.counter.config.tokens_per_name,
                "assistant_reply_overhead": self.counter.config.assistant_reply_overhead,
                "count_message_metadata": self.counter.config.count_message_metadata,
            },
            "budget_config": self.budget_config.to_dict(),
            "estimated": self.counter.is_estimated,
        }


def create_token_budget_manager(
    *,
    model: str | None = None,
    max_context_tokens: int = 128_000,
    reserved_output_tokens: int = 4096,
    reserved_tool_tokens: int = 1024,
    safety_margin: int = 512,
) -> TokenBudgetManager:
    return TokenBudgetManager(
        counter_config=TokenCounterConfig(
            model=model,
        ),
        budget_config=TokenBudgetConfig(
            max_context_tokens=max_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
            reserved_tool_tokens=reserved_tool_tokens,
            safety_margin=safety_margin,
        ),
    )


def count_text_tokens(
    text: str,
    *,
    model: str | None = None,
) -> int:
    counter = TokenCounter(
        TokenCounterConfig(
            model=model,
        )
    )

    return counter.count_text(text).token_count


def count_messages_tokens(
    messages: list[AnyMessage],
    *,
    model: str | None = None,
) -> int:
    counter = TokenCounter(
        TokenCounterConfig(
            model=model,
        )
    )

    total, _ = counter.count_messages(messages)
    return total


def demo() -> None:
    messages: list[AnyMessage] = [
        create_system_message("You are PyWork, a coding agent."),
    ]

    for i in range(1, 12):
        messages.append(
            create_user_message(
                f"这是用户第 {i} 条消息。" + " 请记住这些上下文。" * 8
            )
        )
        messages.append(
            create_assistant_message(
                f"这是助手第 {i} 条回复。" + " 我会继续帮助你完成 PyWork。" * 8
            )
        )

    tool_definitions = [
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
    ]

    manager = TokenBudgetManager(
        counter_config=TokenCounterConfig(
            model="gpt-4o",
        ),
        budget_config=TokenBudgetConfig(
            max_context_tokens=350,
            reserved_output_tokens=80,
            reserved_tool_tokens=20,
            safety_margin=20,
        ),
    )

    print("Manager:")
    print(
        json.dumps(
            manager.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\nBefore report:")
    before = manager.build_report(
        messages,
        tool_definitions=tool_definitions,
    )
    print(before.to_json(indent=2))

    print("\nTrim result:")
    trimmed = manager.trim_messages(
        messages,
        tool_definitions=tool_definitions,
    )
    print(trimmed.to_json(indent=2))

    print("\nAfter messages:")
    for message in trimmed.messages:
        print(f"- {role_value(message)}: {message.content[:60]}")


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())