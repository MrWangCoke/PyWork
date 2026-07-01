from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.permission.policy import (
    PermissionDecision,
    PermissionDecisionType,
)
from pywork.permission.risk import RiskLevel


DEFAULT_AUDIT_DIR = ".pywork/audit"
DEFAULT_AUDIT_FILE = "permissions.jsonl"
DEFAULT_MAX_ARGUMENT_TEXT_CHARS = 2_000


SECRET_KEY_PATTERN = re.compile(
    r"(password|passwd|pwd|token|secret|api[_-]?key|auth|credential|private[_-]?key)",
    re.IGNORECASE,
)

SECRET_TEXT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{12,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{12,}"),
    re.compile(r"AKIA[0-9A-Z]{12,}"),
]


class PermissionAuditError(Exception):
    """审计日志基础异常。"""


class PermissionAuditStorageError(PermissionAuditError):
    """审计日志读写异常。"""


class PermissionAuditEventType(str, Enum):
    """审计事件类型。"""

    POLICY_DECISION = "policy_decision"
    USER_DECISION = "user_decision"
    EXECUTION_RESULT = "execution_result"


class PermissionAuditUserAction(str, Enum):
    """用户在审批弹窗里的选择。"""

    ALLOW = "allow"
    DENY = "deny"
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"


@dataclass(slots=True, frozen=True)
class PermissionAuditRecord:
    """一条权限审计记录。"""

    audit_id: str
    timestamp: str
    event_type: PermissionAuditEventType

    tool_name: str
    action: str | None

    mode: str | None
    risk: str | None
    decision: str | None
    allowed: bool | None

    reason: str | None = None

    call_id: str | None = None
    session_id: str | None = None

    user_action: str | None = None

    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data

    def to_json_line(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PermissionAuditRecord:
        return cls(
            audit_id=str(data["audit_id"]),
            timestamp=str(data["timestamp"]),
            event_type=PermissionAuditEventType(str(data["event_type"])),
            tool_name=str(data["tool_name"]),
            action=optional_str(data.get("action")),
            mode=optional_str(data.get("mode")),
            risk=optional_str(data.get("risk")),
            decision=optional_str(data.get("decision")),
            allowed=optional_bool(data.get("allowed")),
            reason=optional_str(data.get("reason")),
            call_id=optional_str(data.get("call_id")),
            session_id=optional_str(data.get("session_id")),
            user_action=optional_str(data.get("user_action")),
            arguments=dict(data.get("arguments") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json_line(cls, line: str) -> PermissionAuditRecord:
        return cls.from_dict(json.loads(line))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_audit_id() -> str:
    return f"pa_{uuid.uuid4().hex}"


def optional_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None

    return bool(value)


def enum_value(value: Any) -> str | None:
    if value is None:
        return None

    raw = getattr(value, "value", value)

    return str(raw)


def redact_secret_text(value: str) -> str:
    text = value

    for pattern in SECRET_TEXT_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)

    return text


def should_redact_key(key: str) -> bool:
    return bool(SECRET_KEY_PATTERN.search(key))


def truncate_text(
    value: str,
    *,
    max_chars: int = DEFAULT_MAX_ARGUMENT_TEXT_CHARS,
) -> str:
    if len(value) <= max_chars:
        return value

    omitted = len(value) - max_chars

    return value[:max_chars] + f"\n... [truncated {omitted} chars]"


def sanitize_audit_value(
    value: Any,
    *,
    key: str | None = None,
    max_text_chars: int = DEFAULT_MAX_ARGUMENT_TEXT_CHARS,
) -> Any:
    """
    清理要写入审计日志的数据。

    目标：
    - 尽量保留操作上下文
    - 避免把 token / password / secret 写进日志
    - 避免超长文本撑爆 JSONL
    """
    if key is not None and should_redact_key(key):
        return "[REDACTED]"

    if value is None:
        return None

    if isinstance(value, str):
        return truncate_text(
            redact_secret_text(value),
            max_chars=max_text_chars,
        )

    if isinstance(value, bool | int | float):
        return value

    if isinstance(value, Path):
        return str(value)

    raw = getattr(value, "value", None)
    if raw is not None and isinstance(raw, str | int | float | bool):
        return raw

    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_audit_value(
                item_value,
                key=str(item_key),
                max_text_chars=max_text_chars,
            )
            for item_key, item_value in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [
            sanitize_audit_value(
                item,
                max_text_chars=max_text_chars,
            )
            for item in value
        ]

    return truncate_text(
        redact_secret_text(str(value)),
        max_chars=max_text_chars,
    )


def sanitize_arguments(
    arguments: Mapping[str, Any] | None,
    *,
    max_text_chars: int = DEFAULT_MAX_ARGUMENT_TEXT_CHARS,
) -> dict[str, Any]:
    if not arguments:
        return {}

    return {
        str(key): sanitize_audit_value(
            value,
            key=str(key),
            max_text_chars=max_text_chars,
        )
        for key, value in arguments.items()
    }


def merge_metadata(
    *items: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for item in items:
        if not item:
            continue

        for key, value in item.items():
            merged[str(key)] = sanitize_audit_value(value)

    return merged


def create_policy_decision_record(
    decision: PermissionDecision,
    *,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PermissionAuditRecord:
    """把 PermissionDecision 转成审计记录。"""
    request = decision.request

    return PermissionAuditRecord(
        audit_id=new_audit_id(),
        timestamp=utc_now_iso(),
        event_type=PermissionAuditEventType.POLICY_DECISION,
        tool_name=request.tool_name,
        action=request.action,
        mode=enum_value(decision.mode),
        risk=enum_value(decision.risk),
        decision=enum_value(decision.decision),
        allowed=decision.allowed,
        reason=decision.reason,
        call_id=request.call_id,
        session_id=session_id,
        user_action=None,
        arguments=sanitize_arguments(request.arguments),
        metadata=merge_metadata(
            request.metadata,
            decision.metadata,
            metadata,
        ),
    )


def create_user_decision_record(
    decision: PermissionDecision,
    *,
    user_action: PermissionAuditUserAction | str,
    session_id: str | None = None,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PermissionAuditRecord:
    """记录用户在审批弹窗里的选择。"""
    request = decision.request
    user_action_text = enum_value(user_action)

    return PermissionAuditRecord(
        audit_id=new_audit_id(),
        timestamp=utc_now_iso(),
        event_type=PermissionAuditEventType.USER_DECISION,
        tool_name=request.tool_name,
        action=request.action,
        mode=enum_value(decision.mode),
        risk=enum_value(decision.risk),
        decision=enum_value(decision.decision),
        allowed=user_action_text in {
            PermissionAuditUserAction.ALLOW.value,
            PermissionAuditUserAction.ALWAYS_ALLOW.value,
        },
        reason=reason or decision.reason,
        call_id=request.call_id,
        session_id=session_id,
        user_action=user_action_text,
        arguments=sanitize_arguments(request.arguments),
        metadata=merge_metadata(
            request.metadata,
            decision.metadata,
            metadata,
        ),
    )


def create_execution_result_record(
    decision: PermissionDecision,
    *,
    executed: bool,
    success: bool | None = None,
    exit_code: int | None = None,
    session_id: str | None = None,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PermissionAuditRecord:
    """记录权限通过后工具是否执行，以及执行结果摘要。"""
    request = decision.request

    return PermissionAuditRecord(
        audit_id=new_audit_id(),
        timestamp=utc_now_iso(),
        event_type=PermissionAuditEventType.EXECUTION_RESULT,
        tool_name=request.tool_name,
        action=request.action,
        mode=enum_value(decision.mode),
        risk=enum_value(decision.risk),
        decision=enum_value(decision.decision),
        allowed=executed,
        reason=reason or decision.reason,
        call_id=request.call_id,
        session_id=session_id,
        user_action=None,
        arguments=sanitize_arguments(request.arguments),
        metadata=merge_metadata(
            request.metadata,
            decision.metadata,
            {
                "executed": executed,
                "success": success,
                "exit_code": exit_code,
            },
            metadata,
        ),
    )


class PermissionAuditLog:
    """
    权限审计日志。

    默认写入：
        <workspace>/.pywork/audit/permissions.jsonl
    """

    def __init__(
        self,
        workspace_path: str | Path,
        *,
        audit_dir: str | Path | None = None,
        filename: str = DEFAULT_AUDIT_FILE,
    ) -> None:
        self.workspace_path = Path(workspace_path).expanduser().resolve()

        if audit_dir is None:
            self.audit_dir = self.workspace_path / DEFAULT_AUDIT_DIR
        else:
            raw_audit_dir = Path(audit_dir).expanduser()
            self.audit_dir = (
                raw_audit_dir
                if raw_audit_dir.is_absolute()
                else self.workspace_path / raw_audit_dir
            ).resolve()

        self.path = self.audit_dir / filename

    def ensure_dir(self) -> None:
        self.audit_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
        self,
        record: PermissionAuditRecord,
    ) -> PermissionAuditRecord:
        self.ensure_dir()

        try:
            with self.path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as file:
                file.write(record.to_json_line())
                file.write("\n")
        except OSError as exc:
            raise PermissionAuditStorageError(
                f"failed to append permission audit log: {self.path}"
            ) from exc

        return record

    def record_policy_decision(
        self,
        decision: PermissionDecision,
        *,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PermissionAuditRecord:
        return self.append(
            create_policy_decision_record(
                decision,
                session_id=session_id,
                metadata=metadata,
            )
        )

    def record_user_decision(
        self,
        decision: PermissionDecision,
        *,
        user_action: PermissionAuditUserAction | str,
        session_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PermissionAuditRecord:
        return self.append(
            create_user_decision_record(
                decision,
                user_action=user_action,
                session_id=session_id,
                reason=reason,
                metadata=metadata,
            )
        )

    def record_execution_result(
        self,
        decision: PermissionDecision,
        *,
        executed: bool,
        success: bool | None = None,
        exit_code: int | None = None,
        session_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PermissionAuditRecord:
        return self.append(
            create_execution_result_record(
                decision,
                executed=executed,
                success=success,
                exit_code=exit_code,
                session_id=session_id,
                reason=reason,
                metadata=metadata,
            )
        )

    def iter_records(
        self,
        *,
        strict: bool = False,
    ) -> Iterable[PermissionAuditRecord]:
        if not self.path.exists():
            return

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    text = line.strip()

                    if not text:
                        continue

                    try:
                        yield PermissionAuditRecord.from_json_line(text)
                    except Exception:
                        if strict:
                            raise

                        continue
        except OSError as exc:
            raise PermissionAuditStorageError(
                f"failed to read permission audit log: {self.path}"
            ) from exc

    def list_records(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        event_type: PermissionAuditEventType | str | None = None,
        tool_name: str | None = None,
        call_id: str | None = None,
        session_id: str | None = None,
    ) -> list[PermissionAuditRecord]:
        records = list(self.iter_records())

        if event_type is not None:
            event_type_value = enum_value(event_type)
            records = [
                record
                for record in records
                if record.event_type.value == event_type_value
            ]

        if tool_name is not None:
            normalized_tool_name = tool_name.strip().lower()
            records = [
                record
                for record in records
                if record.tool_name.strip().lower() == normalized_tool_name
            ]

        if call_id is not None:
            records = [
                record
                for record in records
                if record.call_id == call_id
            ]

        if session_id is not None:
            records = [
                record
                for record in records
                if record.session_id == session_id
            ]

        if reverse:
            records.reverse()

        if limit is not None:
            records = records[:limit]

        return records

    def find_by_call_id(
        self,
        call_id: str,
    ) -> list[PermissionAuditRecord]:
        return self.list_records(call_id=call_id)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def record_permission_decision(
    decision: PermissionDecision,
    *,
    workspace_path: str | Path,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PermissionAuditRecord:
    """快捷记录一次策略判断。"""
    return PermissionAuditLog(workspace_path).record_policy_decision(
        decision,
        session_id=session_id,
        metadata=metadata,
    )


def render_permission_audit_record(
    record: PermissionAuditRecord,
) -> str:
    """渲染审计记录，给日志 / 调试输出用。"""
    parts = [
        f"{record.timestamp}",
        f"event={record.event_type.value}",
        f"tool={record.tool_name}",
    ]

    if record.action:
        parts.append(f"action={record.action}")

    if record.mode:
        parts.append(f"mode={record.mode}")

    if record.risk:
        parts.append(f"risk={record.risk}")

    if record.decision:
        parts.append(f"decision={record.decision}")

    if record.user_action:
        parts.append(f"user_action={record.user_action}")

    if record.call_id:
        parts.append(f"call_id={record.call_id}")

    if record.reason:
        parts.append(f"reason={record.reason}")

    return " | ".join(parts)


def demo() -> None:
    from pywork.permission.policy import evaluate_permission

    workspace = Path.cwd()
    audit_log = PermissionAuditLog(workspace)

    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
            "content": "hello",
            "api_key": "sk-demo-secret-value",
        },
        call_id="demo_call_1",
    )

    record = audit_log.record_policy_decision(
        decision,
        session_id="demo_session",
    )

    print(render_permission_audit_record(record))
    print("audit path:", audit_log.path)


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())