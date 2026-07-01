from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """
    权限系统的统一风险等级。

    safe:
        基本无风险，例如 echo、纯内存操作。

    low:
        只读类操作，例如读取文件、搜索文件。

    medium:
        有一定副作用，或者可能消耗较多资源，但通常不直接破坏文件。

    high:
        会修改 workspace 文件，例如 file_write、file_edit。

    critical:
        可执行任意命令或高危系统操作，例如 bash、powershell。
    """

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


RISK_ALIASES: dict[str, RiskLevel] = {
    "safe": RiskLevel.SAFE,
    "none": RiskLevel.SAFE,
    "readonly": RiskLevel.LOW,
    "read_only": RiskLevel.LOW,
    "read-only": RiskLevel.LOW,
    "low": RiskLevel.LOW,
    "medium": RiskLevel.MEDIUM,
    "moderate": RiskLevel.MEDIUM,
    "high": RiskLevel.HIGH,
    "danger": RiskLevel.CRITICAL,
    "dangerous": RiskLevel.CRITICAL,
    "critical": RiskLevel.CRITICAL,
}


DEFAULT_UNKNOWN_TOOL_RISK = RiskLevel.MEDIUM


DEFAULT_TOOL_RISK_MAP: dict[str, RiskLevel] = {
    # safe
    "echo": RiskLevel.SAFE,

    # read-only tools
    "glob": RiskLevel.LOW,
    "grep": RiskLevel.LOW,
    "file_read": RiskLevel.LOW,
    "read": RiskLevel.LOW,
    "ls": RiskLevel.LOW,

    # write/edit tools
    "file_write": RiskLevel.HIGH,
    "file_edit": RiskLevel.HIGH,
    "write": RiskLevel.HIGH,
    "edit": RiskLevel.HIGH,

    # shell tools
    "bash": RiskLevel.CRITICAL,
    "shell": RiskLevel.CRITICAL,
    "powershell": RiskLevel.CRITICAL,
    "pwsh": RiskLevel.CRITICAL,
    "cmd": RiskLevel.CRITICAL,
}


@dataclass(slots=True, frozen=True)
class RiskInfo:
    """风险等级说明。"""

    level: RiskLevel
    score: int
    label: str
    description: str
    requires_confirmation: bool
    requires_elevated_confirmation: bool


RISK_INFO_MAP: dict[RiskLevel, RiskInfo] = {
    RiskLevel.SAFE: RiskInfo(
        level=RiskLevel.SAFE,
        score=RISK_ORDER[RiskLevel.SAFE],
        label="Safe",
        description="Harmless operation with no meaningful side effects.",
        requires_confirmation=False,
        requires_elevated_confirmation=False,
    ),
    RiskLevel.LOW: RiskInfo(
        level=RiskLevel.LOW,
        score=RISK_ORDER[RiskLevel.LOW],
        label="Low",
        description="Read-only or low-impact operation.",
        requires_confirmation=False,
        requires_elevated_confirmation=False,
    ),
    RiskLevel.MEDIUM: RiskInfo(
        level=RiskLevel.MEDIUM,
        score=RISK_ORDER[RiskLevel.MEDIUM],
        label="Medium",
        description="Operation with moderate side effects or resource usage.",
        requires_confirmation=True,
        requires_elevated_confirmation=False,
    ),
    RiskLevel.HIGH: RiskInfo(
        level=RiskLevel.HIGH,
        score=RISK_ORDER[RiskLevel.HIGH],
        label="High",
        description="Operation that may modify workspace files or persistent state.",
        requires_confirmation=True,
        requires_elevated_confirmation=False,
    ),
    RiskLevel.CRITICAL: RiskInfo(
        level=RiskLevel.CRITICAL,
        score=RISK_ORDER[RiskLevel.CRITICAL],
        label="Critical",
        description="Operation that may execute arbitrary commands or affect the system.",
        requires_confirmation=True,
        requires_elevated_confirmation=True,
    ),
}


def normalize_risk_level(
    value: RiskLevel | str | Any,
    *,
    default: RiskLevel | None = None,
) -> RiskLevel:
    """
    把外部传入的风险等级规范化成 RiskLevel。

    支持：
    - RiskLevel.HIGH
    - "high"
    - "dangerous" -> critical
    - ToolRiskLevel.DANGEROUS -> critical
    """
    if isinstance(value, RiskLevel):
        return value

    if value is None:
        if default is not None:
            return default

        raise ValueError("risk level cannot be None")

    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower().replace(" ", "_")

    if normalized in RISK_ALIASES:
        return RISK_ALIASES[normalized]

    if default is not None:
        return default

    raise ValueError(f"unknown risk level: {value!r}")


def risk_score(value: RiskLevel | str | Any) -> int:
    """返回风险等级分数，越大越危险。"""
    level = normalize_risk_level(value)

    return RISK_ORDER[level]


def get_risk_info(value: RiskLevel | str | Any) -> RiskInfo:
    """获取风险等级说明。"""
    level = normalize_risk_level(value)

    return RISK_INFO_MAP[level]


def compare_risk(
    left: RiskLevel | str | Any,
    right: RiskLevel | str | Any,
) -> int:
    """
    比较两个风险等级。

    返回：
        -1  left < right
         0  left == right
         1  left > right
    """
    left_score = risk_score(left)
    right_score = risk_score(right)

    if left_score < right_score:
        return -1

    if left_score > right_score:
        return 1

    return 0


def risk_at_least(
    value: RiskLevel | str | Any,
    threshold: RiskLevel | str | Any,
) -> bool:
    """判断 value 是否大于等于 threshold。"""
    return compare_risk(value, threshold) >= 0


def risk_above(
    value: RiskLevel | str | Any,
    threshold: RiskLevel | str | Any,
) -> bool:
    """判断 value 是否大于 threshold。"""
    return compare_risk(value, threshold) > 0


def risk_at_most(
    value: RiskLevel | str | Any,
    threshold: RiskLevel | str | Any,
) -> bool:
    """判断 value 是否小于等于 threshold。"""
    return compare_risk(value, threshold) <= 0


def max_risk(
    *values: RiskLevel | str | Any,
    default: RiskLevel = RiskLevel.SAFE,
) -> RiskLevel:
    """返回多个风险等级里最高的一个。"""
    if not values:
        return default

    levels = [
        normalize_risk_level(
            value,
            default=default,
        )
        for value in values
    ]

    return max(
        levels,
        key=lambda level: RISK_ORDER[level],
    )


def min_risk(
    *values: RiskLevel | str | Any,
    default: RiskLevel = RiskLevel.SAFE,
) -> RiskLevel:
    """返回多个风险等级里最低的一个。"""
    if not values:
        return default

    levels = [
        normalize_risk_level(
            value,
            default=default,
        )
        for value in values
    ]

    return min(
        levels,
        key=lambda level: RISK_ORDER[level],
    )


def risk_requires_confirmation(value: RiskLevel | str | Any) -> bool:
    """是否需要普通确认。"""
    return get_risk_info(value).requires_confirmation


def risk_requires_elevated_confirmation(value: RiskLevel | str | Any) -> bool:
    """是否需要高风险确认。"""
    return get_risk_info(value).requires_elevated_confirmation


def normalize_tool_name(tool_name: str) -> str:
    """规范化工具名。"""
    return str(tool_name).strip().lower().replace("-", "_")


def get_tool_risk(
    tool_name: str,
    *,
    default: RiskLevel = DEFAULT_UNKNOWN_TOOL_RISK,
    extra_mapping: dict[str, RiskLevel | str] | None = None,
) -> RiskLevel:
    """
    根据工具名获取默认风险等级。

    extra_mapping 可以让 policy.py 或配置覆盖默认映射。
    """
    normalized_name = normalize_tool_name(tool_name)

    if extra_mapping:
        normalized_extra = {
            normalize_tool_name(name): normalize_risk_level(level)
            for name, level in extra_mapping.items()
        }

        if normalized_name in normalized_extra:
            return normalized_extra[normalized_name]

    return DEFAULT_TOOL_RISK_MAP.get(
        normalized_name,
        default,
    )


def get_tool_risk_from_object(
    tool: Any,
    *,
    default: RiskLevel = DEFAULT_UNKNOWN_TOOL_RISK,
) -> RiskLevel:
    """
    从工具对象上读取风险等级。

    兼容：
    - tool.risk_level = RiskLevel.HIGH
    - tool.risk_level = ToolRiskLevel.DANGEROUS
    - tool.name = "file_write"
    """
    raw_risk = getattr(
        tool,
        "risk_level",
        None,
    )

    if raw_risk is not None:
        try:
            return normalize_risk_level(
                raw_risk,
                default=default,
            )
        except ValueError:
            return default

    raw_name = getattr(
        tool,
        "name",
        None,
    )

    if raw_name:
        return get_tool_risk(
            str(raw_name),
            default=default,
        )

    return default


def risk_to_dict(value: RiskLevel | str | Any) -> dict[str, Any]:
    """把风险等级转成可序列化字典。"""
    info = get_risk_info(value)

    return {
        "level": info.level.value,
        "score": info.score,
        "label": info.label,
        "description": info.description,
        "requires_confirmation": info.requires_confirmation,
        "requires_elevated_confirmation": info.requires_elevated_confirmation,
    }


def list_risk_infos() -> list[RiskInfo]:
    """按风险从低到高列出所有等级说明。"""
    return [
        RISK_INFO_MAP[level]
        for level in sorted(
            RISK_ORDER,
            key=lambda item: RISK_ORDER[item],
        )
    ]


def render_risk_label(value: RiskLevel | str | Any) -> str:
    """渲染简短风险标签。"""
    info = get_risk_info(value)

    return f"{info.label}({info.level.value})"


def demo() -> None:
    for info in list_risk_infos():
        print(
            info.level.value,
            info.score,
            info.label,
            "confirm=",
            info.requires_confirmation,
            "elevated=",
            info.requires_elevated_confirmation,
        )

    print("file_read:", get_tool_risk("file_read").value)
    print("file_write:", get_tool_risk("file_write").value)
    print("bash:", get_tool_risk("bash").value)


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())