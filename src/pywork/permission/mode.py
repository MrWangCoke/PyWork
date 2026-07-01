from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pywork.permission.risk import (
    RiskLevel,
    normalize_risk_level,
    risk_at_most,
    risk_score,
)


class PermissionMode(str, Enum):
    """
    PyWork 权限模式。

    default:
        默认模式。读类工具自动允许，高风险操作需要确认。

    readonly:
        只读模式。只允许读取、搜索、分析，不允许写文件或执行 shell。

    plan:
        计划模式。允许读取和分析，但不允许真正修改文件或执行命令。
        适合“先给我方案，不要动代码”。

    accept_edits:
        接受编辑模式。允许读工具和文件编辑工具自动执行。
        但 shell / powershell 这种 critical 操作仍然不自动放行。

    bypass:
        跳过权限模式。所有操作默认允许。
        这个模式非常危险，只应该在用户明确授权时使用。
    """

    DEFAULT = "default"
    READONLY = "readonly"
    PLAN = "plan"
    ACCEPT_EDITS = "accept_edits"
    BYPASS = "bypass"


MODE_ALIASES: dict[str, PermissionMode] = {
    "default": PermissionMode.DEFAULT,
    "normal": PermissionMode.DEFAULT,

    "readonly": PermissionMode.READONLY,
    "read_only": PermissionMode.READONLY,
    "read-only": PermissionMode.READONLY,
    "ro": PermissionMode.READONLY,

    "plan": PermissionMode.PLAN,
    "planning": PermissionMode.PLAN,
    "dry_run": PermissionMode.PLAN,
    "dry-run": PermissionMode.PLAN,

    "accept_edits": PermissionMode.ACCEPT_EDITS,
    "accept-edits": PermissionMode.ACCEPT_EDITS,
    "accept edits": PermissionMode.ACCEPT_EDITS,
    "auto_edit": PermissionMode.ACCEPT_EDITS,
    "auto-edit": PermissionMode.ACCEPT_EDITS,

    "bypass": PermissionMode.BYPASS,
    "unsafe": PermissionMode.BYPASS,
    "dangerous": PermissionMode.BYPASS,
    "no_confirm": PermissionMode.BYPASS,
    "no-confirm": PermissionMode.BYPASS,
}


@dataclass(slots=True, frozen=True)
class PermissionModeInfo:
    """权限模式说明。"""

    mode: PermissionMode
    label: str
    description: str

    auto_allow_max_risk: RiskLevel
    deny_above_risk: RiskLevel | None = None

    allows_read: bool = True
    allows_write: bool = False
    allows_shell: bool = False

    requires_confirmation: bool = True
    is_readonly: bool = False
    is_planning: bool = False
    is_bypass: bool = False


MODE_INFO_MAP: dict[PermissionMode, PermissionModeInfo] = {
    PermissionMode.DEFAULT: PermissionModeInfo(
        mode=PermissionMode.DEFAULT,
        label="Default",
        description=(
            "Default mode. Read-only operations are allowed automatically; "
            "write operations and shell commands require confirmation."
        ),
        auto_allow_max_risk=RiskLevel.LOW,
        deny_above_risk=None,
        allows_read=True,
        allows_write=False,
        allows_shell=False,
        requires_confirmation=True,
    ),
    PermissionMode.READONLY: PermissionModeInfo(
        mode=PermissionMode.READONLY,
        label="Readonly",
        description=(
            "Readonly mode. Only safe and read-only operations are allowed. "
            "File edits, writes, and shell commands are denied."
        ),
        auto_allow_max_risk=RiskLevel.LOW,
        deny_above_risk=RiskLevel.LOW,
        allows_read=True,
        allows_write=False,
        allows_shell=False,
        requires_confirmation=False,
        is_readonly=True,
    ),
    PermissionMode.PLAN: PermissionModeInfo(
        mode=PermissionMode.PLAN,
        label="Plan",
        description=(
            "Plan mode. PyWork may inspect and analyze, but should not modify "
            "files or execute shell commands. It is intended for planning only."
        ),
        auto_allow_max_risk=RiskLevel.LOW,
        deny_above_risk=RiskLevel.LOW,
        allows_read=True,
        allows_write=False,
        allows_shell=False,
        requires_confirmation=False,
        is_planning=True,
    ),
    PermissionMode.ACCEPT_EDITS: PermissionModeInfo(
        mode=PermissionMode.ACCEPT_EDITS,
        label="Accept Edits",
        description=(
            "Accept edits mode. Read operations and workspace file edits may be "
            "allowed automatically. Critical shell commands still require elevated "
            "confirmation."
        ),
        auto_allow_max_risk=RiskLevel.HIGH,
        deny_above_risk=None,
        allows_read=True,
        allows_write=True,
        allows_shell=False,
        requires_confirmation=True,
    ),
    PermissionMode.BYPASS: PermissionModeInfo(
        mode=PermissionMode.BYPASS,
        label="Bypass",
        description=(
            "Bypass mode. All permission checks are bypassed. This is dangerous "
            "and should only be used with explicit user approval."
        ),
        auto_allow_max_risk=RiskLevel.CRITICAL,
        deny_above_risk=None,
        allows_read=True,
        allows_write=True,
        allows_shell=True,
        requires_confirmation=False,
        is_bypass=True,
    ),
}


DEFAULT_PERMISSION_MODE = PermissionMode.DEFAULT


def normalize_permission_mode(
    value: PermissionMode | str | Any,
    *,
    default: PermissionMode | None = None,
) -> PermissionMode:
    """
    把外部传入的模式规范化成 PermissionMode。

    支持：
    - PermissionMode.DEFAULT
    - "default"
    - "read-only" -> readonly
    - "accept edits" -> accept_edits
    """
    if isinstance(value, PermissionMode):
        return value

    if value is None:
        if default is not None:
            return default

        raise ValueError("permission mode cannot be None")

    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower().replace(" ", "_")

    if normalized in MODE_ALIASES:
        return MODE_ALIASES[normalized]

    if default is not None:
        return default

    raise ValueError(f"unknown permission mode: {value!r}")


def get_permission_mode_info(
    value: PermissionMode | str | Any,
) -> PermissionModeInfo:
    """获取权限模式说明。"""
    mode = normalize_permission_mode(value)

    return MODE_INFO_MAP[mode]


def list_permission_mode_infos() -> list[PermissionModeInfo]:
    """列出所有权限模式说明。"""
    return [
        MODE_INFO_MAP[PermissionMode.DEFAULT],
        MODE_INFO_MAP[PermissionMode.READONLY],
        MODE_INFO_MAP[PermissionMode.PLAN],
        MODE_INFO_MAP[PermissionMode.ACCEPT_EDITS],
        MODE_INFO_MAP[PermissionMode.BYPASS],
    ]


def permission_mode_to_dict(
    value: PermissionMode | str | Any,
) -> dict[str, Any]:
    """把权限模式转成可序列化字典。"""
    info = get_permission_mode_info(value)

    return {
        "mode": info.mode.value,
        "label": info.label,
        "description": info.description,
        "auto_allow_max_risk": info.auto_allow_max_risk.value,
        "deny_above_risk": (
            info.deny_above_risk.value
            if info.deny_above_risk is not None
            else None
        ),
        "allows_read": info.allows_read,
        "allows_write": info.allows_write,
        "allows_shell": info.allows_shell,
        "requires_confirmation": info.requires_confirmation,
        "is_readonly": info.is_readonly,
        "is_planning": info.is_planning,
        "is_bypass": info.is_bypass,
    }


def render_permission_mode_label(
    value: PermissionMode | str | Any,
) -> str:
    """渲染简短模式标签。"""
    info = get_permission_mode_info(value)

    return f"{info.label}({info.mode.value})"


def mode_is_bypass(
    value: PermissionMode | str | Any,
) -> bool:
    return get_permission_mode_info(value).is_bypass


def mode_is_readonly(
    value: PermissionMode | str | Any,
) -> bool:
    return get_permission_mode_info(value).is_readonly


def mode_is_planning(
    value: PermissionMode | str | Any,
) -> bool:
    return get_permission_mode_info(value).is_planning


def mode_allows_read(
    value: PermissionMode | str | Any,
) -> bool:
    return get_permission_mode_info(value).allows_read


def mode_allows_write(
    value: PermissionMode | str | Any,
) -> bool:
    return get_permission_mode_info(value).allows_write


def mode_allows_shell(
    value: PermissionMode | str | Any,
) -> bool:
    return get_permission_mode_info(value).allows_shell


def get_mode_auto_allow_max_risk(
    value: PermissionMode | str | Any,
) -> RiskLevel:
    """获取该模式下自动允许的最高风险等级。"""
    return get_permission_mode_info(value).auto_allow_max_risk


def get_mode_deny_above_risk(
    value: PermissionMode | str | Any,
) -> RiskLevel | None:
    """
    获取该模式下超过哪个风险等级就直接拒绝。

    readonly / plan:
        deny_above_risk = low
        表示 medium / high / critical 直接拒绝。

    default / accept_edits / bypass:
        None
        表示不在 mode.py 里直接拒绝，由 policy.py 继续判断 ask / allow。
    """
    return get_permission_mode_info(value).deny_above_risk


def mode_auto_allows_risk(
    mode: PermissionMode | str | Any,
    risk: RiskLevel | str | Any,
) -> bool:
    """
    判断某个模式是否自动允许某个风险等级。

    例子：
        default + low      -> True
        default + high     -> False
        accept_edits + high -> True
        bypass + critical  -> True
    """
    info = get_permission_mode_info(mode)
    risk_level = normalize_risk_level(risk)

    return risk_at_most(
        risk_level,
        info.auto_allow_max_risk,
    )


def mode_denies_risk(
    mode: PermissionMode | str | Any,
    risk: RiskLevel | str | Any,
) -> bool:
    """
    判断某个模式是否直接拒绝某个风险等级。

    readonly / plan 会拒绝 low 以上风险。
    """
    info = get_permission_mode_info(mode)
    risk_level = normalize_risk_level(risk)

    if info.deny_above_risk is None:
        return False

    return risk_score(risk_level) > risk_score(info.deny_above_risk)


def mode_may_ask_for_risk(
    mode: PermissionMode | str | Any,
    risk: RiskLevel | str | Any,
) -> bool:
    """
    判断该模式遇到该风险时是否可能进入 ask / ask_elevated。

    这个不是最终决策，只是给 policy.py 使用的辅助判断。
    """
    info = get_permission_mode_info(mode)

    if info.is_bypass:
        return False

    if mode_denies_risk(
        info.mode,
        risk,
    ):
        return False

    if mode_auto_allows_risk(
        info.mode,
        risk,
    ):
        return False

    return info.requires_confirmation


def choose_stricter_mode(
    left: PermissionMode | str | Any,
    right: PermissionMode | str | Any,
) -> PermissionMode:
    """
    从两个模式里选择更严格的一个。

    严格程度大致为：
        readonly / plan > default > accept_edits > bypass

    readonly 和 plan 都很严格，但语义不同。
    这里如果二者比较，readonly 更严格。
    """
    left_mode = normalize_permission_mode(left)
    right_mode = normalize_permission_mode(right)

    order: dict[PermissionMode, int] = {
        PermissionMode.BYPASS: 0,
        PermissionMode.ACCEPT_EDITS: 1,
        PermissionMode.DEFAULT: 2,
        PermissionMode.PLAN: 3,
        PermissionMode.READONLY: 4,
    }

    return left_mode if order[left_mode] >= order[right_mode] else right_mode


def choose_looser_mode(
    left: PermissionMode | str | Any,
    right: PermissionMode | str | Any,
) -> PermissionMode:
    """
    从两个模式里选择更宽松的一个。

    宽松程度大致为：
        bypass > accept_edits > default > plan > readonly
    """
    left_mode = normalize_permission_mode(left)
    right_mode = normalize_permission_mode(right)

    order: dict[PermissionMode, int] = {
        PermissionMode.BYPASS: 0,
        PermissionMode.ACCEPT_EDITS: 1,
        PermissionMode.DEFAULT: 2,
        PermissionMode.PLAN: 3,
        PermissionMode.READONLY: 4,
    }

    return left_mode if order[left_mode] <= order[right_mode] else right_mode


def demo() -> None:
    for info in list_permission_mode_infos():
        print(
            info.mode.value,
            info.label,
            "auto<=",
            info.auto_allow_max_risk.value,
            "deny_above=",
            info.deny_above_risk.value if info.deny_above_risk else None,
            "write=",
            info.allows_write,
            "shell=",
            info.allows_shell,
        )


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())