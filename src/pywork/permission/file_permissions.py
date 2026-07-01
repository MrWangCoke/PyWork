from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.permission.policy import PermissionDecisionType
from pywork.permission.risk import RiskLevel, risk_score


class FilePermissionError(Exception):
    """文件权限基础异常。"""


class FilePermissionValidationError(FilePermissionError):
    """文件权限参数异常。"""


class FileOperation(str, Enum):
    READ = "read"
    LIST = "list"
    SEARCH = "search"
    CREATE = "create"
    WRITE = "write"
    OVERWRITE = "overwrite"
    EDIT = "edit"
    DELETE = "delete"
    MOVE = "move"
    RENAME = "rename"


READ_OPERATIONS: set[FileOperation] = {
    FileOperation.READ,
    FileOperation.LIST,
    FileOperation.SEARCH,
}


WRITE_OPERATIONS: set[FileOperation] = {
    FileOperation.CREATE,
    FileOperation.WRITE,
    FileOperation.OVERWRITE,
    FileOperation.EDIT,
}


DESTRUCTIVE_OPERATIONS: set[FileOperation] = {
    FileOperation.DELETE,
    FileOperation.MOVE,
    FileOperation.RENAME,
}


DENIED_DIR_NAMES: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".pywork/file_history",
    ".pywork/sessions",
    ".pywork/tmp",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".uv-cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
}


SENSITIVE_FILE_NAMES: set[str] = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
}


SENSITIVE_SUFFIXES: set[str] = {
    ".pem",
    ".key",
    ".crt",
    ".cer",
    ".p12",
    ".pfx",
}


IMPORTANT_PROJECT_FILES: set[str] = {
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    "pytest.ini",
    "README.md",
    "PYWORK.md",
}


CRITICAL_SYSTEM_LIKE_NAMES: set[str] = {
    "passwd",
    "shadow",
    "sudoers",
    "hosts",
}


@dataclass(slots=True, frozen=True)
class FilePermissionRequest:
    path: str
    operation: FileOperation
    workspace_path: str | Path
    target_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FilePermissionResult:
    path: str
    operation: FileOperation
    workspace_path: str
    absolute_path: str

    decision: PermissionDecisionType
    risk: RiskLevel
    reason: str

    matched_rules: tuple[str, ...] = ()
    target_path: str | None = None
    absolute_target_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecisionType.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == PermissionDecisionType.DENY

    @property
    def should_ask(self) -> bool:
        return self.decision in {
            PermissionDecisionType.ASK,
            PermissionDecisionType.ASK_ELEVATED,
        }

    @property
    def requires_elevated_confirmation(self) -> bool:
        return self.decision == PermissionDecisionType.ASK_ELEVATED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operation"] = self.operation.value
        data["decision"] = self.decision.value
        data["risk"] = self.risk.value
        return data


def normalize_file_operation(operation: FileOperation | str) -> FileOperation:
    if isinstance(operation, FileOperation):
        return operation

    value = str(operation).strip().lower().replace("-", "_")

    aliases = {
        "read_file": FileOperation.READ,
        "file_read": FileOperation.READ,
        "ls": FileOperation.LIST,
        "glob": FileOperation.LIST,
        "grep": FileOperation.SEARCH,
        "search": FileOperation.SEARCH,
        "create_file": FileOperation.CREATE,
        "write_file": FileOperation.WRITE,
        "file_write": FileOperation.WRITE,
        "modify": FileOperation.EDIT,
        "edit_file": FileOperation.EDIT,
        "file_edit": FileOperation.EDIT,
        "remove": FileOperation.DELETE,
        "rm": FileOperation.DELETE,
        "delete_file": FileOperation.DELETE,
        "mv": FileOperation.MOVE,
        "move_file": FileOperation.MOVE,
        "rename_file": FileOperation.RENAME,
    }

    if value in aliases:
        return aliases[value]

    try:
        return FileOperation(value)
    except ValueError as exc:
        raise FilePermissionValidationError(
            f"unknown file operation: {operation!r}"
        ) from exc


def normalize_path_text(path: str | Path) -> str:
    return str(path).strip()


def resolve_workspace_path(workspace_path: str | Path) -> Path:
    return Path(workspace_path).expanduser().resolve()


def resolve_path_in_workspace(
    path: str | Path,
    workspace_path: str | Path,
) -> Path:
    workspace = resolve_workspace_path(workspace_path)
    raw = Path(normalize_path_text(path)).expanduser()

    if raw.is_absolute():
        return raw.resolve()

    return (workspace / raw).resolve()


def path_is_inside_workspace(
    path: Path,
    workspace_path: Path,
) -> bool:
    try:
        path.relative_to(workspace_path)
        return True
    except ValueError:
        return False


def relative_parts(
    path: Path,
    workspace_path: Path,
) -> tuple[str, ...]:
    try:
        relative = path.relative_to(workspace_path)
    except ValueError:
        return tuple(part.lower() for part in path.parts)

    return tuple(part.lower() for part in relative.parts)


def normalized_relative_text(
    path: Path,
    workspace_path: Path,
) -> str:
    parts = relative_parts(path, workspace_path)

    return "/".join(parts)


def path_has_denied_dir(
    path: Path,
    workspace_path: Path,
) -> bool:
    parts = relative_parts(path, workspace_path)
    relative_text = normalized_relative_text(path, workspace_path)

    for denied in DENIED_DIR_NAMES:
        denied_lower = denied.lower()

        if "/" in denied_lower:
            if relative_text == denied_lower or relative_text.startswith(
                denied_lower + "/"
            ):
                return True
        elif denied_lower in parts:
            return True

    return False


def matched_denied_dir(
    path: Path,
    workspace_path: Path,
) -> str | None:
    parts = relative_parts(path, workspace_path)
    relative_text = normalized_relative_text(path, workspace_path)

    for denied in DENIED_DIR_NAMES:
        denied_lower = denied.lower()

        if "/" in denied_lower:
            if relative_text == denied_lower or relative_text.startswith(
                denied_lower + "/"
            ):
                return denied
        elif denied_lower in parts:
            return denied

    return None


def is_sensitive_file(path: Path) -> bool:
    name = path.name.lower()

    if name in SENSITIVE_FILE_NAMES:
        return True

    if name.startswith(".env."):
        return True

    return path.suffix.lower() in SENSITIVE_SUFFIXES


def is_important_project_file(path: Path) -> bool:
    return path.name in IMPORTANT_PROJECT_FILES


def is_critical_system_like_file(path: Path) -> bool:
    return path.name.lower() in CRITICAL_SYSTEM_LIKE_NAMES


def stronger_decision(
    left: PermissionDecisionType,
    right: PermissionDecisionType,
) -> PermissionDecisionType:
    order = {
        PermissionDecisionType.ALLOW: 0,
        PermissionDecisionType.ASK: 1,
        PermissionDecisionType.ASK_ELEVATED: 2,
        PermissionDecisionType.DENY: 3,
    }

    return left if order[left] >= order[right] else right


def max_file_risk(
    left: RiskLevel,
    right: RiskLevel,
) -> RiskLevel:
    return left if risk_score(left) >= risk_score(right) else right


def base_file_decision(operation: FileOperation) -> PermissionDecisionType:
    if operation in READ_OPERATIONS:
        return PermissionDecisionType.ALLOW

    if operation in WRITE_OPERATIONS:
        return PermissionDecisionType.ASK

    if operation in DESTRUCTIVE_OPERATIONS:
        return PermissionDecisionType.ASK_ELEVATED

    return PermissionDecisionType.ASK


def base_file_risk(operation: FileOperation) -> RiskLevel:
    if operation in READ_OPERATIONS:
        return RiskLevel.LOW

    if operation in WRITE_OPERATIONS:
        return RiskLevel.HIGH

    if operation in DESTRUCTIVE_OPERATIONS:
        return RiskLevel.CRITICAL

    return RiskLevel.MEDIUM


class FilePermissionMatrix:
    """
    文件权限矩阵。

    规则优先级从高到低：

    1. workspace 外路径直接 deny
    2. .git / .pywork/file_history 等保护目录写入直接 deny
    3. .env / 私钥 / 证书类敏感文件 ask_elevated
    4. pyproject.toml / lockfile / README 等关键项目文件写入 ask_elevated
    5. 普通读/list/search allow
    6. 普通写/edit ask
    7. delete/move/rename ask_elevated
    """

    def evaluate(
        self,
        request: FilePermissionRequest,
    ) -> FilePermissionResult:
        operation = normalize_file_operation(request.operation)
        workspace = resolve_workspace_path(request.workspace_path)
        absolute_path = resolve_path_in_workspace(
            request.path,
            workspace,
        )

        absolute_target_path: Path | None = None

        if request.target_path is not None:
            absolute_target_path = resolve_path_in_workspace(
                request.target_path,
                workspace,
            )

        matched_rules: list[str] = []
        metadata = dict(request.metadata)

        if not path_is_inside_workspace(absolute_path, workspace):
            return FilePermissionResult(
                path=request.path,
                operation=operation,
                workspace_path=str(workspace),
                absolute_path=str(absolute_path),
                decision=PermissionDecisionType.DENY,
                risk=RiskLevel.CRITICAL,
                reason="path is outside workspace",
                matched_rules=("outside_workspace",),
                target_path=request.target_path,
                absolute_target_path=(
                    str(absolute_target_path)
                    if absolute_target_path is not None
                    else None
                ),
                metadata=metadata,
            )

        if absolute_target_path is not None and not path_is_inside_workspace(
            absolute_target_path,
            workspace,
        ):
            return FilePermissionResult(
                path=request.path,
                operation=operation,
                workspace_path=str(workspace),
                absolute_path=str(absolute_path),
                decision=PermissionDecisionType.DENY,
                risk=RiskLevel.CRITICAL,
                reason="target path is outside workspace",
                matched_rules=("target_outside_workspace",),
                target_path=request.target_path,
                absolute_target_path=str(absolute_target_path),
                metadata=metadata,
            )

        denied_dir = matched_denied_dir(
            absolute_path,
            workspace,
        )

        if denied_dir is not None:
            if operation in WRITE_OPERATIONS or operation in DESTRUCTIVE_OPERATIONS:
                return FilePermissionResult(
                    path=request.path,
                    operation=operation,
                    workspace_path=str(workspace),
                    absolute_path=str(absolute_path),
                    decision=PermissionDecisionType.DENY,
                    risk=RiskLevel.CRITICAL,
                    reason=f"operation targets protected directory: {denied_dir}",
                    matched_rules=("protected_directory", f"dir:{denied_dir}"),
                    target_path=request.target_path,
                    absolute_target_path=(
                        str(absolute_target_path)
                        if absolute_target_path is not None
                        else None
                    ),
                    metadata=metadata,
                )

            return FilePermissionResult(
                path=request.path,
                operation=operation,
                workspace_path=str(workspace),
                absolute_path=str(absolute_path),
                decision=PermissionDecisionType.ASK_ELEVATED,
                risk=RiskLevel.CRITICAL,
                reason=f"read/search targets protected directory: {denied_dir}",
                matched_rules=("protected_directory_read", f"dir:{denied_dir}"),
                target_path=request.target_path,
                absolute_target_path=(
                    str(absolute_target_path)
                    if absolute_target_path is not None
                    else None
                ),
                metadata=metadata,
            )

        if absolute_target_path is not None:
            target_denied_dir = matched_denied_dir(
                absolute_target_path,
                workspace,
            )

            if target_denied_dir is not None:
                return FilePermissionResult(
                    path=request.path,
                    operation=operation,
                    workspace_path=str(workspace),
                    absolute_path=str(absolute_path),
                    decision=PermissionDecisionType.DENY,
                    risk=RiskLevel.CRITICAL,
                    reason=f"target path uses protected directory: {target_denied_dir}",
                    matched_rules=(
                        "target_protected_directory",
                        f"dir:{target_denied_dir}",
                    ),
                    target_path=request.target_path,
                    absolute_target_path=str(absolute_target_path),
                    metadata=metadata,
                )

        if is_critical_system_like_file(absolute_path):
            return FilePermissionResult(
                path=request.path,
                operation=operation,
                workspace_path=str(workspace),
                absolute_path=str(absolute_path),
                decision=PermissionDecisionType.DENY,
                risk=RiskLevel.CRITICAL,
                reason="operation targets critical system-like file name",
                matched_rules=("critical_system_like_file",),
                target_path=request.target_path,
                absolute_target_path=(
                    str(absolute_target_path)
                    if absolute_target_path is not None
                    else None
                ),
                metadata=metadata,
            )

        if is_sensitive_file(absolute_path):
            return FilePermissionResult(
                path=request.path,
                operation=operation,
                workspace_path=str(workspace),
                absolute_path=str(absolute_path),
                decision=PermissionDecisionType.ASK_ELEVATED,
                risk=RiskLevel.CRITICAL,
                reason="operation targets sensitive file",
                matched_rules=("sensitive_file", f"file:{absolute_path.name}"),
                target_path=request.target_path,
                absolute_target_path=(
                    str(absolute_target_path)
                    if absolute_target_path is not None
                    else None
                ),
                metadata=metadata,
            )

        if operation in WRITE_OPERATIONS and is_important_project_file(absolute_path):
            return FilePermissionResult(
                path=request.path,
                operation=operation,
                workspace_path=str(workspace),
                absolute_path=str(absolute_path),
                decision=PermissionDecisionType.ASK_ELEVATED,
                risk=RiskLevel.CRITICAL,
                reason="write/edit targets important project file",
                matched_rules=("important_project_file", f"file:{absolute_path.name}"),
                target_path=request.target_path,
                absolute_target_path=(
                    str(absolute_target_path)
                    if absolute_target_path is not None
                    else None
                ),
                metadata=metadata,
            )

        base_decision = base_file_decision(operation)
        base_risk = base_file_risk(operation)

        matched_rules.append(f"operation:{operation.value}")

        return FilePermissionResult(
            path=request.path,
            operation=operation,
            workspace_path=str(workspace),
            absolute_path=str(absolute_path),
            decision=base_decision,
            risk=base_risk,
            reason=f"{operation.value} operation uses default file permission rule",
            matched_rules=tuple(matched_rules),
            target_path=request.target_path,
            absolute_target_path=(
                str(absolute_target_path)
                if absolute_target_path is not None
                else None
            ),
            metadata=metadata,
        )


def evaluate_file_permission(
    path: str | Path,
    *,
    operation: FileOperation | str,
    workspace_path: str | Path,
    target_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> FilePermissionResult:
    request = FilePermissionRequest(
        path=str(path),
        operation=normalize_file_operation(operation),
        workspace_path=workspace_path,
        target_path=str(target_path) if target_path is not None else None,
        metadata=metadata or {},
    )

    return FilePermissionMatrix().evaluate(request)


def render_file_permission_result(
    result: FilePermissionResult,
) -> str:
    matched = ", ".join(result.matched_rules) or "(none)"

    return (
        f"{result.decision.value}: "
        f"path={result.path} "
        f"operation={result.operation.value} "
        f"risk={result.risk.value} "
        f"reason={result.reason} "
        f"matched={matched}"
    )


def demo() -> None:
    workspace = Path.cwd()

    examples = [
        ("src/utils/helper.py", "write"),
        (".env", "write"),
        (".git/config", "edit"),
        ("../outside.txt", "read"),
        ("pyproject.toml", "edit"),
        ("README.md", "read"),
    ]

    for path, operation in examples:
        result = evaluate_file_permission(
            path,
            operation=operation,
            workspace_path=workspace,
        )

        print(render_file_permission_result(result))


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())