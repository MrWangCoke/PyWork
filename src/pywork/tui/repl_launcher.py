from __future__ import annotations

from pathlib import Path
from typing import Any

from pywork.tui.app import PyWorkApp


def _get_attr_or_key(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    同时支持对象属性和 dict key。

    这样 repl_launcher 可以兼容：
    - WorkspaceInfo 对象
    - dict
    - 普通 Path / str
    """
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def _extract_config(config: Any) -> dict[str, Any]:
    """
    从不同形式的 config 对象里提取 dict。

    兼容：
    - dict
    - ConfigLoadResult(config=...)
    - None
    """
    if config is None:
        return {}

    if isinstance(config, dict):
        return config

    loaded_config = getattr(config, "config", None)

    if isinstance(loaded_config, dict):
        return loaded_config

    return {}


def _extract_workspace_path(workspace: Any) -> str:
    """
    从 workspace 对象中提取 workspace_path。
    """
    if workspace is None:
        return "."

    if isinstance(workspace, str | Path):
        return str(workspace)

    for field_name in (
        "workspace_path",
        "path",
        "root",
        "project_root",
    ):
        value = _get_attr_or_key(workspace, field_name)

        if value:
            return str(value)

    return "."


def _extract_project_root(workspace: Any) -> str | None:
    """
    从 workspace 对象中提取 project_root。
    """
    if workspace is None:
        return None

    if isinstance(workspace, str | Path):
        return str(workspace)

    for field_name in (
        "project_root",
        "root",
        "workspace_path",
        "path",
    ):
        value = _get_attr_or_key(workspace, field_name)

        if value:
            return str(value)

    return None


def launch_repl(
    workspace: Any = None,
    config: Any = None,
    *,
    workspace_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> int:
    """
    启动 PyWork TUI。

    这个函数是 CLI 层进入 TUI 的统一入口。

    entrypoints/cli.py 里会调用：

        launch_repl(workspace=workspace, config=config)

    所以这里要负责把 workspace/config 转换成 PyWorkApp 能使用的参数。
    """
    resolved_workspace_path = str(
        workspace_path
        if workspace_path is not None
        else _extract_workspace_path(workspace)
    )

    resolved_project_root = (
        str(project_root)
        if project_root is not None
        else _extract_project_root(workspace)
    )

    resolved_config = _extract_config(config)

    app = PyWorkApp(
        workspace_path=resolved_workspace_path,
        project_root=resolved_project_root,
        config=resolved_config,
    )

    app.run()

    return 0


def main() -> int:
    """
    方便直接测试：

        uv run python -m pywork.tui.repl_launcher
    """
    return launch_repl(
        workspace_path=".",
        project_root=".",
        config={
            "default": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
            },
            "permissions": {
                "mode": "default",
            },
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())