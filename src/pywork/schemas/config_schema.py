from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


PermissionMode = Literal[
    "default",
    "accept_edits",
    "bypass_permissions",
    "plan",
]

ModelProvider = Literal[
    "openai",
    "anthropic",
    "openai_compatible",
    "deepseek",
    "local",
]


class AppConfig(BaseModel):
    name: str = "pywork"
    version: str = "0.1.0"
    default_workspace: str = "."
    permission_mode: PermissionMode = "default"
    log_level: str = "INFO"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("app.name cannot be empty")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in allowed:
            raise ValueError(f"app.log_level must be one of {sorted(allowed)}")
        return value


class DefaultModelConfig(BaseModel):
    provider: ModelProvider = "openai_compatible"
    model: str = "deepseek-chat"
    base_url: str | None = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: int = 120
    max_retries: int = 2

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("default.model cannot be empty")
        return value

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("default.api_key_env cannot be empty")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("default.timeout_seconds must be greater than 0")
        return value

    @field_validator("max_retries")
    @classmethod
    def validate_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("default.max_retries cannot be negative")
        return value


class PermissionsConfig(BaseModel):
    mode: PermissionMode = "default"
    allow_read: bool = True
    require_approval_for_write: bool = True
    require_approval_for_shell: bool = True
    require_approval_for_network: bool = True


class RiskConfig(BaseModel):
    allow_read: bool = True
    require_approval_for_write: bool = True
    require_approval_for_shell: bool = True
    require_approval_for_network: bool = True
    allow_destructive_commands: bool = False
    allow_outside_workspace: bool = False


class ToolsConfig(BaseModel):
    enable_file_tools: bool = True
    enable_shell_tools: bool = True
    enable_git_tools: bool = True
    enable_mcp_tools: bool = False
    enable_web_tools: bool = False
    enable_task_tools: bool = True
    enable_agent_tools: bool = True


class MCPConfig(BaseModel):
    enabled: bool = False
    config_file: str | None = None
    timeout_seconds: int = 30

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("mcp.timeout_seconds must be greater than 0")
        return value


class SandboxConfig(BaseModel):
    enabled: bool = True
    mode: Literal["process", "filesystem", "container", "wsl"] = "process"
    allow_network: bool = False
    allow_outside_workspace: bool = False


class MemoryConfig(BaseModel):
    enabled: bool = True
    project_memory_file: str = ".pywork/MEMORY.md"
    instructions_file: str = "PYWORK.md"


class StorageConfig(BaseModel):
    sqlite_path: str = ".pywork/pywork.sqlite3"
    sessions_dir: str = ".pywork/sessions"
    logs_dir: str = ".pywork/logs"
    cache_dir: str = ".pywork/cache"


class AgentConfig(BaseModel):
    enable_subagents: bool = True
    enable_coordinator: bool = True
    enable_swarm: bool = False
    max_parallel_agents: int = 4

    @field_validator("max_parallel_agents")
    @classmethod
    def validate_max_parallel_agents(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("agent.max_parallel_agents must be greater than 0")
        return value


class PyWorkConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    default: DefaultModelConfig = Field(default_factory=DefaultModelConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @model_validator(mode="after")
    def sync_permission_mode(self) -> "PyWorkConfig":
        """
        兼容旧配置：
        app.permission_mode 和 permissions.mode 如果不同，以 permissions.mode 为准。
        """
        if self.app.permission_mode != self.permissions.mode:
            self.app.permission_mode = self.permissions.mode

        return self

    @model_validator(mode="after")
    def sync_risk_with_permissions(self) -> "PyWorkConfig":
        """
        兼容现在已有的 risk 配置。
        permissions 是高层权限模式；
        risk 是具体风险开关。
        """
        self.permissions.allow_read = self.risk.allow_read
        self.permissions.require_approval_for_write = self.risk.require_approval_for_write
        self.permissions.require_approval_for_shell = self.risk.require_approval_for_shell
        self.permissions.require_approval_for_network = self.risk.require_approval_for_network

        return self


def validate_config(data: dict[str, Any]) -> PyWorkConfig:
    return PyWorkConfig.model_validate(data)


def config_to_dict(config: PyWorkConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def config_to_json(config: PyWorkConfig) -> str:
    return json.dumps(
        config_to_dict(config),
        ensure_ascii=False,
        indent=2,
    )


def load_and_validate_config(workspace: str | Path = ".") -> PyWorkConfig:
    """
    读取 config_loader 的 dict，然后转换成强类型 PyWorkConfig。
    """
    from pywork.bootstrap.config_loader import load_config

    result = load_config(workspace=Path(workspace))
    return validate_config(result.config)


def main() -> int:
    config = load_and_validate_config(".")
    print("PyWork config schema validation OK")
    print(config_to_json(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())