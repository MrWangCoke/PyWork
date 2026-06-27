from __future__ import annotations

import importlib
import sys


MODULES = [
    "typer",
    "rich",
    "textual",
    "pydantic",
    "pydantic_settings",
    "yaml",
    "aiosqlite",
    "httpx",
    "anyio",
    "psutil",
    "orjson",
    "openai",
    "anthropic",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "langgraph.checkpoint.sqlite",
    "langgraph_supervisor",
    "deepagents",
    "mcp",
    "git",
    "unidiff",
    "tree_sitter",
    "nbformat",
]


def main() -> int:
    failed: list[str] = []

    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:
            print(f"[FAIL] {module_name}: {exc}")
            failed.append(module_name)

    if failed:
        print()
        print("Some dependencies failed to import:")
        for name in failed:
            print(f"  - {name}")
        return 1

    print()
    print("All PyWork dependencies imported successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
