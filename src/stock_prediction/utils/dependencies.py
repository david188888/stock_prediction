from __future__ import annotations

import importlib


def require_dependency(module_name: str, install_hint: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = install_hint or f"Install the `{module_name}` package first."
        raise RuntimeError(f"Missing dependency `{module_name}`. {hint}") from exc

