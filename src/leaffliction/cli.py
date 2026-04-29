"""Shared typer/rich helpers for the 5 entrypoints."""
from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

THEME = Theme({"info": "cyan", "warn": "yellow", "err": "bold red", "ok": "bold green"})
console = Console(theme=THEME)


def die(msg: str, code: int = 1) -> None:
    """Print an error message in red and exit with the given code."""
    console.print(f"[err]error:[/err] {msg}")
    sys.exit(code)
