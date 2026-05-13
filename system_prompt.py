"""Load and parameterize the system prompt from prompt.md.

Placeholders:
  {{FIRST_NAME}}     — overridden by USER_NAME env, default "Анонимный"
  {{LANGUAGE_CODE}}  — overridden by USER_LOCALE env, default "ru"
"""
from __future__ import annotations

import os
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def load_system_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8").strip()
    text = text.replace("{{FIRST_NAME}}", os.environ.get("USER_NAME", "Анонимный"))
    text = text.replace("{{LANGUAGE_CODE}}", os.environ.get("USER_LOCALE", "ru"))
    return text
