"""
fluxrouter.config — load/save ~/.fluxrouter/config.yaml.

Config load priority (highest to lowest):
    1. ~/.fluxrouter/config.yaml   (written by `fluxrouter init`)
    2. Environment variables       (GROQ_API_KEY, etc.)
    3. Hard-coded defaults         (model names, thresholds)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR  = Path.home() / ".fluxrouter"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

# ── Defaults (mirror controller.py module-level constants) ────────────────────
DEFAULT_FAST_MODEL   = "llama-3.1-8b-instant"
DEFAULT_STRONG_MODEL = "qwen/qwen3.6-27b"
DEFAULT_MARGIN       = 0.04
DEFAULT_JUDGE_THRESHOLD = 3




class ConfigNotFoundError(FileNotFoundError):
    """Raised when config.yaml is missing and no env-var fallback exists."""


@dataclass
class FluxConfig:
    api_key:                    str
    fast_model:                 str   = DEFAULT_FAST_MODEL
    strong_model:               str   = DEFAULT_STRONG_MODEL
    margin_threshold:           float = DEFAULT_MARGIN
    escalation_judge_threshold: int   = DEFAULT_JUDGE_THRESHOLD

    def masked_key(self) -> str:
        """Return API key with middle characters redacted."""
        k = self.api_key
        if len(k) <= 8:
            return "****"
        return k[:4] + ("*" * (len(k) - 8)) + k[-4:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_key":                    self.api_key,
            "fast_model":                 self.fast_model,
            "strong_model":               self.strong_model,
            "margin_threshold":           self.margin_threshold,
            "escalation_judge_threshold": self.escalation_judge_threshold,
        }


def load() -> FluxConfig:
    """
    Load FluxConfig from ~/.fluxrouter/config.yaml.

    Falls back to the GROQ_API_KEY environment variable for the key,
    keeping model names / thresholds at defaults if the file is absent.

    Raises ConfigNotFoundError when neither the file nor the env var exist.
    """
    if CONFIG_PATH.exists():
        try:
            raw: dict = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigNotFoundError(
                f"Config file at {CONFIG_PATH} is malformed: {exc}\n"
                "Run 'fluxrouter init' to regenerate it."
            ) from exc

        api_key = raw.get("api_key") or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ConfigNotFoundError(
                "No API key found in config. Run 'fluxrouter init'."
            )
        return FluxConfig(
            api_key=api_key,
            fast_model=raw.get("fast_model", DEFAULT_FAST_MODEL),
            strong_model=raw.get("strong_model", DEFAULT_STRONG_MODEL),
            margin_threshold=float(raw.get("margin_threshold", DEFAULT_MARGIN)),
            escalation_judge_threshold=int(raw.get("escalation_judge_threshold", DEFAULT_JUDGE_THRESHOLD)),
        )


    # No config file — try env var only
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ConfigNotFoundError(
            "No FluxRouter config found. Run 'fluxrouter init' to set up."
        )
    return FluxConfig(api_key=api_key)


def save(cfg: FluxConfig) -> None:
    """Write config to ~/.fluxrouter/config.yaml, creating the dir if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(cfg.to_dict(), default_flow_style=False))
