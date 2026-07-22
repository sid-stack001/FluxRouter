"""
Smoke tests for the FluxRouter CLI.

Tests only the CLI surface — no API calls are made.

Run with:
    python -m pytest tests/test_cli.py -v
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from fluxrouter.cli import main


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def runner():
    return CliRunner()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNoArgs:
    def test_exits_zero(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code == 0, result.output

    def test_shows_usage(self, runner):
        result = runner.invoke(main, [])
        assert "Usage" in result.output or "FluxRouter" in result.output

    def test_help_flag(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "route" in result.output
        assert "init" in result.output
        assert "serve" in result.output


class TestRouteWithoutConfig:
    """When no config file and no GROQ_API_KEY env var, route must exit 1
    with a helpful 'run fluxrouter init' message — NOT a crash/traceback."""

    def test_exits_nonzero(self, runner, tmp_path, monkeypatch):
        # Point config to a dir that definitely has no config.yaml
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(
            "fluxrouter.config.CONFIG_PATH",
            fake_home / ".fluxrouter" / "config.yaml",
        )
        # Remove env var fallback too
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        result = runner.invoke(main, ["route", "test query"])
        assert result.exit_code != 0

    def test_shows_init_hint(self, runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(
            "fluxrouter.config.CONFIG_PATH",
            fake_home / ".fluxrouter" / "config.yaml",
        )
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        result = runner.invoke(main, ["route", "test query"])
        combined = result.output + (result.stderr or "")
        assert "init" in combined.lower(), \
            f"Expected 'init' hint in output, got:\n{combined}"


class TestConfigCommand:
    """config command with missing config must exit 1 cleanly."""

    def test_missing_config_exits_nonzero(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "fluxrouter.config.CONFIG_PATH",
            tmp_path / "nonexistent" / "config.yaml",
        )
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        result = runner.invoke(main, ["config"])
        assert result.exit_code != 0
