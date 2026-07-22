"""
Unit tests for the FluxRouter Python SDK interface.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fluxrouter import (
    FluxRouter,
    RoutingDecision,
    OrchestrationResult,
    Complexity,
    GroqAdapter,
    ModelResponse,
    FluxConfig,
)


def test_sdk_exports():
    """Verify all key symbols are exposed at top-level fluxrouter module."""
    import fluxrouter
    assert hasattr(fluxrouter, "FluxRouter")
    assert hasattr(fluxrouter, "RoutingDecision")
    assert hasattr(fluxrouter, "OrchestrationResult")
    assert hasattr(fluxrouter, "Complexity")
    assert hasattr(fluxrouter, "GroqAdapter")
    assert hasattr(fluxrouter, "FluxConfig")


@patch("fluxrouter.GroqAdapter")
@patch("fluxrouter.ComplexityClassifier")
def test_sdk_init_with_params(mock_clf, mock_adapter, tmp_path, monkeypatch):
    """Test FluxRouter initialization with explicit parameters."""
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text("api_key: test_key\n")
    monkeypatch.setattr("fluxrouter.config.CONFIG_PATH", fake_config)

    router = FluxRouter(
        api_key="custom_key",
        fast_model="llama-3.1-8b-instant",
        strong_model="qwen/qwen3.6-27b",
        margin_threshold=0.1,
        escalation_judge_threshold=4,
    )
    assert router.controller.margin_threshold == 0.1
    assert router.controller.judge_escalation_threshold == 4
    assert router.controller.fast_model == "llama-3.1-8b-instant"
    assert router.controller.strong_model == "qwen/qwen3.6-27b"



@patch.object(FluxRouter, "__init__", return_value=None)
def test_sdk_route_method_signature(mock_init):
    """Test router.route signature presence."""
    router = FluxRouter()
    assert hasattr(router, "route")
    assert hasattr(router, "agent")
