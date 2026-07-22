"""
FluxRouter Python SDK — Complexity-aware LLM routing layer.

Usage:
    from fluxrouter import FluxRouter

    router = FluxRouter()   # loads ~/.fluxrouter/config.yaml automatically

    # Route a query
    decision = router.route("What is the capital of France?")
    print(decision.final_response.text)
    print(decision.predicted_tier, decision.escalated)

    # Agent multi-task orchestration
    result = router.agent("Write a Python function to reverse a string, add a docstring, and write a unit test")
    print(result.final_answer)
"""

from __future__ import annotations

import os
from typing import Any

from router.classifier.complexity import Complexity, ComplexityClassifier
from router.adapters.groq_adapter import GroqAdapter, ModelResponse
from router.cascade.controller import (
    CascadeController,
    RoutingDecision,
    FAST_MODEL,
    STRONG_MODEL,
    MARGIN_THRESHOLD,
    JUDGE_ESCALATION_THRESHOLD,
)
from router.orchestrator.agent import AgentOrchestrator, OrchestrationResult, CostBreakdown
from fluxrouter.config import FluxConfig, load as load_config, ConfigNotFoundError


class FluxRouter:
    """
    Python SDK for FluxRouter.

    Provides in-process, zero-HTTP-overhead complexity routing and multi-task
    agent orchestration.

    Args:
        config:                      FluxConfig instance. If None, loaded from ~/.fluxrouter/config.yaml.
        api_key:                     Groq API key (overrides config/env var).
        fast_model:                  Fast model identifier (e.g., "llama-3.1-8b-instant").
        strong_model:                Strong model identifier (e.g., "qwen/qwen3.6-27b").

        margin_threshold:            Classifier tier separation margin threshold.
        escalation_judge_threshold:  Judge score <= this value triggers escalation (default: 4).
        classifier:                  Optional pre-warmed ComplexityClassifier instance.
    """

    def __init__(
        self,
        config: FluxConfig | None = None,
        api_key: str | None = None,
        fast_model: str | None = None,
        strong_model: str | None = None,
        margin_threshold: float | None = None,
        escalation_judge_threshold: int | None = None,
        classifier: ComplexityClassifier | None = None,
    ) -> None:
        if config is None:
            try:
                config = load_config()
            except ConfigNotFoundError:
                key = api_key or os.getenv("GROQ_API_KEY", "")
                if not key:
                    raise ConfigNotFoundError(
                        "FluxRouter configuration not found. Run 'fluxrouter init' "
                        "or pass api_key to FluxRouter(api_key=...)."
                    )
                config = FluxConfig(api_key=key)

        effective_key          = api_key or config.api_key
        effective_fast         = fast_model or config.fast_model
        effective_strong       = strong_model or config.strong_model
        effective_margin       = margin_threshold if margin_threshold is not None else config.margin_threshold
        effective_judge_thresh = (
            escalation_judge_threshold
            if escalation_judge_threshold is not None
            else getattr(config, "escalation_judge_threshold", DEFAULT_JUDGE_THRESHOLD if 'DEFAULT_JUDGE_THRESHOLD' in globals() else 4)
        )

        self.classifier = classifier or ComplexityClassifier()
        self.adapter = GroqAdapter(api_key=effective_key)
        self.controller = CascadeController(
            classifier=self.classifier,
            adapter=self.adapter,
            fast_model=effective_fast,
            strong_model=effective_strong,
            margin_threshold=effective_margin,
            judge_escalation_threshold=effective_judge_thresh,
        )
        self.orchestrator = AgentOrchestrator(
            adapter=self.adapter,
            cascade_controller=self.controller,
            strong_model=effective_strong,
        )

    def route(self, query: str) -> RoutingDecision:
        """
        Classify and route a single query through the cascade pipeline.

        Returns:
            RoutingDecision containing tier, margin, models called, latency,
            escalation state, judge score, and final ModelResponse.
        """
        return self.controller.route(query)

    def agent(self, task: str) -> OrchestrationResult:
        """
        Decompose a multi-part task into subtasks, route each subtask,
        and synthesize a final coherent answer.

        Returns:
            OrchestrationResult containing original task, subtasks, per-subtask
            RoutingDecisions, final answer, latency, and cost breakdown.
        """
        return self.orchestrator.execute(task)


__all__ = [
    "FluxRouter",
    "RoutingDecision",
    "OrchestrationResult",
    "CostBreakdown",
    "Complexity",
    "ComplexityClassifier",
    "CascadeController",
    "AgentOrchestrator",
    "GroqAdapter",
    "ModelResponse",
    "FluxConfig",
]
