"""
AgentOrchestrator — Phase 3 of FluxRouter.

Flow
────
    execute(task)
        └─ decompose(task)          LLM: break into 2-5 subtasks (or [task] if atomic)
        └─ for each subtask:
               cascade_controller.route(subtask)   → RoutingDecision
        └─ synthesize(task, decisions)  LLM: merge subtask answers into one coherent reply
        └─ return OrchestrationResult

Design notes
────────────
* decompose() explicitly instructs the model to return [task] unchanged when the
  task is already simple and atomic.  This is the primary cost-saving gate: trivial
  requests must not waste a strong-model call for decomposition.

* synthesize() is only called when len(subtasks) > 1.  Single-subtask results are
  returned directly without a second synthesis call.

* All LLM calls go through GroqAdapter so they share the same retry / error surface.

* OrchestrationResult is fully serialisable (no Complexity enums in the top level)
  so it can be returned verbatim from the FastAPI endpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from router.adapters.groq_adapter import GroqAdapter, ModelResponse
from router.cascade.controller import CascadeController, RoutingDecision, FAST_MODEL, STRONG_MODEL

# ── Model identifiers ────────────────────────────────────────────────────────
DECOMPOSE_MODEL  = "qwen/qwen3.6-27b"   # needs reasoning to decompose well
SYNTHESIZE_MODEL = "qwen/qwen3.6-27b"   # needs coherence across subtask answers


# ── Prompts ───────────────────────────────────────────────────────────────────

# Strict JSON-only prompt. Key design: the "already atomic" instruction is the
# first rule listed so the model sees it before any examples.
_DECOMPOSE_PROMPT = """\
You are a task decomposition assistant.

Rules (follow EXACTLY):
1. If the task is already simple, specific, and atomic (a single focused question
   or action), return it unchanged as a single-element list. DO NOT split it.
2. Only decompose tasks that are genuinely multi-part (e.g. "write X, then test Y,
   then document Z") into 2–5 independent subtasks.
3. Each subtask must be self-contained — answerable without knowing the other subtasks.
4. Return ONLY valid JSON with no explanation, no markdown fences, no extra text:
   {{"subtasks": ["subtask 1", "subtask 2", ...]}}

Task: {task}"""

_SYNTHESIZE_PROMPT = """\
You are a senior technical writer synthesizing research from multiple sources.

Original task:
{task}

Subtask results:
{subtask_qa}

Write one coherent, complete answer to the original task using the subtask results above.
Be concise. Do not repeat subtask numbering. Output only the final answer."""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CostBreakdown:
    fast_model_calls: int = 0
    strong_model_calls: int = 0
    judge_calls: int = 0       # judge is always JUDGE_MODEL (same as fast)

    @property
    def total_calls(self) -> int:
        return self.fast_model_calls + self.strong_model_calls + self.judge_calls


@dataclass
class OrchestrationResult:
    """
    Complete record of an agent execution run.

    Serialisable to JSON via .to_dict() — suitable for API responses and
    Phase 4 benchmarking logs.
    """
    original_task: str
    subtasks: list[str]
    routing_decisions: list[RoutingDecision]
    final_answer: str
    total_latency_ms: float
    total_models_called: int            # total LLM API calls across the run
    cost_breakdown: CostBreakdown

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        def _rd(d: RoutingDecision) -> dict:
            return {
                "subtask":          d.query,
                "predicted_tier":   d.predicted_tier.value,
                "margin":           round(d.margin, 6),
                "escalated":        d.escalated,
                "judge_score":      d.judge_score,
                "models_called":    d.models_called,
                "final_model":      d.final_response.model,
                "answer_preview":   d.final_response.text[:200],
                "latency_ms":       round(d.total_latency_ms, 2),
            }
        return {
            "original_task":      self.original_task,
            "subtasks":           self.subtasks,
            "routing_decisions":  [_rd(d) for d in self.routing_decisions],
            "final_answer":       self.final_answer,
            "total_latency_ms":   round(self.total_latency_ms, 2),
            "total_models_called": self.total_models_called,
            "cost_breakdown": {
                "fast_model_calls":   self.cost_breakdown.fast_model_calls,
                "strong_model_calls": self.cost_breakdown.strong_model_calls,
                "judge_calls":        self.cost_breakdown.judge_calls,
                "total_calls":        self.cost_breakdown.total_calls,
            },
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Decomposes complex tasks into subtasks, routes each through the cascade
    controller, then synthesizes a final coherent answer.

    Args:
        adapter:            GroqAdapter instance (shared with cascade controller).
        cascade_controller: Pre-warmed CascadeController instance.
    """

    def __init__(
        self,
        adapter: GroqAdapter,
        cascade_controller: CascadeController,
        strong_model: str = STRONG_MODEL,
    ) -> None:
        self.adapter = adapter
        self.cc = cascade_controller
        self.strong_model = strong_model   # used for decompose + synthesize

    # ── Public API ────────────────────────────────────────────────────────────

    def decompose(self, task: str) -> list[str]:
        """
        Break a task into independent subtasks (or return [task] if already atomic).

        The model is explicitly instructed to return [task] unchanged for
        simple/atomic requests — preventing spurious decomposition that would
        waste strong-model calls on trivial queries.

        Returns:
            List of subtask strings (always at least [task]).
        """
        prompt = _DECOMPOSE_PROMPT.format(task=task)
        resp = self.adapter.complete(prompt, self.strong_model)
        raw = resp.text.strip()

        try:
            data = json.loads(raw)
            subtasks = data.get("subtasks", [])
            if not isinstance(subtasks, list) or not subtasks:
                raise ValueError("empty or malformed subtasks list")
            # Validate each entry is a non-empty string
            subtasks = [str(s).strip() for s in subtasks if str(s).strip()]
            if not subtasks:
                raise ValueError("all subtasks were empty strings")
            return subtasks
        except (json.JSONDecodeError, ValueError, KeyError):
            # Parse failure — treat task as atomic rather than crashing
            return [task]

    def execute(self, task: str) -> OrchestrationResult:
        """
        Full orchestration pipeline: decompose → route each subtask → synthesize.

        Returns:
            OrchestrationResult with full audit trail.
        """
        t_start = time.perf_counter()

        # 1. Decompose
        subtasks = self.decompose(task)
        atomic = len(subtasks) == 1 and subtasks[0] == task

        # 2. Route each subtask
        decisions: list[RoutingDecision] = []
        for subtask in subtasks:
            decision = self.cc.route(subtask)
            decisions.append(decision)

        # 3. Synthesize (skip if atomic — avoid an unnecessary LLM call)
        if atomic or len(decisions) == 1:
            final_answer = decisions[0].final_response.text
        else:
            final_answer = self.synthesize(task, decisions)

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        # 4. Cost accounting
        cb = self._tally_costs(decisions)

        return OrchestrationResult(
            original_task=task,
            subtasks=subtasks,
            routing_decisions=decisions,
            final_answer=final_answer,
            total_latency_ms=total_latency_ms,
            total_models_called=cb.total_calls,
            cost_breakdown=cb,
        )

    def synthesize(
        self,
        original_task: str,
        subtask_results: list[RoutingDecision],
    ) -> str:
        """
        Merge subtask answers into one coherent reply using the strong model.

        Only called when len(subtask_results) > 1.
        """
        qa_pairs = "\n\n".join(
            f"Subtask {i+1}: {d.query}\nAnswer: {d.final_response.text}"
            for i, d in enumerate(subtask_results)
        )
        prompt = _SYNTHESIZE_PROMPT.format(
            task=original_task,
            subtask_qa=qa_pairs,
        )
        resp = self.adapter.complete(prompt, self.strong_model)
        return resp.text

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tally_costs(self, decisions: list[RoutingDecision]) -> CostBreakdown:
        """Count fast, strong, and judge model calls across all routing decisions."""
        cb = CostBreakdown()
        fast_m   = self.cc.fast_model
        strong_m = self.cc.strong_model
        for d in decisions:
            for model in d.models_called:
                if model == fast_m:
                    cb.fast_model_calls += 1
                elif model == strong_m:
                    cb.strong_model_calls += 1
            if d.judge_score is not None:
                cb.judge_calls += 1
        return cb
