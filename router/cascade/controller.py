"""
CascadeController — Phase 2 routing logic for FluxRouter.

Routing strategy
────────────────
Given a query, the classifier returns (tier, margin) where:
    tier   ∈ {SIMPLE, MEDIUM, COMPLEX}
    margin = max_sim(winning tier) − max_sim(runner-up tier)

Decision tree (evaluated in order):

    tier == SIMPLE  AND margin >= MARGIN_THRESHOLD  →  fast model only
    tier == COMPLEX AND margin >= MARGIN_THRESHOLD  →  strong model only
    else (tier == MEDIUM, or low-margin simple/complex)
                                                    →  cascade path:
        If avg_logprob is available from the adapter:
            avg_logprob < LOGPROB_ESCALATION_THRESHOLD  →  escalate
            else                                        →  return fast response
        If avg_logprob is unavailable (Groq hosted models):
            return fast model response  ← margin-only routing, no post-hoc check
            (see Known Limitations in README)

Models
    fast   = llama-3.1-8b-instant
    strong = llama-3.3-70b-versatile

All tuneable thresholds are module-level named constants.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from router.classifier.complexity import Complexity, ComplexityClassifier
from router.adapters.groq_adapter import GroqAdapter, ModelResponse

# ── Model identifiers ────────────────────────────────────────────────────────
FAST_MODEL   = "llama-3.1-8b-instant"
STRONG_MODEL = "qwen/qwen3.6-27b"
JUDGE_MODEL  = "llama-3.1-8b-instant"   # Fast LLM judge — 100ms, 8B token quota


# ── Routing thresholds ───────────────────────────────────────────────────────

# Minimum tier-separation margin to trust the classifier and skip cascade.
# Queries below this threshold always go through the cascade path.
MARGIN_THRESHOLD: float = 0.04

# avg_logprob threshold for escalation when an adapter exposes logprobs.
LOGPROB_ESCALATION_THRESHOLD: float = -0.5

# LLM-judge escalation threshold.
# Scores <= this value trigger escalation to the strong model.
# Scale: 1 (completely inadequate) … 5 (flawless).
JUDGE_ESCALATION_THRESHOLD: int = 3


# Strict prompt sent to the strong judge model.
# {query} and {response} are filled at call time.
JUDGE_PROMPT: str = (
    "You are a strict quality evaluation judge.\n\n"
    "User Query: {query}\n"
    "Model Response: {response}\n\n"
    "Critically evaluate the response for:\n"
    "1. Factual correctness and accuracy\n"
    "2. Completeness (no missing steps, missing edge cases, or key details)\n"
    "3. Depth of reasoning (no hand-waving or oversimplification)\n"
    "4. Relevance and adherence to instructions\n\n"
    "Rate the response strictly on a 1-5 scale:\n"
    "5 = Flawless, complete, fully accurate, and addresses all subtleties/edge cases.\n"
    "4 = Mostly correct but missing minor details, slight oversimplification, or subtle edge cases.\n"
    "3 = Partial answer, notable gaps, hand-waving, or questionable reasoning.\n"
    "2 = Significantly incomplete, misleading, or contains factual/logic errors.\n"
    "1 = Completely inadequate, wrong, or off-topic.\n\n"
    "Reply with ONLY a single integer score from 1 to 5."
)


# ── Structured logger ────────────────────────────────────────────────────────
# Emits one JSON-lines record per routing decision.  Phase 4 benchmarking
# reads these lines directly for latency/quality/cost analysis.
_log = logging.getLogger("fluxrouter.cascade")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


@dataclass
class RoutingDecision:
    """
    Full record of how a query was routed.

    Designed for benchmarking and observability — every field is serialisable
    and included in both the API response metadata and the JSON-lines log.
    """
    query: str
    predicted_tier: Complexity
    margin: float
    models_called: list[str]
    final_response: ModelResponse
    total_latency_ms: float
    escalated: bool
    # Per-model responses in call order (fast first if called, then strong).
    responses: list[ModelResponse] = field(default_factory=list)
    # LLM-judge score (1-5) assigned to the fast model's response.
    # None if the cascade path was not taken (direct routing) or if the
    # fast model was not called (strong-only path).
    judge_score: int | None = None


class CascadeController:
    """
    Routes queries to the appropriate model tier with optional cascade
    escalation based on classifier margin and (when available) logprobs.

    Args:
        classifier:       Pre-warmed ComplexityClassifier (shared at startup).
        adapter:          GroqAdapter instance; created automatically if None.
        margin_threshold: Override MARGIN_THRESHOLD.
    """

    def __init__(
        self,
        classifier: ComplexityClassifier,
        adapter: GroqAdapter | None = None,
        margin_threshold: float = MARGIN_THRESHOLD,
        fast_model: str = FAST_MODEL,
        strong_model: str = STRONG_MODEL,
        judge_escalation_threshold: int = JUDGE_ESCALATION_THRESHOLD,
    ) -> None:
        self.classifier = classifier
        self.adapter = adapter or GroqAdapter()
        self.margin_threshold = margin_threshold
        self.fast_model  = fast_model
        self.strong_model = strong_model
        self.judge_escalation_threshold = judge_escalation_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, query: str) -> RoutingDecision:
        """
        Classify the query and route to the appropriate model(s).

        Returns:
            RoutingDecision with full audit trail.
        """
        t_start = time.perf_counter()

        # 1. Classify
        tier, info = self.classifier.classify(query)
        margin: float = info["confidence"]  # tier-separation margin

        # 2. Route — three strictly ordered branches
        if tier == Complexity.SIMPLE and margin >= self.margin_threshold:
            decision = self._fast_only(query, tier, margin)

        elif tier == Complexity.COMPLEX and margin >= self.margin_threshold:
            decision = self._strong_only(query, tier, margin)

        else:
            # tier == MEDIUM, or low-margin simple/complex
            decision = self._cascade(query, tier, margin)

        decision.total_latency_ms = (time.perf_counter() - t_start) * 1000
        self._emit_log(decision)
        return decision

    # ── Private routing paths ─────────────────────────────────────────────────

    def _fast_only(
        self, query: str, tier: Complexity, margin: float
    ) -> RoutingDecision:
        resp = self.adapter.complete(query, self.fast_model)
        return RoutingDecision(
            query=query,
            predicted_tier=tier,
            margin=margin,
            models_called=[self.fast_model],
            final_response=resp,
            total_latency_ms=0.0,   # filled by route()
            escalated=False,
            responses=[resp],
        )

    def _strong_only(
        self, query: str, tier: Complexity, margin: float
    ) -> RoutingDecision:
        resp = self.adapter.complete(query, self.strong_model)
        return RoutingDecision(
            query=query,
            predicted_tier=tier,
            margin=margin,
            models_called=[self.strong_model],
            final_response=resp,
            total_latency_ms=0.0,
            escalated=False,
            responses=[resp],
        )

    def _cascade(
        self, query: str, tier: Complexity, margin: float
    ) -> RoutingDecision:
        """
        Call the fast model, run the LLM judge, escalate if quality is poor.
        """
        fast_resp = self.adapter.complete(query, self.fast_model)
        responses = [fast_resp]

        should_escalate, judge_score = self._needs_escalation(query, fast_resp)

        if should_escalate:
            strong_resp = self.adapter.complete(query, self.strong_model)
            responses.append(strong_resp)
            return RoutingDecision(
                query=query,
                predicted_tier=tier,
                margin=margin,
                models_called=[self.fast_model, self.strong_model],
                final_response=strong_resp,
                total_latency_ms=0.0,
                escalated=True,
                responses=responses,
                judge_score=judge_score,
            )

        return RoutingDecision(
            query=query,
            predicted_tier=tier,
            margin=margin,
            models_called=[self.fast_model],
            final_response=fast_resp,
            total_latency_ms=0.0,
            escalated=False,
            responses=responses,
            judge_score=judge_score,
        )


        return RoutingDecision(
            query=query,
            predicted_tier=tier,
            margin=margin,
            models_called=[self.fast_model],
            final_response=fast_resp,
            total_latency_ms=0.0,
            escalated=False,
            responses=responses,
            judge_score=judge_score,
        )


    def _needs_escalation(
        self, query: str, resp: ModelResponse
    ) -> tuple[bool, int | None]:
        """
        Decide whether to escalate; return (should_escalate, judge_score).

        judge_score is None when logprobs were used (no judge call made).

        Escalation logic:
            logprobs available  →  use avg_logprob directly; skip judge call.
            logprobs unavailable →  call LLM judge; parse 1-5 score.
                score <= JUDGE_ESCALATION_THRESHOLD  →  escalate
                parse failure                         →  escalate (fail-safe)
        """
        # ── Logprob path (future adapters) ────────────────────────────────────
        if resp.logprobs_available and resp.avg_logprob is not None:
            return resp.avg_logprob < LOGPROB_ESCALATION_THRESHOLD, None

        # ── LLM-judge path (Groq / no-logprob adapters) ───────────────────────
        judge_score = self._call_judge(query, resp.text)
        if judge_score is None:
            # Parse failed — fail safe: escalate rather than silently accept.
            return True, None
        return judge_score <= self.judge_escalation_threshold, judge_score

    def _call_judge(self, query: str, fast_response: str) -> int | None:
        """
        Ask the judge model to rate the fast model's response on a 1-5 scale.

        Uses fast_model for judging — cheap (~$0.000002), fast (~100ms), and free of
        70B daily TPD rate limits. Returns the integer score or None on failure.
        """
        prompt = JUDGE_PROMPT.format(query=query, response=fast_response)
        try:
            judge_resp = self.adapter.complete(prompt, self.fast_model, max_tokens=5)
            raw = judge_resp.text.strip()
            # Accept the first digit found — handles "4", "4/5", "Score: 4", etc.
            for token in raw.split():
                token = token.strip(".,:/()")
                if token.isdigit() and 1 <= int(token) <= 5:
                    return int(token)
            return None   # no valid digit found
        except Exception:  # noqa: BLE001
            return None   # network / API error → escalate (fail-safe)


    # ── Structured logging ────────────────────────────────────────────────────

    def _emit_log(self, d: RoutingDecision) -> None:
        """
        Emit one JSON-lines record per routing decision.

        Schema (all fields present on every record — None where unavailable):

          ts                   ISO-8601 UTC timestamp
          query                Full query text
          predicted_tier       "simple" | "medium" | "complex"
          margin               Classifier tier-separation margin
          escalated            bool
          judge_score          int (1-5) or null — LLM-judge rating of fast model response.
                               null when direct routing (no cascade) or logprob path taken.
          escalation_reason    Why this routing path was taken:
                               "high_margin_simple"   — margin >= threshold, simple tier
                               "high_margin_complex"  — margin >= threshold, complex tier
                               "judge_escalation"     — judge score <= threshold, escalated
                               "judge_parse_fail"     — judge response unparseable, escalated (fail-safe)
                               "judge_passed"         — judge score > threshold, fast model kept
                               "logprob_escalation"   — logprob below threshold, escalated
          models_called        List of model IDs in call order
          fast_model           Fields for the fast-model call (null if not called)
            .model, .input_tokens, .output_tokens, .latency_ms,
            .avg_logprob, .logprobs_available
          strong_model         Same fields for strong-model call (null if not called)
          final_model          Model ID that produced the returned response
          total_latency_ms     End-to-end wall-clock time incl. classifier (ms)
        """
        def _resp_dict(r: ModelResponse | None) -> dict | None:
            if r is None:
                return None
            return {
                "model":              r.model,
                "input_tokens":       r.input_tokens,
                "output_tokens":      r.output_tokens,
                "latency_ms":         round(r.latency_ms, 2),
                "avg_logprob":        r.avg_logprob,
                "logprobs_available": r.logprobs_available,
            }

        # Map response list → named slots
        fast_resp   = d.responses[0] if len(d.responses) >= 1 else None
        strong_resp = d.responses[1] if len(d.responses) >= 2 else None

        # Determine the precise reason for the routing decision
        reason: str
        if not d.escalated:
            if d.predicted_tier == Complexity.SIMPLE \
                    and d.margin >= self.margin_threshold \
                    and d.models_called == [self.fast_model]:
                reason = "high_margin_simple"
            elif d.predicted_tier == Complexity.COMPLEX \
                    and d.margin >= self.margin_threshold \
                    and d.models_called == [self.strong_model]:
                reason = "high_margin_complex"
            elif d.judge_score is not None:
                reason = "judge_passed"          # judge ran and approved fast model
            elif fast_resp and fast_resp.logprobs_available:
                reason = "logprob_passed"        # logprob signal used, no escalation
            else:
                reason = "fast_direct"           # fast model used directly

        else:
            if fast_resp and fast_resp.logprobs_available:
                reason = "logprob_escalation"
            elif d.judge_score is None:
                reason = "judge_parse_fail"      # parse failed → fail-safe escalation
            else:
                reason = "judge_escalation"      # judge score <= threshold

        record = {
            "ts":                datetime.now(timezone.utc).isoformat(),
            "query":             d.query,
            "predicted_tier":    d.predicted_tier.value,
            "margin":            round(d.margin, 6),
            "escalated":         d.escalated,
            "escalation_reason": reason,
            "judge_score":       d.judge_score,
            "models_called":     d.models_called,
            "fast_model":        _resp_dict(fast_resp),
            "strong_model":      _resp_dict(strong_resp),
            "final_model":       d.final_response.model,
            "total_latency_ms":  round(d.total_latency_ms, 2),
        }
        _log.info(json.dumps(record))
