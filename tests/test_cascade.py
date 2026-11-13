"""
Cascade routing test — Phase 2 validation.

Runs varied queries through CascadeController and prints the full
RoutingDecision for each. Validates:
    - Simple queries with clear margin do NOT escalate.
    - Complex queries with clear margin skip the fast model entirely.
    - Medium / ambiguous / hard edge-case queries go through the cascade path.

Run with:
    python -m tests.test_cascade
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier.complexity import ComplexityClassifier, Complexity
from router.cascade.controller import CascadeController

# ── Test queries ──────────────────────────────────────────────────────────────
# Each entry: (query, expected_tier, expect_escalation)
# expect_escalation:
#   False = must NOT escalate (direct routing)
#   True  = must escalate
#   None  = don't assert (medium / ambiguous / hard — depends on fast model performance)
TEST_CASES: list[tuple[str, Complexity, bool | None]] = [
    # Simple — high margin → fast only, no escalation
    ("What is the capital of Germany?",                  Complexity.SIMPLE,  False),
    ("What does 'RAM' stand for?",                        Complexity.SIMPLE,  False),
    ("Translate 'goodbye' to Japanese.",                  Complexity.SIMPLE,  False),

    # Complex — high margin → strong only, no escalation (skips fast model entirely)
    ("Design a distributed rate limiter that works across 50 nodes with "
     "consistent hashing and Redis Cluster, handling 100k req/s.",
                                                          Complexity.COMPLEX, False),
    ("Debug why this async Rust code deadlocks under concurrent load: "
     "explain the ownership model violation and propose a fix.",
                                                          Complexity.COMPLEX, False),
    ("Derive the backpropagation equations for a convolutional layer and "
     "explain gradient flow through pooling operations.",  Complexity.COMPLEX, False),

    # Medium / ambiguous — cascade logic applies
    ("Write a Python function to merge two sorted lists.",Complexity.MEDIUM,  None),
    ("Explain the tradeoffs between REST and GraphQL.",   Complexity.MEDIUM,  None),
    ("Refactor this class to use dependency injection.",  Complexity.MEDIUM,  None),
    ("Summarize this five-page technical design document in three bullet "
     "points, then suggest one improvement.",             Complexity.MEDIUM,  None),

    # Hard edge-cases — designed to test fast model slips & judge trigger
    ("What's 17 * 24 - 8 / 2 + 3?",                       Complexity.MEDIUM,  None),  # Arithmetic check
    ("Explain why my recursive Fibonacci function is slow and give the fix, in detail with complexity analysis.", 
                                                          Complexity.MEDIUM,  None),
    ("Is it true that the Great Wall of China is visible from space? Explain precisely why or why not.", 
                                                          Complexity.MEDIUM,  None),
    ("Walk through, step by step, how a hash table resizes when it exceeds its load factor.", 
                                                          Complexity.MEDIUM,  None),
]


def run() -> None:
    print("\n" + "=" * 70)
    print("  FluxRouter — Phase 2 Cascade Routing Test")
    print("=" * 70)

    clf = ComplexityClassifier()
    controller = CascadeController(classifier=clf, strong_model="qwen/qwen3.6-27b")


    results = []
    for query, expected_tier, expect_escalation in TEST_CASES:
        print(f"\n  Query: {query[:72]}")
        decision = controller.route(query)

        # Soft assertions
        tier_ok  = decision.predicted_tier == expected_tier
        esc_ok   = (
            True if expect_escalation is None          # don't assert for medium/hard queries
            else decision.escalated == expect_escalation
        )

        tier_mark = "✓" if tier_ok  else "✗"
        esc_mark  = "✓" if esc_ok   else "✗"

        print(f"  {tier_mark} tier={decision.predicted_tier.value:<7}  "
              f"(expected {expected_tier.value})")
        print(f"  {esc_mark} escalated={decision.escalated}  "
              f"models={decision.models_called}")
        judge_info = (
            f"judge={decision.judge_score}/5"
            if decision.judge_score is not None
            else "judge=n/a (direct)"
        )
        print(f"    margin={decision.margin:.3f}  total={decision.total_latency_ms:.0f}ms  "
              f"output_tok={decision.final_response.output_tokens}  "
              f"logprobs={decision.final_response.logprobs_available}  {judge_info}")

        print(f"    response: {decision.final_response.text[:100]!r}")

        results.append((tier_ok, esc_ok, decision))

    # Summary
    print("\n" + "=" * 70)
    tier_passes = sum(1 for t, _, _ in results if t)
    esc_passes  = sum(1 for _, e, _ in results if e)
    escalated   = sum(1 for _, _, d in results if d.escalated)

    print(f"  Tier accuracy      : {tier_passes}/{len(results)}")
    print(f"  Escalation correct : {esc_passes}/{len(results)}")
    print(f"  Escalated calls    : {escalated}/{len(results)} "
          f"(called strong model)")

    direct_simple  = [d for _, _, d in results
                      if d.predicted_tier == Complexity.SIMPLE and not d.escalated]
    direct_complex = [d for _, _, d in results
                      if d.predicted_tier == Complexity.COMPLEX and not d.escalated]
    cascade_calls  = [d for _, _, d in results if len(d.models_called) == 2]

    if direct_simple:
        avg = sum(d.total_latency_ms for d in direct_simple) / len(direct_simple)
        print(f"  Simple avg latency : {avg:.0f}ms  (fast model only)")
    if direct_complex:
        avg = sum(d.total_latency_ms for d in direct_complex) / len(direct_complex)
        print(f"  Complex avg latency: {avg:.0f}ms  (strong model only)")
    if cascade_calls:
        avg = sum(d.total_latency_ms for d in cascade_calls) / len(cascade_calls)
        print(f"  Cascade avg latency: {avg:.0f}ms  (fast + strong)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()