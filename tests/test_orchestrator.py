"""
Orchestrator test — Phase 3 validation.

Tests three tasks of increasing complexity:
    a) Atomic simple task      → decompose() must return [task] unchanged
    b) Multi-part coding task  → should decompose into ~3 subtasks
    c) Genuinely complex task  → multi-step design question, multiple subtasks

Key assertion: decompose() must NOT split atomic tasks.
If it always decomposes everything, we flag it — spurious decompositions
burn strong-model calls on things that didn't need it.

Run with:
    python -m tests.test_orchestrator
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier.complexity import ComplexityClassifier
from router.adapters.groq_adapter import GroqAdapter
from router.cascade.controller import CascadeController
from router.orchestrator.agent import AgentOrchestrator

# ── Test cases ────────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "a",
        "label": "Atomic / simple",
        "task": "What's the capital of France?",
        # decompose() must NOT split this — it's a single focused question.
        "expect_atomic": True,
    },
    {
        "id": "b",
        "label": "Multi-part coding",
        "task": (
            "Write a Python function to reverse a string, "
            "add a docstring to it, "
            "and write a unit test for it."
        ),
        # Should decompose into ~3 subtasks.
        "expect_atomic": False,
    },
    {
        "id": "c",
        "label": "Complex multi-step design",
        "task": (
            "Design a REST API for a todo app: "
            "define the endpoints, write the database schema, "
            "and explain the authentication approach."
        ),
        # Should decompose into ~3 subtasks.
        "expect_atomic": False,
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run() -> None:
    print("\n" + "=" * 72)
    print("  FluxRouter — Phase 3 Orchestrator Test")
    print("=" * 72)

    clf = ComplexityClassifier()
    adapter = GroqAdapter()
    controller = CascadeController(classifier=clf, adapter=adapter)
    orchestrator = AgentOrchestrator(adapter=adapter, cascade_controller=controller)

    results = []

    for spec in TASKS:
        task_id    = spec["id"]
        label      = spec["label"]
        task       = spec["task"]
        expect_atomic = spec["expect_atomic"]

        print(f"\n{'─'*72}")
        print(f"  [{task_id}] {label}")
        print(f"  Task: {task}")
        print(f"{'─'*72}")

        result = orchestrator.execute(task)

        # ── Atomicity check ──────────────────────────────────────────────────
        actually_atomic = (
            len(result.subtasks) == 1 and result.subtasks[0].strip() == task.strip()
        )
        if expect_atomic and not actually_atomic:
            atomic_flag = "⚠ SPURIOUS DECOMPOSITION — burned strong-model call unnecessarily"
            atomic_ok = False
        elif expect_atomic and actually_atomic:
            atomic_flag = "✓ correctly kept atomic"
            atomic_ok = True
        elif not expect_atomic and actually_atomic:
            atomic_flag = "⚠ expected decomposition but got atomic — may under-serve"
            atomic_ok = False
        else:
            atomic_flag = f"✓ decomposed into {len(result.subtasks)} subtasks"
            atomic_ok = True

        print(f"\n  Decomposition: {atomic_flag}")
        for i, st in enumerate(result.subtasks, 1):
            print(f"    {i}. {st}")

        # ── Per-subtask routing ──────────────────────────────────────────────
        print(f"\n  Routing decisions:")
        for d in result.routing_decisions:
            judge_str = f"judge={d.judge_score}/5" if d.judge_score is not None else "judge=n/a"
            esc_str   = "⬆ escalated" if d.escalated else "direct"
            print(f"    • [{d.predicted_tier.value:<7}] {esc_str:<11}  "
                  f"margin={d.margin:.3f}  {judge_str}  "
                  f"model={d.final_response.model.split('-')[0]+'-'+d.final_response.model.split('-')[1]}  "
                  f"{d.total_latency_ms:.0f}ms")
            print(f"      subtask: {d.query[:80]}")

        # ── Cost breakdown ───────────────────────────────────────────────────
        cb = result.cost_breakdown
        print(f"\n  Cost breakdown:")
        print(f"    fast calls  : {cb.fast_model_calls}")
        print(f"    strong calls: {cb.strong_model_calls}")
        print(f"    judge calls : {cb.judge_calls}")
        print(f"    total calls : {cb.total_calls}")
        print(f"    total time  : {result.total_latency_ms:.0f}ms")

        # ── Final answer ─────────────────────────────────────────────────────
        synthesized = len(result.subtasks) > 1
        print(f"\n  Final answer ({'synthesized' if synthesized else 'direct'}):")
        # Print first 600 chars
        answer_preview = result.final_answer[:600]
        if len(result.final_answer) > 600:
            answer_preview += "…"
        for line in answer_preview.split("\n")[:15]:
            print(f"    {line}")

        results.append({"id": task_id, "atomic_ok": atomic_ok, "result": result})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Summary")
    print(f"{'='*72}")
    for r in results:
        mark = "✓" if r["atomic_ok"] else "✗"
        res  = r["result"]
        print(f"  {mark} [{r['id']}]  subtasks={len(res.subtasks)}  "
              f"calls={res.cost_breakdown.total_calls}  "
              f"time={res.total_latency_ms:.0f}ms")

    # Explicit decomposition safety check
    spurious = [r for r in results if not r["atomic_ok"]]
    if spurious:
        print(f"\n  ⚠  {len(spurious)} decomposition issue(s) detected — see flags above.")
        print("     Tune _DECOMPOSE_PROMPT or MARGIN_THRESHOLD if spurious splits persist.")
    else:
        print("\n  ✓  All decomposition checks passed.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    run()
