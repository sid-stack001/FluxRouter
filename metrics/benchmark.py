"""
FluxRouter Phase 4 — Benchmark Harness

Runs the eval_set through three routing configurations and
compares quality, cost, and latency using an LLM-as-judge scorer.

Configurations
──────────────
  always_fast   : direct call to llama-3.1-8b-instant for every query
  always_strong : direct call to qwen/qwen3.6-27b for every query (sample)
  fluxrouter    : CascadeController (margin → judge → optional escalation)

Pricing (Groq on-demand, verified 2026-07)
──────────────────────────────────────────
  llama-3.1-8b-instant : $0.05 / 1M input tokens, $0.08 / 1M output tokens
  qwen/qwen3.6-27b     : $0.39 / 1M input tokens, $0.49 / 1M output tokens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier.complexity import ComplexityClassifier
from router.adapters.groq_adapter import GroqAdapter, ModelResponse
from router.cascade.controller import CascadeController, FAST_MODEL, STRONG_MODEL
from metrics.stratified_sample import stratified_sample
from tests.eval_set import EVAL_SET as full_eval_set

# ── Pricing constants ($/1M tokens) ──────────────────────────────────────────
PRICE_FAST_INPUT:    float = 0.05   # llama-3.1-8b-instant input ($/1M)
PRICE_FAST_OUTPUT:   float = 0.08   # llama-3.1-8b-instant output ($/1M)
PRICE_STRONG_INPUT:  float = 0.39   # qwen/qwen3.6-27b input ($/1M)
PRICE_STRONG_OUTPUT: float = 0.49   # qwen/qwen3.6-27b output ($/1M)

JUDGE_MODEL = FAST_MODEL

# ── Judge prompt ─────────────────────────────────────────────────────────────
_JUDGE_PROMPT = (
    "Query: {query}\n"
    "Response: {response}\n\n"
    "Rate how completely and correctly this response answers the query.\n"
    "Reply with ONLY a number 1-5 (5 = excellent, 1 = completely inadequate)."
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    label: str                      # "simple" / "medium" / "complex"
    config: str                     # "always_fast" / "always_strong" / "fluxrouter"
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    judge_score: int | None
    escalated: bool = False         # fluxrouter only
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ConfigSummary:
    config: str
    n_queries: int
    total_cost_usd: float
    cost_per_query_usd: float
    avg_latency_ms: float
    mean_judge_score: float
    escalation_rate: float | None   # None for non-fluxrouter configs
    total_input_tokens: int
    total_output_tokens: int
    fast_calls: int
    strong_calls: int
    judge_calls: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if "27b" in model or "70b" in model or "versatile" in model:
        p_in, p_out = PRICE_STRONG_INPUT, PRICE_STRONG_OUTPUT
    else:
        p_in, p_out = PRICE_FAST_INPUT, PRICE_FAST_OUTPUT
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def _judge(adapter: GroqAdapter, query: str, response: str) -> tuple[int | None, int, int]:
    prompt = _JUDGE_PROMPT.format(query=query, response=response)
    try:
        resp = adapter.complete(prompt, JUDGE_MODEL, max_tokens=5)
        raw = resp.text.strip()
        score: int | None = None
        for token in raw.split():
            token = token.strip(".,:/()")
            if token.isdigit() and 1 <= int(token) <= 5:
                score = int(token)
                break
        return score, resp.input_tokens, resp.output_tokens
    except Exception:
        return None, 0, 0


def _print_progress(current: int, total: int, config: str, query: str) -> None:
    bar_len = 30
    filled  = int(bar_len * current / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    q_short = query[:40].ljust(40)
    sys.stdout.write(f"\r  [{bar}] {current}/{total}  {config:<14} {q_short}")
    sys.stdout.flush()


# ── Config runners ────────────────────────────────────────────────────────────

def run_always_fast(queries: list[tuple[str, str]], adapter: GroqAdapter) -> list[QueryResult]:
    results = []
    total = len(queries)
    for i, (query, label) in enumerate(queries, 1):
        _print_progress(i, total, "always_fast", query)
        t0 = time.perf_counter()
        resp = adapter.complete(query, FAST_MODEL)
        lat = (time.perf_counter() - t0) * 1000

        score, j_in, j_out = _judge(adapter, query, resp.text)
        cost = _compute_cost(FAST_MODEL, resp.input_tokens, resp.output_tokens)

        results.append(QueryResult(
            query=query, label=label, config="always_fast",
            model_used=FAST_MODEL,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_ms=lat, judge_score=score,
            judge_input_tokens=j_in, judge_output_tokens=j_out,
            cost_usd=cost,
        ))
    print()
    return results


def run_always_strong(queries: list[tuple[str, str]], adapter: GroqAdapter) -> list[QueryResult]:
    results = []
    total = len(queries)
    for i, (query, label) in enumerate(queries, 1):
        _print_progress(i, total, "always_strong", query)
        t0 = time.perf_counter()
        resp = adapter.complete(query, STRONG_MODEL)
        lat = (time.perf_counter() - t0) * 1000

        score, j_in, j_out = _judge(adapter, query, resp.text)
        cost = _compute_cost(STRONG_MODEL, resp.input_tokens, resp.output_tokens)

        results.append(QueryResult(
            query=query, label=label, config="always_strong",
            model_used=STRONG_MODEL,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_ms=lat, judge_score=score,
            judge_input_tokens=j_in, judge_output_tokens=j_out,
            cost_usd=cost,
        ))
    print()
    return results


def run_fluxrouter(
    queries: list[tuple[str, str]], controller: CascadeController, adapter: GroqAdapter
) -> list[QueryResult]:
    results = []
    total = len(queries)
    for i, (query, label) in enumerate(queries, 1):
        _print_progress(i, total, "fluxrouter", query)
        decision = controller.route(query)
        resp = decision.final_response

        score, j_in, j_out = _judge(adapter, query, resp.text)

        routing_cost = sum(
            _compute_cost(m, r.input_tokens, r.output_tokens)
            for m, r in zip(decision.models_called, decision.responses)
        )
        if decision.judge_score is not None:
            routing_cost += _compute_cost(FAST_MODEL, 50, 2)
        judge_cost = _compute_cost(JUDGE_MODEL, j_in, j_out)

        results.append(QueryResult(
            query=query, label=label, config="fluxrouter",
            model_used=resp.model,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_ms=decision.total_latency_ms,
            judge_score=score,
            escalated=decision.escalated,
            judge_input_tokens=j_in, judge_output_tokens=j_out,
            cost_usd=routing_cost + judge_cost,
        ))
    print()
    return results


# ── Summarise ─────────────────────────────────────────────────────────────────

def summarise(results: list[QueryResult]) -> ConfigSummary:
    n = len(results)
    total_cost = sum(r.cost_usd for r in results)
    cost_per_q = total_cost / n if n > 0 else 0.0
    avg_latency = sum(r.latency_ms for r in results) / n
    scored = [r.judge_score for r in results if r.judge_score is not None]
    mean_score = sum(scored) / len(scored) if scored else 0.0
    escalated = [r for r in results if r.escalated]
    esc_rate = len(escalated) / n if any(r.config == "fluxrouter" for r in results) else None

    fast_calls   = sum(1 for r in results if r.model_used == FAST_MODEL)
    strong_calls = sum(1 for r in results if r.model_used == STRONG_MODEL)
    judge_calls  = sum(1 for r in results if r.judge_score is not None)

    return ConfigSummary(
        config=results[0].config,
        n_queries=n,
        total_cost_usd=total_cost,
        cost_per_query_usd=cost_per_q,
        avg_latency_ms=avg_latency,
        mean_judge_score=mean_score,
        escalation_rate=esc_rate,
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        fast_calls=fast_calls,
        strong_calls=strong_calls,
        judge_calls=judge_calls,
    )


# ── Pretty table ──────────────────────────────────────────────────────────────

def print_table(summaries: list[ConfigSummary]) -> None:
    header = (
        f"{'Config':<16} {'Queries':>7} {'Cost($)':>9} {'Cost/Q($)':>11} {'Avg Latency':>12} "
        f"{'Mean Quality':>13} {'Esc Rate':>9} {'Fast':>6} {'Strong':>7}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for s in summaries:
        esc = f"{s.escalation_rate*100:.1f}%" if s.escalation_rate is not None else "   n/a"
        print(
            f"{s.config:<16} {s.n_queries:>7} {s.total_cost_usd:>9.4f} {s.cost_per_query_usd:>11.6f} "
            f"{s.avg_latency_ms:>10.0f}ms {s.mean_judge_score:>12.2f}/5 "
            f"{esc:>9} {s.fast_calls:>6} {s.strong_calls:>7}"
        )
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FluxRouter benchmark harness")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N queries from eval_set (default: all 100)"
    )
    parser.add_argument(
        "--configs", nargs="+",
        choices=["always_fast", "always_strong", "fluxrouter"],
        default=["always_fast", "always_strong", "fluxrouter"],
        help="Which configs to run (default: all three)"
    )
    args = parser.parse_args()

    sampled = stratified_sample(full_eval_set, n_total=30)
    raw = [(q, c.value) for q, c in sampled]
    if args.limit:
        raw = raw[:args.limit]
        print(f"[benchmark] Running on first {args.limit} queries (--limit flag).")
    else:
        print(f"[benchmark] Running full eval set ({len(raw)} queries).")

    print("[benchmark] Initialising adapter and classifier…")
    adapter    = GroqAdapter()
    classifier = ComplexityClassifier()
    controller = CascadeController(classifier=classifier, adapter=adapter)
    print("[benchmark] Ready.\n")

    all_results: list[QueryResult] = []
    summaries:   list[ConfigSummary] = []

    if "always_fast" in args.configs:
        print("── Config 1/3: always_fast ──────────────────────────────────────────")
        r = run_always_fast(raw, adapter)
        all_results.extend(r)
        summaries.append(summarise(r))

    if "always_strong" in args.configs:
        print("── Config 2/3: always_strong (capped at 12 queries) ────────────────")
        r = run_always_strong(raw[:12], adapter)
        all_results.extend(r)
        summaries.append(summarise(r))

    if "fluxrouter" in args.configs:
        print("── Config 3/3: fluxrouter ───────────────────────────────────────────")
        r = run_fluxrouter(raw, controller, adapter)
        all_results.extend(r)
        summaries.append(summarise(r))

    # Print comparison table
    print_table(summaries)

    # Print normalized per-query cost savings vs always_strong & always_fast
    af = next((s for s in summaries if s.config == "always_fast"),   None)
    as_ = next((s for s in summaries if s.config == "always_strong"), None)
    fr = next((s for s in summaries if s.config == "fluxrouter"),     None)
    if as_ and fr:
        saving_pct = (as_.cost_per_query_usd - fr.cost_per_query_usd) / as_.cost_per_query_usd * 100
        quality_delta = fr.mean_judge_score - as_.mean_judge_score
        print(f"\n  FluxRouter vs always_strong (Per-Query Normalized):")
        print(f"    Always Strong Cost/Q: ${as_.cost_per_query_usd:.6f}")
        print(f"    FluxRouter Cost/Q   : ${fr.cost_per_query_usd:.6f}")
        print(f"    Cost savings (per q): {saving_pct:+.1f}%")
        print(f"    Quality delta       : {quality_delta:+.2f}/5")
    if af and fr:
        quality_gain = fr.mean_judge_score - af.mean_judge_score
        cost_overhead = (fr.cost_per_query_usd - af.cost_per_query_usd) / af.cost_per_query_usd * 100
        print(f"\n  FluxRouter vs always_fast (Per-Query Normalized):")
        print(f"    Always Fast Cost/Q  : ${af.cost_per_query_usd:.6f}")
        print(f"    FluxRouter Cost/Q   : ${fr.cost_per_query_usd:.6f}")
        print(f"    Cost overhead (per q): {cost_overhead:+.1f}%")
        print(f"    Quality gain        : {quality_gain:+.2f}/5")

    # Save JSON
    out_path = Path(__file__).parent / "benchmark_results.json"
    output = {
        "meta": {
            "n_queries":    len(raw),
            "configs":      args.configs,
            "pricing": {
                "fast_input_per_1m":    PRICE_FAST_INPUT,
                "fast_output_per_1m":   PRICE_FAST_OUTPUT,
                "strong_input_per_1m":  PRICE_STRONG_INPUT,
                "strong_output_per_1m": PRICE_STRONG_OUTPUT,
                "source": "https://groq.com/pricing/ — verified 2026-07",
            },
        },
        "summaries": [asdict(s) for s in summaries],
        "per_query":  [asdict(r) for r in all_results],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
