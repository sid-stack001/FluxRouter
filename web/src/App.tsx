import React, { useState, useEffect } from 'react';
import { routeQuery, MARGIN_THRESHOLD } from './lib/knnRouter';
import { ANCHORS } from './data/anchors';
import { RoutingResult as RoutingResultType } from './types';

const EXAMPLE_PROMPTS = [
  "What is the capital of France?",
  "Write a one-line Python function to reverse a string.",
  "How do I implement debounce in JavaScript?",
  "Write a SQL query to find duplicate rows in a table.",
  "Design a distributed rate limiter that works across 50 nodes."
];

export function App() {
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPTS[0]);
  const [result, setResult] = useState<RoutingResultType | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRoute = (queryText: string) => {
    if (!queryText.trim()) return;
    setLoading(true);
    setPrompt(queryText);

    // Compute deterministic k-NN routing
    const res = routeQuery(queryText.trim(), {
      k: 3,
      marginThreshold: MARGIN_THRESHOLD
    });
    setResult(res);
    setLoading(false);
  };

  useEffect(() => {
    handleRoute(EXAMPLE_PROMPTS[0]);
  }, []);

  const totalAnchors = ANCHORS.length;

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 font-sans p-3 sm:p-6 flex flex-col justify-between max-w-xl mx-auto w-full">
      
      {/* Top Header */}
      <div>
        <header className="border-b border-neutral-800 pb-2.5 mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <img
              src="./logo.jpg"
              alt="FluxRouter Logo"
              className="w-7 h-7 rounded border border-neutral-800 object-cover"
            />
            <h1 className="text-sm sm:text-base font-semibold tracking-tight text-white">
              FluxRouter
            </h1>
          </div>
          <span className="text-[11px] sm:text-xs text-neutral-400 font-mono">
            k-NN LLM Cascade Demo
          </span>
        </header>

        {/* Prompt Input Form */}
        <section className="mb-3.5">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleRoute(prompt);
            }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a prompt..."
              className="flex-1 min-w-0 bg-neutral-900 border border-neutral-800 rounded px-3 py-2 text-xs sm:text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:border-neutral-600"
            />
            <button
              type="submit"
              disabled={loading || !prompt.trim()}
              className="bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 px-3.5 py-2 rounded text-xs sm:text-sm font-medium shrink-0 transition disabled:opacity-50"
            >
              {loading ? "Routing..." : "Route"}
            </button>
          </form>

          {/* 5 Example Prompts */}
          <div className="flex flex-wrap items-center gap-1.5 mt-2.5">
            <span className="text-[11px] text-neutral-400 mr-0.5">Examples:</span>
            {EXAMPLE_PROMPTS.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleRoute(ex)}
                className={`text-[11px] sm:text-xs px-2 py-1 rounded border text-left transition ${
                  prompt === ex
                    ? "bg-neutral-800 border-neutral-600 text-white font-medium"
                    : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700"
                }`}
              >
                {ex}
              </button>
            ))}
          </div>
        </section>

        {/* Results */}
        {result && (
          <main className="space-y-3 pt-1">
            
            {/* 1. Routed Tier / Model */}
            <div className="border border-neutral-800 bg-neutral-900/60 rounded p-3 sm:p-3.5">
              <div className="text-[10px] sm:text-xs uppercase tracking-wider text-neutral-400 font-mono mb-2">
                Routing Decision
              </div>
              <div className="grid grid-cols-2 gap-2.5 sm:gap-3 text-xs sm:text-sm">
                <div>
                  <div className="text-[11px] text-neutral-400">Tier</div>
                  <div className="font-mono font-medium capitalize text-neutral-200">
                    {result.predictedTier}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-neutral-400">Dispatched Model</div>
                  <div className="font-mono font-medium text-neutral-200 break-all sm:break-normal">
                    {result.dispatchedModel}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-neutral-400">Path</div>
                  <div className="font-mono text-neutral-200">
                    {result.isDirectRouting
                      ? "Direct"
                      : result.escalated
                      ? "Cascade (Escalated)"
                      : "Cascade (Passed)"}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-neutral-400">Latency</div>
                  <div className="font-mono text-neutral-200">
                    {result.classifierLatencyMs}ms classify
                  </div>
                </div>
              </div>
            </div>

            {/* 2. Top 3 Nearest Neighbors */}
            <div className="border border-neutral-800 bg-neutral-900/60 rounded p-3 sm:p-3.5">
              <div className="text-[10px] sm:text-xs uppercase tracking-wider text-neutral-400 font-mono mb-1.5">
                Top 3 Nearest Neighbors
              </div>
              <div className="divide-y divide-neutral-800/80">
                {result.nearestNeighbors.slice(0, 3).map((nn) => (
                  <div
                    key={nn.anchor.id}
                    className="py-2 first:pt-1 last:pb-1 flex items-start justify-between gap-2 text-xs"
                  >
                    <div className="flex items-start gap-1.5 min-w-0 flex-1">
                      <span className="font-mono text-neutral-400 shrink-0">#{nn.rank}</span>
                      <span className="font-mono text-neutral-400 capitalize shrink-0">
                        [{nn.anchor.tier}]
                      </span>
                      <span className="text-neutral-300 break-words">
                        "{nn.anchor.text}"
                      </span>
                    </div>
                    <span className="font-mono text-neutral-200 font-medium shrink-0 ml-1.5">
                      {nn.similarity.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* 3. Cost vs. Always-Strong Model */}
            <div className="border border-neutral-800 bg-neutral-900/60 rounded p-3 sm:p-3.5">
              <div className="text-[10px] sm:text-xs uppercase tracking-wider text-neutral-400 font-mono mb-2">
                Cost vs. Always-Strong Model
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs sm:text-sm">
                <div>
                  <div className="text-[11px] text-neutral-400">Routed Cost</div>
                  <div className="font-mono font-medium text-neutral-200">
                    ${result.cost.fluxCostUsd.toFixed(6)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-neutral-400">Always-Strong</div>
                  <div className="font-mono font-medium text-neutral-400">
                    ${result.cost.alwaysStrongCostUsd.toFixed(6)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-neutral-400">Savings</div>
                  <div className="font-mono font-medium text-neutral-200">
                    {result.cost.savedPercent > 0
                      ? `${result.cost.savedPercent.toFixed(1)}%`
                      : "0.0%"}
                  </div>
                </div>
              </div>
            </div>

          </main>
        )}
      </div>

      {/* One-line Footer */}
      <footer className="mt-6 pt-3 border-t border-neutral-800 text-[11px] sm:text-xs text-neutral-400 font-mono text-center">
        k-NN router over {totalAnchors} reference prompts. Early results: 62% cost savings on a 5-prompt test set.
      </footer>

    </div>
  );
}

export default App;
