"""
Complexity classifier for FluxRouter.
Classifies incoming queries as simple / medium / complex
using k-NN cosine similarity against a large anchor set.

Algorithm:
  - All anchor examples are embedded individually at startup (no mean pooling).
  - For each query, cosine similarity is computed against every anchor embedding.
  - Top-k (default k=5) nearest anchors are selected; majority vote determines tier.
  - Confidence = average cosine similarity of the top-k anchors.

CPU-only, target <10ms per classification.
"""

from __future__ import annotations

import logging
import time
from collections import Counter


import numpy as np
from sentence_transformers import SentenceTransformer
from enum import Enum


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# ---------------------------------------------------------------------------
# Anchor examples per tier — 45 each, diverse domains:
#   coding, general QA, summarisation, architecture/design,
#   debugging, math, creative writing.
# These are embedded individually; no mean-pooling is done.
# ---------------------------------------------------------------------------
ANCHORS: dict[Complexity, list[str]] = {
    Complexity.SIMPLE: [
        # General QA
        "What is the capital of France?",
        "Who invented the telephone?",
        "What does HTTP stand for?",
        "How many days are in a leap year?",
        "What is the boiling point of water in Celsius?",
        "What color is the sky?",
        "What is the speed of light?",
        "What language do people speak in Brazil?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in the solar system?",
        # Coding — simple
        "What does the 'len' function do in Python?",
        "How do I declare a variable in JavaScript?",
        "Write a docstring for this function.",
        "Convert this JSON to a Python dictionary.",
        "What is a null pointer?",
        "How do I print 'Hello World' in Go?",
        "What is an array?",
        "Fix the spelling in this code comment.",
        "Format this date as ISO 8601.",
        "Write a one-line function that returns the absolute value of x.",
        # Summarisation
        "Summarize this paragraph in one sentence.",
        "Give me a three-word title for this passage.",
        "What is the main point of this tweet?",
        "Rewrite this sentence to be shorter.",
        "Condense this bullet list into one sentence.",
        # Math — simple
        "What is 2 + 2?",
        "What is 15% of 200?",
        "Convert 100 Fahrenheit to Celsius.",
        "What is the square root of 144?",
        "How many seconds are in an hour?",
        # Creative writing — simple
        "Write a one-line birthday message for a colleague.",
        "Give me a fun slogan for a coffee shop.",
        "Write a short caption for this photo of a sunset.",
        "Suggest a name for a friendly robot character.",
        "Write a two-sentence product tagline for running shoes.",
        # Miscellaneous simple
        "Translate 'hello' to Spanish.",
        "List the planets in the solar system.",
        "What is the plural of 'mouse'?",
        "Convert this list to bullet points.",
        "What does the acronym 'NASA' stand for?",
        "Spell-check this sentence.",
        "What is a synonym for 'happy'?",
        "What time zone is Tokyo in?",
        "What country uses the euro?",
        "Round 3.14159 to two decimal places.",
    ],

    Complexity.MEDIUM: [
        # Coding — medium
        "Write a Python function to parse CSV files.",
        "Implement a binary search algorithm in Java.",
        "Refactor this class to use dependency injection.",
        "Write unit tests for this authentication module.",
        "Write a regex to validate email addresses.",
        "Create a REST API endpoint that returns paginated results.",
        "How do I implement debounce in JavaScript?",
        "Write a SQL query to find duplicate rows in a table.",
        "Explain the difference between shallow and deep copy in Python.",
        "Implement a stack data structure with push, pop, and peek.",
        # Architecture / design
        "Design a simple URL shortener.",
        "What are the tradeoffs between SQL and NoSQL databases?",
        "Compare REST and GraphQL APIs.",
        "How would you structure a monorepo for a mid-size team?",
        "What design pattern would you use for a plugin system?",
        "Explain the MVC architecture pattern with an example.",
        "How does a CDN improve web performance?",
        "What is the difference between horizontal and vertical scaling?",
        # Debugging — medium
        "Why is my React component re-rendering too often?",
        "My API returns 401 Unauthorized even though I'm passing a token — why?",
        "Why does my Python script run fine locally but fail in Docker?",
        "Explain why this SQL query is slow and how to fix it.",
        "Why is this JavaScript promise not resolving?",
        # NLP / ML concepts
        "Explain how transformers work in NLP.",
        "What is the difference between precision and recall?",
        "Explain the intuition behind gradient descent.",
        "What is overfitting and how do you prevent it?",
        "Compare supervised and unsupervised learning.",
        # Summarisation — medium
        "Summarize this 5-page research paper.",
        "Write an executive summary of this financial report.",
        "Extract the key action items from this meeting transcript.",
        "Summarize this GitHub issue thread and identify the root cause.",
        # Math — medium
        "Solve this system of two linear equations: 2x + y = 10, x - y = 2.",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1.",
        "Explain the central limit theorem in simple terms.",
        "How do you calculate confidence intervals?",
        # Creative writing — medium
        "Write a 200-word product description for a noise-cancelling headset.",
        "Write a short story opening (3 paragraphs) about a time traveller.",
        "Draft a professional email declining a meeting invitation.",
        "Write a LinkedIn bio for a senior software engineer.",
        "Create five interview questions for a data scientist role.",
        "Write a persuasive paragraph arguing for four-day work weeks.",
        "Draft a README introduction for an open-source CLI tool.",
    ],

    Complexity.COMPLEX: [
        # Debugging — complex
        "Debug why this async Rust code deadlocks under concurrent load.",
        "Why does this CUDA kernel have warp divergence and how do I fix it?",
        "Debug this race condition in a distributed transaction system.",
        "My Kafka consumer group is experiencing partition rebalance storms — diagnose and fix.",
        "Identify why this memory leak occurs in a long-running C++ service under heavy load.",
        "Explain why this lock-free ring buffer has an ABA problem and propose a fix.",
        "Our Postgres query planner is choosing a seq scan instead of an index scan — investigate.",
        "Diagnose why our WebSocket server drops connections under 10k concurrent users.",
        # Architecture / design — complex
        "Design a distributed rate limiter that works across 50 nodes.",
        "Architect a real-time fraud detection system for 1M transactions/sec.",
        "Design a multi-tenant SaaS with row-level security and audit logs.",
        "Design a recommendation system that handles cold-start problems.",
        "How would you design a globally distributed key-value store like DynamoDB?",
        "Architect a data pipeline that processes 10TB/day with exactly-once semantics.",
        "Design a CQRS+event-sourcing system for a financial ledger.",
        "How would you implement zero-downtime database schema migrations at scale?",
        "Design an autoscaling strategy for a stateful stream-processing cluster.",
        "Architect a multi-region active-active deployment with conflict resolution.",
        # Coding — complex
        "How would you implement a lock-free concurrent hashmap?",
        "Implement a persistent segment tree with lazy propagation.",
        "Write a Raft consensus leader election implementation in Go.",
        "Implement a query optimizer for a simple SQL engine.",
        "Build a WebAssembly JIT compiler for a subset of Lua.",
        "Implement a garbage collector using tri-color mark-and-sweep.",
        # Math — complex
        "Explain the mathematical intuition behind attention mechanisms.",
        "Analyze the time and space complexity of this dynamic programming solution.",
        "Prove that the halting problem is undecidable.",
        "Derive the backpropagation equations for a convolutional layer.",
        "Explain the epsilon-delta definition of a limit and prove a limit from first principles.",
        "Solve this combinatorics problem: how many ways can 8 non-attacking rooks be placed on a chessboard?",
        # ML / AI — complex
        "How would you fine-tune a large language model with RLHF?",
        "Design a training pipeline for a multi-modal model that fuses vision and text.",
        "Explain how FlashAttention reduces memory complexity of self-attention.",
        "How would you detect and mitigate data poisoning attacks in a federated learning setup?",
        # Creative writing — complex
        "Write a 1000-word short story exploring themes of identity and memory from the perspective of an AI.",
        "Draft a detailed worldbuilding document for a hard science-fiction universe where FTL travel is impossible.",
        "Write a Socratic dialogue between a utilitarian and a Kantian philosopher debating AI ethics.",
        # Systems
        "Explain how Linux's CFS scheduler achieves fair CPU sharing across cgroups.",
        "How does TLS 1.3 differ from TLS 1.2 and what are the security improvements?",
        "Design a kernel-bypass networking stack for ultra-low latency trading.",
        "Explain how MVCC works in PostgreSQL and why it leads to table bloat.",
    ],
}


_logger = logging.getLogger("fluxrouter.classifier")


class ComplexityClassifier:
    """
    Complexity classifier for FluxRouter using k-NN cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 5,
    ) -> None:
        """
        Load the sentence transformer model and pre-compute anchor embeddings.

        Args:
            model_name: Sentence-transformers model to use.
                        'all-MiniLM-L6-v2' — 22M params, fast on CPU, good quality.
            k: Number of nearest neighbours to use for majority vote.
        """
        _logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.k = k
        # List of (Complexity tier, np.ndarray embedding) — one entry per anchor.
        self._anchor_index: list[tuple[Complexity, np.ndarray]] = (
            self._build_anchor_index()
        )
        # Pre-stack anchor embeddings into a matrix for fast batch dot-product.
        self._anchor_matrix: np.ndarray = np.stack(
            [emb for _, emb in self._anchor_index], axis=0
        )  # shape (N_anchors, embed_dim)
        _logger.info(
            f"Ready — {len(self._anchor_index)} anchors indexed (k={self.k})."
        )

    def _build_anchor_index(self) -> list[tuple[Complexity, np.ndarray]]:
        """
        Embed every anchor example individually and store with its tier label.
        No mean-pooling — each anchor is a separate nearest-neighbour candidate.
        """
        index: list[tuple[Complexity, np.ndarray]] = []
        for tier, examples in ANCHORS.items():
            embeddings = self.model.encode(examples, normalize_embeddings=True)
            for emb in embeddings:
                index.append((tier, emb))
        return index

    def classify(self, query: str) -> tuple[Complexity, dict]:
        """
        Classify a query into simple / medium / complex using k-NN majority vote.

        Algorithm:
            1. Embed the query (L2-normalised).
            2. Compute cosine similarity against all anchor embeddings via a
               single matrix dot-product (fast on CPU).
            3. Select top-k most similar anchors.
            4. Majority-vote their tier labels → predicted Complexity.
            5. Per-tier max similarity is computed for the three tiers.
            6. Margin = max_sim(winning tier) − max_sim(best losing tier).
               A large margin means the query sits unambiguously in one tier;
               a small (or negative) margin means it straddles two tiers and
               the cascade escalation logic should treat it with lower confidence.

        Returns:
            (Complexity, info_dict)

            info_dict keys:
                "confidence"      — float: margin between top-1 and top-2 tier-level
                                    max similarities.  Primary ambiguity signal for
                                    Phase 2 cascade escalation.  Range ≈ (−1, 1);
                                    values < ~0.05 indicate a borderline query.
                "knn_confidence"  — float: mean cosine similarity of the top-k anchors
                                    (kept for reference / backward compat).
                "simple"          — float: best cosine sim seen among SIMPLE anchors.
                "medium"          — float: best cosine sim seen among MEDIUM anchors.
                "complex"         — float: best cosine sim seen among COMPLEX anchors.
                "elapsed_ms"      — float: wall-clock time for this call.
        """
        t0 = time.perf_counter()

        # 1. Embed query
        query_emb: np.ndarray = self.model.encode(
            [query], normalize_embeddings=True
        )[0]  # shape (embed_dim,)

        # 2. Cosine similarities against all anchors (dot product on L2-normed vecs)
        sims: np.ndarray = self._anchor_matrix @ query_emb  # shape (N_anchors,)

        # 3. Top-k indices (for knn_confidence reference value)
        k = min(self.k, len(self._anchor_index))
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]  # sort desc
        top_k_sims = sims[top_k_idx]
        top_k_tiers = [self._anchor_index[i][0] for i in top_k_idx]

        # 4. Majority vote
        vote_counts = Counter(top_k_tiers)
        predicted: Complexity = vote_counts.most_common(1)[0][0]

        # 5. Per-tier max similarity
        tier_max: dict[Complexity, float] = {t: -1.0 for t in Complexity}
        for i, (tier, _) in enumerate(self._anchor_index):
            if sims[i] > tier_max[tier]:
                tier_max[tier] = float(sims[i])

        # 6. Margin = winning tier score − best runner-up tier score
        #    Sort tier_max values descending; margin = rank-1 minus rank-2.
        sorted_scores = sorted(tier_max.values(), reverse=True)
        margin = float(sorted_scores[0] - sorted_scores[1])

        # Legacy mean-of-top-k (kept for reference)
        knn_confidence = float(np.mean(top_k_sims))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.info(
            f"'{query[:60]}' → {predicted.value} "
            f"(margin={margin:.3f}, knn_conf={knn_confidence:.3f}, {elapsed_ms:.1f}ms)"
        )


        info: dict = {
            "confidence": margin,           # PRIMARY signal: tier separation margin
            "knn_confidence": knn_confidence,  # secondary: mean sim of top-k anchors
            "simple": tier_max[Complexity.SIMPLE],
            "medium": tier_max[Complexity.MEDIUM],
            "complex": tier_max[Complexity.COMPLEX],
            "elapsed_ms": elapsed_ms,
        }
        return predicted, info