"""
Evaluation test for the FluxRouter complexity classifier.

Runs the classifier against the 100-example labeled eval set and reports:
  - Overall accuracy %
  - Per-tier precision, recall, F1
  - Confusion matrix (3×3)
  - Confidence statistics (mean/min/max for correct vs incorrect predictions)

Run with:
    python -m pytest tests/test_classifier_eval.py -v -s
or standalone:
    python tests/test_classifier_eval.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from router.classifier.complexity import Complexity, ComplexityClassifier
from tests.eval_set import EVAL_SET

TIERS = [Complexity.SIMPLE, Complexity.MEDIUM, Complexity.COMPLEX]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _compute_metrics(
    y_true: list[Complexity],
    y_pred: list[Complexity],
) -> tuple[dict, np.ndarray]:
    """
    Returns:
        per_tier_metrics: {Complexity: {"precision", "recall", "f1", "support"}}
        confusion:        np.ndarray shape (3, 3)
                          confusion[i][j] = #examples where true=i, pred=j
    """
    idx = {t: i for i, t in enumerate(TIERS)}
    n = len(TIERS)
    confusion = np.zeros((n, n), dtype=int)

    for true, pred in zip(y_true, y_pred):
        confusion[idx[true]][idx[pred]] += 1

    per_tier: dict[Complexity, dict] = {}
    for tier in TIERS:
        i = idx[tier]
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_tier[tier] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }

    return per_tier, confusion


def _print_report(
    y_true: list[Complexity],
    y_pred: list[Complexity],
    margins: list[float],
    knn_confs: list[float],
    elapsed_list: list[float],
) -> None:
    correct_mask = [t == p for t, p in zip(y_true, y_pred)]
    accuracy = sum(correct_mask) / len(correct_mask) * 100

    per_tier, confusion = _compute_metrics(y_true, y_pred)

    def _split(vals: list[float]) -> tuple[list[float], list[float]]:
        correct = [v for v, ok in zip(vals, correct_mask) if ok]
        wrong   = [v for v, ok in zip(vals, correct_mask) if not ok]
        return correct, wrong

    correct_margins, wrong_margins = _split(margins)
    correct_knns,    wrong_knns    = _split(knn_confs)

    print("\n" + "=" * 65)
    print("  FluxRouter Complexity Classifier — Evaluation Report")
    print("=" * 65)
    print(f"  Examples evaluated : {len(y_true)}")
    print(f"  Overall accuracy   : {accuracy:.1f}%  "
          f"({sum(correct_mask)}/{len(y_true)} correct)")
    print(f"  Mean latency       : {np.mean(elapsed_list):.1f} ms")
    print(f"  p95 latency        : {np.percentile(elapsed_list, 95):.1f} ms")

    print("\n  Per-tier metrics:")
    print(f"  {'Tier':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 50)
    for tier in TIERS:
        m = per_tier[tier]
        print(
            f"  {tier.value:<10} {m['precision']:>10.3f} "
            f"{m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>9}"
        )

    print("\n  Confusion matrix (rows = true tier, cols = predicted tier):")
    header = "  " + " " * 12 + "  ".join(f"{t.value[:7]:>7}" for t in TIERS)
    print(header)
    for i, true_tier in enumerate(TIERS):
        row = "  " + f"{true_tier.value[:10]:<12}"
        row += "  ".join(f"{confusion[i, j]:>7}" for j in range(len(TIERS)))
        print(row)

    # ------------------------------------------------------------------
    # Confidence signal comparison: margin vs knn_confidence
    # ------------------------------------------------------------------
    def _stats(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        return f"mean={np.mean(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}"

    print("\n  Confidence signal comparison:")
    print(f"  {'Signal':<18} {'Correct predictions':<34} {'Incorrect predictions'}")
    print("  " + "-" * 80)
    print(f"  {'margin (primary)':<18} {_stats(correct_margins):<34} {_stats(wrong_margins)}")
    print(f"  {'knn_confidence':<18} {_stats(correct_knns):<34} {_stats(wrong_knns)}")

    # Delta = mean(correct) - mean(incorrect) — higher is better discriminator
    if correct_margins and wrong_margins:
        delta_margin = np.mean(correct_margins) - np.mean(wrong_margins)
        delta_knn    = np.mean(correct_knns)    - np.mean(wrong_knns)
        print(f"\n  Discrimination delta (correct_mean - incorrect_mean):")
        print(f"    margin:          {delta_margin:+.4f}")
        print(f"    knn_confidence:  {delta_knn:+.4f}")
        winner = "margin" if abs(delta_margin) > abs(delta_knn) else "knn_confidence"
        print(f"    → better signal: {winner}")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Test entry point (pytest discovers this function)
# ---------------------------------------------------------------------------

def test_classifier_eval() -> None:
    """
    Full eval run on the 100-example held-out set.

    Assertions:
      - overall accuracy >= 70%  (hard pass threshold)
      - per-tier F1 >= 0.50 for each tier

    Note on medium recall: medium is a deliberately fuzzy middle band — queries
    naturally bleed into simple/complex at the margins.  See README for the
    design decision: cascade uses margin to escalate borderline medium calls
    rather than trying to force high recall through anchor tuning alone.
    """
    clf = ComplexityClassifier()

    y_true: list[Complexity] = []
    y_pred: list[Complexity] = []
    margins: list[float] = []
    knn_confs: list[float] = []
    elapsed_list: list[float] = []

    for query, label in EVAL_SET:
        predicted, info = clf.classify(query)
        y_true.append(label)
        y_pred.append(predicted)
        margins.append(info["confidence"])       # margin is now the primary signal
        knn_confs.append(info["knn_confidence"])
        elapsed_list.append(info["elapsed_ms"])

    _print_report(y_true, y_pred, margins, knn_confs, elapsed_list)

    # --- Assertions ---
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) * 100
    per_tier, _ = _compute_metrics(y_true, y_pred)

    assert accuracy >= 70.0, (
        f"Overall accuracy {accuracy:.1f}% is below the 70% threshold."
    )
    for tier in TIERS:
        f1 = per_tier[tier]["f1"]
        assert f1 >= 0.50, (
            f"F1 for '{tier.value}' is {f1:.3f}, below the 0.50 threshold."
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_classifier_eval()
