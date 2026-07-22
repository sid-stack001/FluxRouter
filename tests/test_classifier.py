"""Smoke test for the complexity classifier."""
from router.classifier.complexity import ComplexityClassifier, Complexity


def test_classifier():
    clf = ComplexityClassifier()

    cases = [
        ("What is the capital of France?", Complexity.SIMPLE),
        ("Write a Python function to parse CSV files.", Complexity.MEDIUM),
        ("Debug why this async Rust code deadlocks under concurrent load.", Complexity.COMPLEX),
        ("Convert this JSON to markdown.", Complexity.SIMPLE),
        ("Design a distributed rate limiter across 50 nodes.", Complexity.COMPLEX),
    ]

    passed = 0
    for query, expected in cases:
        predicted, scores = clf.classify(query)
        status = "✓" if predicted == expected else "✗"
        if predicted == expected:
            passed += 1
        print(f"  {status} [{expected.value:7}] {query[:55]}")
        print(f"      scores: {scores}")

    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    test_classifier()