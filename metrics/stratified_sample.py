import random
from collections import defaultdict
from router.classifier.complexity import Complexity

random.seed(42)  # reproducible sample — same 30 queries every run


def stratified_sample(
    eval_set: list[tuple[str, Complexity]], 
    n_total: int = 30
) -> list[tuple[str, Complexity]]:
    """
    eval_set: list of (query, complexity_enum)
    Returns a stratified subset of size n_total, proportional per tier.
    """
    by_tier = defaultdict(list)
    for query, tier in eval_set:
        # Group by enum value (e.g., "simple", "medium", "complex") or tier enum itself
        by_tier[tier.value].append((query, tier))

    tiers = list(by_tier.keys())
    n_per_tier = n_total // len(tiers)  # e.g. 30 // 3 = 10 per tier
    remainder = n_total % len(tiers)

    sample = []
    for i, tier in enumerate(tiers):
        take = n_per_tier + (1 if i < remainder else 0)
        pool = by_tier[tier]
        take = min(take, len(pool))
        sample.extend(random.sample(pool, take))

    random.shuffle(sample)
    return sample