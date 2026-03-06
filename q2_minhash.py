"""
Q2: Min-Hashing
CSL7110 Assignment 2
"""

import os
import time
import random
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Large prime > 10,000 for hash function modulus
PRIME = 104729  # prime > 10,000


def load_document(filepath: str) -> str:
    with open(filepath, "r") as f:
        text = f.read().strip()
    return "".join(ch for ch in text.lower() if ch.isalpha() or ch == " ")


def char_kgrams(text: str, k: int) -> set:
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def build_universe(all_gram_sets: list) -> list:
    """Build a sorted universe list from multiple k-gram sets."""
    universe = sorted(set().union(*all_gram_sets))
    return universe


def gram_to_index(universe: list) -> dict:
    return {g: i for i, g in enumerate(universe)}


def generate_hash_params(t: int, seed: int = 42) -> list:
    """Generate t sets of (a, b) for hash functions h(x) = (a*x + b) % PRIME."""
    rng = random.Random(seed)
    params = []
    for _ in range(t):
        a = rng.randint(1, PRIME - 1)
        b = rng.randint(0, PRIME - 1)
        params.append((a, b))
    return params


def minhash_signature(gram_set: set, universe_index: dict, hash_params: list) -> np.ndarray:
    """Compute MinHash signature vector for a set."""
    t = len(hash_params)
    sig = np.full(t, np.inf)
    for gram in gram_set:
        if gram not in universe_index:
            continue
        x = universe_index[gram]
        for i, (a, b) in enumerate(hash_params):
            h = (a * x + b) % PRIME
            if h < sig[i]:
                sig[i] = h
    return sig


def approx_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Approximate Jaccard similarity from MinHash signatures."""
    return float(np.mean(sig_a == sig_b))


def exact_jaccard(set_a: set, set_b: set) -> float:
    u = len(set_a | set_b)
    return len(set_a & set_b) / u if u > 0 else 1.0


def run_minhash_experiment(t_values: list, gram_sets: dict, doc_pair=("D1", "D2")):
    """Q2A: Run minhash for multiple t values on a pair."""
    d1, d2 = doc_pair
    all_sets = list(gram_sets.values())
    universe = build_universe(all_sets)
    u_index = gram_to_index(universe)

    true_j = exact_jaccard(gram_sets[d1], gram_sets[d2])

    print("=" * 65)
    print(f"Q2A: MIN-HASH APPROXIMATE JACCARD SIMILARITY ({d1} vs {d2})")
    print("=" * 65)
    print(f"\n  True Jaccard similarity: {true_j:.4f}")
    print(f"  Universe size: {len(universe)} unique 3-grams\n")
    print(f"  {'t':>6} {'Approx Jaccard':>16} {'Error':>10} {'Time (s)':>10}")
    print(f"  {'-'*45}")

    results = {}
    for t in t_values:
        start = time.time()
        params = generate_hash_params(t)
        sig1 = minhash_signature(gram_sets[d1], u_index, params)
        sig2 = minhash_signature(gram_sets[d2], u_index, params)
        approx = approx_jaccard(sig1, sig2)
        elapsed = time.time() - start
        error = abs(approx - true_j)
        results[t] = approx
        print(f"  {t:>6} {approx:>16.4f} {error:>10.4f} {elapsed:>10.4f}")

    return results, true_j


def find_best_t(gram_sets: dict, doc_pair=("D1", "D2"), n_trials=5):
    """Q2B: Experiment to find good t value."""
    d1, d2 = doc_pair
    all_sets = list(gram_sets.values())
    universe = build_universe(all_sets)
    u_index = gram_to_index(universe)
    true_j = exact_jaccard(gram_sets[d1], gram_sets[d2])

    t_candidates = [20, 60, 100, 150, 200, 300, 400, 600]
    print("\n" + "=" * 65)
    print("Q2B: FINDING THE BEST VALUE OF t")
    print("=" * 65)
    print(f"\n  True Jaccard: {true_j:.4f}")
    print(f"\n  Running {n_trials} trials per t value...\n")
    print(f"  {'t':>6} {'Mean Error':>12} {'Std Error':>12} {'Mean Time(s)':>14}")
    print(f"  {'-'*48}")

    for t in t_candidates:
        errors = []
        times = []
        for seed in range(n_trials):
            start = time.time()
            params = generate_hash_params(t, seed=seed)
            sig1 = minhash_signature(gram_sets[d1], u_index, params)
            sig2 = minhash_signature(gram_sets[d2], u_index, params)
            approx = approx_jaccard(sig1, sig2)
            elapsed = time.time() - start
            errors.append(abs(approx - true_j))
            times.append(elapsed)
        print(f"  {t:>6} {np.mean(errors):>12.4f} {np.std(errors):>12.4f} {np.mean(times):>14.4f}")


def main():
    doc_names = ["D1", "D2", "D3", "D4"]
    docs = {}
    for name in doc_names:
        path = os.path.join(DATA_DIR, f"{name}.txt")
        docs[name] = load_document(path)

    gram_sets = {name: char_kgrams(docs[name], 3) for name in doc_names}

    t_values = [20, 60, 150, 300, 600]
    results, true_j = run_minhash_experiment(t_values, gram_sets)

    find_best_t(gram_sets)

    print("\nOBSERVATION:")
    print("  - Error decreases as t increases (more hash functions = better estimate)")
    print("  - t=150 offers a good balance: low error with manageable computation time")
    print("  - Beyond t=300, gains in accuracy diminish while compute cost grows linearly")


if __name__ == "__main__":
    main()
