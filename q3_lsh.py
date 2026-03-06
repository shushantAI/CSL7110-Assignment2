"""
Q3: Locality Sensitive Hashing (LSH)
CSL7110 Assignment 2
"""

import os
import math
import random
import itertools
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRIME = 104729
T_LSH = 160
TAU = 0.7


def load_document(filepath: str) -> str:
    with open(filepath, "r") as f:
        text = f.read().strip()
    return "".join(ch for ch in text.lower() if ch.isalpha() or ch == " ")


def char_kgrams(text: str, k: int) -> set:
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def build_universe(all_gram_sets: list) -> list:
    return sorted(set().union(*all_gram_sets))


def gram_to_index(universe: list) -> dict:
    return {g: i for i, g in enumerate(universe)}


def generate_hash_params(t: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [(rng.randint(1, PRIME - 1), rng.randint(0, PRIME - 1)) for _ in range(t)]


def minhash_signature(gram_set: set, universe_index: dict, hash_params: list) -> np.ndarray:
    t = len(hash_params)
    sig = np.full(t, float("inf"))
    for gram in gram_set:
        if gram not in universe_index:
            continue
        x = universe_index[gram]
        for i, (a, b) in enumerate(hash_params):
            h = (a * x + b) % PRIME
            if h < sig[i]:
                sig[i] = h
    return sig


def exact_jaccard(set_a: set, set_b: set) -> float:
    u = len(set_a | set_b)
    return len(set_a & set_b) / u if u > 0 else 1.0


def scurve(s: float, b: int, r: int) -> float:
    """LSH S-curve probability: P(candidate pair) = 1 - (1 - s^r)^b"""
    return 1.0 - (1.0 - s ** r) ** b


def find_best_band_row(t: int, tau: float):
    """
    Q3A: Find (b, r) such that b*r = t and the S-curve has
    inflection close to tau with good separation.
    """
    print("=" * 65)
    print(f"Q3A: FINDING BEST (b, r) FOR t={t}, τ={tau}")
    print("=" * 65)

    candidates = []
    for r in range(1, t + 1):
        if t % r == 0:
            b = t // r
            # S-curve threshold (inflection point approx at (1/b)^(1/r))
            threshold = (1.0 / b) ** (1.0 / r)
            # Probability at tau (should be high) and at tau-0.2 (should be low)
            p_above = scurve(tau, b, r)
            p_below = scurve(tau - 0.2, b, r)
            separation = p_above - p_below
            candidates.append((b, r, threshold, p_above, p_below, separation))

    # Sort by separation descending (best separation = sharpest curve at tau)
    candidates.sort(key=lambda x: -x[5])

    print(f"\n  Top candidates (sorted by separation at τ={tau}):")
    print(f"  {'b':>6} {'r':>6} {'Thresh':>10} {'P(τ)':>10} {'P(τ-0.2)':>12} {'Sep':>10}")
    print(f"  {'-'*58}")
    for row in candidates[:10]:
        b, r, thr, pa, pb, sep = row
        print(f"  {b:>6} {r:>6} {thr:>10.4f} {pa:>10.4f} {pb:>12.4f} {sep:>10.4f}")

    # Best choice
    best = candidates[0]
    b_best, r_best = best[0], best[1]
    print(f"\n  ✓ Best choice: b={b_best}, r={r_best}")
    print(f"    Threshold ≈ {best[2]:.4f}, P(similarity={tau}) = {best[3]:.4f}")
    print(f"    S-curve inflects sharply around τ={tau}\n")

    # Print a few S-curve values for best choice
    print(f"  S-curve f(s) for b={b_best}, r={r_best}:")
    for s in [0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        print(f"    f({s:.2f}) = {scurve(s, b_best, r_best):.4f}")

    return b_best, r_best


def lsh_candidate_pairs(signatures: dict, b: int, r: int) -> set:
    """Apply LSH banding to find candidate pairs."""
    doc_names = list(signatures.keys())
    candidate_pairs = set()
    t = len(next(iter(signatures.values())))

    for band_idx in range(b):
        start = band_idx * r
        end = start + r
        buckets = {}
        for name in doc_names:
            band_hash = tuple(signatures[name][start:end])
            buckets.setdefault(band_hash, []).append(name)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidate_pairs.add(tuple(sorted(pair)))

    return candidate_pairs


def approx_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def main():
    doc_names = ["D1", "D2", "D3", "D4"]
    docs = {name: load_document(os.path.join(DATA_DIR, f"{name}.txt"))
            for name in doc_names}
    gram_sets = {name: char_kgrams(docs[name], 3) for name in doc_names}

    universe = build_universe(list(gram_sets.values()))
    u_index = gram_to_index(universe)

    # Q3A: find best b, r
    b_best, r_best = find_best_band_row(T_LSH, TAU)

    # Q3B: compute probability for each pair
    print("=" * 65)
    print("Q3B: PROBABILITY OF EACH PAIR BEING A CANDIDATE (using LSH)")
    print("=" * 65)

    hash_params = generate_hash_params(T_LSH)
    sigs = {name: minhash_signature(gram_sets[name], u_index, hash_params)
            for name in doc_names}

    pairs = list(itertools.combinations(doc_names, 2))

    print(f"\n  Using b={b_best}, r={r_best}, t={T_LSH}, τ={TAU}\n")
    print(f"  {'Pair':<12} {'Exact J':>10} {'Approx J':>10} {'P(candidate)':>14} {'Predicted':>10}")
    print(f"  {'-'*60}")

    for d1, d2 in pairs:
        exact_j = exact_jaccard(gram_sets[d1], gram_sets[d2])
        approx_j = approx_jaccard(sigs[d1], sigs[d2])
        p_cand = scurve(exact_j, b_best, r_best)
        predicted = "YES" if p_cand >= 0.5 else "NO"
        print(f"  {d1+'-'+d2:<12} {exact_j:>10.4f} {approx_j:>10.4f} {p_cand:>14.4f} {predicted:>10}")

    # Also run actual LSH to verify
    print(f"\n  Actual LSH candidate pairs found (b={b_best}, r={r_best}):")
    cands = lsh_candidate_pairs(sigs, b_best, r_best)
    if cands:
        for pair in sorted(cands):
            ej = exact_jaccard(gram_sets[pair[0]], gram_sets[pair[1]])
            print(f"    {pair[0]}-{pair[1]}  (Exact J = {ej:.4f})")
    else:
        print("    No candidate pairs found above threshold.")


if __name__ == "__main__":
    main()
