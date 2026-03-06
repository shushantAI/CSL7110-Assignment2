
import os
import random
import time
import numpy as np
import itertools
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRIME = 1000003
N_TRIALS = 5

def load_movielens(filepath: str) -> dict:
    user_movies = defaultdict(set)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            user_id, movie_id = int(parts[0]), int(parts[1])
            user_movies[user_id].add(movie_id)
    return dict(user_movies)

def exact_jaccard(set_a: set, set_b: set) -> float:
    u = len(set_a | set_b)
    return len(set_a & set_b) / u if u > 0 else 1.0

def generate_hash_params(t: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [(rng.randint(1, PRIME - 1), rng.randint(0, PRIME - 1)) for _ in range(t)]

def minhash_signature(movie_set: set, hash_params: list) -> np.ndarray:
    t = len(hash_params)
    sig = np.full(t, np.inf)
    for movie_id in movie_set:
        for i, (a, b) in enumerate(hash_params):
            h = (a * movie_id + b) % PRIME
            if h < sig[i]:
                sig[i] = h
    return sig

def lsh_candidate_pairs(signatures: dict, b: int, r: int) -> set:
    users = list(signatures.keys())
    candidates = set()
    for band_idx in range(b):
        start = band_idx * r
        end = start + r
        buckets = defaultdict(list)
        for u in users:
            band_key = tuple(signatures[u][start:end])
            buckets[band_key].append(u)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(sorted(bucket), 2):
                    candidates.add(pair)
    return candidates

def compute_exact_pairs(user_movies: dict, threshold: float) -> set:
    users = sorted(user_movies.keys())
    exact = set()
    for u1, u2 in itertools.combinations(users, 2):
        if exact_jaccard(user_movies[u1], user_movies[u2]) >= threshold:
            exact.add((u1, u2))
    return exact

def lsh_experiment(user_movies: dict, t: int, b: int, r: int,
                   exact_pairs: set, threshold: float,
                   seed: int = 42):
    hash_params = generate_hash_params(t, seed=seed)
    sigs = {u: minhash_signature(user_movies[u], hash_params)
            for u in user_movies}
    candidates = lsh_candidate_pairs(sigs, b, r)

    found_pairs = set()
    for u1, u2 in candidates:
        aj = np.mean(sigs[u1] == sigs[u2])
        if aj >= threshold:
            found_pairs.add((min(u1, u2), max(u1, u2)))

    fp = len(found_pairs - exact_pairs)
    fn = len(exact_pairs - found_pairs)
    return fp, fn, len(candidates)

def run_lsh_config(user_movies, config_label, t, b, r, exact_pairs_06, exact_pairs_08):
    print(f"\n  Config: {config_label}  (t={t}, b={b}, r={r})")
    print(f"  {'-'*55}")

    for threshold, ep_label, exact_pairs in [
        (0.6, "τ=0.6", exact_pairs_06),
        (0.8, "τ=0.8", exact_pairs_08)
    ]:
        fps, fns, cands = [], [], []
        for trial in range(N_TRIALS):
            fp, fn, nc = lsh_experiment(user_movies, t, b, r, exact_pairs, threshold, seed=trial)
            fps.append(fp)
            fns.append(fn)
            cands.append(nc)
        print(f"  {ep_label}  Avg FP={np.mean(fps):.1f}  Avg FN={np.mean(fns):.1f}  "
              f"Exact pairs={len(exact_pairs)}  Avg candidates={np.mean(cands):.0f}")

def main():
    data_path = os.path.join(DATA_DIR, "u.data")

    if not os.path.exists(data_path):
        print("=" * 65)
        print("Q5: LSH ON MOVIELENS DATASET")
        print("=" * 65)
        print(f"\n  ⚠ Dataset not found at: {data_path}")
        print(f"  Download MovieLens 100k from http://www.grouplens.org/node/73")
        print(f"  Extract and place 'u.data' in the data/ directory.\n")
        return

    print("=" * 65)
    print("Q5: LSH ON MOVIELENS DATASET")
    print("=" * 65)

    user_movies = load_movielens(data_path)
    print(f"\n  Loaded {len(user_movies)} users")

    print(f"\n  Pre-computing exact pairs for τ=0.6 and τ=0.8...")
    exact_pairs_06 = compute_exact_pairs(user_movies, 0.6)
    exact_pairs_08 = compute_exact_pairs(user_movies, 0.8)
    print(f"  Exact pairs (τ=0.6): {len(exact_pairs_06)}")
    print(f"  Exact pairs (τ=0.8): {len(exact_pairs_08)}")

    configs = [
        ("r=5, b=10", 50, 10, 5),
        ("r=5, b=20", 100, 20, 5),
        ("r=5, b=40", 200, 40, 5),
        ("r=10, b=20", 200, 20, 10),
    ]

    print(f"\n  Running {N_TRIALS} trials per config...\n")
    for label, t, b, r in configs:
        run_lsh_config(user_movies, label, t, b, r, exact_pairs_06, exact_pairs_08)

    print("\n  OBSERVATION:")
    print("  - Higher r (rows per band) → stricter match → fewer FP but more FN")
    print("  - Higher b (bands) → more candidates → fewer FN but more FP")
    print("  - At τ=0.8, fewer true pairs exist, so FN is harder to control")
    print("  - r=5, b=40 (t=200) gives best recall at τ=0.6")

if __name__ == "__main__":
    main()
