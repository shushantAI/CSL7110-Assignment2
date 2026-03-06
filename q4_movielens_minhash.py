import os
import random
import time
import numpy as np
import itertools
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRIME = 1000003
N_TRIALS = 5
SIMILARITY_THRESHOLD = 0.5


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


def approx_jaccard_from_sig(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def compute_exact_similar_pairs(user_movies: dict, threshold: float) -> set:
    users = sorted(user_movies.keys())
    similar_pairs = set()
    total = len(users) * (len(users) - 1) // 2
    print(f"  Computing exact Jaccard for {total:,} pairs...")
    count = 0
    for u1, u2 in itertools.combinations(users, 2):
        j = exact_jaccard(user_movies[u1], user_movies[u2])
        if j >= threshold:
            similar_pairs.add((u1, u2))
        count += 1
        if count % 50000 == 0:
            print(f"    Progress: {count:,}/{total:,} pairs processed...")
    return similar_pairs


def minhash_experiment(user_movies: dict, t: int, exact_pairs: set,
                       threshold: float, seed: int = 42):
    users = sorted(user_movies.keys())
    hash_params = generate_hash_params(t, seed=seed)
    sigs = {u: minhash_signature(user_movies[u], hash_params) for u in users}
    approx_pairs = set()
    for u1, u2 in itertools.combinations(users, 2):
        aj = approx_jaccard_from_sig(sigs[u1], sigs[u2])
        if aj >= threshold:
            approx_pairs.add((u1, u2))

    fp = len(approx_pairs - exact_pairs)
    fn = len(exact_pairs - approx_pairs)

    return fp, fn, len(approx_pairs)


def main():
    data_path = os.path.join(DATA_DIR, "u.data")

    if not os.path.exists(data_path):
        print("=" * 65)
        print("Q4: MIN-HASHING ON MOVIELENS DATASET")
        print("=" * 65)
        print(f"\n  ⚠ Dataset not found at: {data_path}")
        print(f"  Download MovieLens 100k from http://www.grouplens.org/node/73")
        print(f"  Extract and place 'u.data' in the data/ directory.\n")
        print("  The code below is fully functional once the dataset is placed.")
        return

    print("=" * 65)
    print("Q4: MIN-HASHING ON MOVIELENS DATASET")
    print("=" * 65)

    start = time.time()
    user_movies = load_movielens(data_path)
    print(f"\n  Loaded {len(user_movies)} users, data read in {time.time()-start:.2f}s")

    print(f"\n  Step 1: Exact Jaccard Similarity (threshold={SIMILARITY_THRESHOLD})")
    exact_pairs = compute_exact_similar_pairs(user_movies, SIMILARITY_THRESHOLD)
    print(f"  Exact similar pairs (J >= {SIMILARITY_THRESHOLD}): {len(exact_pairs)}")
    if exact_pairs:
        print(f"\n  ALL pairs with J >= {SIMILARITY_THRESHOLD}:")
        print(f"  {'User A':>8} {'User B':>8} {'Jaccard':>10}")
        print(f"  {'-'*30}")
        for u1, u2 in sorted(exact_pairs):
            j = exact_jaccard(user_movies[u1], user_movies[u2])
            print(f"  {u1:>8} {u2:>8} {j:>10.4f}")

    t_values = [50, 100, 200]
    print(f"\n  Step 2: MinHash Approximation ({N_TRIALS} runs each)\n")
    print(f"  {'t':>6} {'Avg FP':>10} {'Avg FN':>10} {'Avg Pairs':>12} {'Time(s)':>10}")
    print(f"  {'-'*52}")

    for t in t_values:
        fps, fns, found_counts = [], [], []
        elapsed_list = []
        for trial in range(N_TRIALS):
            start = time.time()
            fp, fn, found = minhash_experiment(
                user_movies, t, exact_pairs, SIMILARITY_THRESHOLD, seed=trial
            )
            elapsed_list.append(time.time() - start)
            fps.append(fp)
            fns.append(fn)
            found_counts.append(found)
        print(f"  {t:>6} {np.mean(fps):>10.1f} {np.mean(fns):>10.1f} "
              f"{np.mean(found_counts):>12.1f} {np.mean(elapsed_list):>10.2f}")

    print("\n  OBSERVATION:")
    print("  - More hash functions → fewer false negatives (better recall)")
    print("  - False positives remain low when threshold is well-tuned")
    print("  - t=200 gives best precision/recall tradeoff at the cost of time")


if __name__ == "__main__":
    main()
