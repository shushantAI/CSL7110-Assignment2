
import os
import itertools

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_document(filepath: str) -> str:
    with open(filepath, "r") as f:
        text = f.read().strip()
                                           
    cleaned = "".join(ch for ch in text.lower() if ch.isalpha() or ch == " ")
    return cleaned

def char_kgrams(text: str, k: int) -> set:
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def word_kgrams(text: str, k: int) -> set:
    words = text.split()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

def main():
    doc_names = ["D1", "D2", "D3", "D4"]
    docs = {}
    for name in doc_names:
        path = os.path.join(DATA_DIR, f"{name}.txt")
        docs[name] = load_document(path)

    print("=" * 65)
    print("Q1A: K-GRAM CONSTRUCTION")
    print("=" * 65)

    gram_types = {
        "char_2gram": {name: char_kgrams(docs[name], 2) for name in doc_names},
        "char_3gram": {name: char_kgrams(docs[name], 3) for name in doc_names},
        "word_2gram": {name: word_kgrams(docs[name], 2) for name in doc_names},
    }

    for gram_type, gram_sets in gram_types.items():
        print(f"\n{gram_type.upper()}:")
        for name, grams in gram_sets.items():
            print(f"  {name}: {len(grams)} unique k-grams")
                                       
        sample = list(gram_sets["D1"])[:5]
        print(f"  Sample from D1: {sample}")

    print("\n" + "=" * 65)
    print("Q1B: JACCARD SIMILARITY BETWEEN ALL PAIRS (exact)")
    print("=" * 65)

    pairs = list(itertools.combinations(doc_names, 2))

    results = {}
    for gram_type, gram_sets in gram_types.items():
        results[gram_type] = {}
        print(f"\n{gram_type.upper()}:")
        print(f"  {'Pair':<12} {'|A|':>6} {'|B|':>6} {'|A∩B|':>8} {'|A∪B|':>8} {'Jaccard':>10}")
        print(f"  {'-'*55}")
        for d1, d2 in pairs:
            s1 = gram_sets[d1]
            s2 = gram_sets[d2]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            j = inter / union if union > 0 else 1.0
            results[gram_type][(d1, d2)] = j
            print(f"  {d1+'-'+d2:<12} {len(s1):>6} {len(s2):>6} {inter:>8} {union:>8} {j:>10.4f}")

    print("\n\nSUMMARY TABLE (18 Jaccard values: 3 gram types × 6 pairs):")
    print(f"\n  {'Pair':<12}", end="")
    for g in gram_types:
        print(f"  {g:>15}", end="")
    print()
    print(f"  {'-'*60}")
    for d1, d2 in pairs:
        print(f"  {d1+'-'+d2:<12}", end="")
        for g in gram_types:
            print(f"  {results[g][(d1,d2)]:>15.4f}", end="")
        print()

    return results, gram_types

if __name__ == "__main__":
    main()
