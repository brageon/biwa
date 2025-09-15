import numpy as np
from collections import Counter

# --- Base BIWA weights ---
weights = {
    "DC": 0.065, "CH": 0.143, "ND": 0.156, "NH": 0.104, "DH": 0.026, "NC": 0.078,
    "HC": 0.13, "CD": 0.052, "HN": 0.117, "HD": 0.026, "CN": 0.091, "DN": 0.013
}
intentions = list(weights.keys())
prob_universal = np.array([weights[i] for i in intentions], dtype=float)
prob_universal /= prob_universal.sum()

# --- Simple transition model (can refine later) ---
n = len(intentions)
prob_next = np.zeros((n, n))
bias, fwd, bwd, opp, diag_p = 0.25, 0.30, 0.15, 0.10, 0.20
for i in range(n):
    prob_next[i, i] = bias
    prob_next[i, (i+1) % n] = fwd
    prob_next[i, (i-1) % n] = bwd
    prob_next[i, (i+n//2) % n] = opp
    prob_next[i, (i+2) % n] = diag_p / 2
    prob_next[i, (i-2) % n] = diag_p / 2
prob_next /= prob_next.sum(axis=1, keepdims=True)

# --- Sequence generator ---
def sample_sequence(length=10, mix=0.6, seed=None):
    if seed is not None:
        np.random.seed(seed)
    seq = []
    state = np.random.choice(intentions, p=prob_universal)
    seq.append(state)
    for _ in range(length-1):
        i = intentions.index(state)
        blended = (1-mix) * prob_universal + mix * prob_next[i]
        blended /= blended.sum()
        state = np.random.choice(intentions, p=blended)
        seq.append(state)
    return seq

# --- Groupings for TEE/TET ---
TEE = {"CD", "ND", "NH", "HC", "NC", "DH"}
TET = {"DC", "HN", "DN", "CH", "CN", "HD"}

# --- Scoring function ---
def score_sequence(seq):
    counts = Counter(seq)
    total = len(seq)
    tee_ratio = sum(counts[i] for i in TEE) / total
    tet_ratio = sum(counts[i] for i in TET) / total
    # entropy = balance measure
    probs = np.array(list(counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs))
    # symmetry = closeness of tee vs tet
    symmetry = 1 - abs(tee_ratio - tet_ratio)
    return {"tee": tee_ratio, "tet": tet_ratio,
            "entropy": entropy, "symmetry": symmetry}

# --- Monte Carlo test ---
def monte_carlo(n_runs=10000, length=12, mix=0.6):
    results = []
    for r in range(n_runs):
        seq = sample_sequence(length, mix)
        scores = score_sequence(seq)
        results.append((seq, scores))
    # rank by symmetry first, then entropy
    results.sort(key=lambda x: (x[1]["symmetry"], x[1]["entropy"]), reverse=True)
    return results

# --- Example run ---
best = monte_carlo(n_runs=1000, length=12, mix=0.6)[:12]
for seq, score in best:
    print(seq, score)
