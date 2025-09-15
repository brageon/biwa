import numpy as np

# --- Step 1. Base weights (your +scores) ---
weights = {
    "DC": 0.065, "CH": 0.143, "ND": 0.156, "NH": 8, "DH": 0.026, "NC": 0.078,
    "HC": 0.13, "CD": 0.052, "HN": 0.117, "HD": 0.026, "CN": 0.091, "DN": 0.013
}
intentions = list(weights.keys())

# Universal prior
prob_universal = np.array([weights[i] for i in intentions], dtype=float)
prob_universal /= prob_universal.sum()

# --- Step 2. Conditional transitions (toy example, refine later) ---
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

# --- Step 3. Monte Carlo sampler ---
def sample_sequence(length=10, mix=0.5, seed=None):
    """
    length : how many steps
    mix    : 0=only prior, 1=only transitions, in between = blend
    """
    if seed is not None:
        np.random.seed(seed)

    seq = []
    # Initial state from prior
    state = np.random.choice(intentions, p=prob_universal)
    seq.append(state)

    for _ in range(length-1):
        i = intentions.index(state)
        # Blend universal prior + conditional transition
        blended = (1-mix) * prob_universal + mix * prob_next[i]
        blended /= blended.sum()  # renormalize
        state = np.random.choice(intentions, p=blended)
        seq.append(state)

    return seq

# --- Example run ---
print("Monte Carlo sequence:")
print(sample_sequence(length=12, mix=0.6, seed=47))
