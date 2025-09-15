import numpy as np

# Node scores (from your mapping)
weights = {
    "DC": 5,   # hyperbolic [std]
    "CH": 11,  # elliptic curve [cdf]
    "ND": 12,  # cancellation [covariance]
    "NH": 8,   # path [jumps]
    "DH": 2,   # spiral [alignment]
    "NC": 6,   # sine wave [fourier]
    "HC": 10,  # star graph
    "CD": 4,   # zig-zag [modal_shift]
    "HN": 9,   # DAG [cadence]
    "HD": 2,   # decay spiral [color_jumps]
    "CN": 7,   # lattice [mean]
    "DN": 1    # modus [kde]
}

# Convert to numpy vector (sorted in canonical BIWA order)
intentions = ["HC", "CD", "HD", "CH", "DC", "DN", "DH", "NH", "HN", "CN", "NC", "ND"]
prob_universal = np.array([weights[i] for i in intentions], dtype=float)

# Normalize into probabilities
prob_universal /= prob_universal.sum()

print("Universal node probabilities:")
for i, p in zip(intentions, prob_universal):
    print(f"{i}: {p:.3f}")
