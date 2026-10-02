import numpy as np
import pickle


# ==========================================
# Load train + validation data
# ==========================================
X_train = np.load("X_train_binary.npy")
y_train = np.load("y_train_binary.npy")

X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")

X = np.concatenate([X_train, X_val], axis=0)
y = np.concatenate([y_train, y_val], axis=0)

print("Combined data:", X.shape)
print("Combined labels:", y.shape)


# ==========================================
# Load best ORIGINAL CTM
# ==========================================
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Original CTM loaded successfully!")


# ==========================================
# Get clause activations
# ==========================================
raw_output = ctm.transform(X)

clause_outputs = raw_output.reshape(
    len(X),
    2,
    1000
).astype(np.float32)


# ==========================================
# Analyze positive-weight clauses
# ==========================================
for cls in [0, 1]:

    weights = np.array(
        ctm.weight_banks[cls].get_weights()
    )

    true_mask = (y == cls)
    opposite_mask = (y != cls)

    results = []

    for clause in range(1000):

        # We only investigate positive clauses
        if weights[clause] <= 0:
            continue

        fired = clause_outputs[:, cls, clause] == 1

        true_fires = np.sum(
            fired & true_mask
        )

        opposite_fires = np.sum(
            fired & opposite_mask
        )

        true_total = np.sum(true_mask)
        opposite_total = np.sum(opposite_mask)

        true_rate = true_fires / true_total
        false_rate = opposite_fires / opposite_total

        gap = false_rate - true_rate

        results.append(
            (
                clause,
                int(weights[clause]),
                true_fires,
                opposite_fires,
                true_rate,
                false_rate,
                gap
            )
        )

    # Largest opposite-class firing first
    results.sort(
        key=lambda x: x[6],
        reverse=True
    )

    print("\n" + "=" * 85)
    print(f"CLASS {cls} — STABLE POSITIVE-WEIGHT MISFIRING CLAUSES")
    print("=" * 85)

    print(
        "Clause | Weight | TrueFire | FalseFire | "
        "TrueRate | FalseRate | Gap"
    )

    print("-" * 85)

    for r in results[:20]:

        clause, weight, tf, ff, tr, fr, gap = r

        print(
            f"{clause:6d} | "
            f"{weight:6d} | "
            f"{tf:8d} | "
            f"{ff:9d} | "
            f"{tr:8.3f} | "
            f"{fr:9.3f} | "
            f"{gap:7.3f}"
        )


print("\nStable clause analysis complete!")