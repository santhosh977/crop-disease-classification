import numpy as np
import pickle


# ==========================================
# Load validation data
# ==========================================
X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")

print("Validation data:", X_val.shape)
print("Labels:", y_val.shape)


# ==========================================
# Load best CTM
# ==========================================
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("CTM model loaded successfully!")


# ==========================================
# Get clause activations
# ==========================================
raw_output = ctm.transform(X_val)

clause_outputs = raw_output.reshape(
    len(X_val),
    2,
    1000
).astype(np.float32)


# ==========================================
# Analyze positive-weight clauses only
# ==========================================
for cls in [0, 1]:

    weights = np.array(
        ctm.weight_banks[cls].get_weights()
    )

    true_mask = (y_val == cls)
    opposite_mask = (y_val != cls)

    results = []

    for clause in range(1000):

        # ONLY positive-weight clauses
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
                weights[clause],
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

    print("\n" + "=" * 80)
    print(f"CLASS {cls} — POSITIVE-WEIGHT MISFIRING CLAUSES")
    print("=" * 80)

    print(
        "Clause | Weight | TrueFire | FalseFire | "
        "TrueRate | FalseRate | Gap"
    )

    print("-" * 80)

    for r in results[:20]:

        clause, weight, tf, ff, tr, fr, gap = r

        print(
            f"{clause:6d} | "
            f"{weight:6.0f} | "
            f"{tf:8d} | "
            f"{ff:9d} | "
            f"{tr:8.3f} | "
            f"{fr:9.3f} | "
            f"{gap:7.3f}"
        )


print("\nAnalysis complete!")