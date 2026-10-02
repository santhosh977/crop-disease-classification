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
# Load saved best CTM
# ==========================================
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("CTM model loaded successfully!")


# Clause activation output
raw_output = ctm.transform(X_val)

clause_outputs = raw_output.reshape(
    len(X_val),
    2,
    1000
)

clause_outputs = clause_outputs == 1


# Calculate class votes directly from clause outputs
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
])

class_sums = np.sum(
    raw_output.reshape(len(X_val), 2, 1000) * weights[np.newaxis, :, :],
    axis=2
)

y_pred = np.argmax(class_sums, axis=1)


# ==========================================
# Analyze each class
# ==========================================
for cls in [0, 1]:

    print("\n" + "=" * 80)
    print(f"CLASS {cls} VALIDATION CLAUSE ANALYSIS")
    print("=" * 80)

    true_mask = (y_val == cls)
    opposite_mask = (y_val != cls)

    wrong_pred_mask = (
        (y_val != cls) &
        (y_pred == cls)
    )

    results = []

    for clause in range(1000):

        fired = clause_outputs[:, cls, clause]

        true_fires = np.sum(
            fired & true_mask
        )

        opposite_fires = np.sum(
            fired & opposite_mask
        )

        wrong_prediction_fires = np.sum(
            fired & wrong_pred_mask
        )

        true_total = np.sum(true_mask)
        opposite_total = np.sum(opposite_mask)

        true_fire_rate = (
            true_fires / true_total
            if true_total > 0 else 0
        )

        false_fire_rate = (
            opposite_fires / opposite_total
            if opposite_total > 0 else 0
        )

        misfire_gap = (
            false_fire_rate -
            true_fire_rate
        )

        results.append(
            (
                clause,
                true_fires,
                opposite_fires,
                wrong_prediction_fires,
                true_fire_rate,
                false_fire_rate,
                misfire_gap
            )
        )

    # Largest false-firing gap first
    results.sort(
        key=lambda x: x[6],
        reverse=True
    )

    print("\nTop 20 candidate unreliable clauses:")
    print(
        "Clause | TrueFire | FalseFire | WrongPred | "
        "TrueRate | FalseRate | Gap"
    )
    print("-" * 80)

    for r in results[:20]:

        clause, tf, ff, wp, tr, fr, gap = r

        print(
            f"{clause:6d} | "
            f"{tf:8d} | "
            f"{ff:9d} | "
            f"{wp:9d} | "
            f"{tr:8.3f} | "
            f"{fr:9.3f} | "
            f"{gap:7.3f}"
        )


print("\nValidation clause analysis complete!")