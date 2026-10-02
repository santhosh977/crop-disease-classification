import numpy as np
import pickle


# ==========================================
# Load test data
# ==========================================
X_test = np.load("X_test_binary.npy")
y_test = np.load("y_test_binary.npy")

# Load saved best CTM
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("CTM model loaded successfully!")


# ==========================================
# Predictions
# ==========================================
y_pred = ctm.predict(X_test)

# Clause outputs
raw_output = ctm.transform(X_test)

# 2 classes × 1000 clauses
clause_outputs = raw_output.reshape(
    len(X_test),
    2,
    1000
)

clause_outputs = clause_outputs == 1


# ==========================================
# Analyze each class
# ==========================================
for cls in [0, 1]:

    print("\n" + "=" * 80)
    print(f"CLASS {cls} CLAUSE ANALYSIS")
    print("=" * 80)

    true_mask = (y_test == cls)
    opposite_mask = (y_test != cls)
    wrong_pred_mask = (y_test != cls) & (y_pred == cls)

    results = []

    for clause in range(1000):

        fired = clause_outputs[:, cls, clause]

        true_fires = np.sum(fired & true_mask)
        opposite_fires = np.sum(fired & opposite_mask)
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

        # Higher means the clause fires relatively more
        # on the opposite class than on its own class.
        misfire_gap = false_fire_rate - true_fire_rate

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

    # Rank clauses by largest false-firing gap
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


print("\nAnalysis complete!")