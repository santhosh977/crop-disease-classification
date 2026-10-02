import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# Load validation data
# ==========================================
X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")


# ==========================================
# Load refined CTM
# ==========================================
with open("refined_binary_ctm_v1.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined CTM loaded successfully!")


# ==========================================
# Clause activations
# ==========================================
raw_output = ctm.transform(X_val)

clause_outputs = raw_output.reshape(
    len(X_val),
    2,
    1000
).astype(np.float32)


# ==========================================
# Weights
# ==========================================
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)


# ==========================================
# Original predictions
# ==========================================
base_scores = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)

base_pred = np.argmax(
    base_scores,
    axis=1
)

base_accuracy = accuracy_score(y_val, base_pred)
base_f1 = f1_score(
    y_val,
    base_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# Analyze every positive clause
# ==========================================
results = []

for cls in [0, 1]:

    for clause in range(1000):

        if weights[cls, clause] <= 0:
            continue

        modified_scores = base_scores.copy()

        # Flip +1 -> -1
        modified_scores[:, cls] -= (
            2 * clause_outputs[:, cls, clause]
        )

        modified_pred = np.argmax(
            modified_scores,
            axis=1
        )

        accuracy = accuracy_score(
            y_val,
            modified_pred
        )

        f1 = f1_score(
            y_val,
            modified_pred,
            average="macro",
            zero_division=0
        )

        # Which predictions changed?
        changed = np.where(
            base_pred != modified_pred
        )[0]

        corrected = 0
        harmed = 0

        for idx in changed:

            original_wrong = (
                base_pred[idx] != y_val[idx]
            )

            new_correct = (
                modified_pred[idx] == y_val[idx]
            )

            original_correct = (
                base_pred[idx] == y_val[idx]
            )

            new_wrong = (
                modified_pred[idx] != y_val[idx]
            )

            if original_wrong and new_correct:
                corrected += 1

            if original_correct and new_wrong:
                harmed += 1

        acc_change = accuracy - base_accuracy
        f1_change = f1 - base_f1

        # Keep only flips that improve accuracy
        if acc_change > 0:

            results.append(
                (
                    cls,
                    clause,
                    accuracy,
                    f1,
                    acc_change,
                    f1_change,
                    len(changed),
                    corrected,
                    harmed
                )
            )


# ==========================================
# Sort by:
# Accuracy improvement
# Then corrections
# Then F1 improvement
# ==========================================
results.sort(
    key=lambda x: (x[4], x[7], x[5]),
    reverse=True
)


# ==========================================
# Display
# ==========================================
print("\n" + "=" * 100)
print("ALL POSITIVE CLAUSE FLIPS THAT IMPROVE VALIDATION ACCURACY")
print("=" * 100)

print(
    "Class | Clause | Accuracy | MacroF1 | "
    "AccGain | F1Gain | Changed | Corrected | Harmed"
)

print("-" * 100)

for r in results:

    cls, clause, acc, f1, ag, fg, changed, corrected, harmed = r

    print(
        f"{cls:5d} | "
        f"{clause:6d} | "
        f"{acc:.4f}   | "
        f"{f1:.4f}  | "
        f"{ag:+.4f}  | "
        f"{fg:+.4f} | "
        f"{changed:7d} | "
        f"{corrected:9d} | "
        f"{harmed:6d}"
    )


print("\nTotal improving clauses:", len(results))
print("\nAnalysis complete!")