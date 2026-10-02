import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# Load validation data
# ==========================================
X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")

print("Validation data:", X_val.shape)


# ==========================================
# Load current refined CTM
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
# Clause weights
# ==========================================
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)


# ==========================================
# Original prediction
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

base_accuracy = accuracy_score(
    y_val,
    base_pred
)

base_f1 = f1_score(
    y_val,
    base_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 80)
print("CURRENT REFINED MODEL")
print("=" * 80)
print(f"Accuracy : {base_accuracy:.4f}")
print(f"Macro F1 : {base_f1:.4f}")


# ==========================================
# Test flipping POSITIVE clauses
# +1  --->  -1
# ==========================================
results = []

for cls in [0, 1]:

    for clause in range(1000):

        weight = weights[cls, clause]

        # Only positive clauses are candidates
        if weight <= 0:
            continue

        modified_scores = base_scores.copy()

        # +1 to -1 changes the contribution by -2
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

        macro_f1 = f1_score(
            y_val,
            modified_pred,
            average="macro",
            zero_division=0
        )

        accuracy_change = accuracy - base_accuracy
        f1_change = macro_f1 - base_f1

        changed_predictions = np.sum(
            modified_pred != base_pred
        )

        results.append(
            (
                cls,
                clause,
                int(weight),
                accuracy,
                macro_f1,
                accuracy_change,
                f1_change,
                int(changed_predictions)
            )
        )


# ==========================================
# Sort by Accuracy first
# Macro F1 second
# ==========================================
results.sort(
    key=lambda x: (x[5], x[6]),
    reverse=True
)


# ==========================================
# Summary
# ==========================================
improved = [
    r for r in results if r[5] > 0
]

same = [
    r for r in results if r[5] == 0
]

worse = [
    r for r in results if r[5] < 0
]


print("\n" + "=" * 80)
print("WEIGHT-FLIP SCAN SUMMARY")
print("=" * 80)

print(f"Positive clauses tested : {len(results)}")
print(f"Improve accuracy        : {len(improved)}")
print(f"No accuracy change      : {len(same)}")
print(f"Reduce accuracy         : {len(worse)}")


# ==========================================
# Top 20 flips
# ==========================================
print("\n" + "=" * 100)
print("TOP 20 POSITIVE CLAUSE WEIGHT FLIPS")
print("=" * 100)

print(
    "Class | Clause | OldWeight | Accuracy | MacroF1 | "
    "AccChange | F1Change | PredictionsChanged"
)

print("-" * 100)

for r in results[:20]:

    cls, clause, weight, acc, f1, acc_ch, f1_ch, changed = r

    print(
        f"{cls:5d} | "
        f"{clause:6d} | "
        f"{weight:9d} | "
        f"{acc:.4f}   | "
        f"{f1:.4f}  | "
        f"{acc_ch:+.4f}    | "
        f"{f1_ch:+.4f}   | "
        f"{changed:18d}"
    )


print("\nAnalysis complete!")