import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score


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
# Get clause weights
# ==========================================
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)


# ==========================================
# Original prediction
# ==========================================
original_scores = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)

y_pred_original = np.argmax(
    original_scores,
    axis=1
)

original_accuracy = accuracy_score(
    y_val,
    y_pred_original
)

original_f1 = f1_score(
    y_val,
    y_pred_original,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 80)
print("ORIGINAL VALIDATION RESULT")
print("=" * 80)
print(f"Accuracy : {original_accuracy:.4f}")
print(f"Macro F1 : {original_f1:.4f}")


# ==========================================
# Top positive-weight misfiring candidates
# ==========================================
candidates = {
    0: [286, 488, 393, 34, 418],
    1: [420, 159, 355, 306, 364]
}


# ==========================================
# Test one clause at a time
# ==========================================
results = []

for cls in [0, 1]:

    for clause in candidates[cls]:

        modified_weights = weights.copy()

        # Temporarily remove this positive clause
        modified_weights[cls, clause] = 0

        modified_scores = np.sum(
            clause_outputs *
            modified_weights[np.newaxis, :, :],
            axis=2
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

        accuracy_change = accuracy - original_accuracy
        f1_change = macro_f1 - original_f1

        results.append(
            (
                cls,
                clause,
                accuracy,
                macro_f1,
                accuracy_change,
                f1_change
            )
        )


# ==========================================
# Accuracy is primary selection metric
# Macro F1 is secondary
# ==========================================
results.sort(
    key=lambda x: (x[4], x[5]),
    reverse=True
)


# ==========================================
# Display results
# ==========================================
print("\n" + "=" * 80)
print("POSITIVE CLAUSE REMOVAL RESULTS")
print("=" * 80)

print(
    "Class | Clause | Accuracy | Macro F1 | "
    "Acc Change | F1 Change"
)

print("-" * 80)

for r in results:

    cls, clause, accuracy, macro_f1, acc_change, f1_change = r

    print(
        f"{cls:5d} | "
        f"{clause:6d} | "
        f"{accuracy:.4f}   | "
        f"{macro_f1:.4f}  | "
        f"{acc_change:+.4f}    | "
        f"{f1_change:+.4f}"
    )


print("\nAnalysis complete!")