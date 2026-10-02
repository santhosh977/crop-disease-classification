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
# Load trained CTM
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
)

clause_outputs = clause_outputs.astype(np.float32)


# ==========================================
# Get clause weights
# ==========================================
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)


# ==========================================
# Original validation prediction
# ==========================================
class_sums = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)

y_pred_original = np.argmax(
    class_sums,
    axis=1
)


# ==========================================
# Original validation metrics
# ==========================================
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

print("\n==============================")
print("ORIGINAL VALIDATION RESULT")
print("==============================")
print(f"Accuracy : {original_accuracy:.4f}")
print(f"Macro F1 : {original_f1:.4f}")


# ==========================================
# Top 5 candidate unreliable clauses
# ==========================================
candidates = {
    0: [963, 812, 702, 632, 810],
    1: [787, 506, 963, 805, 653]
}


# ==========================================
# Remove top 5 clauses of each class
# ==========================================
results = []

for cls in [0, 1]:

    modified_weights = weights.copy()

    # Temporarily disable the top 5 candidate clauses
    for clause in candidates[cls]:
        modified_weights[cls, clause] = 0

    # Recalculate class scores
    modified_sums = np.sum(
        clause_outputs *
        modified_weights[np.newaxis, :, :],
        axis=2
    )

    # New predictions
    modified_pred = np.argmax(
        modified_sums,
        axis=1
    )

    # Metrics
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

    results.append(
        (
            cls,
            candidates[cls],
            accuracy,
            macro_f1
        )
    )


# ==========================================
# Display results
# ==========================================
print("\n" + "=" * 80)
print("TOP-5 CLAUSE GROUP REMOVAL RESULTS")
print("=" * 80)

for cls, removed_clauses, accuracy, macro_f1 in results:

    print(f"\nClass {cls}")
    print(f"Removed clauses : {removed_clauses}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(
        f"Accuracy change : "
        f"{accuracy - original_accuracy:+.4f}"
    )
    print(
        f"Macro F1 change : "
        f"{macro_f1 - original_f1:+.4f}"
    )


print("\nAnalysis complete!")