import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_val = np.load("X_val_multithreshold.npy").astype(np.uint32)
y_val = np.load("y_val_multithreshold.npy").astype(np.uint32)

print("Validation data:", X_val.shape)
print("Validation labels:", y_val.shape)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Multi-threshold CTM loaded successfully!")

# ============================================================
# GET CLAUSE OUTPUTS
# ============================================================

clause_outputs = ctm.transform(X_val)

print("Clause output shape:", clause_outputs.shape)

# ============================================================
# GET CLAUSE WEIGHTS
# ============================================================

weights_class0 = np.array(ctm.weight_banks[0].weights)
weights_class1 = np.array(ctm.weight_banks[1].weights)

print("Class 0 weights:", weights_class0.shape)
print("Class 1 weights:", weights_class1.shape)

# ============================================================
# COMPUTE CLASS SCORES
# ============================================================

n_clauses = len(weights_class0)

class0_outputs = clause_outputs[:, :n_clauses]
class1_outputs = clause_outputs[:, n_clauses:n_clauses * 2]

scores_class0 = np.sum(
    class0_outputs * weights_class0,
    axis=1
)

scores_class1 = np.sum(
    class1_outputs * weights_class1,
    axis=1
)

predictions = np.where(
    scores_class0 > scores_class1,
    0,
    1
).astype(np.uint32)

# ============================================================
# VALIDATION RESULTS
# ============================================================

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(y_val, predictions, average="macro")

print()
print("=" * 60)
print("MULTI-THRESHOLD CTM VALIDATION RESULTS")
print("=" * 60)
print(f"Validation Accuracy : {accuracy:.4f}")
print(f"Validation Macro F1 : {macro_f1:.4f}")

print()
print(classification_report(
    y_val,
    predictions,
    target_names=[
        "Pepper_Bacterial_spot",
        "Pepper_healthy"
    ]
))

# ============================================================
# FIND MISCLASSIFIED VALIDATION SAMPLES
# ============================================================

wrong_indices = np.where(predictions != y_val)[0]

print()
print("=" * 60)
print("MISCLASSIFIED VALIDATION SAMPLES")
print("=" * 60)

print("Total wrong predictions:", len(wrong_indices))

for idx in wrong_indices:
    print(
        f"Index: {idx:3d} | "
        f"True: {y_val[idx]} | "
        f"Predicted: {predictions[idx]} | "
        f"Score0: {scores_class0[idx]:5.1f} | "
        f"Score1: {scores_class1[idx]:5.1f}"
    )

# ============================================================
# CLAUSE FIRING ON WRONG PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("TOP FIRING CLAUSES ON MISCLASSIFIED SAMPLES")
print("=" * 60)

for idx in wrong_indices:

    print()
    print(
        f"Validation index {idx} | "
        f"True class = {y_val[idx]} | "
        f"Predicted class = {predictions[idx]}"
    )

    # Clauses firing for class 0
    fired0 = np.where(class0_outputs[idx] == 1)[0]

    # Clauses firing for class 1
    fired1 = np.where(class1_outputs[idx] == 1)[0]

    # Sort by absolute weight contribution
    fired0_sorted = sorted(
        fired0,
        key=lambda c: abs(weights_class0[c]),
        reverse=True
    )

    fired1_sorted = sorted(
        fired1,
        key=lambda c: abs(weights_class1[c]),
        reverse=True
    )

    print("Class 0 firing clauses:")
    print([
        (int(c), int(weights_class0[c]))
        for c in fired0_sorted[:20]
    ])

    print("Class 1 firing clauses:")
    print([
        (int(c), int(weights_class1[c]))
        for c in fired1_sorted[:20]
    ])