import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# Load training data
# ==========================================
X_train = np.load("X_train_binary.npy")
y_train = np.load("y_train_binary.npy")

# ==========================================
# Load current refined model
# ==========================================
with open("refined_binary_ctm_v1.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined CTM loaded successfully!")


# ==========================================
# Clause activations
# ==========================================
raw_output = ctm.transform(X_train)

clause_outputs = raw_output.reshape(
    len(X_train),
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
scores = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)

original_pred = np.argmax(scores, axis=1)

original_acc = accuracy_score(
    y_train,
    original_pred
)

original_f1 = f1_score(
    y_train,
    original_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# Flip Class 0 Clause 3
# +1 -> -1
# ==========================================
modified_scores = scores.copy()

modified_scores[:, 0] -= (
    2 * clause_outputs[:, 0, 3]
)

modified_pred = np.argmax(
    modified_scores,
    axis=1
)


# ==========================================
# New metrics
# ==========================================
new_acc = accuracy_score(
    y_train,
    modified_pred
)

new_f1 = f1_score(
    y_train,
    modified_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# Changed predictions
# ==========================================
changed = np.where(
    original_pred != modified_pred
)[0]

corrected = 0
harmed = 0

for idx in changed:

    if (
        original_pred[idx] != y_train[idx]
        and
        modified_pred[idx] == y_train[idx]
    ):
        corrected += 1

    if (
        original_pred[idx] == y_train[idx]
        and
        modified_pred[idx] != y_train[idx]
    ):
        harmed += 1


# ==========================================
# Results
# ==========================================
print("\n" + "=" * 70)
print("CLASS 0 CLAUSE 3 — TRAINING CHECK")
print("=" * 70)

print(f"Original Accuracy : {original_acc:.4f}")
print(f"New Accuracy      : {new_acc:.4f}")
print(f"Accuracy Change   : {new_acc - original_acc:+.4f}")

print(f"\nOriginal Macro F1 : {original_f1:.4f}")
print(f"New Macro F1      : {new_f1:.4f}")
print(f"F1 Change         : {new_f1 - original_f1:+.4f}")

print(f"\nPredictions changed: {len(changed)}")
print(f"Corrected          : {corrected}")
print(f"Harmed             : {harmed}")

print("\nAnalysis complete!")