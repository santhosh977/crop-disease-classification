import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score


# Load validation data
X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")


# Load refined v1 model
with open("refined_binary_ctm_v1.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined v1 CTM loaded successfully!")


# Get clause activations
raw_output = ctm.transform(X_val)

clause_outputs = raw_output.reshape(
    len(X_val),
    2,
    1000
).astype(np.float32)


# Get weights
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)


# Current refined v1 scores
scores = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)

base_pred = np.argmax(scores, axis=1)

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


# ------------------------------------------
# Add Clause 3 refinement
# Class 0: +1 -> -1
# ------------------------------------------
combined_scores = scores.copy()

combined_scores[:, 0] -= (
    2 * clause_outputs[:, 0, 3]
)

combined_pred = np.argmax(
    combined_scores,
    axis=1
)


# Metrics
combined_accuracy = accuracy_score(
    y_val,
    combined_pred
)

combined_f1 = f1_score(
    y_val,
    combined_pred,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 70)
print("COMBINED REFINEMENT")
print("=" * 70)

print("Existing refinement : Class 0 Clause 488 = 0")
print("New refinement      : Class 0 Clause 3 = -1")

print("\nCurrent refined v1:")
print(f"Accuracy : {base_accuracy:.4f}")
print(f"Macro F1 : {base_f1:.4f}")

print("\nAfter adding Clause 3 flip:")
print(f"Accuracy : {combined_accuracy:.4f}")
print(f"Macro F1 : {combined_f1:.4f}")

print(
    f"\nAccuracy Change : "
    f"{combined_accuracy - base_accuracy:+.4f}"
)

print(
    f"Macro F1 Change : "
    f"{combined_f1 - base_f1:+.4f}"
)