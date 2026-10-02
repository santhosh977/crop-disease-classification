import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score


# Load validation data
X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")


# Load refined model
with open("refined_binary_ctm_v1.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined CTM loaded successfully!")


# Get clause outputs
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


# Current refined prediction
scores = np.sum(
    clause_outputs * weights[np.newaxis, :, :],
    axis=2
)

y_pred = np.argmax(scores, axis=1)

current_accuracy = accuracy_score(y_val, y_pred)
current_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)


# Test removing clause 286 from Class 0
modified_weights = weights.copy()
modified_weights[0, 66] = 0

modified_scores = np.sum(
    clause_outputs *
    modified_weights[np.newaxis, :, :],
    axis=2
)

modified_pred = np.argmax(
    modified_scores,
    axis=1
)

new_accuracy = accuracy_score(
    y_val,
    modified_pred
)

new_f1 = f1_score(
    y_val,
    modified_pred,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("CLAUSE 286 REFINEMENT TEST")
print("=" * 60)

print(f"Current Accuracy : {current_accuracy:.4f}")
print(f"Current Macro F1 : {current_f1:.4f}")

print("\nAfter removing Class 0 Clause 66:")

print(f"New Accuracy     : {new_accuracy:.4f}")
print(f"New Macro F1     : {new_f1:.4f}")

print(
    f"\nAccuracy Change  : "
    f"{new_accuracy - current_accuracy:+.4f}"
)

print(
    f"Macro F1 Change  : "
    f"{new_f1 - current_f1:+.4f}"
)
