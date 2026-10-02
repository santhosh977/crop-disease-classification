import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
# Get clause activations
# ==========================================
raw_output = ctm.transform(X_val)

clause_outputs = raw_output.reshape(
    len(X_val),
    2,
    1000
).astype(np.float32)


# ==========================================
# Get weights
# ==========================================
weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)


# ==========================================
# Calculate class scores
# ==========================================
class_scores = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)


# ==========================================
# Predictions
# ==========================================
y_pred = np.argmax(
    class_scores,
    axis=1
)


# ==========================================
# Metrics
# ==========================================
accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# Display
# ==========================================
print("\n" + "=" * 60)
print("REFINED CTM VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")