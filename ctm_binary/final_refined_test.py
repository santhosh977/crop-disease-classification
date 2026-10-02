import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# Load untouched test data
# ==========================================
X_test = np.load("X_test_binary.npy")
y_test = np.load("y_test_binary.npy")


# ==========================================
# Load refined CTM
# ==========================================
with open("refined_binary_ctm_v2.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined   V2 CTM loaded successfully!")


# ==========================================
# Get clause activations
# ==========================================
raw_output = ctm.transform(X_test)

clause_outputs = raw_output.reshape(
    len(X_test),
    2,
    1000
).astype(np.float32)


# ==========================================
# Get refined weights
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
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# Final result
# ==========================================
print("\n" + "=" * 60)
print("FINAL REFINED CTM TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test Macro F1 : {macro_f1:.4f}")