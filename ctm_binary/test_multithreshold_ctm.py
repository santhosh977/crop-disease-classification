import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# Load untouched test data
# ==========================================
X_test = np.load("X_test_multithreshold.npy")
y_test = np.load("y_test_multithreshold.npy")

print("Test data:", X_test.shape)
print("Test labels:", y_test.shape)


# ==========================================
# Load best 9-channel CTM
# ==========================================
with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Multi-threshold CTM loaded successfully!")


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
# Get clause weights
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
# Results
# ==========================================
print("\n" + "=" * 60)
print("FINAL MULTI-THRESHOLD CTM TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test Macro F1 : {macro_f1:.4f}")