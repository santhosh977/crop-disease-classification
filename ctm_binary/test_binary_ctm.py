import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load test data
X_test = np.load("X_test_binary.npy")
y_test = np.load("y_test_binary.npy")

print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)


# Load the best trained CTM model
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Best CTM model loaded successfully!")


# Predict on test data
y_pred = ctm.predict(X_test)


# Calculate metrics
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


print("\n==============================")
print("FINAL BINARY CTM TEST RESULTS")
print("==============================")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test Macro F1 : {macro_f1:.4f}")