import numpy as np
import pickle


# Load test data
X_test = np.load("X_test_binary.npy")
y_test = np.load("y_test_binary.npy")
paths_test = np.load("paths_test_binary.npy", allow_pickle=True)


# Load best CTM model
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Best CTM model loaded successfully!")


# Predictions
y_pred = ctm.predict(X_test)


# Find misclassified samples
wrong_indices = np.where(y_test != y_pred)[0]

print("\n==============================")
print("MISCLASSIFIED TEST SAMPLES")
print("==============================")

print(f"Total test samples: {len(y_test)}")
print(f"Correct predictions: {np.sum(y_test == y_pred)}")
print(f"Wrong predictions: {len(wrong_indices)}")

print("\nDetails:\n")

for i in wrong_indices:
    print(f"Test Index : {i}")
    print(f"True Label : {y_test[i]}")
    print(f"Predicted  : {y_pred[i]}")
    print(f"Image Path : {paths_test[i]}")
    print("-" * 60)