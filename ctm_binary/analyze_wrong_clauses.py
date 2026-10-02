import numpy as np
import pickle


# ==============================
# Load test data
# ==============================
X_test = np.load("X_test_binary.npy")
y_test = np.load("y_test_binary.npy")

# ==============================
# Load trained CTM
# ==============================
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("CTM model loaded successfully!")


# ==============================
# Get predictions
# ==============================
y_pred = ctm.predict(X_test)

wrong_indices = np.where(y_test != y_pred)[0]

print(f"\nWrong predictions: {len(wrong_indices)}")


# ==============================
# Get clause activations
# ==============================
# For 2 classes and 1000 clauses:
# shape = (samples, 2, 1000)

clause_outputs = ctm.transform(X_test)

clause_outputs = clause_outputs.reshape(
    X_test.shape[0],
    2,
    ctm.number_of_clauses
)


# ==============================
# Analyze wrong predictions
# ==============================
for idx in wrong_indices:

    true_class = int(y_test[idx])
    predicted_class = int(y_pred[idx])

    true_clauses = np.where(
        clause_outputs[idx, true_class] == 1
    )[0]

    predicted_clauses = np.where(
        clause_outputs[idx, predicted_class] == 1
    )[0]

    print("\n" + "=" * 70)
    print(f"Test Index       : {idx}")
    print(f"True Class       : {true_class}")
    print(f"Predicted Class  : {predicted_class}")

    print(f"\nClauses firing for TRUE class ({true_class}):")
    print(true_clauses.tolist())
    print(f"Total: {len(true_clauses)}")

    print(f"\nClauses firing for PREDICTED class ({predicted_class}):")
    print(predicted_clauses.tolist())
    print(f"Total: {len(predicted_clauses)}")


print("\nAnalysis complete!")