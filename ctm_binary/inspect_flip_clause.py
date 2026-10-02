import numpy as np
import pickle


# ==========================================
# Load validation data
# ==========================================
X_val = np.load("X_val_binary.npy")
y_val = np.load("y_val_binary.npy")
paths_val = np.load(
    "paths_val_binary.npy",
    allow_pickle=True
)


# ==========================================
# Load refined CTM
# ==========================================
with open("refined_binary_ctm_v1.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined CTM loaded successfully!")


# ==========================================
# Get clause outputs
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
# Original predictions
# ==========================================
original_scores = np.sum(
    clause_outputs *
    weights[np.newaxis, :, :],
    axis=2
)

original_pred = np.argmax(
    original_scores,
    axis=1
)


# ==========================================
# Flip Class 0, Clause 3
# +1 -> -1
# ==========================================
modified_scores = original_scores.copy()

modified_scores[:, 0] -= (
    2 * clause_outputs[:, 0, 7]
)

modified_pred = np.argmax(
    modified_scores,
    axis=1
)


# ==========================================
# Find changed predictions
# ==========================================
changed = np.where(
    original_pred != modified_pred
)[0]


print("\n" + "=" * 70)
print("CLASS 0 CLAUSE 7 FLIP ANALYSIS")
print("=" * 70)

print(f"Changed predictions: {len(changed)}")


for idx in changed:

    print("\nValidation Index :", idx)
    print("True Class       :", int(y_val[idx]))
    print("Original         :", int(original_pred[idx]))
    print("After Flip       :", int(modified_pred[idx]))
    print("Image Path       :", paths_val[idx])

    print(
        "Original Scores  :",
        original_scores[idx]
    )

    print(
        "New Scores       :",
        modified_scores[idx]
    )

    if original_pred[idx] != y_val[idx]:
        print("RESULT: ✅ Corrected a wrong prediction")
    else:
        print("RESULT: ❌ Changed a correct prediction")


print("\nAnalysis complete!")