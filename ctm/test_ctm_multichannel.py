import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

# Load a small sample
X = np.load("X_train_ctm_multi.npy").astype(np.uint32)
y = np.load("y_train_ctm_multi.npy").astype(np.uint32)

# Use only 10 samples for testing
X_small = X[:10]
y_small = y[:10]

print("Input shape:", X_small.shape)
print("Input dtype:", X_small.dtype)
print("Labels dtype:", y_small.dtype)

# Create CTM
ctm = TMClassifier(
    number_of_clauses=100,
    T=50,
    s=3.9,
    patch_dim=(8, 8),
    platform="CPU"
)

print("\nCTM created!")

# Test one training epoch
ctm.fit(X_small, y_small, epochs=1)

print("Multichannel CTM training successful!")