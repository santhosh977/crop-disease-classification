import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

# Create dummy binary images
X = np.random.randint(0, 2, size=(10, 32, 32), dtype=np.uint8)
y = np.random.randint(0, 5, size=10)

# Convolutional Tsetlin Machine
ctm = TMClassifier(
    number_of_clauses=100,
    T=50,
    s=3.0,
    patch_dim=(8, 8),
    platform="CPU"
)

print("CTM created successfully!")
print("Input shape:", X.shape)
print("Classes:", np.unique(y))