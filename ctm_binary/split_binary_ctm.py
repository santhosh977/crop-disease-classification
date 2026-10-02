import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load our binary CTM dataset
X = np.load("X_binary_ctm.npy")
y = np.load("y_binary_ctm.npy")
paths = np.load(
    "paths_binary_ctm.npy",
    allow_pickle=True
)

print("Total images:", len(X))

# --------------------------------------------------
# First split: 70% training, 30% temporary
# --------------------------------------------------

X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
    X,
    y,
    paths,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# Second split: 15% validation, 15% test
# --------------------------------------------------

X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
    X_temp,
    y_temp,
    paths_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nSplit completed!")

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# --------------------------------------------------
# Save
# --------------------------------------------------

np.save("X_train_binary.npy", X_train)
np.save("y_train_binary.npy", y_train)
np.save("paths_train_binary.npy", paths_train)

np.save("X_val_binary.npy", X_val)
np.save("y_val_binary.npy", y_val)
np.save("paths_val_binary.npy", paths_val)

np.save("X_test_binary.npy", X_test)
np.save("y_test_binary.npy", y_test)
np.save("paths_test_binary.npy", paths_test)

print("\nSaved all binary CTM splits successfully!")