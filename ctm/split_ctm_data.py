import numpy as np
import pandas as pd
import os

CTM_PATH = r"C:\CropDisease\ctm"
SPLIT_FILE = r"C:\CropDisease\experiments\dataset_split.csv"

# Load CTM data
X = np.load(os.path.join(CTM_PATH, "X_ctm.npy"))
y = np.load(os.path.join(CTM_PATH, "y_ctm.npy"))
paths = np.load(
    os.path.join(CTM_PATH, "paths_ctm.npy"),
    allow_pickle=True
)

# Load original ResNet18 split
split_df = pd.read_csv(SPLIT_FILE)

# Normalize paths
paths = np.array([os.path.normcase(os.path.abspath(p)) for p in paths])

split_df["image_path"] = split_df["image_path"].apply(
    lambda p: os.path.normcase(os.path.abspath(p))
)

# Create mapping:
# image path -> split (train / validation / test)
path_to_split = dict(
    zip(split_df["image_path"], split_df["split"])
)

# Find split for every CTM image
splits = np.array([
    path_to_split.get(path, "NOT_FOUND")
    for path in paths
])

# Safety check
not_found = np.sum(splits == "NOT_FOUND")

print("Images not found in split file:", not_found)

if not_found > 0:
    raise ValueError("Some images could not be matched!")

# Create train / validation / test masks
train_mask = splits == "train"
val_mask = splits == "validation"
test_mask = splits == "test"

# Split data
X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nFinal CTM Split:")
print("Training:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Testing:", X_test.shape[0])

# Save splits
np.save("X_train_ctm.npy", X_train)
np.save("y_train_ctm.npy", y_train)

np.save("X_val_ctm.npy", X_val)
np.save("y_val_ctm.npy", y_val)

np.save("X_test_ctm.npy", X_test)
np.save("y_test_ctm.npy", y_test)

print("\nAll CTM splits saved successfully!")