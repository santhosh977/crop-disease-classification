import numpy as np
import pandas as pd
import os

CTM_PATH = r"C:\CropDisease\ctm"
SPLIT_FILE = r"C:\CropDisease\experiments\dataset_split.csv"

# Load multichannel CTM data
X = np.load(os.path.join(CTM_PATH, "X_ctm_multi.npy"))
y = np.load(os.path.join(CTM_PATH, "y_ctm_multi.npy"))
paths = np.load(
    os.path.join(CTM_PATH, "paths_ctm_multi.npy"),
    allow_pickle=True
)

# Load original split information
split_df = pd.read_csv(SPLIT_FILE)

# Normalize paths
paths = np.array([
    os.path.normcase(os.path.abspath(p))
    for p in paths
])

split_df["image_path"] = split_df["image_path"].apply(
    lambda p: os.path.normcase(os.path.abspath(p))
)

# Create mapping
path_to_split = dict(
    zip(split_df["image_path"], split_df["split"])
)

# Get split for every image
splits = np.array([
    path_to_split.get(path, "NOT_FOUND")
    for path in paths
])

# Safety check
not_found = np.sum(splits == "NOT_FOUND")

print("Images not found:", not_found)

if not_found > 0:
    raise ValueError("Some images could not be matched!")

# Masks
train_mask = splits == "train"
val_mask = splits == "validation"
test_mask = splits == "test"

# Split
X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# Display
print("\nMultichannel CTM Split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Save
np.save("X_train_ctm_multi.npy", X_train)
np.save("y_train_ctm_multi.npy", y_train)

np.save("X_val_ctm_multi.npy", X_val)
np.save("y_val_ctm_multi.npy", y_val)

np.save("X_test_ctm_multi.npy", X_test)
np.save("y_test_ctm_multi.npy", y_test)

print("\nMultichannel CTM splits saved successfully!")