import numpy as np


# ==========================================
# Load multi-threshold dataset
# ==========================================
X = np.load("X_multithreshold_ctm.npy")
y = np.load("y_multithreshold_ctm.npy")

# Original paths
all_paths = np.load(
    "paths_binary_ctm.npy",
    allow_pickle=True
)

# Paths from the original split
train_paths = np.load(
    "paths_train_binary.npy",
    allow_pickle=True
)

val_paths = np.load(
    "paths_val_binary.npy",
    allow_pickle=True
)

test_paths = np.load(
    "paths_test_binary.npy",
    allow_pickle=True
)


# ==========================================
# Create path -> index mapping
# ==========================================
path_to_index = {
    str(path): i
    for i, path in enumerate(all_paths)
}


def get_indices(paths):
    return np.array(
        [path_to_index[str(path)] for path in paths],
        dtype=np.int64
    )


train_idx = get_indices(train_paths)
val_idx = get_indices(val_paths)
test_idx = get_indices(test_paths)


# ==========================================
# Create splits
# ==========================================
X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


# ==========================================
# Save
# ==========================================
np.save("X_train_multithreshold.npy", X_train)
np.save("y_train_multithreshold.npy", y_train)

np.save("X_val_multithreshold.npy", X_val)
np.save("y_val_multithreshold.npy", y_val)

np.save("X_test_multithreshold.npy", X_test)
np.save("y_test_multithreshold.npy", y_test)


print("\n====================================")
print("MULTI-THRESHOLD SPLIT CREATED")
print("====================================")

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)

print("\nClass counts:")
print("Train class 0:", np.sum(y_train == 0))
print("Train class 1:", np.sum(y_train == 1))

print("Val class 0:", np.sum(y_val == 0))
print("Val class 1:", np.sum(y_val == 1))

print("Test class 0:", np.sum(y_test == 0))
print("Test class 1:", np.sum(y_test == 1))