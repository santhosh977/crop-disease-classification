import cv2
import numpy as np


# ==========================================
# Classes
# ==========================================
classes = {
    "Pepper__bell___Bacterial_spot": 0,
    "Pepper__bell___healthy": 1
}


# ==========================================
# Load image paths created earlier
# ==========================================
paths = np.load(
    "paths_binary_ctm.npy",
    allow_pickle=True
)


X = []
y = []


# ==========================================
# Thresholds
# ==========================================
thresholds = [32, 96, 160]


# ==========================================
# Process images
# ==========================================
for path in paths:

    path = str(path)

    # Determine class from folder name
    if "Pepper__bell___Bacterial_spot" in path:
        label = 0

    elif "Pepper__bell___healthy" in path:
        label = 1

    else:
        print("Unknown class:", path)
        continue

    image = cv2.imread(path)

    if image is None:
        print("Could not read:", path)
        continue

    # OpenCV loads BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (32, 32)
    )

    image = image.astype(np.uint8)


    # ======================================
    # Create 9 binary channels
    # ======================================
    channels = []

    for threshold in thresholds:

        binary = (
            image >= threshold
        ).astype(np.uint32)

        channels.append(binary)


    # Concatenate:
    # (32,32,3) × 3 thresholds
    # -> (32,32,9)
    multi = np.concatenate(
        channels,
        axis=2
    )

    X.append(multi)
    y.append(label)


# ==========================================
# Convert to NumPy arrays
# ==========================================
X = np.array(
    X,
    dtype=np.uint32
)

y = np.array(
    y,
    dtype=np.uint32
)


# ==========================================
# Save
# ==========================================
np.save(
    "X_multithreshold_ctm.npy",
    X
)

np.save(
    "y_multithreshold_ctm.npy",
    y
)


print("\n====================================")
print("MULTI-THRESHOLD CTM DATA CREATED")
print("====================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Class 0:",
    np.sum(y == 0)
)

print(
    "Class 1:",
    np.sum(y == 1)
)

print(
    "Channels: RGB × 3 thresholds = 9"
)

print(
    "Thresholds:",
    thresholds
)