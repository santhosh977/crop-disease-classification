import os
import cv2
import numpy as np

DATASET_PATH = r"C:\CropDisease\PlantVillage"

CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy"
]

IMAGE_SIZE = 32

X = []
y = []
paths = []

print("Preparing 2-class CTM dataset...\n")

for label, class_name in enumerate(CLASSES):

    class_path = os.path.join(DATASET_PATH, class_name)

    print(f"Loading {class_name}...")

    for filename in os.listdir(class_path):

        image_path = os.path.join(class_path, filename)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Binary RGB representation
        binary_image = (image >= 128).astype(np.uint32)

        X.append(binary_image)
        y.append(label)
        paths.append(os.path.abspath(image_path))

X = np.array(X, dtype=np.uint32)
y = np.array(y, dtype=np.uint32)
paths = np.array(paths)

print("\nDataset preparation complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique values:", np.unique(X))

print("\nClass distribution:")

for label, class_name in enumerate(CLASSES):
    print(
        class_name,
        ":",
        np.sum(y == label)
    )

np.save("X_binary_ctm.npy", X)
np.save("y_binary_ctm.npy", y)
np.save("paths_binary_ctm.npy", paths)

print("\nSaved:")
print("X_binary_ctm.npy")
print("y_binary_ctm.npy")
print("paths_binary_ctm.npy")