import os
import numpy as np
from PIL import Image

DATASET_PATH = r"C:\CropDisease\PlantVillage"

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot"
]

IMAGE_SIZE = 32

X = []
y = []
paths = []

for label, class_name in enumerate(CLASSES):

    class_path = os.path.join(DATASET_PATH, class_name)

    print(f"Loading {class_name}...")

    for file_name in os.listdir(class_path):

        image_path = os.path.join(class_path, file_name)

        try:
            image = Image.open(image_path)

            # Grayscale
            image = image.convert("L")

            # Resize
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            # Convert to NumPy
            image_array = np.array(image)

            # Convert to binary
            image_binary = (image_array > 128).astype(np.uint8)

            X.append(image_binary)
            y.append(label)

            # Save exact image path
            paths.append(os.path.abspath(image_path))

        except Exception as e:
            print(f"Could not load {image_path}: {e}")

X = np.array(X)
y = np.array(y)
paths = np.array(paths)

print("\nDataset preparation complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Paths shape:", paths.shape)

# Save everything
np.save("X_ctm.npy", X)
np.save("y_ctm.npy", y)
np.save("paths_ctm.npy", paths)

print("\nSaved:")
print("X_ctm.npy")
print("y_ctm.npy")
print("paths_ctm.npy")