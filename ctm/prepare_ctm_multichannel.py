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
            image = Image.open(image_path).convert("L")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            image_array = np.array(image)

            # Create multiple binary representations
            channel_1 = (image_array > 64).astype(np.uint32)
            channel_2 = (image_array > 128).astype(np.uint32)
            channel_3 = (image_array > 192).astype(np.uint32)

            # Stack channels
            image_features = np.stack(
                [channel_1, channel_2, channel_3],
                axis=-1
            )

            X.append(image_features)
            y.append(label)
            paths.append(os.path.abspath(image_path))

        except Exception as e:
            print(f"Could not load {image_path}: {e}")

X = np.array(X, dtype=np.uint32)
y = np.array(y, dtype=np.uint32)
paths = np.array(paths)

print("\nMultichannel dataset preparation complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nUnique values:", np.unique(X))

# Save separately so the baseline remains untouched
np.save("X_ctm_multi.npy", X)
np.save("y_ctm_multi.npy", y)
np.save("paths_ctm_multi.npy", paths)

print("\nSaved:")
print("X_ctm_multi.npy")
print("y_ctm_multi.npy")
print("paths_ctm_multi.npy")
