from pathlib import Path

# PlantVillage dataset location
DATASET_DIR = Path(r"C:\CropDisease\PlantVillage")

# Classes we are using for the first experiment
CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot"
]

# Check that the dataset folder exists
if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

print("Dataset found:", DATASET_DIR)
print()

total_images = 0

for class_name in CLASSES:
    class_path = DATASET_DIR / class_name

    if not class_path.exists():
        print(f"ERROR: Class not found: {class_name}")
        continue

    # Count image files
    images = [
        file for file in class_path.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print(f"{class_name}: {len(images)} images")
    total_images += len(images)

print()
print("Total images:", total_images)