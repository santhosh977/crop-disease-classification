from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = Path(r"C:\CropDisease\PlantVillage")

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot"
]

data = []

# Collect all image paths and labels
for label, class_name in enumerate(CLASSES):

    class_path = DATASET_DIR / class_name

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            data.append({
                "image_path": str(image_path),
                "label": label,
                "class_name": class_name
            })

df = pd.DataFrame(data)

print("Total images:", len(df))

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: half validation, half test
# 15% validation + 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Add split column
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# Combine everything
final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# Shuffle rows
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
output_path = Path(r"C:\CropDisease\experiments\dataset_split.csv")

final_df.to_csv(output_path, index=False)

print("\nSplit completed!")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nClass distribution:")
print(
    final_df.groupby(["split", "class_name"]).size()
)