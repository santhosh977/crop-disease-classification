import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PlantDiseaseDataset(Dataset):

    def __init__(self, csv_file, split):

        self.data = pd.read_csv(csv_file)

        # Keep only the requested split
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data.iloc[index]["image_path"]
        label = self.data.iloc[index]["label"]

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Transform image
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# CSV location
CSV_FILE = r"C:\CropDisease\experiments\dataset_split.csv"


# Create datasets
train_dataset = PlantDiseaseDataset(
    CSV_FILE,
    "train"
)

validation_dataset = PlantDiseaseDataset(
    CSV_FILE,
    "validation"
)

test_dataset = PlantDiseaseDataset(
    CSV_FILE,
    "test"
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# Test loading
images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)