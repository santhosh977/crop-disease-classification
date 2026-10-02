import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from torchvision.models import resnet18

from data_loader import test_loader


# ==================================================
# Class names
# ==================================================

CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot"
]


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==================================================
# Create ResNet18 architecture
# ==================================================

model = resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    5
)


# ==================================================
# Load our BEST trained model
# ==================================================

model.load_state_dict(
    torch.load(
        r"C:\CropDisease\results\best_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ==================================================
# Test
# ==================================================

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==================================================
# Metrics
# ==================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


print("\n==============================")
print("FINAL RESNET18 TEST RESULTS")
print("==============================")

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test Macro F1 : {f1:.4f}")


# ==================================================
# Classification report
# ==================================================

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ==================================================
# Confusion Matrix
# ==================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)