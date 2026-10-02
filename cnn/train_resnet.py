import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score

from torchvision.models import resnet18, ResNet18_Weights

from data_loader import train_loader, validation_loader


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
# Load pretrained ResNet18
# ==================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

# Replace ImageNet's 1000-class output
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    5
)

model = model.to(device)


# ==================================================
# Loss
# ==================================================

criterion = nn.CrossEntropyLoss()


# ==================================================
# Optimizer
# ==================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==================================================
# Training settings
# ==================================================

num_epochs = 10

best_val_f1 = 0.0


# ==================================================
# Training loop
# ==================================================

for epoch in range(num_epochs):

    model.train()

    train_predictions = []
    train_labels = []

    running_loss = 0.0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Predictions
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_labels.extend(
            labels.cpu().numpy()
        )


    # ==================================================
    # Training metrics
    # ==================================================

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    train_f1 = f1_score(
        train_labels,
        train_predictions,
        average="macro"
    )


    # ==================================================
    # Validation
    # ==================================================

    model.eval()

    val_predictions = []
    val_labels = []

    val_loss = 0.0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )


    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro"
    )


    # ==================================================
    # Print results
    # ==================================================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Train F1: {train_f1:.4f} "
        f"Val Acc: {val_accuracy:.4f} "
        f"Val F1: {val_f1:.4f}"
    )


    # ==================================================
    # Save best model
    # ==================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            r"C:\CropDisease\results\best_resnet18.pth"
        )

        print("Best model saved!")


print("\nTraining complete.")
print("Best validation F1:", best_val_f1)