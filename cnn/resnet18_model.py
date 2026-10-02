import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# Load ImageNet pretrained ResNet18
weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

# Replace the final classification layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    5
)


if __name__ == "__main__":

    print(model)

    # Test input
    x = torch.randn(4, 3, 224, 224)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)