import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load multichannel CTM data
X_train = np.load("X_train_ctm_multi.npy").astype(np.uint32)
y_train = np.load("y_train_ctm_multi.npy").astype(np.uint32)

X_val = np.load("X_val_ctm_multi.npy").astype(np.uint32)
y_val = np.load("y_val_ctm_multi.npy").astype(np.uint32)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# Create Multichannel CTM
ctm = TMClassifier(
    number_of_clauses=500,
    T=100,
    s=3.9,
    patch_dim=(8, 8),
    platform="CPU"
)

print("\nMultichannel CTM created successfully!")
print("Starting training...\n")

EPOCHS = 20
best_f1 = 0.0
best_epoch = 0

for epoch in range(EPOCHS):

    # Train for one epoch
    ctm.fit(X_train, y_train, epochs=1)

    # Predict on validation set
    y_pred = ctm.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Val Accuracy: {accuracy:.4f} "
        f"Val Macro F1: {macro_f1:.4f}"
    )

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_epoch = epoch + 1
        print("Best result so far!")

print("\nTraining complete!")
print("Best Validation Macro F1:", best_f1)
print("Best Epoch:", best_epoch)