import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# Load data
# --------------------------------------------------

X_train = np.load("X_train_multithreshold.npy").astype(np.uint32)
y_train = np.load("y_train_multithreshold.npy").astype(np.uint32)

X_val = np.load("X_val_multithreshold.npy").astype(np.uint32)
y_val = np.load("y_val_multithreshold.npy").astype(np.uint32)


print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# --------------------------------------------------
# Create CTM
# --------------------------------------------------

ctm = TMClassifier(
    number_of_clauses=400,
    T=12,
    s=5.01,
    patch_dim=(9, 9),
    platform="CPU"
)


print("\n2-class CTM created successfully!")
print("Clauses: 400")
print("T: 12")
print("s: 5.01")
print("Patch: 9 × 9")

print("\nStarting training...\n")


# --------------------------------------------------
# Training
# --------------------------------------------------

EPOCHS = 20

best_f1 = 0.0
best_epoch = 0

for epoch in range(EPOCHS):

    ctm.fit(
        X_train,
        y_train,
        epochs=1
    )

    # Validation prediction
    y_pred = ctm.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Val Accuracy: {accuracy:.4f} "
        f"Precision: {precision:.4f} "
        f"Recall: {recall:.4f} "
        f"Macro F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch + 1

        import pickle
        with open("best_multithreshold_ctm.pkl", "wb") as f:
            pickle.dump(ctm, f)

        print("Best validation result so far!")
        print("Best multi-threshold CTM model saved!")

print("\nTraining complete!")

print(
    f"Best Validation Macro F1: {best_f1:.4f}"
)

print(
    f"Best Epoch: {best_epoch}"
)