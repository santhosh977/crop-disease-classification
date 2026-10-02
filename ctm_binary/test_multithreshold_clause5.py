import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load("X_test_multithreshold.npy")
y_test = np.load("y_test_multithreshold.npy")

# ============================================================
# LOAD MODEL
# ============================================================

with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Model loaded successfully!")

# ============================================================
# CLAUSE ACTIVATIONS
# ============================================================

raw_output = ctm.transform(X_test)

clause_outputs = raw_output.reshape(
    len(X_test),
    2,
    1000
).astype(np.float32)

# ============================================================
# WEIGHTS
# ============================================================

weights = np.array([
    ctm.weight_banks[0].get_weights(),
    ctm.weight_banks[1].get_weights()
], dtype=np.float32)

# ============================================================
# ORIGINAL SCORES
# ============================================================

class_scores = np.sum(
    clause_outputs * weights[np.newaxis, :, :],
    axis=2
)

original_pred = np.argmax(class_scores, axis=1)

original_acc = accuracy_score(y_test, original_pred)
original_f1 = f1_score(
    y_test,
    original_pred,
    average="macro",
    zero_division=0
)

# ============================================================
# FLIP CLASS-1 CLAUSE 5
# +1 -> -1
# ============================================================

clause = 5

new_weights = weights.copy()
new_weights[1, clause] = -new_weights[1, clause]

new_scores = np.sum(
    clause_outputs * new_weights[np.newaxis, :, :],
    axis=2
)

new_pred = np.argmax(new_scores, axis=1)

new_acc = accuracy_score(y_test, new_pred)
new_f1 = f1_score(
    y_test,
    new_pred,
    average="macro",
    zero_division=0
)

# ============================================================
# CHANGED PREDICTIONS
# ============================================================

changed = np.where(original_pred != new_pred)[0]

corrected = 0
harmed = 0

print()
print("=" * 60)
print("CLAUSE 5 TEST-SET EFFECT")
print("=" * 60)

print(f"Original Accuracy : {original_acc:.4f}")
print(f"New Accuracy      : {new_acc:.4f}")
print(f"Accuracy Change   : {(new_acc - original_acc) * 100:+.2f} pp")

print()

print(f"Original Macro F1 : {original_f1:.4f}")
print(f"New Macro F1      : {new_f1:.4f}")
print(f"F1 Change         : {(new_f1 - original_f1) * 100:+.2f} pp")

print()

print("Changed predictions:", len(changed))

for idx in changed:

    if new_pred[idx] == y_test[idx]:
        corrected += 1
        result = "CORRECTED"
    else:
        harmed += 1
        result = "HARMED"

    print(
        f"Index: {idx:3d} | "
        f"True: {y_test[idx]} | "
        f"Old: {original_pred[idx]} | "
        f"New: {new_pred[idx]} | "
        f"{result}"
    )

print()
print("Corrected:", corrected)
print("Harmed   :", harmed)