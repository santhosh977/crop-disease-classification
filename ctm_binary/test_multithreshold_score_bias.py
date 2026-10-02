import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_val = np.load("X_val_multithreshold.npy")
y_val = np.load("y_val_multithreshold.npy")

# ============================================================
# LOAD MODEL
# ============================================================

with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

# ============================================================
# CLAUSE OUTPUTS
# ============================================================

raw_output = ctm.transform(X_val)

clause_outputs = raw_output.reshape(
    len(X_val),
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
# CLASS SCORES
# ============================================================

class_scores = np.sum(
    clause_outputs * weights[np.newaxis, :, :],
    axis=2
)

# Original result
original_pred = np.argmax(class_scores, axis=1)

original_acc = accuracy_score(y_val, original_pred)
original_f1 = f1_score(
    y_val,
    original_pred,
    average="macro",
    zero_division=0
)

print("=" * 60)
print("ORIGINAL VALIDATION RESULT")
print("=" * 60)

print(f"Accuracy : {original_acc:.4f}")
print(f"Macro F1 : {original_f1:.4f}")

# ============================================================
# TEST CLASS-0 BIAS
#
# Positive bias gives Class 0 an advantage.
# ============================================================

results = []

for bias in np.arange(-5.0, 5.1, 0.5):

    adjusted_scores = class_scores.copy()

    adjusted_scores[:, 0] += bias

    predictions = np.argmax(
        adjusted_scores,
        axis=1
    )

    acc = accuracy_score(y_val, predictions)

    f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append((bias, acc, f1))

# ============================================================
# SORT BY VALIDATION ACCURACY
# ============================================================

results.sort(
    key=lambda x: (x[1], x[2]),
    reverse=True
)

print()
print("=" * 60)
print("BEST SCORE BIAS VALUES")
print("=" * 60)

for bias, acc, f1 in results[:10]:

    print(
        f"Bias: {bias:+.1f} | "
        f"Accuracy: {acc:.4f} | "
        f"Macro F1: {f1:.4f}"
    )

# ============================================================
# BEST
# ============================================================

best_bias, best_acc, best_f1 = results[0]

print()
print("=" * 60)
print("BEST VALIDATION BIAS")
print("=" * 60)

print(f"Best Bias     : {best_bias:+.1f}")
print(f"Best Accuracy : {best_acc:.4f}")
print(f"Best Macro F1 : {best_f1:.4f}")

print(
    f"Accuracy gain : {(best_acc - original_acc) * 100:+.2f} pp"
)