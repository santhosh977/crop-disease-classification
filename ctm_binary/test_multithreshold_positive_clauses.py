import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_val = np.load("X_val_multithreshold.npy").astype(np.uint32)
y_val = np.load("y_val_multithreshold.npy").astype(np.uint32)

# ============================================================
# LOAD MODEL
# ============================================================

with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Model loaded successfully!")

# ============================================================
# CLAUSE OUTPUTS
# ============================================================

clause_outputs = ctm.transform(X_val)

n_clauses = 1000

class0_outputs = clause_outputs[:, :n_clauses]
class1_outputs = clause_outputs[:, n_clauses:]

weights0 = np.array(ctm.weight_banks[0].weights).copy()
weights1 = np.array(ctm.weight_banks[1].weights).copy()

# ============================================================
# ORIGINAL SCORES
# ============================================================

scores0 = np.sum(class0_outputs * weights0, axis=1)
scores1 = np.sum(class1_outputs * weights1, axis=1)

original_pred = np.where(scores0 > scores1, 0, 1)

original_acc = accuracy_score(y_val, original_pred)
original_f1 = f1_score(y_val, original_pred, average="macro")

print()
print("=" * 60)
print("ORIGINAL VALIDATION RESULT")
print("=" * 60)
print(f"Accuracy : {original_acc:.4f}")
print(f"Macro F1 : {original_f1:.4f}")

# ============================================================
# TEST FLIPPING EACH POSITIVE CLASS-0 CLAUSE
# +1 -> -1
# ============================================================

results = []

print()
print("Testing positive Class 0 clauses...")

for clause in range(n_clauses):

    if weights0[clause] <= 0:
        continue

    # Flipping +1 to -1 changes score by -2 when clause fires
    new_scores0 = scores0 - (2 * class0_outputs[:, clause])

    new_pred = np.where(new_scores0 > scores1, 0, 1)

    acc = accuracy_score(y_val, new_pred)
    f1 = f1_score(y_val, new_pred, average="macro")

    results.append((
        acc,
        f1,
        0,
        clause
    ))

# ============================================================
# TEST FLIPPING EACH POSITIVE CLASS-1 CLAUSE
# +1 -> -1
# ============================================================

print("Testing positive Class 1 clauses...")

for clause in range(n_clauses):

    if weights1[clause] <= 0:
        continue

    new_scores1 = scores1 - (2 * class1_outputs[:, clause])

    new_pred = np.where(scores0 > new_scores1, 0, 1)

    acc = accuracy_score(y_val, new_pred)
    f1 = f1_score(y_val, new_pred, average="macro")

    results.append((
        acc,
        f1,
        1,
        clause
    ))

# ============================================================
# SORT RESULTS
# ============================================================

results.sort(
    key=lambda x: (x[0], x[1]),
    reverse=True
)

# ============================================================
# DISPLAY TOP 20
# ============================================================

print()
print("=" * 60)
print("TOP CLAUSE FLIP RESULTS")
print("=" * 60)

for acc, f1, class_id, clause in results[:20]:

    improvement = (acc - original_acc) * 100

    print(
        f"Class {class_id} | "
        f"Clause {clause:4d} | "
        f"Accuracy: {acc:.4f} | "
        f"Macro F1: {f1:.4f} | "
        f"Improvement: {improvement:+.2f} pp"
    )

# ============================================================
# SUMMARY
# ============================================================

better = [r for r in results if r[0] > original_acc]
same = [r for r in results if r[0] == original_acc]
worse = [r for r in results if r[0] < original_acc]

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("Positive clauses tested:", len(results))
print("Improved accuracy     :", len(better))
print("No accuracy change    :", len(same))
print("Reduced accuracy      :", len(worse))

if better:
    best = results[0]

    print()
    print("BEST CLAUSE TO INVESTIGATE")
    print(f"Class  : {best[2]}")
    print(f"Clause : {best[3]}")
    print(f"Accuracy: {best[0]:.4f}")
    print(f"Macro F1: {best[1]:.4f}")