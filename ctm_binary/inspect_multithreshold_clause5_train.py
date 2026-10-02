import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# LOAD TRAINING DATA
# ============================================================

X_train = np.load("X_train_multithreshold.npy").astype(np.uint32)
y_train = np.load("y_train_multithreshold.npy").astype(np.uint32)

# ============================================================
# LOAD MODEL
# ============================================================

with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

# ============================================================
# CLAUSE OUTPUTS
# ============================================================

clause_outputs = ctm.transform(X_train)

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

original_acc = accuracy_score(y_train, original_pred)
original_f1 = f1_score(y_train, original_pred, average="macro")

# ============================================================
# FLIP CLASS-1 CLAUSE 5
# ============================================================

clause = 5

new_scores1 = scores1 - (2 * class1_outputs[:, clause])

new_pred = np.where(scores0 > new_scores1, 0, 1)

new_acc = accuracy_score(y_train, new_pred)
new_f1 = f1_score(y_train, new_pred, average="macro")

# ============================================================
# CHANGED PREDICTIONS
# ============================================================

changed = np.where(original_pred != new_pred)[0]

corrected = 0
harmed = 0

for idx in changed:
    if new_pred[idx] == y_train[idx]:
        corrected += 1
    else:
        harmed += 1

# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("CLAUSE 5 TRAINING-SET EFFECT")
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
print("Corrected:", corrected)
print("Harmed   :", harmed)

print()
print("Clause 5 training firing count:",
      int(np.sum(class1_outputs[:, clause])))