import numpy as np
import pickle

# ============================================================
# LOAD DATA
# ============================================================

X_val = np.load("X_val_multithreshold.npy").astype(np.uint32)
y_val = np.load("y_val_multithreshold.npy").astype(np.uint32)

with open("best_multithreshold_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

# ============================================================
# CLAUSE OUTPUTS AND WEIGHTS
# ============================================================

clause_outputs = ctm.transform(X_val)

n_clauses = 1000

class0_outputs = clause_outputs[:, :n_clauses]
class1_outputs = clause_outputs[:, n_clauses:]

weights0 = np.array(ctm.weight_banks[0].weights).copy()
weights1 = np.array(ctm.weight_banks[1].weights).copy()

# ============================================================
# ORIGINAL PREDICTIONS
# ============================================================

scores0 = np.sum(class0_outputs * weights0, axis=1)
scores1 = np.sum(class1_outputs * weights1, axis=1)

original_pred = np.where(scores0 > scores1, 0, 1)

# ============================================================
# FLIP CLASS-1 CLAUSE 5
# +1 -> -1
# ============================================================

clause = 5

print("Clause 5 original weight:", weights1[clause])
print("Clause 5 firing count:", int(np.sum(class1_outputs[:, clause])))

new_scores1 = scores1 - (2 * class1_outputs[:, clause])

new_pred = np.where(scores0 > new_scores1, 0, 1)

# ============================================================
# FIND CHANGED PREDICTIONS
# ============================================================

changed = np.where(original_pred != new_pred)[0]

print()
print("=" * 60)
print("CLAUSE 5 EFFECT")
print("=" * 60)

print("Number of changed predictions:", len(changed))

for idx in changed:

    result = "CORRECTED" if new_pred[idx] == y_val[idx] else "HARMED"

    print(
        f"Index: {idx:3d} | "
        f"True: {y_val[idx]} | "
        f"Old Pred: {original_pred[idx]} | "
        f"New Pred: {new_pred[idx]} | "
        f"Score0: {scores0[idx]:5.1f} | "
        f"Old Score1: {scores1[idx]:5.1f} | "
        f"New Score1: {new_scores1[idx]:5.1f} | "
        f"{result}"
    )