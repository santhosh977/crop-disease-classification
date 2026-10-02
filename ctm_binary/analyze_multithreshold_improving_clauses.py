import numpy as np
import pickle

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
# ORIGINAL PREDICTIONS
# ============================================================

class_scores = np.sum(
    clause_outputs * weights[np.newaxis, :, :],
    axis=2
)

original_pred = np.argmax(class_scores, axis=1)

original_wrong = np.where(original_pred != y_val)[0]

print("=" * 60)
print("ORIGINAL VALIDATION ERRORS")
print("=" * 60)
print("Total errors:", len(original_wrong))
print("Error indices:", original_wrong.tolist())

# ============================================================
# TEST EVERY POSITIVE CLAUSE
# ============================================================

improving = []

for class_id in range(2):

    for clause in range(1000):

        if weights[class_id, clause] <= 0:
            continue

        new_scores = class_scores.copy()

        # +1 -> -1
        new_scores[:, class_id] -= (
            2 * clause_outputs[:, class_id, clause]
        )

        new_pred = np.argmax(new_scores, axis=1)

        changed = np.where(new_pred != original_pred)[0]

        if len(changed) == 0:
            continue

        corrected = [
            int(i) for i in changed
            if new_pred[i] == y_val[i]
            and original_pred[i] != y_val[i]
        ]

        harmed = [
            int(i) for i in changed
            if new_pred[i] != y_val[i]
            and original_pred[i] == y_val[i]
        ]

        if len(corrected) > len(harmed):

            improving.append(
                (
                    class_id,
                    clause,
                    corrected,
                    harmed
                )
            )

# ============================================================
# DISPLAY IMPROVING CLAUSES
# ============================================================

print()
print("=" * 60)
print("IMPROVING CLAUSES")
print("=" * 60)

print("Number of improving clauses:", len(improving))

for class_id, clause, corrected, harmed in improving:

    print(
        f"Class {class_id} | "
        f"Clause {clause:4d} | "
        f"Corrected: {corrected} | "
        f"Harmed: {harmed}"
    )

# ============================================================
# GROUP BY CORRECTED VALIDATION SAMPLES
# ============================================================

groups = {}

for class_id, clause, corrected, harmed in improving:

    key = tuple(corrected)

    if key not in groups:
        groups[key] = []

    groups[key].append(
        (class_id, clause)
    )

print()
print("=" * 60)
print("CORRECTION GROUPS")
print("=" * 60)

for corrected_samples, clauses in groups.items():

    print(
        f"Corrected validation samples: {corrected_samples}"
    )

    print(
        "Clauses:",
        clauses
    )