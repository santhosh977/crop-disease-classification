import pickle
import numpy as np


# Load best CTM
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("CTM model loaded successfully!")


# Candidate clauses from validation analysis
candidates = {
    0: [963, 812, 702, 632, 810],
    1: [787, 506, 963, 805, 653]
}


print("\n" + "=" * 70)
print("CANDIDATE CLAUSE WEIGHTS")
print("=" * 70)


for cls in [0, 1]:

    weights = np.array(
        ctm.weight_banks[cls].get_weights()
    )

    print(f"\nClass {cls}")

    for clause in candidates[cls]:

        print(
            f"Clause {clause:4d} "
            f"Weight = {weights[clause]}"
        )


print("\nAnalysis complete!")