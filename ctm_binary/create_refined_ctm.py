import pickle


# Load original best CTM
with open("best_binary_ctm.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Original CTM loaded!")


# ------------------------------------------
# Refine clause 488 of Class 0
# ------------------------------------------
print(
    "Before refinement:",
    ctm.weight_banks[0].weights[488]
)

ctm.weight_banks[0].weights[488] = 0

print(
    "After refinement:",
    ctm.weight_banks[0].weights[488]
)


# ------------------------------------------
# Save refined CTM as a new model
# ------------------------------------------
with open("refined_binary_ctm_v1.pkl", "wb") as f:
    pickle.dump(ctm, f)

print("\nRefined CTM saved as:")
print("refined_binary_ctm_v1.pkl")