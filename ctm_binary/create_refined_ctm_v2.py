import pickle


# Load refined v1 model
with open("refined_binary_ctm_v1.pkl", "rb") as f:
    ctm = pickle.load(f)

print("Refined v1 model loaded!")


# ==========================================
# Apply Clause 3 refinement
# Class 0: +1 -> -1
# ==========================================
print(
    "Clause 3 before:",
    ctm.weight_banks[0].weights[3]
)

ctm.weight_banks[0].weights[3] = -1

print(
    "Clause 3 after:",
    ctm.weight_banks[0].weights[3]
)


# ==========================================
# Clause 488 is already 0 in refined v1
# ==========================================
print(
    "Clause 488:",
    ctm.weight_banks[0].weights[488]
)


# ==========================================
# Save combined refined model
# ==========================================
with open("refined_binary_ctm_v2.pkl", "wb") as f:
    pickle.dump(ctm, f)

print("\nCombined refined model saved as:")
print("refined_binary_ctm_v2.pkl")