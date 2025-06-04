import numpy as np
import os


# --- Step 2: Load the B0 and B1 matrices ---
try:
    B0 = np.load('../data/violated/violated_b0.npy')
    B1 = np.load('../data/violated/violated_b1.npy')
    
    print("\n--- Loaded B0 Matrix ---")
    print(B0)
    
    print("\n--- Loaded B1 Matrix ---")
    print(B1)

except FileNotFoundError:
    print("Error: Ensure 'B0.npy' and 'B1.npy' are in the correct directory.")
    exit()


# --- Step 3: Combine matrices and convert non-zero values to 1 ---

# Check where either B0 or B1 is not zero. This creates a boolean (True/False) matrix.
# The '|' operator performs an element-wise logical OR.
combined_boolean_matrix = (B0 != 0) | (B1 != 0)

# Convert the boolean matrix (True/False) to an integer matrix (1/0)
summary_matrix = combined_boolean_matrix.astype(int)

print("\n--- Integrated Summary Matrix (non-zero values are 1) ---")
print(summary_matrix)


# --- Step 4: Save the summary matrix to a new .npy file ---
output_filename = '../data/violated/summary_matrix.npy'
np.save(output_filename, summary_matrix)

print(f"\nSummary matrix saved to '{output_filename}'")