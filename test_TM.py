import sys
import os

# Get the absolute path of the directory containing the 'examples' and 'lingam' folders
# This assumes your notebook is directly inside the 'examples' folder.
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.var_lingam_TM import VARLiNGAM
from lingam.utils import make_dot, print_causal_directions, print_dagc

print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])




B0_true = np.load("./data/b0.npy")
B1_true = np.load("./data/b1.npy")


X = pd.read_csv('data/data_1.csv')

model = VARLiNGAM(
    lags=1,
    criterion='bic',
    prune=True,
    pruning_threshold=0.05,
    use_tm=True,
    tm_clauses=100,
    tm_states=100,
    tm_s=3.0,
    tm_T=10,
    bin_threshold=0.5,
    random_state=42
)
model.fit(X)

estimated_matrices = model.adjacency_matrices_
if estimated_matrices is not None and len(estimated_matrices) >= 2:
    B0_est = estimated_matrices[0]
    B1_est = estimated_matrices[1]

    # 4. Calculate Mean Squared Error (MSE)
    # Ensure dimensions match
    if B0_true.shape == B0_est.shape and B1_true.shape == B1_est.shape:
        # Calculate squared errors for B0 and B1
        squared_error_b0 = np.sum((B0_true - B0_est)**2)
        squared_error_b1 = np.sum((B1_true - B1_est)**2)

        # Total number of elements in B0 and B1
        total_elements = B0_true.size + B1_true.size

        # Calculate MSE
        mse = (squared_error_b0 + squared_error_b1) / total_elements

        print(f"\nMean Squared Error (MSE) between true and estimated B0 & B1: {mse:.6f}")
        # Lower MSE indicates higher accuracy in estimating the matrix coefficients
    else:
        print("Error: Shape mismatch between true and estimated B matrices.")
        if estimated_matrices is not None:
             print(f"  True B0 shape: {B0_true.shape}, Estimated B0 shape: {B0_est.shape}")
             print(f"  True B1 shape: {B1_true.shape}, Estimated B1 shape: {B1_est.shape}")


    # You can still print the estimated order if needed
    causal_order_estimated = model.causal_order_
    print(f"Estimated Causal Order:      {causal_order_estimated}")

else:
    print("Could not retrieve estimated adjacency matrices from the model.")


estimated_matrices = model.adjacency_matrices_ # Assumes this IS the pruned result
# -------------------------------------------------


# --- Helper function to calculate metrics (No Threshold) ---
def calculate_edge_metrics_pruned(B_true, B_est_pruned):
    """
    Calculates edge detection metrics (TP, FP, FN, Precision, Recall, F1)
    based on non-zero entries in the pruned estimated matrix.
    Ignores diagonal elements.
    """
    if B_true.shape != B_est_pruned.shape:
        print(f"Error: Shape mismatch - True: {B_true.shape}, Estimated: {B_est_pruned.shape}")
        return None

    # Create boolean masks for existing edges (ignore diagonal)
    true_edges = np.abs(B_true) > 0 # Or B_true != 0
    # Use non-zero check for the pruned matrix
    est_edges = np.abs(B_est_pruned) > 0 # Or B_est_pruned != 0
    np.fill_diagonal(true_edges, False)
    np.fill_diagonal(est_edges, False)

    # Calculate TP, FP, FN
    tp = np.sum(true_edges & est_edges)
    fp = np.sum(~true_edges & est_edges)
    fn = np.sum(true_edges & ~est_edges)

    # Calculate Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Correct Edge Ratio
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate total true edges for ratio calculations
    total_true_edges = np.sum(true_edges)
    total_est_edges = np.sum(est_edges)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Total_True_Edges': total_true_edges,
        'Total_Est_Edges': total_est_edges
    }
# --- End Helper function ---


# --- Main Logic (Using the pruned helper function) ---


total_tp = 0
total_fp = 0
total_fn = 0
results = {}

if estimated_matrices is not None and len(estimated_matrices) >= 2:
    # Assuming these are the PRUNED matrices
    B0_est_pruned = estimated_matrices[0]
    B1_est_pruned = estimated_matrices[1]

    print("\n--- Edge Discovery Metrics (Post-Pruning) ---")

    # --- Metrics for Pruned B0 ---
    print("\nMetrics for Pruned B0 (Instantaneous Effects):")
    # Use the new helper function
    metrics_b0 = calculate_edge_metrics_pruned(B0_true, B0_est_pruned)
    if metrics_b0:
        results['B0'] = metrics_b0
        print(f"  True Edges:       {metrics_b0['Total_True_Edges']}")
        print(f"  Estimated Edges:  {metrics_b0['Total_Est_Edges']}")
        print(f"  Correct Edges (TP): {metrics_b0['TP']}")
        print(f"  Incorrect Edges (FP): {metrics_b0['FP']}")
        print(f"  Missed Edges (FN):  {metrics_b0['FN']}")
        print(f"  Precision:        {metrics_b0['Precision']:.4f}")
        print(f"  Recall (Correct Edge Ratio): {metrics_b0['Recall']:.4f}")
        print(f"  F1 Score:         {metrics_b0['F1']:.4f}")
        total_tp += metrics_b0['TP']
        total_fp += metrics_b0['FP']
        total_fn += metrics_b0['FN']
    else:
        print("  Could not calculate metrics for B0 (shape mismatch).")


    # --- Metrics for Pruned B1 ---
    print("\nMetrics for Pruned B1 (Lagged Effects):")
    # Use the new helper function
    metrics_b1 = calculate_edge_metrics_pruned(B1_true, B1_est_pruned)
    if metrics_b1:
        results['B1'] = metrics_b1
        print(f"  True Edges:       {metrics_b1['Total_True_Edges']}")
        print(f"  Estimated Edges:  {metrics_b1['Total_Est_Edges']}")
        print(f"  Correct Edges (TP): {metrics_b1['TP']}")
        print(f"  Incorrect Edges (FP): {metrics_b1['FP']}")
        print(f"  Missed Edges (FN):  {metrics_b1['FN']}")
        print(f"  Precision:        {metrics_b1['Precision']:.4f}")
        print(f"  Recall (Correct Edge Ratio): {metrics_b1['Recall']:.4f}")
        print(f"  F1 Score:         {metrics_b1['F1']:.4f}")
        total_tp += metrics_b1['TP']
        total_fp += metrics_b1['FP']
        total_fn += metrics_b1['FN']
    else:
        print("  Could not calculate metrics for B1 (shape mismatch).")

    # --- Overall Metrics (Combining Pruned B0 and B1, excluding diagonals) ---
    if 'B0' in results or 'B1' in results:
        print("\nOverall Metrics (Pruned B0 + B1):")
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        print(f"  Total True Edges:       {results.get('B0', {}).get('Total_True_Edges', 0) + results.get('B1', {}).get('Total_True_Edges', 0)}")
        print(f"  Total Estimated Edges:  {results.get('B0', {}).get('Total_Est_Edges', 0) + results.get('B1', {}).get('Total_Est_Edges', 0)}")
        print(f"  Total Correct Edges (TP): {total_tp}")
        print(f"  Total Incorrect Edges (FP): {total_fp}")
        print(f"  Total Missed Edges (FN):  {total_fn}")
        print(f"  Overall Precision:        {overall_precision:.4f}")
        print(f"  Overall Recall:           {overall_recall:.4f}")
        print(f"  Overall F1 Score:         {overall_f1:.4f}")

    # You can still print the estimated order if needed
    causal_order_estimated = model.causal_order_
    print(f"\nEstimated Causal Order: {causal_order_estimated}") # Causal order often comes from the pre-pruning B0

else:
    print("Could not retrieve estimated adjacency matrices (or not enough matrices found) from the model.")
