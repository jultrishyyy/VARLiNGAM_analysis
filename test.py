import argparse
import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot, print_causal_directions, print_dagc

import argparse
import os
import sys


from utils.helper import convert_Btaus_to_summary_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

def parse_arguments():
    """Parse command-line arguments for the script."""
    # Get the current directory (root of the script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set default file path to <current_directory>/data/sample_data.csv
    default_data_path = os.path.join(data_dir, "sample_data.csv")
    default_label_path = os.path.join(data_dir, "sample_summary_matrix.npy")
    
    parser = argparse.ArgumentParser(description="Test VARLiNGAM with a specified file.")
    parser.add_argument("--data", type=str, help="Path to the input data", default=default_data_path)
    parser.add_argument("--evaluate", type=int, help="Whether to evaluate with labels", default=True, choices=[0, 1])
    parser.add_argument("--label", type=str, help="Path to the input label", default=default_label_path)
    parser.add_argument("--set_diagonal_0", type=int, help="Whether to set diagonal as 0 when evaluating", default=False, choices=[0, 1])
    return parser.parse_args()

def load_data_and_model(file_path):
    """Load data from CSV and initialize the VARLiNGAM model."""
    try:
        data = pd.read_csv(file_path)
        model = lingam.VARLiNGAM()
        model.fit(data)
        return data, model
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None
    except Exception as e:
        print(f"Error loading data or fitting model: {e}")
        return None, None

def calculate_edge_metrics(B_true, B_est, set_diagonal_0=False):
    """
    Calculate edge detection metrics (TP, FP, FN, Precision, Recall, F1)
    based on non-zero entries in theestimated matrix.
    Ignores diagonal elements.
    """
    if B_true.shape != B_est.shape:
        print(f"Error: Shape mismatch - True: {B_true.shape}, Estimated: {B_est.shape}")
        return None

    true_edges = np.abs(B_true) > 0
    est_edges = np.abs(B_est) > 0

    if set_diagonal_0:
        np.fill_diagonal(true_edges, False)
        np.fill_diagonal(est_edges, False)

    tp = np.sum(true_edges & est_edges)
    fp = np.sum(~true_edges & est_edges)
    fn = np.sum(true_edges & ~est_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Total_True_Edges': np.sum(true_edges),
        'Total_Est_Edges': np.sum(est_edges)
    }

def print_metrics(metrics, matrix_name):
    """Print edge detection metrics for a given matrix."""
    if metrics:
        print(f"\nMetrics for {matrix_name}:")
        print(f"  True Edges:       {metrics['Total_True_Edges']}")
        print(f"  Estimated Edges:  {metrics['Total_Est_Edges']}")
        print(f"  Correct Edges (TP): {metrics['TP']}")
        print(f"  Incorrect Edges (FP): {metrics['FP']}")
        print(f"  Missed Edges (FN):  {metrics['FN']}")
        print(f"  Precision:        {metrics['Precision']:.4f}")
        print(f"  Recall (Correct Edge Ratio): {metrics['Recall']:.4f}")
        print(f"  F1 Score:         {metrics['F1']:.4f}")
    else:
        print(f"  Could not calculate metrics for {matrix_name} (shape mismatch).")

def main():
    """Main function to execute the VARLiNGAM analysis."""
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

    # Parse arguments
    args = parse_arguments()
    print(f"Using data: {args.data}")
    print(f"Using label: {args.label}")

    # Load data and fit model
    _, model = load_data_and_model(args.data)
    if model is None:
        return

    if args.evaluate:
        label = np.load(args.label)
        if label is None:
            print(f"Error: Label data file {args.label} is empty or not found.")
            return
        print("\n--- Label Summary Matrix ---")
        print(label)
        
        if model.adjacency_matrices_ is not None:
            estimated_matrix = convert_Btaus_to_summary_matrix(model.adjacency_matrices_)
            print("\n--- Estimated Summary Matrix (non-zero values are 1) ---")
            print(estimated_matrix)
            metrics = calculate_edge_metrics(label, estimated_matrix, args.set_diagonal_0)
            print_metrics(metrics, "input summary matrix")
        else:
            print("Could not retrieve estimated adjacency matrices (or not enough matrices found) from the model.")

        print(f"\nEstimated Causal Order: {model.causal_order_}")

        true_causal_order = list(np.load(os.path.join(data_dir, "sample_causal_order.npy")))
        print(f"True Causal Order: {true_causal_order}")

if __name__ == "__main__":
    main()