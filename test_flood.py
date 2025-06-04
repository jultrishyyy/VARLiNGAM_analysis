import sys
import os
import pickle
import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot # Not used in this version, but kept from original
import networkx as nx # For topological sort

# --- Path Setup (from your original script) ---
# Get the absolute path of the directory containing the 'examples' and 'lingam' folders
# This assumes your script might be run from a similar structure or you adjust as needed.
# For this specific task, we'll assume the script is run from a location where
# './causalriver/' path is valid.
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# if project_root not in sys.path:
# sys.path.insert(0, project_root)

print(f"LiNGAM version: {lingam.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NetworkX version: {nx.__version__}")
print(f"Graphviz version: {graphviz.__version__ if graphviz else 'Not found'}")


# --- Helper function to convert graph to Adjacency Matrix (for B0_true) ---
def H_graph_to_adjacency_matrix(G_sample, sorted_node_list_df_columns):
    """
    Converts a graph sample (e.g., NetworkX graph) to an adjacency matrix (B0_true).
    The order of nodes in the matrix is determined by sorted_node_list_df_columns.
    Args:
        G_sample: The graph object (should have .edges and .nodes attributes).
                  Nodes in G_sample.edges are expected to match names in sorted_node_list_df_columns.
        sorted_node_list_df_columns: A list of node names (strings) representing the
                                     order of columns in the input data DataFrame.
    Returns:
        A NumPy array (adjacency matrix for B0) where B[i, j] = 1 if j -> i.
    """
    num_nodes = len(sorted_node_list_df_columns)
    node_to_idx = {node_name: i for i, node_name in enumerate(sorted_node_list_df_columns)}
    
    adj_matrix = np.zeros((num_nodes, num_nodes))

    if not hasattr(G_sample, 'edges'):
        print("Error: G_sample does not have an 'edges' attribute.")
        return adj_matrix

    for u, v in G_sample.edges: # Edge u -> v (u is cause, v is effect)
        if str(u) in node_to_idx and str(v) in node_to_idx:
            idx_u_cause = node_to_idx[str(u)]
            idx_v_effect = node_to_idx[str(v)]
            adj_matrix[idx_v_effect, idx_u_cause] = 1 # B0[effect, cause] = 1
        # else:
            # print(f"Warning: Edge ({u} -> {v}) contains nodes not in the provided data columns. Skipping this edge.")
    return adj_matrix

# --- Helper function to calculate metrics (from your original script) ---
def calculate_edge_metrics_pruned(B_true, B_est_pruned):
    if B_true.shape != B_est_pruned.shape:
        print(f"Error: Shape mismatch - True: {B_true.shape}, Estimated: {B_est_pruned.shape}")
        return None

    true_edges = np.abs(B_true) > 0 
    est_edges = np.abs(B_est_pruned) > 0 
    np.fill_diagonal(true_edges, False)
    np.fill_diagonal(est_edges, False)

    tp = np.sum(true_edges & est_edges)
    fp = np.sum(~true_edges & est_edges)
    fn = np.sum(true_edges & ~est_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    total_true_edges = np.sum(true_edges)
    total_est_edges = np.sum(est_edges)

    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision,
        'Recall': recall, 'F1': f1, 'Total_True_Edges': total_true_edges,
        'Total_Est_Edges': total_est_edges
    }

# --- Main Logic ---

# 1. Load Preprocessed Data
data_path = "./causalriver/rivers_ts_flood_preprocessed.csv"
label_path = "./causalriver/rivers_flood.p" # Ground truth graph data

try:
    X_df = pd.read_csv(data_path, index_col='datetime')
    X_df.index = pd.to_datetime(X_df.index)
    print(f"Successfully loaded preprocessed data from: {data_path}")
    print(f"Data shape: {X_df.shape}")
    # Ensure all data columns are numeric, convert if necessary, or handle errors
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    if X_df.isnull().any().any():
        print("Warning: NaNs found after converting data to numeric. Consider imputation or dropping.")
        # Example: X_df = X_df.fillna(X_df.mean()) # Or dropna()
        # For now, we proceed, but LiNGAM might fail with NaNs.
        X_df = X_df.fillna(0) # Simplistic NaN handling for now
        print("NaNs filled with 0 for VARLiNGAM.")


except FileNotFoundError:
    print(f"Error: Preprocessed data file not found at {data_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading preprocessed data: {e}")
    sys.exit(1)

# 2. Load Ground Truth Label (Graph) and derive B0_true and true_causal_order
B0_true = None
true_causal_order_indices = None
true_node_names_ordered_by_data = list(X_df.columns) # Order of nodes as in the DataFrame

try:
    with open(label_path, 'rb') as f:
        ground_truth_data_loaded = pickle.load(f)
    
    G_true_sample = None # Initialize G_true_sample

    # Check if the loaded data is a list of graphs or a single graph
    if isinstance(ground_truth_data_loaded, list):
        if len(ground_truth_data_loaded) > 0:
            G_true_sample = ground_truth_data_loaded[0] # Use the first graph if it's a list
            print(f"Successfully loaded ground truth: list of graphs found in {label_path}, using the first graph.")
        else:
            print(f"Error: Label file {label_path} contained an empty list.")
            sys.exit(1)
    elif hasattr(ground_truth_data_loaded, 'edges') and hasattr(ground_truth_data_loaded, 'nodes'):
        # If not a list, check if it looks like a single graph object (e.g., NetworkX graph)
        G_true_sample = ground_truth_data_loaded
        print(f"Successfully loaded ground truth: single graph object found directly in {label_path}.")
    else:
        print(f"Error: Label file {label_path} is not in the expected format. Expected a list of graphs or a single graph object. Type found: {type(ground_truth_data_loaded)}")
        sys.exit(1)

    # Ensure G_true_sample was successfully assigned
    if G_true_sample is None:
        print(f"Critical Error: G_true_sample could not be determined from {label_path}. Exiting.")
        sys.exit(1)
        
    print(f"Ground truth sample type: {type(G_true_sample)}")

    # Convert graph to B0_true adjacency matrix
    B0_true = H_graph_to_adjacency_matrix(G_true_sample, true_node_names_ordered_by_data)
    print(f"Derived B0_true with shape: {B0_true.shape}")

    # Derive true causal order (indices) from B0_true using topological sort
    rows, cols = np.where(B0_true != 0)
    true_graph_for_order_nx = nx.DiGraph(list(zip(cols, rows))) 
    true_graph_for_order_nx.add_nodes_from(range(B0_true.shape[0]))

    if nx.is_directed_acyclic_graph(true_graph_for_order_nx):
        true_causal_order_indices = list(nx.topological_sort(true_graph_for_order_nx))
        print(f"Derived True Causal Order (indices): {true_causal_order_indices}")
        true_causal_order_names = [true_node_names_ordered_by_data[i] for i in true_causal_order_indices]
        print(f"Derived True Causal Order (names):  {true_causal_order_names}")
    else:
        print("Warning: The ground truth B0 matrix (derived from the label file) does not form a DAG. Cannot derive a unique topological sort.")
        true_causal_order_indices = list(range(B0_true.shape[0])) 
        print(f"Using default order as True Causal Order (indices): {true_causal_order_indices}")

except FileNotFoundError:
    print(f"Error: Label file not found at {label_path}")
    sys.exit(1)
except pickle.UnpicklingError:
    print(f"Error: Could not unpickle the label file at {label_path}. It might be corrupted or not a pickle file.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading or processing label file: {e}")
    sys.exit(1)

# 3. Fit VARLiNGAM model
print("\nFitting VARLiNGAM model...")
# Consider parameters like lags, prune. Default lags=1 (B0, B1). Default prune=False.
# If your data has many variables or time points, this can take time.
model = lingam.VARLiNGAM(lags=1, prune=True) # Using prune=True as per metrics function context
try:
    model.fit(X_df.to_numpy()) # VARLiNGAM expects a NumPy array
except Exception as e:
    print(f"Error during VARLiNGAM fitting: {e}")
    print("This might be due to data issues (e.g., all-zero columns, collinearity, NaNs not handled).")
    sys.exit(1)

print("VARLiNGAM fitting complete.")

# 4. Get Estimated Causal Order and Adjacency Matrices
causal_order_estimated_indices = model.causal_order_
estimated_matrices = model.adjacency_matrices_ # [B0_est, B1_est, ...]

print(f"\nEstimated Causal Order (indices): {causal_order_estimated_indices}")
if causal_order_estimated_indices is not None:
    estimated_causal_order_names = [true_node_names_ordered_by_data[i] for i in causal_order_estimated_indices]
    print(f"Estimated Causal Order (names):  {estimated_causal_order_names}")


# 5. Calculate and Print Metrics for B0 (Instantaneous Effects)
if B0_true is not None and estimated_matrices is not None and len(estimated_matrices) > 0:
    B0_est_pruned = estimated_matrices[0] # Assuming the first matrix is B0

    print("\n--- Edge Discovery Metrics for B0 (Instantaneous Effects, Post-Pruning by VARLiNGAM) ---")
    metrics_b0 = calculate_edge_metrics_pruned(B0_true, B0_est_pruned)
    if metrics_b0:
        print(f"  True Edges (B0):            {metrics_b0['Total_True_Edges']}")
        print(f"  Estimated Edges (B0):       {metrics_b0['Total_Est_Edges']}")
        print(f"  Correct Edges (TP for B0):  {metrics_b0['TP']}")
        print(f"  Incorrect Edges (FP for B0):{metrics_b0['FP']}")
        print(f"  Missed Edges (FN for B0):   {metrics_b0['FN']}")
        print(f"  Precision (B0):             {metrics_b0['Precision']:.4f}")
        print(f"  Recall (B0):                {metrics_b0['Recall']:.4f}")
        print(f"  F1 Score (B0):              {metrics_b0['F1']:.4f}")
    else:
        print("  Could not calculate metrics for B0 (e.g., shape mismatch or error in function).")
    
    if len(estimated_matrices) > 1:
        B1_est_pruned = estimated_matrices[1]
        print(f"\nNote: Model also estimated B1 (lagged effects matrix) with shape {B1_est_pruned.shape}.")
        print("However, ground truth for B1 is not available from the provided label file for direct comparison.")

else:
    print("\nCould not retrieve estimated adjacency matrices from the model, or B0_true was not derived.")

# Optional: Visualize B0_true and B0_est if needed, e.g., using graphviz or matplotlib
# from lingam.utils import make_dot
# dot_b0_true = make_dot(B0_true, labels=true_node_names_ordered_by_data)
# dot_b0_true.render('results/B0_true_flood', cleanup=True)

# if 'B0_est_pruned' in locals():
#    dot_b0_est = make_dot(B0_est_pruned, labels=true_node_names_ordered_by_data)
#    dot_b0_est.render('results/B0_est_flood_pruned', cleanup=True)
#    print("\nVisualizations of B0_true and B0_est_pruned saved to 'results' directory (if Graphviz is installed).")
print(f"Derived True Causal Order (indices): {true_causal_order_indices}")

print(f"Derived True Causal Order (names):  {true_causal_order_names}")

print("\n--- Script Finished ---")