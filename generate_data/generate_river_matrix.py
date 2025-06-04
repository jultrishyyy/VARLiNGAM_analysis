import sys
import os
import pickle
import numpy as np
import pandas as pd
import graphviz
import networkx as nx # For topological sort# 1. Load Preprocessed Data


data_path = "../causalriver/rivers_ts_bavaria_preprocessed.csv"
label_path = "../causalriver/rivers_bavaria.p" # Ground truth graph data


# label_path = "../causalriver/east_germany_matrix.npy" # Ground truth graph data

# matrix = np.load(label_path)
# print("Loaded matrix shape:", matrix.shape)


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
    print("B0_true adjacency matrix:")
    print(B0_true)
    np.save('../causalriver/bavaria_matrix.npy', B0_true)  # Save B0_true for later use

    # Derive true causal order (indices) from B0_true using topological sort
    rows, cols = np.where(B0_true != 0)
    true_graph_for_order_nx = nx.DiGraph(list(zip(cols, rows))) 
    true_graph_for_order_nx.add_nodes_from(range(B0_true.shape[0]))

    if nx.is_directed_acyclic_graph(true_graph_for_order_nx):
        true_causal_order_indices = list(nx.topological_sort(true_graph_for_order_nx))
        print(f"Derived True Causal Order (indices): {true_causal_order_indices}")
        true_causal_order_names = [true_node_names_ordered_by_data[i] for i in true_causal_order_indices]
        print(f"Derived True Causal Order (names):  {true_causal_order_names}")
        np.save('../causalriver/bavaria_order.npy', true_causal_order_indices) 
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