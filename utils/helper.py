
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def convert_Btaus_to_summary_matrix(Btaus):
    """
    Convert a list of Btau matrices (2D or 3D) to a 2D summary matrix.
    
    Args:
        Btaus (list): List of numpy arrays (2D or 3D matrices) to combine.
        
    Returns:
        np.ndarray: 2D summary matrix where non-zero elements across all input matrices are 1, others are 0.
        
    Raises:
        ValueError: If Btaus is empty, matrices have incompatible shapes, or input arrays have invalid dimensions.
    """
    if Btaus is None or len(Btaus) == 0:
        raise ValueError("Input list of matrices (Btaus) is empty.")
    
    # Validate and convert matrices to 3D
    processed_Btaus = []
    ref_shape = None
    
    for Btau in Btaus:
        Btau = np.asarray(Btau)  # Ensure input is a NumPy array
        if Btau.ndim == 2:
            # Convert 2D matrix to 3D with singleton first dimension
            Btau = Btau[np.newaxis, :, :]
        elif Btau.ndim != 3:
            raise ValueError(f"Input matrix must be 2D or 3D, got {Btau.ndim}D")
        
        # Check shape compatibility (ignoring first dimension for 3D)
        current_shape = Btau.shape[1:]  # (n, m) for 3D arrays
        if ref_shape is None:
            ref_shape = current_shape
        elif current_shape != ref_shape:
            raise ValueError(f"Incompatible matrix shapes: expected {ref_shape}, got {current_shape}")
        
        processed_Btaus.append(Btau)
    
    # Combine non-zero elements across all matrices
    combined_boolean_matrix = None
    for Btau in processed_Btaus:
        # Create boolean mask for non-zero elements across all slices
        matrix_nonzero = np.any(np.abs(Btau) > 0, axis=0)  # Reduce along first dimension
        if combined_boolean_matrix is None:
            combined_boolean_matrix = matrix_nonzero
        else:
            combined_boolean_matrix |= matrix_nonzero
    
    # Convert boolean matrix to integer (1/0)
    summary_matrix = combined_boolean_matrix.astype(int)
    
    return summary_matrix


def plot_summary_causal_graph(B_taus, filename="summary_causal_graph.png"):
    """
    Plots and saves a summary causal graph from a 3D array of Btau matrices.
    Includes larger nodes, attempts shorter edges, and adds a legend.
    
    Args:
        B_taus (np.ndarray): 3D array of shape (k, n, n) where k is the number of matrices,
                             and n is the number of variables (square matrices).
        filename (str): File path to save the plot.
        
    Raises:
        ValueError: If B_taus is not a 3D array or matrices are not square.
    """
    # Validate input
    B_taus = np.asarray(B_taus)
    if B_taus.ndim != 3:
        raise ValueError(f"B_taus must be a 3D array, got {B_taus.ndim}D")
    if B_taus.shape[1] != B_taus.shape[2]:
        raise ValueError(f"B_taus matrices must be square, got shape {B_taus.shape[1:]}")
    
    # Extract number of variables
    n_vars = B_taus.shape[1]
    
    # Initialize directed graph
    G = nx.DiGraph()
    nodes = list(range(n_vars))
    G.add_nodes_from(nodes)
    
    # Collect edges from all matrices
    edge_lists = []
    for k in range(B_taus.shape[0]):
        edge_list = []
        B_tau = B_taus[k]
        for i in range(n_vars):
            for j in range(n_vars):
                if B_tau[i, j] != 0:
                    edge_list.append((j, i))  # Edge from j to i if B_tau[i,j] is non-zero
        edge_lists.append(set(edge_list))
    
    # Combine all unique edges
    all_edges = set()
    for edge_list in edge_lists:
        all_edges.update(edge_list)
    G.add_edges_from(list(all_edges))
    
    # Determine edge colors
    edge_colors = []
    for u, v in G.edges():
        edge_in_matrices = [k for k, edge_list in enumerate(edge_lists) if (u, v) in edge_list]
        if len(edge_in_matrices) == 1:
            if edge_in_matrices[0] == 0:
                edge_colors.append('red')    # Edge only in first matrix (e.g., B0, instantaneous)
            else:
                edge_colors.append('blue')   # Edge only in other matrices (e.g., B1, lagged)
        elif len(edge_in_matrices) > 1:
            edge_colors.append('green')      # Edge in multiple matrices
        else:
            edge_colors.append('black')      # Fallback, should not occur
    
    # Plotting
    plt.figure(figsize=(7, 7))
    
    # Layout parameters
    if n_vars > 15:
        print(f"Warning: Plotting a summary graph with {n_vars} nodes. "
              "Layout might be slow and cluttered. Using spring_layout.")
        pos = nx.spring_layout(G, k=0.6/np.sqrt(n_vars), iterations=50, seed=42)
        node_size = 350
        font_size = 7
        arrow_size = 12
        edge_width = 1.0
    else:
        if n_vars <= 5:
            k_val = 0.9
        elif n_vars <= 10:
            k_val = 0.6
        else:
            k_val = 0.5
        pos = nx.spring_layout(G, k=k_val, iterations=100, seed=42)
        node_size = 3000
        font_size = 35
        arrow_size = 25
        edge_width = 3.5
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue",
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color=edge_colors, width=edge_width, connectionstyle='arc3,rad=0.2')
    
    plt.title(f"Summary Causal Graph (n={n_vars})", fontsize=25)
    # Create legend handles
    red_line = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Instantaneous (B0)')
    blue_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='None', markersize=10, label='Lagged (B1)')
    purple_line = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Both (B0 & B1)')
    
    plt.legend(handles=[red_line, blue_line, purple_line], loc='upper right', fontsize=15)
    
    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Summary causal graph saved to {filename}")
    except Exception as e:
        print(f"Error saving summary graph: {e}")
    plt.close()

    