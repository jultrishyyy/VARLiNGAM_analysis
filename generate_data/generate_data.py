import numpy as np
import pandas as pd
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# def generate_data(n=5, T=1000, random_state=None, initial_data=None):
#     """
#     Parameter
#     ---------
#     n : int
#         number of variables
#     T : int
#         number of samples
#     random_state : int
#         seed for np.random.seed
#     initial_data : list of np.ndarray
#         dictionary of initial datas
#     """

#     T_spurious = 20
#     expon = 1.5
    
#     if initial_data is None:
#         permutation = np.random.permutation(n)
        
#         value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
#         sign = np.random.choice([-1, 1], size=(n, n))
#         B0 = np.multiply(value, sign)
        
#         B0 = np.multiply(B0, np.random.binomial(1, 0.4, size=(n, n)))
#         B0 = np.tril(B0, k=-1)
#         B0 = B0[permutation][:, permutation]

#         value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
#         sign = np.random.choice([-1, 1], size=(n, n))
#         B1 = np.multiply(value, sign)
#         B1 = np.multiply(B1, np.random.binomial(1, 0.4, size=(n, n)))
        
#         causal_order = np.empty(len(permutation))
#         causal_order[permutation] = np.arange(len(permutation))
#         causal_order = causal_order.astype(int)
#     else:
#         B0 = initial_data['B0']
#         B1 = initial_data['B1']
#         causal_order =initial_data['causal_order'] 
        
#     M1 = np.dot(np.linalg.inv(np.eye(n) - B0), B1)

#     ee = np.empty((n, T + T_spurious))
#     for i in range(n):
#         ee[i, :] = np.random.normal(size=(1, T + T_spurious))
#         ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon)
#         ee[i, :] = ee[i, :] - np.mean(ee[i, :])
#         ee[i, :] = ee[i, :] / np.std(ee[i, :])

#     std_e = np.random.uniform(size=(n,)) + 0.5
#     nn = np.dot(np.dot(np.linalg.inv(np.eye(n) - B0), np.diag(std_e)), ee)

#     xx = np.zeros((n, T + T_spurious))
#     xx[:, 0] = np.random.normal(size=(n, ))

#     for t in range(1, T + T_spurious):
#         xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]

#     data = xx[:, T_spurious + 1 : T_spurious + T]
    
#     return data.T, B0, B1, causal_order


def generate_data(n=5, T=1000, random_state=None, initial_data=None):
    """
    Parameter
    ---------
    n : int
        number of variables
    T : int
        number of samples
    random_state : int
        seed for np.random.seed
    initial_data : dict
        dictionary of initial datas
    """
    if random_state is not None:
        np.random.seed(random_state)

    T_spurious = 20
    expon = 1.5
    
    # Generate B0, B1, causal_order once as per original logic
    if initial_data is None:
        permutation = np.random.permutation(n)
        
        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B0 = np.multiply(value, sign)
        
        B0 = np.multiply(B0, np.random.binomial(1, 0.4, size=(n, n)))
        B0 = np.tril(B0, k=-1)
        B0 = B0[permutation][:, permutation]

        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B1 = np.multiply(value, sign)
        B1 = np.multiply(B1, np.random.binomial(1, 0.4, size=(n, n)))
        
        causal_order = np.empty(len(permutation))
        causal_order[permutation] = np.arange(len(permutation))
        causal_order = causal_order.astype(int)
    else:
        B0 = initial_data['B0']
        B1 = initial_data['B1']
        causal_order = initial_data['causal_order']
    
    # Initialize xx to zeros. It will be filled if generation fails or produces Inf/NaN.
    xx = np.zeros((n, T + T_spurious)) 

    try:
        # Attempt to calculate M1 and nn using matrix inversion.
        # This is the original logic which might produce Inf if (I-B0) is ill-conditioned
        # or raise LinAlgError if it's exactly singular.
        inv_I_B0 = np.linalg.inv(np.eye(n) - B0) 
        M1 = np.dot(inv_I_B0, B1)

        # Generate exogenous noise ee
        ee = np.empty((n, T + T_spurious))
        for i in range(n):
            ee[i, :] = np.random.normal(size=(1, T + T_spurious))
            ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon)
            ee[i, :] = ee[i, :] - np.mean(ee[i, :])
            ee[i, :] = ee[i, :] / np.std(ee[i, :])

        std_e = np.random.uniform(size=(n,)) + 0.5
        nn = np.dot(np.dot(inv_I_B0, np.diag(std_e)), ee) # This might also contain Inf/NaN

        # Proceed with time series generation.
        # M1 or nn might contain Inf/NaN if inv_I_B0 did.
        xx[:, 0] = np.random.normal(size=(n,)) # Initial values
        for t in range(1, T + T_spurious):
            # This step might result in xx containing Inf or NaN due to M1 or nn
            xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]

    except np.linalg.LinAlgError:
        # This occurs if (I - B0) is exactly singular and inv() fails.
        # In this case, M1 and nn were not computed, so xx remains primarily zeros.
        warnings.warn(
            f"Singular matrix (I - B0) encountered during inversion. "
            f"Time series 'xx' could not be fully generated using original VAR dynamics. "
            f"The 'xx' array (initially zeros) will be cleaned of any NaNs/Infs if they arose."
        )
        # xx is already initialized to zeros, so if this path is taken,
        # the subsequent nan_to_num will effectively ensure it stays numerically clean (mostly zeros).
        pass

    # --- Fill any NaNs or Infs in the xx array ---
    # This step cleans up xx regardless of whether the 'try' block produced Inf/NaN
    # or the 'except' block was hit (leaving xx as mostly zeros).
    # We replace nan, positive infinity, and negative infinity with 0.0.
    # You could choose other fill values for posinf/neginf if desired,
    # e.g., finfo = np.finfo(xx.dtype); xx = np.nan_to_num(xx, nan=0.0, posinf=finfo.max/n, neginf=finfo.min/n)
    xx = np.nan_to_num(xx, nan=0.0, posinf=0.0, neginf=0.0, copy=False) # copy=False modifies xx in-place

    # Slice the 'data' from the (now cleaned) xx array
    data = xx[:, T_spurious + 1 : T_spurious + T]
    
    # A final safety check on the sliced 'data', though xx should be clean.
    if np.isnan(data).any() or np.isinf(data).any():
        warnings.warn("Unexpected NaN or Inf in the final 'data' slice after cleaning 'xx'. Applying final cleanup.")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        
    return data.T, B0, B1, causal_order


# def plot_summary_causal_graph(B0, B1, n_vars, filename="summary_causal_graph.png"):
#     """
#     Plots and saves a summary causal graph from B0 (instantaneous) and B1 (lagged) matrices.
#     Includes larger nodes, attempts shorter edges, and adds a legend.
#     """
#     G = nx.DiGraph()
#     nodes = list(range(n_vars))
#     G.add_nodes_from(nodes)

#     edge_list_b0 = []
#     edge_list_b1 = []

#     # Add edges from B0 (instantaneous)
#     for i in range(n_vars):
#         for j in range(n_vars):
#             if B0[i, j] != 0:
#                 edge_list_b0.append((j, i)) # Edge from j to i if B0[i,j] is effect of Xj on Xi

#     # Add edges from B1 (lagged)
#     for i in range(n_vars):
#         for j in range(n_vars):
#             if B1[i, j] != 0:
#                 edge_list_b1.append((j, i)) # Edge from j(t-1) to i(t) if B1[i,j] is effect of Xj(t-1) on Xi(t)
    
#     all_edges = set()
#     # Convert lists of tuples to sets of tuples for efficient checking
#     set_edge_list_b0 = set(edge_list_b0)
#     set_edge_list_b1 = set(edge_list_b1)

#     # Add all unique edges to the graph
#     for edge in set_edge_list_b0:
#         all_edges.add(edge)
#     for edge in set_edge_list_b1:
#         all_edges.add(edge)
    
#     G.add_edges_from(list(all_edges)) # Add unique edges to the graph

#     # Determine edge colors
#     edge_colors = []
#     for u, v in G.edges(): # Iterate over edges in the graph G
#         is_b0 = (u,v) in set_edge_list_b0
#         is_b1 = (u,v) in set_edge_list_b1
#         if is_b0 and is_b1:
#             edge_colors.append('green') # Edge from both B0 and B1
#         elif is_b0:
#             edge_colors.append('red')    # Edge only from B0
#         elif is_b1:
#             edge_colors.append('blue')   # Edge only from B1
#         else:
#             # This case should not happen if all_edges are sourced from b0_list and b1_list
#             edge_colors.append('black') # Fallback, though ideally not reached

#     plt.figure(figsize=(7, 7)) # Slightly larger figure size
    
#     # Layout parameters
#     if n_vars > 15: # Adjusted threshold for spring_layout
#         print(f"Warning: Plotting a summary graph with {n_vars} nodes. "
#               "Layout might be slow and cluttered. Using spring_layout.")
#         # For spring_layout, k controls the optimal distance between nodes. Smaller k = denser.
#         pos = nx.spring_layout(G, k=0.6/np.sqrt(n_vars), iterations=50, seed=42) # k changed from 0.8 to 0.6
#         node_size = 350  # Smaller nodes for many variables, increased from 300
#         font_size = 7
#         arrow_size = 12
#         edge_width = 1.0
#     else:
#         pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
#         node_size = 3000 # Larger nodes, increased from 1500
#         font_size = 35
#         arrow_size = 25
#         edge_width = 3.5

#     nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
#             font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
#             edge_color=edge_colors, width=edge_width, connectionstyle='arc3,rad=0.2') # Added connectionstyle for curved edges
    
#     # plt.title(f"Summary Causal Graph (n={n_vars})", fontsize=25)

#     ## Create legend handles
#     # red_line = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Instantaneous (B0)')
#     # blue_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='None', markersize=10, label='Lagged (B1)')
#     # purple_line = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Both (B0 & B1)')
    
#     # plt.legend(handles=[red_line, blue_line, purple_line], loc='upper right', fontsize=15)
    
#     try:
#         plt.savefig(filename, bbox_inches='tight') # bbox_inches to ensure legend is saved
#         print(f"Summary causal graph saved to {filename}")
#     except Exception as e:
#         print(f"Error saving summary graph: {e}")
#     plt.close()

# def plot_instantaneous_causal_graph(B0, n_vars, filename="instantaneous_causal_graph.png"):
#     """
#     Plots and saves an instantaneous causal graph based only on the B0 matrix.
#     Includes larger nodes, attempts shorter edges, and adds a legend.
#     """
#     G = nx.DiGraph()
#     nodes = list(range(n_vars))
#     G.add_nodes_from(nodes)

#     edge_list_b0 = []
#     for i in range(n_vars):
#         for j in range(n_vars):
#             if B0[i, j] != 0:
#                 edge_list_b0.append((j, i)) 
    
#     G.add_edges_from(edge_list_b0)

#     plt.figure(figsize=(7, 7))
    
#     if n_vars > 15:
#         print(f"Warning: Plotting instantaneous graph with {n_vars} nodes. "
#               "Layout might be slow and cluttered. Using spring_layout.")
#         pos = nx.spring_layout(G, k=0.8/np.sqrt(n_vars), iterations=50, seed=42) 
#         node_size = 300
#         font_size = 7
#         arrow_size = 12
#         edge_width = 1.0
#     else:
#         pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
#         node_size = 3000 # Larger nodes, increased from 1500
#         font_size = 35
#         arrow_size = 25
#         edge_width = 3.5
    
#     nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
#             font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
#             edge_color='red', width=edge_width, connectionstyle='arc3,rad=0.2')
    
#     # plt.title(f"Instantaneous Causal Graph (B0 effects, n={n_vars})", fontsize=20)

#     # # Create legend handles
#     # red_line = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Instantaneous (B0)')
#     # plt.legend(handles=[red_line], loc='upper right', fontsize=12)

#     try:
#         plt.savefig(filename, bbox_inches='tight')
#         print(f"Instantaneous causal graph saved to {filename}")
#     except Exception as e:
#         print(f"Error saving instantaneous graph: {e}")
#     plt.close()

# def plot_B1_causal_graph(B1, n_vars, filename="B1_causal_graph.png"):
#     """
#     Plots and saves an instantaneous causal graph based only on the B0 matrix.
#     Includes larger nodes, attempts shorter edges, and adds a legend.
#     """
#     G = nx.DiGraph()
#     nodes = list(range(n_vars))
#     G.add_nodes_from(nodes)

#     edge_list_b0 = []
#     for i in range(n_vars):
#         for j in range(n_vars):
#             if B1[i, j] != 0:
#                 edge_list_b0.append((j, i)) 
    
#     G.add_edges_from(edge_list_b0)

#     plt.figure(figsize=(7, 7))
    
#     if n_vars > 15:
#         print(f"Warning: Plotting instantaneous graph with {n_vars} nodes. "
#               "Layout might be slow and cluttered. Using spring_layout.")
#         pos = nx.spring_layout(G, k=0.8/np.sqrt(n_vars), iterations=50, seed=42) 
#         node_size = 300
#         font_size = 7
#         arrow_size = 12
#         edge_width = 1.0
#     else:
#         pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
#         node_size = 3000 # Larger nodes, increased from 1500
#         font_size = 35
#         arrow_size = 25
#         edge_width = 3.5
    
#     nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
#             font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
#             edge_color='blue', width=edge_width, connectionstyle='arc3,rad=0.2')
    
#     # plt.title(f"Lagged Causal Graph (B1 effects, n={n_vars})", fontsize=20)

#     # # Create legend handles
#     # red_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='None', markersize=10, label='Lagged (B1)')
#     # plt.legend(handles=[red_line], loc='upper right', fontsize=12)

#     try:
#         plt.savefig(filename, bbox_inches='tight')
#         print(f"B1 causal graph saved to {filename}")
#     except Exception as e:
#         print(f"Error saving B1 graph: {e}")
#     plt.close()

def plot_summary_causal_graph(B0, B1, n_vars, filename):
    """
    Plots and saves a summary causal graph from B0 (instantaneous) and B1 (lagged) matrices.
    Includes larger nodes, attempts shorter edges, and adds a legend.
    """
    G = nx.DiGraph()
    nodes = list(range(n_vars))
    G.add_nodes_from(nodes)

    edge_list_b0 = []
    edge_list_b1 = []

    # Add edges from B0 (instantaneous)
    for i in range(n_vars):
        for j in range(n_vars):
            if B0[i, j] != 0:
                edge_list_b0.append((j, i)) # Edge from j to i if B0[i,j] is effect of Xj on Xi

    # Add edges from B1 (lagged)
    for i in range(n_vars):
        for j in range(n_vars):
            if B1[i, j] != 0:
                edge_list_b1.append((j, i)) # Edge from j(t-1) to i(t) if B1[i,j] is effect of Xj(t-1) on Xi(t)
    
    all_edges = set()
    # Convert lists of tuples to sets of tuples for efficient checking
    set_edge_list_b0 = set(edge_list_b0)
    set_edge_list_b1 = set(edge_list_b1)

    # Add all unique edges to the graph
    for edge in set_edge_list_b0:
        all_edges.add(edge)
    for edge in set_edge_list_b1:
        all_edges.add(edge)
    
    G.add_edges_from(list(all_edges)) # Add unique edges to the graph

    # Determine edge colors
    edge_colors = []
    for u, v in G.edges(): # Iterate over edges in the graph G
        is_b0 = (u,v) in set_edge_list_b0
        is_b1 = (u,v) in set_edge_list_b1
        if is_b0 and is_b1:
            edge_colors.append('green') # Edge from both B0 and B1
        elif is_b0:
            edge_colors.append('red')    # Edge only from B0
        elif is_b1:
            edge_colors.append('blue')   # Edge only from B1
        else:
            # This case should not happen if all_edges are sourced from b0_list and b1_list
            edge_colors.append('black') # Fallback, though ideally not reached

    plt.figure(figsize=(7, 7)) # Slightly larger figure size
    
    # Layout parameters
    if n_vars > 15: # Adjusted threshold for spring_layout
        print(f"Warning: Plotting a summary graph with {n_vars} nodes. "
            "Layout might be slow and cluttered. Using spring_layout.")
        # For spring_layout, k controls the optimal distance between nodes. Smaller k = denser.
        pos = nx.spring_layout(G, k=0.6/np.sqrt(n_vars), iterations=50, seed=42) # k changed from 0.8 to 0.6
        node_size = 350  # Smaller nodes for many variables, increased from 300
        font_size = 7
        arrow_size = 12
        edge_width = 1.0
    else:
        if n_vars <= 5:
            k_val = 0.9  # Increase k for very few nodes to push them apart
        elif n_vars <= 10:
            k_val = 0.6
        else:
            k_val = 0.5
        
        iterations = 100 # More iterations for potentially better layout
        pos = nx.spring_layout(G, k=k_val, iterations=iterations, seed=42)
        node_size = 3000 # Larger nodes, increased from 1500
        font_size = 35
        arrow_size = 25
        edge_width = 3.5

    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color=edge_colors, width=edge_width, connectionstyle='arc3,rad=0.2') # Added connectionstyle for curved edges
    
    # plt.title(f"Summary Causal Graph (n={n_vars})", fontsize=25)

    ## Create legend handles
    # red_line = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Instantaneous (B0)')
    # blue_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='None', markersize=10, label='Lagged (B1)')
    # purple_line = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Both (B0 & B1)')
    
    # plt.legend(handles=[red_line, blue_line, purple_line], loc='upper right', fontsize=15)
    
    try:
        plt.savefig(filename, bbox_inches='tight') # bbox_inches to ensure legend is saved
        print(f"Summary causal graph saved to {filename}")
    except Exception as e:
        print(f"Error saving summary graph: {e}")
    plt.close()

def plot_instantaneous_causal_graph(B0, n_vars, filename):
    """
    Plots and saves an instantaneous causal graph based only on the B0 matrix.
    Includes larger nodes, attempts shorter edges, and adds a legend.
    """
    G = nx.DiGraph()
    nodes = list(range(n_vars))
    G.add_nodes_from(nodes)

    edge_list_b0 = []
    for i in range(n_vars):
        for j in range(n_vars):
            if B0[i, j] != 0:
                edge_list_b0.append((j, i)) 
    
    G.add_edges_from(edge_list_b0)

    plt.figure(figsize=(7, 7))
    
    if n_vars > 15:
        print(f"Warning: Plotting instantaneous graph with {n_vars} nodes. "
            "Layout might be slow and cluttered. Using spring_layout.")
        pos = nx.spring_layout(G, 0.8 / np.sqrt(max(n_vars,1)), iterations=50, seed=42) 
        node_size = 300
        font_size = 7
        arrow_size = 12
        edge_width = 1.0
    else:
        if n_vars <= 5:
            k_val = 0.9  # Increase k for very few nodes to push them apart
        elif n_vars <= 10:
            k_val = 0.6
        else:
            k_val = 0.5
        
        iterations = 100 # More iterations for potentially better layout
        pos = nx.spring_layout(G, k=k_val, iterations=iterations, seed=42)
        node_size = 3000 # Larger nodes, increased from 1500
        font_size = 35
        arrow_size = 25
        edge_width = 3.5
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color='red', width=edge_width, connectionstyle='arc3,rad=0.2')
    
    # plt.title(f"Instantaneous Causal Graph (B0 effects, n={n_vars})", fontsize=20)

    # # Create legend handles
    # red_line = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Instantaneous (B0)')
    # plt.legend(handles=[red_line], loc='upper right', fontsize=12)

    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Instantaneous causal graph saved to {filename}")
    except Exception as e:
        print(f"Error saving instantaneous graph: {e}")
    plt.close()

def plot_B1_causal_graph(B1, n_vars, filename="B1_causal_graph.png"): # Argument is B1
    """
    Plots and saves a causal graph based on the B1 (lagged effects) matrix.
    Attempts to improve node separation.
    """
    G = nx.DiGraph()
    nodes = list(range(n_vars))
    G.add_nodes_from(nodes)

    edge_list_from_B1 = [] # Renamed for clarity
    for i in range(n_vars): # Effect variable index
        for j in range(n_vars): # Cause variable index
            if B1[i, j] != 0:
                edge_list_from_B1.append((j, i)) # Edge from j (cause) to i (effect)
    
    G.add_edges_from(edge_list_from_B1)

    plt.figure(figsize=(7, 7)) # Slightly increased figure size for more space
    
    # Use spring_layout for all cases, but adjust parameters
    if n_vars > 15:
        print(f"Warning: Plotting B1 graph with {n_vars} nodes. "
            "Layout might be slow and cluttered.")
        # k scales the optimal distance between nodes. Smaller k for denser packing.
        # For many nodes, we might want smaller k to fit, but it can cause overlap.
        # Let's try to keep it from getting too small.
        k_val = 0.8 / np.sqrt(max(n_vars,1)) # Ensure n_vars is not zero
        iterations = 50
        pos = nx.spring_layout(G, k=k_val, iterations=iterations, seed=42) 
        node_size = 300
        font_size = 7
        arrow_size = 12
        edge_width = 1.0
    else: # n_vars <= 15 (includes your case of n_vars=5)
        # For fewer nodes, we can afford larger k to spread them out more
        if n_vars <= 5:
            k_val = 0.9  # Increase k for very few nodes to push them apart
        elif n_vars <= 10:
            k_val = 0.6
        else:
            k_val = 0.5
        
        iterations = 100 # More iterations for potentially better layout
        pos = nx.spring_layout(G, k=k_val, iterations=iterations, seed=42)
        # Alternatively, you could try other layouts for small N if spring doesn't satisfy:
        # pos = nx.circular_layout(G)
        # pos = nx.shell_layout(G)
        
        node_size = 3000 # Keep nodes reasonably large for small N
        font_size = 35   # Larger font for fewer nodes
        arrow_size = 25
        edge_width = 3.5
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color='blue', width=edge_width, connectionstyle='arc3,rad=0.1') # slightly less curve
    
    # plt.title(f"Lagged Causal Graph (B1 effects, n={n_vars})", fontsize=16)

    # Optional: Add a legend if needed
    # blue_line = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Lagged Effect (B1)')
    # plt.legend(handles=[blue_line], loc='best', fontsize=10)

    try:
        plt.savefig(filename, bbox_inches='tight', dpi=150) # Added dpi for better resolution
        print(f"B1 causal graph saved to {filename}")
    except Exception as e:
        print(f"Error saving B1 graph: {e}")
    plt.close()

# Main execution
N_VARIABLES = 100 # Kept at 100 for data generation as requested
TIME_SAMPLES = 10000 # Kept at 10000 for data generation

# Create data directory if it doesn't exist
import os
if not os.path.exists("../data"):
    os.makedirs("../data")

generated_X, B0_true, B1_true, C_order_true = generate_data(
    n=N_VARIABLES,
    T=TIME_SAMPLES,
    random_state=42,
    initial_data=None 
)

# Check for NaNs and Infs in the generated data
if np.isnan(generated_X).any():
    print("ERROR: Generated data contains NaNs.")
if np.isinf(generated_X).any():
    print("ERROR: Generated data contains Infs.")

# Save B0
# print("B0_true:\n", B0_true)
np.save("../data/varlingam/100/b0.npy", B0_true)
print("B0_true saved to ../data/varlingam/100/b0.npy")

# Save B1
# print("B1_true:\n", B1_true.shape)
np.save("../data/varlingam/100/b1.npy", B1_true)
print("B1_true saved to ../data/varlingam/100/b1.npy")

# Save Causal Order
print("Causal_order_true:", C_order_true)
np.save("../data/varlingam/100/causal_order.npy", C_order_true)
print("C_order_true saved to ../data/varlingam/100/causal_order.npy")


# Save data X to CSV
if generated_X.shape[0] > 0 and generated_X.shape[1] > 0 : # Check if data is not empty
    column_names = [f'X{i}' for i in range(N_VARIABLES)]
    df_X = pd.DataFrame(generated_X, columns=column_names)
    df_X.to_csv("../data/varlingam/100/data.csv", index=False)
    print(f"Generated data X saved to ../data/varlingam/100/data.csv (Shape: {df_X.shape})")
else:
    print("Generated data X is empty or invalid, not saving to CSV.")

combined_boolean_matrix = (B0_true != 0) | (B1_true != 0)

# Convert the boolean matrix (True/False) to an integer matrix (1/0)
summary_matrix = combined_boolean_matrix.astype(int)

print("\n--- Integrated Summary Matrix (non-zero values are 1) ---")
print(summary_matrix)


# --- Step 4: Save the summary matrix to a new .npy file ---
output_filename = '../data/varlingam/100/summary_matrix.npy'
np.save(output_filename, summary_matrix)


# Plot and save the summary causal graph
# For n=100, the plot will be very dense.
# plot_summary_causal_graph(B0_true, B1_true, N_VARIABLES, filename="../data/varlingam/100/summary_graph.png")
# plot_instantaneous_causal_graph(B0_true, N_VARIABLES, 
#                                 filename=os.path.join("../data/varlingam/100/B0_graph.png"))
# plot_B1_causal_graph(B1_true, N_VARIABLES, 
#                         filename=os.path.join("../data/varlingam/100/B1_graph.png"))
print("Script finished.")