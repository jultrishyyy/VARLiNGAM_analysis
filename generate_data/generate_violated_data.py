import numpy as np
import pandas as pd
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def generate_data(n=5, T=1000, random_state=None, 
                  initial_B_matrices=None, stability_factor=0.98,
                  # Violation parameters
                  use_gaussian_noise=False, expon_for_non_gaussian=1.5,
                  make_B0_cyclic=False, num_cyclic_edges_to_add=1,
                  use_non_linearity=False, nonlin_type='tanh_on_lagged_effect', 
                  nonlin_scale=1.0,
                  use_nonstationarity=False, nonstat_type='std_e_drift', 
                  nonstat_params=None
                  ):
    """
    Generates time series data, potentially violating VarLiNGAM assumptions.

    Parameters for violations:
    --------------------------
    use_gaussian_noise : bool
        If True, innovations e(t) are Gaussian.
    expon_for_non_gaussian : float
        Exponent used to generate non-Gaussian noise if use_gaussian_noise is False.
    make_B0_cyclic : bool
        If True, B0 is modified to include cycles.
    num_cyclic_edges_to_add : int
        Number of bidirectional edges to add to B0 to induce cycles.
    use_non_linearity : bool
        If True, introduces non-linearity.
    nonlin_type : str
        'tanh_on_lagged_effect': applies tanh to M1*x(t-1) term.
        'tanh_on_sum': applies tanh to (M1*x(t-1) + n(t)) term.
    nonlin_scale : float
        Scaling factor for inputs to tanh to prevent immediate saturation.
    use_nonstationarity : bool
        If True, introduces non-stationarity.
    nonstat_type : str
        'std_e_drift': Noise variance drifts over time.
        'M1_switch': M1 matrix switches at T/2.
    nonstat_params : dict
        Parameters for non-stationarity, e.g.:
        For 'std_e_drift': {'amplitude': 0.5, 'period_factor': 4} (period = total_T / period_factor)
        For 'M1_switch': {'B1_alt': alternative B1 matrix for the switch}
    """

    if random_state is not None:
        np.random.seed(random_state)

    T_total_internal = T + 20 # internal total time including burn-in
    density_default = 0.4
    coeff_min_val_default = 0.05
    coeff_max_val_default = 0.4 # Reduced from 0.5 for better base stability

    # Initialize B0, B1, causal_order
    if initial_B_matrices is None:
        permutation = np.random.permutation(n)
        
        # Generate Acyclic B0 initially
        value = np.random.uniform(low=coeff_min_val_default, high=coeff_max_val_default, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B0_permuted_lt = np.multiply(value, sign)
        B0_permuted_lt = np.multiply(B0_permuted_lt, np.random.binomial(1, density_default, size=(n, n)))
        B0_permuted_lt = np.tril(B0_permuted_lt, k=-1)
        
        inv_permutation = np.argsort(permutation)
        B0 = B0_permuted_lt[inv_permutation][:, inv_permutation] # B0 in original var order

        causal_order_for_acyclic_B0 = np.empty(n, dtype=int)
        causal_order_for_acyclic_B0[permutation] = np.arange(n)

        if make_B0_cyclic and n >= 2:
            warnings.warn("Making B0 cyclic. The returned 'causal_order' refers to the base acyclic structure.")
            # Add edges to make B0 cyclic, trying not to make I-B0 singular
            for _ in range(num_cyclic_edges_to_add):
                # Pick two nodes not already connected in a 2-cycle by B0
                # This is a simple way, might not guarantee strong cycles or specific structures
                attempts = 0
                while attempts < n*n: # Limit attempts to avoid infinite loop
                    idx = np.random.choice(n, 2, replace=False)
                    # Add edge idx[0] -> idx[1] and idx[1] -> idx[0] if they don't exist
                    # or if they are not already forming this specific 2-cycle
                    can_add_forward = (B0[idx[1], idx[0]] == 0)
                    can_add_backward = (B0[idx[0], idx[1]] == 0)

                    if can_add_forward :
                         B0[idx[1], idx[0]] = np.random.uniform(coeff_min_val_default, coeff_max_val_default*0.5) * np.random.choice([-1, 1])
                    if can_add_backward: # Add the other direction for a 2-cycle
                         B0[idx[0], idx[1]] = np.random.uniform(coeff_min_val_default, coeff_max_val_default*0.5) * np.random.choice([-1, 1])
                    if can_add_forward or can_add_backward: # if at least one edge was added
                        break 
                    attempts +=1
            np.fill_diagonal(B0, 0) # Ensure diagonal is zero
            try:
                np.linalg.inv(np.eye(n) - B0) # Check invertibility
            except np.linalg.LinAlgError:
                warnings.warn("Cyclic B0 led to singular (I-B0). Reverting B0 to acyclic for this run.")
                # Revert to acyclic B0 if problem
                B0 = B0_permuted_lt[inv_permutation][:, inv_permutation]


        value = np.random.uniform(low=coeff_min_val_default, high=coeff_max_val_default, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B1_initial = np.multiply(value, sign)
        B1_initial = np.multiply(B1_initial, np.random.binomial(1, density_default, size=(n, n)))
    else:
        B0 = initial_B_matrices['B0'].copy() # Use copy to avoid modifying original
        B1_initial = initial_B_matrices['B1'].copy()
        causal_order_for_acyclic_B0 = initial_B_matrices['causal_order']
        n = B0.shape[0]
        # If B0 is made cyclic from initial_B_matrices, it would happen here too.
        # For simplicity, if initial_B_matrices provided, assume they are as desired (cyclic or not).

    # --- Prepare for time evolution ---
    try:
        inv_I_minus_B0 = np.linalg.inv(np.eye(n) - B0)
    except np.linalg.LinAlgError:
        raise ValueError("I-B0 is singular. Cannot proceed. Check B0 generation, especially if cyclic.")

    M1_original = np.dot(inv_I_minus_B0, B1_initial)
    
    # Stability check and adjustment for M1_original
    current_M1 = M1_original.copy()
    eigenvalues = np.linalg.eigvals(current_M1)
    max_abs_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"Initial Max absolute eigenvalue of M1: {max_abs_eigenvalue:.4f}")
    if stability_factor is not None and max_abs_eigenvalue >= 1.0:
        print(f"Scaling initial M1 (max_abs_eig = {max_abs_eigenvalue:.4f}) for stability.")
        current_M1 = current_M1 / (max_abs_eigenvalue / stability_factor)
        # B1_initial would be effectively changed if we maintain B0: B1_stable = (np.eye(n)-B0) @ current_M1

    M1_alt = None
    if use_nonstationarity and nonstat_type == 'M1_switch':
        if nonstat_params and 'B1_alt' in nonstat_params:
            B1_alt = nonstat_params['B1_alt']
            if B1_alt.shape != (n,n):
                raise ValueError("B1_alt shape mismatch for M1_switch.")
        else: # Generate a default B1_alt if not provided
            value_alt = np.random.uniform(low=coeff_min_val_default, high=coeff_max_val_default, size=(n, n))
            sign_alt = np.random.choice([-1, 1], size=(n, n))
            B1_alt = np.multiply(value_alt, sign_alt) * np.random.binomial(1, density_default, size=(n,n))
        
        M1_alt = np.dot(inv_I_minus_B0, B1_alt) # Assuming B0 is stationary for this switch type
        eigenvalues_alt = np.linalg.eigvals(M1_alt)
        max_abs_eigenvalue_alt = np.max(np.abs(eigenvalues_alt))
        print(f"Max absolute eigenvalue of M1_alt: {max_abs_eigenvalue_alt:.4f}")
        if stability_factor is not None and max_abs_eigenvalue_alt >= 1.0:
            print(f"Scaling M1_alt (max_abs_eig = {max_abs_eigenvalue_alt:.4f}) for stability.")
            M1_alt = M1_alt / (max_abs_eigenvalue_alt / stability_factor)
        switch_point = T_total_internal // 2


    # Generate base innovations ee
    ee = np.empty((n, T_total_internal))
    for i in range(n):
        if use_gaussian_noise:
            ee[i, :] = np.random.normal(size=T_total_internal)
        else:
            ee[i, :] = np.random.normal(size=T_total_internal)
            ee[i, :] = np.multiply(np.sign(ee[i, :]), np.abs(ee[i, :]) ** expon_for_non_gaussian)
        ee[i, :] = ee[i, :] - np.mean(ee[i, :])
        std_dev_ee_i = np.std(ee[i, :])
        if std_dev_ee_i > 1e-8: # Avoid division by zero/small number
             ee[i, :] = ee[i, :] / std_dev_ee_i
        # else: it's already zero or near zero, leave as is

    base_std_e = np.random.uniform(size=(n,)) + 0.5

    # --- Time Evolution Loop ---
    xx = np.zeros((n, T_total_internal))
    xx[:, 0] = np.random.normal(size=(n,)) # Initial condition

    for t in range(1, T_total_internal):
        # Determine current M1 for this time step
        M1_t = current_M1 # Start with the (potentially stabilized) original M1
        if use_nonstationarity:
            if nonstat_type == 'M1_switch' and M1_alt is not None and t >= switch_point:
                M1_t = M1_alt
            # Add other M1 non-stationarity types here if needed (e.g. drift)

        # Lagged component
        lagged_input = xx[:, t - 1]
        lagged_effect = np.dot(M1_t, lagged_input)

        if use_non_linearity:
            if nonlin_type == 'tanh_on_lagged_effect':
                lagged_effect = np.tanh(lagged_effect * nonlin_scale)
            # Add other non-linearity types for lagged part here

        # Innovation component n(t) = inv(I-B0) * D_std_e * e(t)
        current_std_e_diag = np.diag(base_std_e)
        if use_nonstationarity and nonstat_type == 'std_e_drift':
            if nonstat_params:
                amp = nonstat_params.get('amplitude', 0.5)
                period_fact = nonstat_params.get('period_factor', 4)
            else: # Default params for std_e_drift
                amp = 0.5
                period_fact = 4
            period = T_total_internal / period_fact
            drift_multiplier = 1.0 + amp * np.sin(2 * np.pi * t / period)
            current_std_e_diag = np.diag(base_std_e * drift_multiplier)
        
        nn_t = np.dot(inv_I_minus_B0, current_std_e_diag @ ee[:, t])

        # Combine and apply overall non-linearity if any
        combined_effect_pre_nl = lagged_effect + nn_t
        if use_non_linearity and nonlin_type == 'tanh_on_sum':
            xx[:, t] = np.tanh(combined_effect_pre_nl * nonlin_scale)
        else:
            xx[:, t] = combined_effect_pre_nl

        # Stability checks during generation
        if t % 500 == 0:
            if np.any(np.isinf(xx[:, t])) or np.any(np.isnan(xx[:, t])):
                warnings.warn(f"NaN or Inf generated at time step {t}. Process unstable. Stopping generation.")
                xx[:, t:] = np.nan # Mark rest as NaN
                break
            if np.max(np.abs(xx[:, t])) > 1e12:
                warnings.warn(f"Large values >1e12 detected at t={t}. Process might be unstable. Stopping generation.")
                xx[:, t:] = np.nan
                break
                
    # Prepare output data (removing burn-in and handling potential early stop)
    first_nan_idx = T_total_internal
    nan_mask = np.isnan(xx).any(axis=0)
    if np.any(nan_mask):
        first_nan_idx = np.where(nan_mask)[0][0]

    valid_T_generated = first_nan_idx
    
    if valid_T_generated <= 20: # Burn in period
        warnings.warn("Data unstable within burn-in. Returning empty/NaN array.")
        data_out = np.full((T, n), np.nan)
    else:
        data_for_output = xx[:, 20:valid_T_generated]
        if data_for_output.shape[1] < T -1 : # T-1 because original slice was T_spurious+1 : T_spurious+T
            warnings.warn(f"Generated data shorter ({data_for_output.shape[1]}) than requested T-1 ({T-1}). Padding with NaNs.")
            padding_len = (T-1) - data_for_output.shape[1]
            if padding_len > 0 :
                padding = np.full((n, padding_len), np.nan)
                data_out = np.concatenate((data_for_output, padding), axis=1).T
            else: # data_for_output.shape[1] is 0 or negative
                 data_out = np.full((T-1,n), np.nan)

        elif data_for_output.shape[1] >= T-1 :
             data_out = data_for_output[:, :(T-1)].T # Slice to T-1 (original) and transpose
        else: # Should not happen
             data_out = np.full((T-1,n), np.nan)


    # The returned B0, B1 are the initial ones. Non-stationarity might change effective B1 over time.
    return data_out, B0, B1_initial, causal_order_for_acyclic_B0




def plot_summary_causal_graph(B0, B1, n_vars, filename="summary_causal_graph.png"):
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
        pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
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

def plot_instantaneous_causal_graph(B0, n_vars, filename="instantaneous_causal_graph.png"):
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
        pos = nx.spring_layout(G, k=0.8/np.sqrt(n_vars), iterations=50, seed=42) 
        node_size = 300
        font_size = 7
        arrow_size = 12
        edge_width = 1.0
    else:
        pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
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

def plot_B1_causal_graph(B1, n_vars, filename="B1_causal_graph.png"):
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
            if B1[i, j] != 0:
                edge_list_b0.append((j, i)) 
    
    G.add_edges_from(edge_list_b0)

    plt.figure(figsize=(7, 7))
    
    if n_vars > 15:
        print(f"Warning: Plotting instantaneous graph with {n_vars} nodes. "
              "Layout might be slow and cluttered. Using spring_layout.")
        pos = nx.spring_layout(G, k=0.8/np.sqrt(n_vars), iterations=50, seed=42) 
        node_size = 300
        font_size = 7
        arrow_size = 12
        edge_width = 1.0
    else:
        pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
        node_size = 3000 # Larger nodes, increased from 1500
        font_size = 35
        arrow_size = 25
        edge_width = 3.5
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color='blue', width=edge_width, connectionstyle='arc3,rad=0.2')
    
    # plt.title(f"Lagged Causal Graph (B1 effects, n={n_vars})", fontsize=20)

    # # Create legend handles
    # red_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='None', markersize=10, label='Lagged (B1)')
    # plt.legend(handles=[red_line], loc='upper right', fontsize=12)

    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"B1 causal graph saved to {filename}")
    except Exception as e:
        print(f"Error saving B1 graph: {e}")
    plt.close()



# Main execution
N_VARIABLES = 5 # Kept at 100 for data generation as requested
TIME_SAMPLES = 10000 # Kept at 10000 for data generation

# Create data directory if it doesn't exist
import os
if not os.path.exists("../data"):
    os.makedirs("../data")

generated_X, B0_true, B1_true, C_order_true = generate_data(
    n=N_VARIABLES,
    T=TIME_SAMPLES,
)

# Check for NaNs and Infs in the generated data
if np.isnan(generated_X).any():
    print("ERROR: Generated data contains NaNs.")
if np.isinf(generated_X).any():
    print("ERROR: Generated data contains Infs.")

# Save B0
print("B0_true:\n", B0_true)
np.save("../data/violated/violated_b0.npy", B0_true)
print("B0_true saved to ../data/violated/violated_b0.npy")

# Save B1
print("B1_true:\n", B1_true.shape)
np.save("../data/violated/violated_b1.npy", B1_true)
print("B1_true saved to ../data/violated/violated_b1.npy")

# Save Causal Order
print("Causal_order_true:", C_order_true)
np.save("../data/violated/violated_causal_order.npy", C_order_true)
print("C_order_true saved to ../data/violated/violated_causal_order.npy")


# Save data X to CSV
if generated_X.shape[0] > 0 and generated_X.shape[1] > 0 : # Check if data is not empty
    column_names = [f'X{i}' for i in range(N_VARIABLES)]
    df_X = pd.DataFrame(generated_X, columns=column_names)
    df_X.to_csv("../data/violated/violated_data.csv", index=False)
    print(f"Generated data X saved to ../data/violated/violated_data.csv (Shape: {df_X.shape})")
else:
    print("Generated data X is empty or invalid, not saving to CSV.")


# Plot and save the summary causal graph
# For n=100, the plot will be very dense.
plot_summary_causal_graph(B0_true, B1_true, N_VARIABLES, filename="../data/violated/violated_summary_graph.png")
plot_instantaneous_causal_graph(B0_true, N_VARIABLES, 
                                filename=os.path.join("../data/violated/violated_B0_graph.png"))
plot_B1_causal_graph(B1_true, N_VARIABLES, 
                        filename=os.path.join("../data/violated/violated_B1_graph.png"))
print("Script finished.")