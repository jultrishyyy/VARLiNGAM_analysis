import numpy as np
import pandas as pd
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import os

# Set up root directory and import helper functions
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from helper.helper_methods import convert_Btaus_to_summary_matrix, plot_summary_causal_graph

# Define the path to save the generated data
SAVE_PATH = os.path.join(ROOT_DIR, "data", "varlingam")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def generate_data_with_varlingam_assumption(n=5, T=1000, random_state=None, initial_data=None):
    """
    Generate synthetic time series data for VARLiNGAM.

    Parameters
    ----------
    n : int
        Number of variables.
    T : int
        Number of samples.
    random_state : int
        Seed for np.random.seed.
    initial_data : dict
        Dictionary of initial data containing B0, B1, and causal_order.

    Returns
    -------
    data : np.ndarray
        Generated data of shape (T, n).
    B0 : np.ndarray
        Instantaneous adjacency matrix.
    B1 : np.ndarray
        Lagged adjacency matrix.
    causal_order : np.ndarray
        Causal order of variables.
    """
    T_spurious = 20
    expon = 1.5
    
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
        
    M1 = np.dot(np.linalg.inv(np.eye(n) - B0), B1)

    ee = np.empty((n, T + T_spurious))
    for i in range(n):
        ee[i, :] = np.random.normal(size=(1, T + T_spurious))
        ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon)
        ee[i, :] = ee[i, :] - np.mean(ee[i, :])
        ee[i, :] = ee[i, :] / np.std(ee[i, :])

    std_e = np.random.uniform(size=(n,)) + 0.5
    nn = np.dot(np.dot(np.linalg.inv(np.eye(n) - B0), np.diag(std_e)), ee)

    xx = np.zeros((n, T + T_spurious))
    xx[:, 0] = np.random.normal(size=(n, ))

    for t in range(1, T + T_spurious):
        xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]

    data = xx[:, T_spurious + 1 : T_spurious + T]
    
    return data.T, B0, B1, causal_order

def generate_data_vialote_varlingam_assumption(n=5, T=1000, random_state=None, 
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


def validate_data(data):
    """
    Check for NaNs and Infs in the generated data.

    Parameters
    ----------
    data : np.ndarray
        Data to validate.

    Returns
    -------
    bool
        True if data is valid (no NaNs or Infs), False otherwise.
    """
    if np.isnan(data).any():
        print("ERROR: Generated data contains NaNs.")
        return False
    if np.isinf(data).any():
        print("ERROR: Generated data contains Infs.")
        return False
    return True

def save_causal_order(causal_order, save_path):
    """
    Save the causal order to a .npy file.

    Parameters
    ----------
    causal_order : np.ndarray
        Causal order to save.
    save_path : str
        Directory path to save the file.
    """
    print("Causal_order_true:", causal_order)
    np.save(os.path.join(save_path, "causal_order.npy"), causal_order)
    print(f"Causal_order_true saved to {os.path.join(save_path, 'causal_order.npy')}")

def save_data_to_csv(data, n_variables, save_path):
    """
    Save the generated data to a CSV file.

    Parameters
    ----------
    data : np.ndarray
        Data to save.
    n_variables : int
        Number of variables (for column naming).
    save_path : str
        Directory path to save the file.
    """
    if data.shape[0] > 0 and data.shape[1] > 0:
        column_names = [f'X{i}' for i in range(n_variables)]
        df = pd.DataFrame(data, columns=column_names)
        csv_path = os.path.join(save_path, "data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Generated data X saved to {csv_path} (Shape: {df.shape})")
    else:
        print("Generated data X is empty or invalid, not saving to CSV.")

def save_summary_matrix(summary_matrix, save_path):
    """
    Save summary_matrix to a .npy file.

    Parameters
    ----------
    summary_matrix : np.ndarray
        Summary adjacency matrix to save.
    save_path : str
        Directory path to save the file.
    """
    np.save(os.path.join(save_path, "summary_matrix.npy"), summary_matrix)
    print(f"Summary matrix saved to {os.path.join(save_path, 'summary_matrix.npy')}")

def plot_and_save_graph(summary_matrix, save_path):
    """
    Plot and save the summary causal graph.

    Parameters
    ----------
    summary_matrix : np.ndarray
        Summary adjacency matrix to plot.
    save_path : str
        Directory path to save the graph.
    """
    graph_path = os.path.join(save_path, "summary_graph.png")
    plot_summary_causal_graph(summary_matrix, filename=graph_path)

def main():
    """
    Main function to execute the VARLiNGAM data generation and processing pipeline.
    """
    # Constants
    N_VARIABLES = 10
    TIME_SAMPLES = 10000

    print(f"Data will be saved to: {SAVE_PATH}")

    # Generate data with VARLiNGAM assumptions
    generated_X, B0_true, B1_true, C_order_true = generate_data_with_varlingam_assumption(
        n=N_VARIABLES,
        T=TIME_SAMPLES,
        random_state=42,
    )

    # # Generate data vialoate VarLiNGAM assumption
    # generated_X, B0_true, B1_true, C_order_true = generate_data_vialote_varlingam_assumption(
    #     n=N_VARIABLES,
    #     T=TIME_SAMPLES,
    #     random_state=42,
    #     use_non_linearity= True,
    #     make_B0_cyclic=True,
    #     use_nonstationarity=True,
    # )

    # Validate data
    if not validate_data(generated_X):
        return

    # Save causal order
    save_causal_order(C_order_true, SAVE_PATH)

    # Save data to CSV
    save_data_to_csv(generated_X, N_VARIABLES, SAVE_PATH)

    summary_matrix = convert_Btaus_to_summary_matrix([B0_true, B1_true])
    print("\n--- Summary Matrix (non-zero values are 1) ---")
    print(summary_matrix)

    # Save summary matrix
    save_summary_matrix(summary_matrix, SAVE_PATH)

    # Plot and save causal graph
    plot_and_save_graph(summary_matrix, SAVE_PATH)

if __name__ == "__main__":
    main()