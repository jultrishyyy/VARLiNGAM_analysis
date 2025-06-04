import numpy as np
import pandas as pd
import warnings

def generate_data_nonlinear(n=100, T=10000, random_state=None, initial_data=None, stability_factor=0.98, nonlinearity_scale=1.0):
    """
    Generates time series data with non-linear lagged effects.
    The contemporaneous effects remain linear, and exogenous shocks are non-Gaussian.

    Parameter
    ---------
    n : int
        number of variables
    T : int
        number of samples
    random_state : int
        seed for np.random.seed
    initial_data : dict
        dictionary of initial datas {'B0': ..., 'B1': ..., 'causal_order': ...}
    stability_factor : float
        Factor to scale M1 (linear part) if it's potentially unstable.
        Note: Stability for non-linear systems is much more complex. This is a heuristic.
    nonlinearity_scale : float
        Scaling factor applied before the tanh function for lagged effects.
        Smaller values make tanh operate more in its linear region.
    """
    if random_state is not None:
        np.random.seed(random_state)

    T_spurious = 200  # Increased burn-in for potentially slower convergence of non-linear systems
    expon = 1.5      # For non-Gaussian noise in e(t)

    if initial_data is None:
        coeff_max_val = 0.4 
        coeff_min_val = 0.05
        density = 0.4

        permutation = np.random.permutation(n)
        
        value = np.random.uniform(low=coeff_min_val, high=coeff_max_val, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B0 = np.multiply(value, sign)
        B0 = np.multiply(B0, np.random.binomial(1, density, size=(n, n))) 
        B0 = np.tril(B0, k=-1)
        B0 = B0[permutation][:, permutation]

        value = np.random.uniform(low=coeff_min_val, high=coeff_max_val, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B1_lag_coeffs = np.multiply(value, sign) # These will be M1 in linear case
        B1_lag_coeffs = np.multiply(B1_lag_coeffs, np.random.binomial(1, density, size=(n, n)))
        
        causal_order = np.empty(len(permutation), dtype=int)
        causal_order[permutation] = np.arange(len(permutation))
    else:
        B0 = initial_data['B0']
        B1_lag_coeffs = initial_data['B1_lag_coeffs'] # Expecting a matrix for lagged term coeffs
        causal_order = initial_data['causal_order'] 
        n = B0.shape[0]
            
    inv_I_minus_B0 = np.linalg.inv(np.eye(n) - B0)
    M1_linear_equivalent = B1_lag_coeffs # In this non-linear case, this matrix is used inside tanh

    # --- Stability Check (heuristic for the linear part M1_linear_equivalent) ---
    eigenvalues = np.linalg.eigvals(M1_linear_equivalent)
    max_abs_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"Max absolute eigenvalue of M1_linear_equivalent (used in non-linear term): {max_abs_eigenvalue:.4f}")

    if stability_factor is not None and max_abs_eigenvalue >= 1.0:
        print(f"Warning: M1_linear_equivalent is potentially unstable (max_abs_eig = {max_abs_eigenvalue:.4f}). Scaling for stability heuristic.")
        M1_linear_equivalent = M1_linear_equivalent / (max_abs_eigenvalue / stability_factor)

    # --- Non-Gaussian Exogenous Shocks e(t) ---
    ee = np.empty((n, T + T_spurious))
    for i in range(n):
        ee[i, :] = np.random.normal(size=(1, T + T_spurious))
        ee[i, :] = np.multiply(np.sign(ee[i, :]), np.abs(ee[i, :]) ** expon)
        ee[i, :] = ee[i, :] - np.mean(ee[i, :])
        std_val = np.std(ee[i, :])
        if std_val > 1e-10:
            ee[i, :] = ee[i, :] / std_val

    std_e = np.random.uniform(size=(n,)) + 0.5
    # Linear contemporaneous structure for n(t) based on e(t)
    nn = np.dot(inv_I_minus_B0, np.diag(std_e) @ ee)

    # --- Time Series Generation with Non-Linear Lagged Effect ---
    xx = np.zeros((n, T + T_spurious))
    initial_cond_std = 0.1 # Smaller initial conditions for non-linear systems
    xx[:, 0] = np.random.normal(loc=0.0, scale=initial_cond_std, size=(n,))

    for t in range(1, T + T_spurious):
        # Non-linear lagged term: tanh(M1_linear_equivalent * x_{t-1})
        # Element-wise tanh, M1_linear_equivalent acts as weights inside
        lagged_effect = np.tanh(np.dot(M1_linear_equivalent, xx[:, t - 1]) * nonlinearity_scale)
        xx[:, t] = lagged_effect + nn[:, t]
        
        if t % 1000 == 0: 
            if np.any(np.isinf(xx[:, t])) or np.any(np.isnan(xx[:, t])):
                raise ValueError(f"NaN or Inf generated at time step {t} in non-linear model. Process likely exploded.")
            max_val_at_t = np.max(np.abs(xx[:,t]))
            if max_val_at_t > 1e10: 
                warnings.warn(f"Warning: Very large values ({max_val_at_t:.2e}) detected at t={t} in non-linear model. Process might be exploding.")

    data = xx[:, T_spurious : T_spurious + T]
    
    if np.any(np.isnan(data)):
        warnings.warn("NaNs found in the final non-linear generated data!")
    if np.any(np.isinf(data)):
        warnings.warn("Infs found in the final non-linear generated data!")
            
    return data.T, B0, M1_linear_equivalent, causal_order # Returning M1_linear_equivalent as the lagged coefficient matrix

# Example usage:
generated_X_nonlinear, B0_nl, B1_coeffs_nl, C_order_nl = generate_data_nonlinear(
    n=100, T=1000, random_state=43, nonlinearity_scale=1.0
)
print("Non-linear data generated.")
print(generated_X_nonlinear.shape)

# Check for NaNs and Infs in the generated data
if np.isnan(generated_X_nonlinear).any():
    print("ERROR: Generated data contains NaNs.")
if np.isinf(generated_X_nonlinear).any():
    print("ERROR: Generated data contains Infs.")

print(B0_nl)
np.save("./data/b0_nonlinear.npy", B0_nl)

print(B1_coeffs_nl)
np.save("./data/b1_nonlinear.npy", B1_coeffs_nl)

column_names = [f'X{i}' for i in range(100)]
df_X = pd.DataFrame(generated_X_nonlinear, columns=column_names)
df_X.to_csv("./data/data_nonlinear.csv", index=False)