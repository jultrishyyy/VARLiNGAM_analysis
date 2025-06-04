import numpy as np
import pandas as pd
import warnings

def generate_data_gaussian_noise(n=100, T=10000, random_state=None, initial_data=None, stability_factor=0.98):
    """
    Generates time series data with Gaussian exogenous shocks e(t).
    The system dynamics (contemporaneous and lagged) are linear.

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
        Factor to scale M1 if it's unstable.
    """
    if random_state is not None:
        np.random.seed(random_state)

    T_spurious = 20  # Burn-in period
    # expon = 1.5 # This is removed to keep noise Gaussian

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
        B1 = np.multiply(value, sign)
        B1 = np.multiply(B1, np.random.binomial(1, density, size=(n, n)))
        
        causal_order = np.empty(len(permutation), dtype=int)
        causal_order[permutation] = np.arange(len(permutation))
    else:
        B0 = initial_data['B0']
        B1 = initial_data['B1']
        causal_order = initial_data['causal_order'] 
        n = B0.shape[0]
            
    inv_I_minus_B0 = np.linalg.inv(np.eye(n) - B0)
    M1 = np.dot(inv_I_minus_B0, B1)

    # --- Stability Check and Adjustment for M1 ---
    eigenvalues = np.linalg.eigvals(M1)
    max_abs_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"Max absolute eigenvalue of M1: {max_abs_eigenvalue:.4f}")

    if stability_factor is not None and max_abs_eigenvalue >= 1.0:
        print(f"Warning: M1 is unstable (max_abs_eig = {max_abs_eigenvalue:.4f}). Scaling M1 for stability.")
        M1 = M1 / (max_abs_eigenvalue / stability_factor)

    # --- Gaussian Exogenous Shocks e(t) ---
    ee = np.empty((n, T + T_spurious))
    for i in range(n):
        ee[i, :] = np.random.normal(size=(1, T + T_spurious)) # Kept as Gaussian
        # No non-Gaussian transformation
        ee[i, :] = ee[i, :] - np.mean(ee[i, :]) # Center
        std_val = np.std(ee[i, :])              # Normalize variance
        if std_val > 1e-10:
            ee[i, :] = ee[i, :] / std_val

    std_e = np.random.uniform(size=(n,)) + 0.5 # Random scaling for each error component
    # nn are the VAR residuals/innovations. Here they are driven by Gaussian e(t).
    nn = np.dot(inv_I_minus_B0, np.diag(std_e) @ ee)


    xx = np.zeros((n, T + T_spurious))
    initial_cond_std = 1.0 
    xx[:, 0] = np.random.normal(loc=0.0, scale=initial_cond_std, size=(n,))

    for t in range(1, T + T_spurious):
        xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]
        if t % 1000 == 0: 
            if np.any(np.isinf(xx[:, t])) or np.any(np.isnan(xx[:, t])):
                raise ValueError(f"NaN or Inf generated at time step {t}. Process likely exploded.")
            max_val_at_t = np.max(np.abs(xx[:,t]))
            if max_val_at_t > 1e12: 
                warnings.warn(f"Warning: Very large values ({max_val_at_t:.2e}) detected at t={t}. Process might be exploding.")

    data = xx[:, T_spurious : T_spurious + T]
    
    if np.any(np.isnan(data)):
        warnings.warn("NaNs found in the final Gaussian noise generated data!")
    if np.any(np.isinf(data)):
        warnings.warn("Infs found in the final Gaussian noise generated data!")
            
    return data.T, B0, B1, causal_order

# Example usage:
generated_X_gaussian, B0_g, B1_g, C_order_g = generate_data_gaussian_noise(
    n=100, T=1000, random_state=44
)
print("Gaussian noise data generated.")
print(generated_X_gaussian.shape)

# Check for NaNs and Infs in the generated data
if np.isnan(generated_X_gaussian).any():
    print("ERROR: Generated data contains NaNs.")
if np.isinf(generated_X_gaussian).any():
    print("ERROR: Generated data contains Infs.")

print(B0_g)
np.save("./data/b0_gaussian.npy", B0_g)

print(B1_g)
np.save("./data/b1_gaussian.npy", B1_g)

column_names = [f'X{i}' for i in range(100)]
df_X = pd.DataFrame(generated_X_gaussian, columns=column_names)
df_X.to_csv("./data/data_gaussian.csv", index=False)