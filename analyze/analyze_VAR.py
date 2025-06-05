import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from helper.helper_methods import (
    plot_summary_causal_graph,
    prune_summary_matrix_with_best_f1_threshold,
    save_results_and_metrics,
    get_acf_ccf_ratio_over_lags
)


DATA_PATH = os.path.join(ROOT_DIR, "data", "Flood")
OUTPUT_PATH = os.path.join(ROOT_DIR, "result", "Flood")
os.makedirs(OUTPUT_PATH, exist_ok=True)

data_filename = DATA_PATH + '/rivers_ts_flood_preprocessed.csv'
label_filename = DATA_PATH + '/summary_matrix.npy'
output_filename = OUTPUT_PATH + '/VAR_result.txt'


def estimate_var_coefs(X, criterion='bic', lags=10):
    """Estimate coefficients of VAR"""
    # XXX: VAR.fit() is not searching lags correctly
    if criterion not in ["aic", "fpe", "hqic", "bic"]:
        var = VAR(X)
        result = var.fit(maxlags=lags, trend="n")
    else:
        min_value = float("Inf")
        result = None

        for lag in range(1, lags + 1):
            var = VAR(X)
            fitted = var.fit(maxlags=lag, ic=None, trend="n")

            value = getattr(fitted, criterion)
            if value < min_value:
                min_value = value
                result = fitted

    return result.coefs, result.k_ar, result.resid   
    


if __name__ == "__main__":
    X = pd.read_csv(data_filename, delimiter=',', index_col=0, header=0)
    X = X.to_numpy()

    MAX_LAG = 10

    VAR_removal_of_time_effect_info = []
    raw_acf_ratio, raw_ccf_ratio = get_acf_ccf_ratio_over_lags(X, lags_to_test=MAX_LAG)
    VAR_removal_of_time_effect_info.append("Raw ACF Ratio before VAR: " + str(raw_acf_ratio))
    VAR_removal_of_time_effect_info.append("Raw CCF Ratio before VAR: " + str(raw_ccf_ratio))

    print(f"\nFitting VAR model with max lag as {MAX_LAG}...")
    B_taus, best_lag, residuals = estimate_var_coefs(X, lags=MAX_LAG)
    print("\nVAR model fitted successfully.")

    var_acf_ratio, var_ccf_ratio = get_acf_ccf_ratio_over_lags(residuals, lags_to_test=MAX_LAG)
    VAR_removal_of_time_effect_info.append("ACF Ratio after VAR: " + str(var_acf_ratio))
    VAR_removal_of_time_effect_info.append("CCF Ratio after VAR: " + str(var_ccf_ratio))

    print("\nNumber of best lags:", best_lag)

    # Combine all lagged effects into a single matrix
    estimated_summary_matrix_continuous = np.max(np.abs(B_taus), axis=0)

    label_summary_matrix = np.load(label_filename)
    
    estimated_summary_matrix = prune_summary_matrix_with_best_f1_threshold(estimated_summary_matrix_continuous, label_summary_matrix)

    print("\nEstimated summary matrix:")
    print(estimated_summary_matrix)

    save_results_and_metrics(
        label_summary_matrix,
        estimated_summary_matrix,
        estimated_summary_matrix_continuous,
        best_lag,
        filename=output_filename,
        additional_info=VAR_removal_of_time_effect_info
    )

    # (Optional) Plot and save the causal graph
    # plot_summary_causal_graph(estimated_summary_matrix, filename=OUTPUT_PATH + '/causal_graph.png')


    

