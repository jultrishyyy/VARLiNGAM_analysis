import numpy as np
import pandas as pd
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from helper.helper_methods import (
    plot_summary_causal_graph,
    prune_summary_matrix_with_best_f1_threshold,
    save_results_and_metrics,
)
from lingam import VARLiNGAM


DATA_PATH = os.path.join(ROOT_DIR, "data", "Storm_Ingestion_Activity")
OUTPUT_PATH = os.path.join(ROOT_DIR, "result", "Storm_Ingestion_Activity")
os.makedirs(OUTPUT_PATH, exist_ok=True)

data_filename = DATA_PATH + '/storm_data_normal.csv'
label_filename = DATA_PATH + '/summary_matrix.npy'
output_filename = OUTPUT_PATH + '/VARLiNGAM_result.txt'


    


if __name__ == "__main__":
    X = pd.read_csv(data_filename, delimiter=',', index_col=0, header=0)
    X = X.to_numpy()

    MAX_LAG = 10

    print(f"\nFitting VARLiNGAM model with max lag as {MAX_LAG}...")
    model = VARLiNGAM(lags=MAX_LAG, prune=False)
    model.fit(X)
    print("\nVAR model fitted successfully.")

    print("\nNumber of best lags:", model._lags)

    estimated_summary_matrix_continuous = np.max(np.abs(model._adjacency_matrices), axis=0)

    label_summary_matrix = np.load(label_filename)
    
    estimated_summary_matrix = prune_summary_matrix_with_best_f1_threshold(estimated_summary_matrix_continuous, label_summary_matrix)

    print("\nEstimated summary matrix:")
    print(estimated_summary_matrix)

    save_results_and_metrics(
        label_summary_matrix,
        estimated_summary_matrix,
        estimated_summary_matrix_continuous,
        order=model._causal_order,
        filename=output_filename
    )

    # (Optional) Plot and save the causal graph
    # plot_summary_causal_graph(estimated_summary_matrix, filename=OUTPUT_PATH + '/causal_graph.png')


    

