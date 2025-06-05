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
from lingam import DirectLiNGAM


DATA_PATH = os.path.join(ROOT_DIR, "data", "Flood")
OUTPUT_PATH = os.path.join(ROOT_DIR, "result", "Flood")
os.makedirs(OUTPUT_PATH, exist_ok=True)

data_filename = DATA_PATH + '/rivers_ts_flood_preprocessed.csv'
label_filename = DATA_PATH + '/summary_matrix.npy'
output_filename = OUTPUT_PATH + '/DirectLiNGAM_result.txt'


    


if __name__ == "__main__":
    X = pd.read_csv(data_filename, delimiter=',', index_col=0, header=0)
    X = X.to_numpy()

    model = DirectLiNGAM()
    model.fit(X)

    B0 = np.abs(model._adjacency_matrix)

    label_summary_matrix = np.load(label_filename)
    
    estimated_summary_matrix = prune_summary_matrix_with_best_f1_threshold(B0, label_summary_matrix)

    print("\nEstimated summary matrix:")
    print(estimated_summary_matrix)

    save_results_and_metrics(
        label_summary_matrix,
        estimated_summary_matrix,
        B0,
        order=model._causal_order,
        filename=output_filename
    )

    # (Optional) Plot and save the causal graph
    # plot_summary_causal_graph(estimated_summary_matrix, filename=OUTPUT_PATH + '/causal_graph.png')


    

