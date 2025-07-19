
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import precision_recall_curve
from statsmodels.tsa.stattools import ccf, acf
from itertools import combinations

def convert_Btaus_to_summary_matrix(B_taus):
    """
    Convert a list of Btau matrices (2D or 3D) to a 2D summary matrix.
    
    Args:
        B_taus (list): List of numpy arrays (2D or 3D matrices) to combine.
        
    Returns:
        np.ndarray: 2D summary matrix where non-zero elements across all input matrices are 1, others are 0.
        
    Raises:
        ValueError: If B_taus is empty, matrices have incompatible shapes, or input arrays have invalid dimensions.
    """
    if B_taus is None or len(B_taus) == 0:
        raise ValueError("Input list of matrices (B_taus) is empty.")
    
    # Validate and convert matrices to 3D
    B_taus_list = []

    B_taus = np.asarray(B_taus)  # Ensure input is a NumPy array
    print(f"Processing matrix with shape {B_taus.shape}")
    if B_taus.ndim == 2:
        # Convert 2D matrix to 3D with singleton first dimension
        B_taus = B_taus[np.newaxis, :, :]
    elif B_taus.ndim != 3:
        raise ValueError(f"Input matrix must be 2D or 3D, got {B_taus.ndim}D")
    
    for Btau in B_taus:
        B_taus_list.append(Btau)
    # print(f"Processe {len(B_taus_list)} matrices with shape {B_taus_list[0].shape}")
    
    # Combine non-zero elements across all matrices
    combined_boolean_matrix = None
    for Btau in B_taus_list:
        # Create boolean mask for non-zero elements across all slices
        matrix_nonzero = np.where(Btau != 0, 1, 0)  # Convert to boolean mask (1 for non-zero, 0 for zero)
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
    if B_taus.ndim == 2:
        # Convert 2D matrix to 3D with singleton first dimension
        B_taus = B_taus[np.newaxis, :, :]
    elif B_taus.ndim != 3:
        raise ValueError(f"Input matrix must be 2D or 3D, got {B_taus.ndim}D")
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
            k_val = 8
        elif n_vars <= 10:
            k_val = 6
        else:
            k_val = 4
        pos = nx.spring_layout(G, k=k_val, iterations=100, seed=42)
        node_size = 3000
        font_size = 35
        arrow_size = 25
        edge_width = 3.5
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue",
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color=edge_colors, width=edge_width, connectionstyle='arc3,rad=0.2')
    
    plt.title(f"Summary Causal Graph (n={n_vars})", fontsize=25)

    # # Create legend handles
    # red_line = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Instantaneous (B0)')
    # blue_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='None', markersize=10, label='Lagged (B1)')
    # purple_line = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Both (B0 & B1)')
    # plt.legend()
    
    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"\nSummary causal graph saved to {filename}")
    except Exception as e:
        print(f"Error saving summary graph: {e}")
    plt.close()

def get_acf_ccf_ratio_over_lags(data, lags_to_test=None, significance_level=0.05):
    """
    Provides a numerical summary of VAR residuals to check for remaining
    temporal structure, focusing on ACF and CCF.

    Args:
        lags_to_test (int, optional): The number of lags to include in the analysis.
                                    If None, it defaults to the number of lags from the VAR model.
        significance_level (float, optional): The significance level for confidence intervals.
                                            Defaults to 0.05 for 95% confidence.
    """
    if data is None:
        print("Model has not been fit yet. Please run .fit(X) first.")
        return

    residuals = data
    n_obs, n_dims = residuals.shape

    

    print("--- Numerical Summary of VAR Residuals ---")

    # --- 1. Autocorrelation Function (ACF) Summary ---
    print(f"\n[1/2] Analyzing Autocorrelation (ACF) up to {lags_to_test} lags...")

    # Calculate the fixed confidence interval boundary for white noise
    # This is a common approximation for ACF plots of residuals
    critical_value = 1.96  # For 95% confidence
    fixed_conf_bound = critical_value / np.sqrt(n_obs)

    acf_ratios = []
    for i in range(n_dims):
        # Calculate ACF values
        acf_values = acf(residuals[:, i], nlags=lags_to_test, fft=True) # No alpha needed here

        # The fixed confidence interval lower/upper bounds
        lower_bound_fixed = -fixed_conf_bound
        upper_bound_fixed = fixed_conf_bound

        # Count significant spikes (ignoring lag 0, which is always 1)
        significant_spikes = 0
        for k in range(1, lags_to_test + 1):
            if acf_values[k] < lower_bound_fixed or acf_values[k] > upper_bound_fixed:
                significant_spikes += 1
                # print(f"  Significant spike at Lag {k} for series {i}: ACF = {acf_values[k]:.4f}, Bounds = [{lower_bound_fixed:.4f}, {upper_bound_fixed:.4f}]")

        ratio = significant_spikes / lags_to_test
        acf_ratios.append(ratio)


    # Calculate and print the mean value as requested, with a warning
    mean_acf_ratio = np.mean(acf_ratios)
    print(f"\nMean ACF Ratio of Significant Spikes: {mean_acf_ratio:.4f}")
    print("  (Warning: A low mean can hide severe autocorrelation in a single series.)")


            # --- 2. Cross-Correlation Function (CCF) Summary ---
    print(f"\n[2/2] Analyzing Cross-Correlation (CCF) up to {lags_to_test} lags...")
    ccf_ratios = []
    # Confidence interval is constant for CCF
    conf_interval_boundary = 1.96 / np.sqrt(n_obs) # For 95% confidence

    num_pairs = 0
    for i, j in combinations(range(n_dims), 2):
        num_pairs += 1
        # Calculate CCF. We test lags from 1 to lags_to_test.
        ccf_values = ccf(residuals[:, i], residuals[:, j], adjusted=False)[:lags_to_test + 1]

        # Count significant spikes (ignoring lag 0)
        significant_spikes = 0
        for k in range(1, lags_to_test + 1):
            if abs(ccf_values[k]) > conf_interval_boundary:
                significant_spikes += 1
        
        ratio = significant_spikes / lags_to_test
        ccf_ratios.append(ratio)

    # Calculate and print the mean value
    mean_ccf_ratio = np.mean(ccf_ratios)
    print(f"\nMean CCF Ratio of Significant Spikes: {mean_ccf_ratio:.4f} (averaged over {num_pairs} pairs)")
    print("  (Warning: This average can obscure strong cross-correlations between specific pairs.)")

    return mean_acf_ratio, mean_ccf_ratio


def get_best_f1_thresholod(labs, preds):
        # F1 MAX
        precision, recall, thresholds = precision_recall_curve(labs, preds)

        denominator = recall + precision
        numerator = 2 * recall * precision

        # Initialize f1_scores as a NumPy array of zeros with the same shape and a float data type.
        # This ensures that if the denominator is zero, the F1 score defaults to 0.0.
        f1_scores = np.zeros_like(denominator, dtype=float)

        # Use np.divide to perform element-wise division.
        # The 'out' argument specifies where to store the result.
        # The 'where' argument ensures division only happens where the denominator is not zero.
        np.divide(numerator, denominator, out=f1_scores, where=(denominator != 0))
        
        best_idx = np.argmax(f1_scores)
        f1_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        return f1_thresh

def prune_summary_matrix_with_best_f1_threshold(summary_matrix, label):
    # Prune summary_matrixs using the best F1 threshold and return a binary summary matrix.
    # you can also use fixed threshold or other methods to prune summary_matrix, just modify this function.
    
    summary_matrix = np.asarray(summary_matrix)

    y_true_flat = label.flatten()
    y_pred_flat = summary_matrix.flatten()
    best_f1_thresh = get_best_f1_thresholod(y_true_flat, y_pred_flat)
    summary_matrix_pruned = (summary_matrix > best_f1_thresh).astype(int) # Convert to binary matrix

    print(f"\nPrune summary matrix with best-F1 threshold: {best_f1_thresh}")

    return summary_matrix_pruned

def save_results_and_metrics(label_summary_matrix, estimated_summary_matrix, estimated_summary_matrix_continuous, lags=None, order=None, filename="results.txt", additional_info=None):
        with open(filename, 'w') as f:
            # --- Write scalar values ---
            if additional_info is not None:
                f.write("--- ADDITIONAL INFO ---\n\n")
                for info in additional_info:
                    f.write(f"{info}\n")


            f.write("\n\n--- METRICS ---\n\n")

            if lags is not None:
                f.write(f"Best Lags: {lags}\n")
            

            num_edges_ground_truth = np.sum(label_summary_matrix)
            f.write(f"Number of edges in ground truth (label summary matrix): {num_edges_ground_truth}\n")

            # True Positives: in both labels and pruned prediction
            true_positives_matrix = (label_summary_matrix == 1) & (estimated_summary_matrix == 1)
            num_correctly_predicted = np.sum(true_positives_matrix)
            f.write(f"Number of correctly predicted edges (True Positives): {num_correctly_predicted}\n")

            # True Negatives: not in both labels and pruned prediction
            true_negatives_matrix = (label_summary_matrix == 0) & (estimated_summary_matrix == 0)
            num_correct_nonedges = np.sum(true_negatives_matrix)
            f.write(f"Number of correct non-edges (True Negatives): {num_correct_nonedges}\n")

            # False Positives: in pruned prediction but not in labels
            false_positives_matrix = (label_summary_matrix == 0) & (estimated_summary_matrix == 1)
            num_incorrectly_predicted = np.sum(false_positives_matrix)
            f.write(f"Number of incorrectly predicted edges (False Positives): {num_incorrectly_predicted}\n")

            # False Negatives: in labels but not in pruned prediction
            false_negatives_matrix = (label_summary_matrix == 1) & (estimated_summary_matrix == 0)
            num_missed_edges = np.sum(false_negatives_matrix)
            f.write(f"Number of correct edges not predicted (False Negatives): {num_missed_edges}\n")

            tp = num_correctly_predicted
            fp = num_incorrectly_predicted
            fn = num_missed_edges

            # Calculate Precision
            if (tp + fp) == 0:
                precision = 0.0  # Avoid division by zero if no positive predictions were made
            else:
                precision = tp / (tp + fp)

            # Calculate Recall
            if (tp + fn) == 0:
                recall = 0.0  # Avoid division by zero if no actual positives exist (or none were predicted)
            else:
                recall = tp / (tp + fn)

            # Calculate F1 Score
            if (precision + recall) == 0:
                f1_score = 0.0  # Avoid division by zero if both precision and recall are zero
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1_score:.4f}\n")

            
            if order is not None:
                f.write("\n--- ORDER ANALYSIS---\n\n")
                f.write(f"Predicted Causal Order: {order}\n")

                num_wrongly_ordered_edges = 0
                wrong_pairs = []
                
                if num_correctly_predicted > 0:
                    # Get indices of true positive edges
                    # np.where returns a tuple of arrays, one for each dimension
                    tp_effect_indices, tp_cause_indices = np.where(label_summary_matrix == 1)
                    # print(f"True Positive Effect Indices: {tp_effect_indices}")
                    # print(f"True Positive Cause Indices: {tp_cause_indices}")
                    
                    for i in range(len(tp_effect_indices)):
                        effect_idx = tp_effect_indices[i]
                        cause_idx = tp_cause_indices[i]
                        
                        # Check order: if cause's order is >= effect's order, it's wrong
                        if order.index(cause_idx) > order.index(effect_idx):
                            num_wrongly_ordered_edges += 1
                            wrong_pairs.append(f"    - Wrongly ordered: {cause_idx} -> {effect_idx} \n")
                           
                f.write(f"Number of wrongly ordered cause-effect pairs: {num_wrongly_ordered_edges}\n")

                print_num = min(len(wrong_pairs), 5)
                f.write(f"{print_num} Wrongly ordered pairs in ground truth causal matrix:\n")
                if num_wrongly_ordered_edges > 0:
                    for pair in wrong_pairs[:print_num]:
                        f.write(pair)

            f.write("\n\n" + "="*50 + "\n\n")
            
            # --- Write the summary_matrix_continuous ---
            f.write("--- ESTIMATED SUMMARY MATRIX ---\n")
            f.write("(Represents the estimated causal effect across all lags after pruning)\n\n")
            np.savetxt(f, estimated_summary_matrix, delimiter=',', fmt='%d')

            
            f.write("\n\n\n" + "="*50 + "\n\n")
            
            # --- Write the summary_matrix_continuous ---
            f.write("--- CONTINUOUS SUMMARY MATRIX ---\n")
            f.write("(Represents the max estimated causal effect across all lags before pruning)\n\n")
            
            # 6. Use numpy.savetxt to write the array to the file handle 'f'
            # 'fmt' controls the number format to keep it clean.
            np.savetxt(f, estimated_summary_matrix_continuous, fmt='%.6f', delimiter=',')

        print(f"All results have been successfully saved to '{filename}'")     

