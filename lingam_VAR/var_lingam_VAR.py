"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""
import itertools
import warnings
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample
from statsmodels.tsa.vector_ar.var_model import VAR
# from statsmodels.tsa.api import VAR

import pandas as pd
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import combinations
from collections import deque
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    roc_curve
)

import sys
import os

# --- Sys.path modification (ensure your lingam_VAR is found) ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    package_parent_dir = os.path.dirname(current_script_dir)
    if package_parent_dir not in sys.path:
        sys.path.insert(0, package_parent_dir)

    from lingam_VAR.base import _BaseLiNGAM
    from lingam_VAR.bootstrap import BootstrapResult # For VARBootstrapResult
    from lingam_VAR.direct_lingam import DirectLiNGAM
    from lingam_VAR.hsic import hsic_test_gamma
    from lingam_VAR.utils import predict_adaptive_lasso, find_all_paths, calculate_total_effect
except ImportError as e:
    print(f"Warning: Could not import some LiNGAM components: {e}. Check sys.path and lingam_VAR.")
# --- End Sys.path modification ---



class VARLiNGAM:
    """Implementation of VAR-LiNGAM Algorithm [1]_

    References
    ----------
    .. [1] Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, Patrik O. Hoyer.
       Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity.
       Journal of Machine Learning Research, 11: 1709-1731, 2010.
    """


    def __init__(self, lags=1, criterion='bic', prune=False, pruning_threshold=0.05, ar_coefs=None, lingam_model=None, random_state=None): # Added pruning_threshold
        """Construct a VARLiNGAM model.

        Parameters
        ----------
        lags : int, optional (default=1)
            Number of lags.
        criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
            Criterion to decide the best lags within ``lags``.
            Searching the best lags is disabled if ``criterion`` is ``None``.
        prune : boolean, optional (default=False)
            Whether to prune the adjacency matrix or not. If True, uses threshold-based pruning.
        pruning_threshold : float, optional (default=0.01)
            The threshold for pruning. Coefficients with absolute values less than
            this threshold will be set to zero. Used only if prune=True.
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR model. Estimating AR model is skipped if specified ``ar_coefs``.
            Shape must be (``lags``, n_features, n_features).
        lingam_model : lingam object inherits 'lingam._BaseLiNGAM', optional (default=None)
            LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._lags = lags
        self._criterion = criterion
        self._prune = prune
        self._pruning_threshold = pruning_threshold # Store the threshold
        self._ar_coefs = check_array(ar_coefs, allow_nd=True) if ar_coefs is not None else None
        self._lingam_model = lingam_model
        self._random_state = random_state


    def fit_VAReffect(self, X):



        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)

        # n_vars = X.shape[-1]
        # print(f"Fitting VARLiNGAM model with {n_vars} variables and {self._lags} lags.")

        # res = VAR(X).fit(self._lags)
        # pred = res.params[1:]
        # pred = np.abs(pred)
        # pred = np.stack(
        #     [pred[:, x].reshape(self._lags, n_vars).T for x in range(pred.shape[1])]
        # )
        # print("Initial VAR coefficients (absolute values):")
        # print(pred)
        # out = self.summary_transform(pred, "max")
        # print("Initial VAR coefficients (max):")
        # print(out)
        # out = self.make_human_readable(out, X)
        # print("Human-readable VAR coefficients:")
        # print(out)

        M_taus = self._ar_coefs
        if M_taus is None:
            M_taus, lags, residuals = self._estimate_var_coefs(X)
        else:
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)
 
        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals

        M_taus = np.abs(M_taus)  # Ensure M_taus is absolute values


        print("M_taus shape:", M_taus.shape)
        print("lags:", lags)


    def fit(self, X):



        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)

        # n_vars = X.shape[-1]
        # print(f"Fitting VARLiNGAM model with {n_vars} variables and {self._lags} lags.")

        # res = VAR(X).fit(self._lags)
        # pred = res.params[1:]
        # pred = np.abs(pred)
        # pred = np.stack(
        #     [pred[:, x].reshape(self._lags, n_vars).T for x in range(pred.shape[1])]
        # )
        # print("Initial VAR coefficients (absolute values):")
        # print(pred)
        # out = self.summary_transform(pred, "max")
        # print("Initial VAR coefficients (max):")
        # print(out)
        # out = self.make_human_readable(out, X)
        # print("Human-readable VAR coefficients:")
        # print(out)

        M_taus = self._ar_coefs
        if M_taus is None:
            M_taus, lags, residuals = self._estimate_var_coefs(X)
        else:
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)
 
        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals

        M_taus = np.abs(M_taus)  # Ensure M_taus is absolute values


        print("M_taus shape:", M_taus.shape)
        print("lags:", lags)

        label_path = "../causalriver/"
        save_path = "../data/VAR/east/"

        labels = np.load(label_path + "east_germany_matrix.npy")
        order_path = label_path + "east_germany_order.npy"

        summary_matrix_continuous = np.max(np.abs(M_taus), axis=0)

        # print("Shape of M_taus:", M_taus.shape)
        print("Summary matrix:", summary_matrix_continuous)

        # We'll use the f1_max function to find the best threshold for F1 score.
        # Flatten both the summary matrix and the labels for the metric functions.
        y_true_flat = labels.flatten()
        y_pred_flat = summary_matrix_continuous.flatten()

        # Find the best threshold that maximizes the F1 score
        best_f1_thresh, best_f1_score = self.f1_max(y_true_flat, y_pred_flat)
        print(f"Best Threshold (from F1-Max): {best_f1_thresh:.4f}")
        print(f"Best F1 Score possible: {best_f1_score:.4f}")
        # Find the best threshold that maximizes the accuracy
        best_acc_thresh, best_acc_score = self.max_accuracy(y_true_flat, y_pred_flat)
        print(f"Best Threshold (from Accuracy-Max): {best_acc_thresh:.4f}")
        print(f"Best Accuracy possible: {best_acc_score:.4f}")


        print("\n--- Step 3: Prune the summary matrix and calculate final metrics ---")
        # Prune the continuous summary matrix to get a binary adjacency matrix
        summary_matrix_pruned = (summary_matrix_continuous >= best_f1_thresh).astype(int)
        print("Pruned Summary Matrix (binary):")
        print(summary_matrix_pruned)

        # Now, calculate the final metrics for this pruned binary matrix
        final_f1_score = self.f1_max(y_true_flat, summary_matrix_pruned.flatten())[1]
        final_accuracy = self.max_accuracy(y_true_flat, summary_matrix_pruned.flatten())[1]
        final_auroc = roc_auc_score(y_true_flat, summary_matrix_pruned.flatten())
        print(f"Final F1 Score after pruning: {final_f1_score:.4f}")
        print(f"Final Accuracy after pruning: {final_accuracy:.4f}")
        print(f"Final AUROC after pruning: {final_auroc:.4f}")


 

        # 2. Recover "causal order" from the pruned matrix
        # This will only work if the graph from pruned_M1 is a DAG.
        lagged_causal_order = self.get_causal_order_from_lagged_matrix(summary_matrix_pruned)

        print(f"\nCausal order from pruned M1_hat: {lagged_causal_order}")

        np.save(save_path+"matrix.npy", summary_matrix_pruned)
        np.save(save_path+"causal_order.npy", lagged_causal_order)

        output_filename = save_path + 'info.txt'

        # Use a 'with' block to safely open and write to the file
        with open(output_filename, 'w') as f:
            # --- Write scalar values ---
            f.write("--- METRICS AND THRESHOLDS ---\n\n")

            f.write(f"Best Lags: {lags}\n")
            
            # 1. Best Threshold (from F1-Max)
            f.write(f"Best Threshold (from F1-Max): {best_f1_thresh:.4f}\n")
            
            # 2. Best Threshold (from Accuracy-Max)
            f.write(f"Best Threshold (from Accuracy-Max): {best_acc_thresh:.4f}\n")
            
            f.write("\n--- FINAL SCORES (after pruning with F1-Max threshold) ---\n\n")

            # 3. Final F1 Score after pruning
            f.write(f"Final F1 Score after pruning: {final_f1_score:.4f}\n")
            
            # 4. Final Accuracy after pruning
            f.write(f"Final Accuracy after pruning: {final_accuracy:.4f}\n")
            
            # 5. Final AUROC after pruning
            f.write(f"Final AUROC after pruning: {final_auroc:.4f}\n")

            f.write("--- EDGE ANALYSIS ---\n\n")

            num_edges_ground_truth = np.sum(labels)
            f.write(f"Number of edges in ground truth (labels): {num_edges_ground_truth}\n")

            # True Positives: in both labels and pruned prediction
            true_positives_matrix = (labels == 1) & (summary_matrix_pruned == 1)
            num_correctly_predicted = np.sum(true_positives_matrix)
            f.write(f"Number of correctly predicted edges (True Positives): {num_correctly_predicted}\n")

            # True Negatives: not in both labels and pruned prediction
            true_negatives_matrix = (labels == 0) & (summary_matrix_pruned == 0)
            num_correct_nonedges = np.sum(true_negatives_matrix)
            f.write(f"Number of correct non-edges (True Negatives): {num_correct_nonedges}\n")

            # False Positives: in pruned prediction but not in labels
            false_positives_matrix = (labels == 0) & (summary_matrix_pruned == 1)
            num_incorrectly_predicted = np.sum(false_positives_matrix)
            f.write(f"Number of incorrectly predicted edges (False Positives): {num_incorrectly_predicted}\n")

            # False Negatives: in labels but not in pruned prediction
            false_negatives_matrix = (labels == 1) & (summary_matrix_pruned == 0)
            num_missed_edges = np.sum(false_negatives_matrix)
            f.write(f"Number of correct edges not predicted (False Negatives): {num_missed_edges}\n")
            

            f.write("\n--- ORDER ANALYSIS---\n\n")
            lagged_causal_order = self.get_causal_order_from_lagged_matrix(summary_matrix_pruned)
            f.write(f"Predicted Causal Order: {lagged_causal_order}\n")
            f.write(f"True Causal Order: {list(np.load(order_path))}\n")

            num_wrongly_ordered_edges = 0
            
            if num_correctly_predicted > 0:
                # Get indices of true positive edges
                # np.where returns a tuple of arrays, one for each dimension
                tp_effect_indices, tp_cause_indices = np.where(labels == 1)
                print(f"True Positive Effect Indices: {tp_effect_indices}")
                print(f"True Positive Cause Indices: {tp_cause_indices}")
                
                for i in range(len(tp_effect_indices)):
                    effect_idx = tp_effect_indices[i]
                    cause_idx = tp_cause_indices[i]
                    
                    # Check order: if cause's order is >= effect's order, it's wrong
                    if lagged_causal_order[cause_idx] >= lagged_causal_order[effect_idx]:
                        num_wrongly_ordered_edges += 1
                        # Optional: print the specific wrongly ordered edge
                        # f.write(f"    - Wrongly ordered: {cause_idx} -> {effect_idx} (Order: {causal_order[cause_idx]} >= {causal_order[effect_idx]})\n")

            f.write(f"Number of wrongly ordered cause-effect pairs: {num_wrongly_ordered_edges}\n")
    
            
            f.write("\n" + "="*50 + "\n\n")
            
            # --- Write the summary_matrix_continuous ---
            f.write("--- CONTINUOUS SUMMARY MATRIX ---\n")
            f.write("(Represents the max causal effect across all lags before pruning)\n\n")
            
            # 6. Use numpy.savetxt to write the array to the file handle 'f'
            # 'fmt' controls the number format to keep it clean.
            np.savetxt(f, summary_matrix_continuous, fmt='%.6f', delimiter=',')

        print(f"All results have been successfully saved to '{output_filename}'")
        
        # N_VARIABLES = M_taus[0].shape[0]
        # if M_taus.shape[0] > 1:
        #     self.plot_summary_causal_graph(M_taus[0], M_taus[1], N_VARIABLES, filename="../data/VAR/varlingam/summary_graph.png")

        # self.plot_instantaneous_causal_graph(M_taus[0], N_VARIABLES, 
        #                                 filename=os.path.join("../data/VAR/varlingam/B0_graph.png"))
        # self.plot_B1_causal_graph(summary_matrix_pruned, N_VARIABLES, filename=os.path.join(save_path+"summary_graph.png"))
        
        return self

    def evaluate_var_residuals(self, lags_to_test=None, significance_level=0.05):
        """
        Analyzes the VAR residuals to check for remaining temporal structure.

        This method performs and visualizes three key diagnostic checks:
        1.  **ACF/PACF Plots**: For checking remaining autocorrelation in each residual series.
        2.  **Ljung-Box Test**: A formal statistical test for autocorrelation.
        3.  **Cross-Correlation Plots**: For checking remaining cross-dependencies between pairs of residual series.

        Args:
            lags_to_test (int, optional): The number of lags to include in plots and tests.
                                        If None, it defaults to the number of lags from the VAR model.
            significance_level (float, optional): The p-value threshold for statistical tests. Defaults to 0.05.
        """
        if self._residuals is None:
            print("Model has not been fit yet. Please run .fit(X) first.")
            return

        residuals = self._residuals
        n_obs, n_dims = residuals.shape

        if lags_to_test is None:
            # Use the number of lags from the fitted VAR model as a default
            lags_to_test = self._lags

        print("--- Evaluating VAR Residuals ---")

        # --- 1. ACF and PACF Analysis ---
        print("\n[1/3] Generating ACF and PACF plots for each residual series...")
        for i in range(n_dims):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            plot_acf(residuals[:, i], ax=axes[0], lags=lags_to_test)
            plot_pacf(residuals[:, i], ax=axes[1], lags=lags_to_test)
            fig.suptitle(f'Autocorrelation Analysis for Residual Series {i}', fontsize=16)
            axes[0].set_title('Autocorrelation Function (ACF)')
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            
            # --- ADDED AXIS LABELS ---
            axes[0].set_xlabel('Lags')
            axes[0].set_ylabel('Autocorrelation')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('Partial Correlation')
            # --- END OF ADDED CODE ---
            from matplotlib.patches import Patch
            confidence_patch = Patch(color='blue', alpha=0.15, label=f'{(1 - significance_level)*100:.0f}% Confidence Interval')
            
            # You can also use a dummy plot if you prefer, but a patch is cleaner for filled areas.
            # E.g., `axes[0].plot([], [], color='blue', alpha=0.15, label=f'{(1 - significance_level)*100:.0f}% Confidence Interval')`
            
            # Combine existing handles/labels if necessary, but plot_acf/pacf don't create them by default for the lines.
            # So, we'll create custom legend for both.
            
            # For ACF:
            # The stems are typically not labeled for legend.
            # If you want to explicitly label "ACF" and "PACF" lines, you'd need to plot them manually
            # or extract the handles from plot_acf/plot_pacf.
            # A simpler way is to just define custom legend entries.
            
            # Custom legend for the ACF plot
            handles_acf = [
                plt.Line2D([0], [0], marker='o', color='tab:blue', linestyle='-', label='Autocorrelation Value'),
                confidence_patch
            ]
            axes[0].legend(handles=handles_acf, loc='upper right')
            axes[0].set_ylim(-0.25, 0.25)
            
            plt.tight_layout() # Adjust layout to prevent title overlap
            plt.show()
        print("Interpretation: Look for significant spikes at non-zero lags (outside the blue area).")
        print("Their presence indicates that the VAR model has not fully captured the time dependencies.\n")

        # --- 2. Portmanteau Test (Ljung-Box) ---
        print(f"\n[2/3] Performing Ljung-Box test for autocorrelation up to {lags_to_test} lags...")
        is_whitened = True
        for i in range(n_dims):
            # statsmodels' acorr_ljungbox returns a pandas DataFrame
            result_df = acorr_ljungbox(residuals[:, i], lags=[lags_to_test], return_df=True)
            p_value = result_df['lb_pvalue'].iloc[0]

            print(f"  - Residual Series {i}: p-value = {p_value:.4f}")
            if p_value < significance_level:
                print(f"    -> WARNING: Null hypothesis rejected. Significant autocorrelation REMAINS.")
                is_whitened = False
            else:
                print(f"    -> OK: Fail to reject null hypothesis. No significant autocorrelation detected.")
        if is_whitened:
             print("\nOverall Ljung-Box Result: The VAR residuals appear to be white noise, which is the desired outcome.")
        else:
             print("\nOverall Ljung-Box Result: At least one residual series still contains significant autocorrelation. The VAR model may be misspecified.")

        # --- 3. Cross-Correlation Function (CCF) Analysis ---
        print("\n[3/3] Generating Cross-Correlation (CCF) plots for residual pairs...")
        if n_dims > 5:
            print(f"Warning: Generating CCF plots for all {n_dims * (n_dims - 1) // 2} pairs. This may take a moment.")

        for i, j in combinations(range(n_dims), 2):
            # Calculate CCF between residuals i and j
            ccf_values = ccf(residuals[:, i], residuals[:, j], adjusted=False)
            lags = np.arange(len(ccf_values)) - (len(ccf_values) // 2) # Center lags around 0

            plt.figure(figsize=(5, 4))
            plt.stem(lags, ccf_values)
            # Add confidence interval lines
            ci = 2 / np.sqrt(n_obs)
            plt.axhline(y=ci, color='r', linestyle='--', label=f'{100*(1-significance_level):.0f}% Confidence Interval')
            plt.axhline(y=-ci, color='r', linestyle='--')
            plt.title(f'Cross-Correlation between Residual Series {i} and {j}')
            plt.xlabel('Lags')
            plt.ylabel('Cross-Correlation')
            plt.ylim(min(np.min(ccf_values)*1.2, -ci*2), max(np.max(ccf_values)*1.2, ci*2))
            handles_ccf = [
                plt.Line2D([0], [0], marker='o', color='tab:blue', linestyle='-', label='Cross-Correlation Value'), # Assuming stem color is 'tab:blue'
                plt.Line2D([0], [0], color='r', linestyle='--', label=f'{(1 - significance_level)*100:.0f}% Confidence Interval')
            ]
            plt.legend(handles=handles_ccf, loc='upper right')
            # plt.legend()
            plt.show()
        print("Interpretation: Look for significant spikes at non-zero lags. Their presence suggests")
        print("that lagged values of one series still influence another, which the VAR model should have captured.\n")


    def f1_max(self, labs, preds):
        # F1 MAX
        precision, recall, thresholds = precision_recall_curve(labs, preds)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores) # Handle potential division by zero
        best_idx = np.argmax(f1_scores)
        f1_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        f1_score = f1_scores[best_idx]
        return f1_thresh, f1_score

    def max_accuracy(self, labs, preds):
        # ACCURACY MAX
        preds = preds.astype(float)
        if preds.min() == preds.max():
            a = []
        else:
            a = list(
                np.arange(
                    preds.min(),
                    preds.max() + preds.min(),
                    (preds.max() - preds.min()) / 100,
                )
            )
        possible_thresholds = [0] + a + [preds.max() + 1e-6]
        acc = [accuracy_score(labs, preds > thresh) for thresh in possible_thresholds]
        acc_thresh = possible_thresholds[np.argmax(acc)]
        acc_score = np.nanmax(acc)
        return acc_thresh, acc_score


    def get_causal_order_from_lagged_matrix(self, pruned_matrix):
        n_vars = pruned_matrix.shape[0]
        if pruned_matrix.shape[1] != n_vars:
            raise ValueError("Input matrix must be square.")

        # Adjacency list: adj[u] = list of nodes v such that u -> v (u causes v)
        adj = [[] for _ in range(n_vars)]
        # In-degree: current_in_degree[v] = number of incoming edges to v from unprocessed nodes
        initial_in_degree = np.zeros(n_vars, dtype=int)

        for r_idx in range(n_vars):  # r_idx is the effect node i
            for c_idx in range(n_vars):  # c_idx is the cause node j
                if pruned_matrix[r_idx, c_idx] != 0:
                    # Edge from c_idx (cause at t-1) to r_idx (effect at t)
                    adj[c_idx].append(r_idx)
                    initial_in_degree[r_idx] += 1
        
        current_in_degree = np.array(initial_in_degree) # Make a mutable copy
        
        causal_order = []
        processed_nodes_mask = np.array([False] * n_vars)
        
        # print("Starting heuristic ordering process...") # Optional: for verbose output

        for _ in range(n_vars):
            # Find candidate nodes: unprocessed nodes with current_in_degree == 0
            # Sort by index for deterministic tie-breaking
            candidate_nodes_zero_in_degree = sorted([
                node for node in range(n_vars) 
                if not processed_nodes_mask[node] and current_in_degree[node] == 0
            ])

            selected_node = -1

            if candidate_nodes_zero_in_degree:
                selected_node = candidate_nodes_zero_in_degree[0] # Pick smallest index
                # print(f"Selected node {selected_node} (in-degree 0).")
            else:
                # No 0-in-degree nodes left among unprocessed ones -> cycle detected
                # Heuristic: pick an unprocessed node with the minimum positive in-degree
                min_pos_in_degree_val = float('inf')
                
                candidate_nodes_min_pos_in_degree = []
                for node_idx in range(n_vars):
                    if not processed_nodes_mask[node_idx]: # Only consider unprocessed nodes
                        if current_in_degree[node_idx] > 0: # Must be positive in-degree
                            if current_in_degree[node_idx] < min_pos_in_degree_val:
                                min_pos_in_degree_val = current_in_degree[node_idx]
                                candidate_nodes_min_pos_in_degree = [node_idx]
                            elif current_in_degree[node_idx] == min_pos_in_degree_val:
                                candidate_nodes_min_pos_in_degree.append(node_idx)
                
                if not candidate_nodes_min_pos_in_degree:
                    # This state implies all remaining unprocessed nodes have in_degree 0,
                    # which should have been caught by the first 'if' block.
                    # Or, all nodes are processed. This is a fallback for safety.
                    remaining_unprocessed = [i for i, processed in enumerate(processed_nodes_mask) if not processed]
                    if not remaining_unprocessed: # Should only happen if _ > n_vars
                        break 
                    selected_node = min(remaining_unprocessed) 
                    # warnings.warn(f"Cycle breaking fallback: selected node {selected_node} from remaining. "
                    #             f"In-degree was {current_in_degree[selected_node] if selected_node < n_vars else 'N/A'}")
                else:
                    candidate_nodes_min_pos_in_degree.sort() # Tie-break by smallest index
                    selected_node = candidate_nodes_min_pos_in_degree[0]
                    # warnings.warn(f"Cycle encountered. Heuristically breaking by selecting node {selected_node} "
                    #             f"with current min positive in-degree {min_pos_in_degree_val}.")

            if selected_node == -1 : # Should ideally not be reached
                # If all nodes are processed, this loop iteration shouldn't happen.
                # If there are unprocessed nodes but none got selected, that's an issue.
                remaining = [i for i,p in enumerate(processed_nodes_mask) if not p]
                if not remaining: break
                raise Exception(f"Could not select a node. Remaining unprocessed: {remaining}. Current in-degrees: {current_in_degree[~processed_nodes_mask]}")


            causal_order.append(selected_node)
            processed_nodes_mask[selected_node] = True

            # "Remove" outgoing edges from selected_node by decrementing in-degrees 
            # of its children (that are not yet processed)
            for neighbor_node in adj[selected_node]:
                if not processed_nodes_mask[neighbor_node]: 
                    current_in_degree[neighbor_node] -= 1
                    
        # Ensure all nodes are in the order, even if loop logic had an issue (should not happen)
        if len(causal_order) != n_vars:
            warnings.warn(f"Ordering process resulted in {len(causal_order)} nodes, expected {n_vars}. "
                        "Appending remaining nodes by index.")
            processed_set = set(causal_order)
            for i in range(n_vars):
                if i not in processed_set:
                    causal_order.append(i)
        
        # print("Finished heuristic ordering process.") # Optional
        return causal_order


    def plot_summary_causal_graph(self, B0, B1, n_vars, filename):
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

    def plot_instantaneous_causal_graph(self, B0, n_vars, filename):
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

    def plot_B1_causal_graph(self, B1, n_vars, filename="B1_causal_graph.png"): # Argument is B1
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
            iterations = 100 
            pos = nx.spring_layout(G, k=8, iterations=iterations, scale=2.0) 
            
            node_size = 3000 
            font_size = 35  # Font size for integer labels
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



    def estimate_total_effect(self, X, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        X = check_array(X)
        n_features = X.shape[1]

        # Check from/to causal order
        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_index={to_index}) "
                    f"is earlier than the source variable (from_index={from_index})."
                )

        # X + lagged X
        X_joined = np.zeros((X.shape[0], X.shape[1] * (1 + self._lags + from_lag)))
        for p in range(1 + self._lags + from_lag):
            pos = n_features * p
            X_joined[:, pos : pos + n_features] = np.roll(X[:, 0:n_features], p, axis=0)

        # from_index + parents indices
        am = np.concatenate([*self._adjacency_matrices], axis=1)
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        from_index = (
            from_index if from_lag == 0 else from_index + (n_features * from_lag)
        )
        parents = parents if from_lag == 0 else parents + (n_features * from_lag)
        predictors = [from_index]
        predictors.extend(parents)

        # estimate total effect
        lr = LinearRegression()
        lr.fit(X_joined[:, predictors], X_joined[:, to_index])

        return lr.coef_[0]

    def estimate_total_effect2(self, n_features, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model.

        Parameters
        ----------
        n_features :
            The number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check from/to causal order
        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_index={to_index}) "
                    f"is earlier than the source variable (from_index={from_index})."
                )

        # from_index + parents indices
        am = np.concatenate([*self._adjacency_matrices], axis=1)
        am = np.pad(am, [(0, am.shape[1] - am.shape[0]), (0, 0)])
        from_index = (
            from_index if from_lag == 0 else from_index + (n_features * from_lag)
        )

        effect = calculate_total_effect(am, from_index, to_index)

        return effect

    def get_error_independence_p_values(self):
        """Calculate the p-value matrix of independence between error variables.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        nn = self.residuals_
        B0 = self._adjacency_matrices[0]
        E = np.dot(np.eye(B0.shape[0]) - B0, nn.T).T
        n_samples = E.shape[0]
        n_features = E.shape[1]

        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            _, p_value = hsic_test_gamma(
                np.reshape(E[:, i], [n_samples, 1]), np.reshape(E[:, j], [n_samples, 1])
            )
            p_values[i, j] = p_value
            p_values[j, i] = p_value

        return p_values

    def _estimate_var_coefs(self, X):
        """Estimate coefficients of VAR"""
        # XXX: VAR.fit() is not searching lags correctly
        if self._criterion not in ["aic", "fpe", "hqic", "bic"]:
            var = VAR(X)
            result = var.fit(maxlags=self._lags, trend="n")
        else:
            min_value = float("Inf")
            result = None

            for lag in range(1, self._lags + 1):
                var = VAR(X)
                fitted = var.fit(maxlags=lag, ic=None, trend="n")

                value = getattr(fitted, self._criterion)
                if value < min_value:
                    min_value = value
                    result = fitted

        return result.coefs, result.k_ar, result.resid

    def _calc_residuals(self, X, M_taus, lags):
        """Calculate residulas"""
        X = X.T
        n_features = X.shape[0]
        n_samples = X.shape[1]

        residuals = np.zeros((n_features, n_samples))
        for t in range(n_samples):
            if t - lags < 0:
                continue

            estimated = np.zeros((X.shape[0], 1))
            for tau in range(1, lags + 1):
                estimated += np.dot(M_taus[tau - 1], X[:, t - tau].reshape((-1, 1)))

            residuals[:, t] = X[:, t] - estimated.reshape((-1,))

        residuals = residuals[:, lags:].T

        return residuals

    def _calc_b(self, X, B0, M_taus):
        """Calculate B"""
        n_features = X.shape[1]

        B_taus = np.array([B0])

        for M in M_taus:
            B_t = np.dot((np.eye(n_features) - B0), M)
            B_taus = np.append(B_taus, [B_t], axis=0)

        return B_taus

    def _pruning(self, X, B_taus, causal_order):
        """Prune edges"""
        n_features = X.shape[1]

        stacked = [np.flip(X, axis=0)]
        for i in range(self._lags):
            stacked.append(np.roll(stacked[-1], -1, axis=0))
        blocks = np.array(list(zip(*stacked)))[: -self._lags]

        for i in range(n_features):
            causal_order_no = causal_order.index(i)
            ancestor_indexes = causal_order[:causal_order_no]

            obj = np.zeros((len(blocks)))
            exp = np.zeros((len(blocks), causal_order_no + n_features * self._lags))
            for j, block in enumerate(blocks):
                obj[j] = block[0][i]
                exp[j:] = np.concatenate(
                    [block[0][ancestor_indexes].flatten(), block[1:][:].flatten()],
                    axis=0,
                )

            # adaptive lasso
            predictors = [i for i in range(exp.shape[1])]
            target = len(predictors)
            X_con = np.concatenate([exp, obj.reshape(-1, 1)], axis=1)
            coef = predict_adaptive_lasso(X_con, predictors, target)

            B_taus[0][i, ancestor_indexes] = coef[:causal_order_no]
            for j in range(len(B_taus[1:])):
                B_taus[j + 1][i, :] = coef[
                    causal_order_no + n_features * j :
                    causal_order_no + n_features * j + n_features
                ]

        return B_taus

    def _pruning_threshold(self, X, B_taus, causal_order): # Signature kept, X and causal_order are unused in this simple version
        """Prune edges by applying an absolute threshold to the B_taus coefficients.
        This is a simplified version of pruning.
        """
        # Get the threshold from the instance attribute, with a default if it's somehow not set.
        threshold = getattr(self, '_pruning_threshold', 0.05)
        if not hasattr(self, '_pruning_threshold'):
            warnings.warn(
                f"VARLiNGAM: _pruning_threshold attribute not found. "
                f"Using default threshold {threshold} for pruning. "
                f"Consider setting it in the __init__ method."
            )

        print(f"[Pruning] Applying threshold-based pruning with threshold={threshold}.")
        
        # Create a copy of B_taus to avoid modifying the original array in place if it's passed around.
        pruned_B_taus = np.copy(B_taus)
        
        # Apply threshold: set elements with absolute value less than threshold to 0.
        pruned_B_taus[np.abs(pruned_B_taus) < threshold] = 0
        
        # Note: B_taus[0] is the contemporaneous adjacency matrix (B0) from LiNGAM.
        # This simple thresholding applies to B0 as well as lagged coefficient matrices B_t.
        # B0 should already be causally ordered (e.g., lower triangular in the permuted order)
        # by the underlying LiNGAM algorithm. Applying a threshold here will further sparsify it.

        return pruned_B_taus

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where
            n_features is the number of features.
        """
        return self._causal_order

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (lags, n_features, n_features)
            The adjacency matrix of fitted model, where
            n_features is the number of features.
        """
        return self._adjacency_matrices

    @property
    def residuals_(self):
        """Residuals of regression.

        Returns
        -------
        residuals_ : array-like, shape (n_samples)
            Residuals of regression, where n_samples is the number of samples.
        """
        return self._residuals


    def summarize_var_residuals(self, lags_to_test=None, significance_level=0.05):
        """
        Provides a numerical summary of VAR residuals to check for remaining
        temporal structure, focusing on ACF and CCF.

        Args:
            lags_to_test (int, optional): The number of lags to include in the analysis.
                                        If None, it defaults to the number of lags from the VAR model.
            significance_level (float, optional): The significance level for confidence intervals.
                                                Defaults to 0.05 for 95% confidence.
        """
        if self._residuals is None:
            print("Model has not been fit yet. Please run .fit(X) first.")
            return

        residuals = self._residuals
        n_obs, n_dims = residuals.shape

        if lags_to_test is None:
            lags_to_test = self._lags

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
            # print(f"  - Residual Series {i}: Ratio of significant spikes = {ratio:.2f} ({significant_spikes}/{lags_to_test})")

            # --- Optional: Plotting with the fixed confidence interval ---
            # plt.figure(figsize=(5, 4))
            # plt.stem(range(lags_to_test + 1), acf_values, use_line_collection=True)
            # plt.axhspan(lower_bound_fixed, upper_bound_fixed, color='blue', alpha=0.15)
            # plt.axhline(0, color='gray', linestyle='--')
            # plt.xlabel('Lags')
            # plt.ylabel('Autocorrelation')
            # plt.title(f'ACF of Residual Series {i} with Fixed CI')
            # plt.ylim(-0.25, 0.25) # As per your previous request
            # plt.tight_layout()
            # plt.show()
            # --- End Optional Plotting ---


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
            # You can uncomment the line below for a detailed report per pair
            # print(f"  - Pair ({i}, {j}): Ratio of significant spikes = {ratio:.2f} ({significant_spikes}/{lags_to_test})")

        # Calculate and print the mean value
        mean_ccf_ratio = np.mean(ccf_ratios)
        print(f"\nMean CCF Ratio of Significant Spikes: {mean_ccf_ratio:.4f} (averaged over {num_pairs} pairs)")
        print("  (Warning: This average can obscure strong cross-correlations between specific pairs.)")


def summarize_var_residuals(data, lags_to_test=None, significance_level=0.05):
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
        # print(f"  - Residual Series {i}: Ratio of significant spikes = {ratio:.2f} ({significant_spikes}/{lags_to_test})")

        # --- Optional: Plotting with the fixed confidence interval ---
        # plt.figure(figsize=(5, 4))
        # plt.stem(range(lags_to_test + 1), acf_values, use_line_collection=True)
        # plt.axhspan(lower_bound_fixed, upper_bound_fixed, color='blue', alpha=0.15)
        # plt.axhline(0, color='gray', linestyle='--')
        # plt.xlabel('Lags')
        # plt.ylabel('Autocorrelation')
        # plt.title(f'ACF of Residual Series {i} with Fixed CI')
        # plt.ylim(-0.25, 0.25) # As per your previous request
        # plt.tight_layout()
        # plt.show()
        # --- End Optional Plotting ---


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
        # You can uncomment the line below for a detailed report per pair
        # print(f"  - Pair ({i}, {j}): Ratio of significant spikes = {ratio:.2f} ({significant_spikes}/{lags_to_test})")

    # Calculate and print the mean value
    mean_ccf_ratio = np.mean(ccf_ratios)
    print(f"\nMean CCF Ratio of Significant Spikes: {mean_ccf_ratio:.4f} (averaged over {num_pairs} pairs)")
    print("  (Warning: This average can obscure strong cross-correlations between specific pairs.)")



if __name__ == "__main__":
    # X = pd.read_csv('../data/varlingam/varlingam_data.csv', header=0)
    # X = pd.read_csv('../data/violated/violated_data.csv', header=0)

    # X = pd.read_csv("../data/Web_Activity/preprocessed_2.csv", delimiter=',', index_col=0, header=0)
    # X.columns = X.columns.str.replace(' ', '_')

    # X = pd.read_csv("../data/Storm_Ingestion_Activity/storm_data_normal.csv", delimiter=',', index_col=0, header=0)
    # X.columns = X.columns.str.replace(' ', '_')    

    # X = pd.read_csv("../data/Middleware_oriented_message_Activity/monitoring_metrics_1.csv", delimiter=',', index_col=0, header=0)
    # X.columns = X.columns.str.replace(' ', '_')

    # X = pd.read_csv("../data/Antivirus_Activity/preprocessed_2.csv", delimiter=',', index_col=0, header=0)
    # X.columns = X.columns.str.replace(' ', '_')
    
    # X = pd.read_csv("../causalriver/rivers_ts_flood_preprocessed.csv", index_col=0)
    # X = X.to_numpy()

    # X = pd.read_csv("../causalriver/rivers_ts_bavaria_preprocessed.csv", index_col=0)
    # X = X.to_numpy()

    X = pd.read_csv("../causalriver/rivers_ts_east_germany_preprocessed.csv", index_col=0)
    X = X.to_numpy()





    # X = pd.read_csv("../causalriver/rivers_ts_flood_shuffled.csv", header=0)
    # X = X.to_numpy()

    # X = pd.read_csv("../causalriver/rivers_ts_bavaria_shuffled.csv", header=0)
    # X = X.to_numpy()

    # X = pd.read_csv("../data/Web_Activity/shuffled_preprocessed_2.csv", header=0)

    # X = pd.read_csv("../data/Storm_Ingestion_Activity/shuffled_storm_data_normal.csv", header=0)

    # X = pd.read_csv("../data/Middleware_oriented_message_Activity/shuffled_monitoring_metrics_1.csv", header=0)

    # X = pd.read_csv("../data/Antivirus_Activity/shuffled_preprocessed_2.csv", delimiter=',', header=0)
    
    summarize_var_residuals(X, lags_to_test=10)
    print("----------------------after VAR----------------------")
    model = VARLiNGAM(lags=10)
    model.fit_VAReffect(X)
    # model.evaluate_var_residuals(lags_to_test=100)
    model.summarize_var_residuals(lags_to_test=10)
    # summarize_var_residuals(X, lags_to_test=10)

