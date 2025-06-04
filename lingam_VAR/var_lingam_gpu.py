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
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    roc_curve
)
import pandas as pd

import sys
import os

# --- Sys.path modification (ensure your lingam_VAR is found) ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    package_parent_dir = os.path.dirname(current_script_dir)
    if package_parent_dir not in sys.path:
        sys.path.insert(0, package_parent_dir)

    from lingam_VAR.base import _BaseLiNGAM
    # from lingam_VAR.direct_lingam import DirectLiNGAM
    from culingam.directlingam import DirectLiNGAM
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

    # def __init__(
    #     self,
    #     lags=1,
    #     criterion="bic",
    #     prune=True,
    #     ar_coefs=None,
    #     lingam_model=None,
    #     random_state=None,
    # ):
    #     """Construct a VARLiNGAM model.

    #     Parameters
    #     ----------
    #     lags : int, optional (default=1)
    #         Number of lags.
    #     criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
    #         Criterion to decide the best lags within ``lags``.
    #         Searching the best lags is disabled if ``criterion`` is ``None``.
    #     prune : boolean, optional (default=True)
    #         Whether to prune the adjacency matrix of lags.
    #     ar_coefs : array-like, optional (default=None)
    #         Coefficients of AR model. Estimating AR model is skipped if specified ``ar_coefs``.
    #         Shape must be (``lags``, n_features, n_features).
    #     lingam_model : lingam object inherits 'lingam._BaseLiNGAM', optional (default=None)
    #         LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.
    #     random_state : int, optional (default=None)
    #         ``random_state`` is the seed used by the random number generator.
    #     """
    #     self._lags = lags
    #     self._criterion = criterion
    #     self._prune = prune
    #     self._ar_coefs = (
    #         check_array(ar_coefs, allow_nd=True) if ar_coefs is not None else None
    #     )
    #     self._lingam_model = lingam_model
    #     self._random_state = random_state

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


    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        returns
        -------
        self : object
            Returns the instance itself.
        """
        fit_start_time = time.time()
        print("[VarLiNGAM fit] Starting...")

        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)

        # Initialize LiNGAM model if needed
        init_lingam_start = time.time()
        lingam_model = self._lingam_model
        if lingam_model is None:
            from culingam.directlingam import DirectLiNGAM
            lingam_model = DirectLiNGAM()
            # lingam_model = DirectLiNGAM(measure="resid_ng")
            print("[VarLiNGAM fit] Initialized default DirectLiNGAM.")
        elif not isinstance(lingam_model, _BaseLiNGAM): # Assuming _BaseLiNGAM is defined
            raise ValueError("lingam_model must be a subclass of _BaseLiNGAM")
        init_lingam_end = time.time()
        print(f"[VarLiNGAM fit] LiNGAM model setup took: {init_lingam_end - init_lingam_start:.4f} seconds")

        M_taus = self._ar_coefs

        # --- Step 1: Estimate VAR coefficients and residuals ---
        var_fit_start = time.time()
        if M_taus is None:
            print("[VarLiNGAM fit] Estimating VAR coefficients...")
            M_taus, lags, residuals = self._estimate_var_coefs(X)
            print(f"[VarLiNGAM fit] VAR estimation complete. Lags={lags}")
        else:
            print("[VarLiNGAM fit] Calculating residuals from provided VAR coefficients...")
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)
            print("[VarLiNGAM fit] Residual calculation complete.")
        var_fit_end = time.time()
        print(f"[VarLiNGAM fit] VAR estimation/Residual calculation took: {var_fit_end - var_fit_start:.4f} seconds")
        # --- End Step 1 ---

        # --- Step 2: Fit LiNGAM on residuals ---
        lingam_fit_start = time.time()
        print("[VarLiNGAM fit] Fitting LiNGAM model on residuals...")
        model = lingam_model
        model.fit(residuals) # This will call the DirectLiNGAM fit method (or other variant)
        lingam_fit_end = time.time()
        print(f"[VarLiNGAM fit] LiNGAM fitting on residuals took: {lingam_fit_end - lingam_fit_start:.4f} seconds")
        # --- End Step 2 ---

        # --- Step 3: Calculate B_taus (lagged adjacency matrices) ---
        calc_b_start = time.time()
        print("[VarLiNGAM fit] Calculating lagged adjacency matrices (B_taus)...")
        B_taus = self._calc_b(X, model.adjacency_matrix_, M_taus)
        calc_b_end = time.time()
        print(f"[VarLiNGAM fit] B_taus calculation took: {calc_b_end - calc_b_start:.4f} seconds")
        # --- End Step 3 ---

        ## ------------------ Save unprunede results to file ------------------
        # data_path = "../data/varlingam/100/"
        # save_path = "../data/lingam/varlingam/100/"
        # order_path = data_path + "causal_order.npy"
        # label_path = data_path + "summary_matrix.npy"

        label_path = "../causalriver/bavaria_matrix.npy"
        save_path = "../data/lingam/shuffle_bavaria/"
        order_path = "../causalriver/bavaria_order.npy"

        print("B0 shape:", model.adjacency_matrix_.shape)
        print("Btaus shape:", B_taus.shape)

        self.save_file(model.adjacency_matrix_, label_path, order_path, lags, save_path, causal_order=model.causal_order_, prune=True, filename="B0_info")
        self.save_file(B_taus, label_path, order_path, lags, save_path, causal_order=None, prune=True, filename="Btaus_info")


        # --- Step 4: Pruning (Optional) ---
        pruning_start = time.time()
        if self._prune:
            print("[VarLiNGAM fit] Pruning adjacency matrices...")
            B_taus = self._pruning(X, B_taus, model.causal_order_)
            pruning_end = time.time()
            print(f"[VarLiNGAM fit] Pruning took: {pruning_end - pruning_start:.4f} seconds")
        else:
            pruning_end = pruning_start # No time taken if not pruning
            print("[VarLiNGAM fit] Pruning step skipped.")
        # --- End Step 4 ---

        print("Btaus shape:", B_taus.shape)
        self.save_file(B_taus, label_path, order_path, lags, save_path, causal_order=None, prune=False, filename="pruned_Btaus_info")
        

        # Store results
        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals
        self._causal_order = model.causal_order_
        self._adjacency_matrices = B_taus

        fit_end_time = time.time()
        print(f"[VarLiNGAM fit] Finished. Total time: {fit_end_time - fit_start_time:.4f} seconds")

        print("Causal Order:", model.causal_order_)

        return self

    def save_file(self, matrix, label_path, order_path, lags, save_path, causal_order=None, prune=True, filename="summary_matrix_info"):
        if matrix.ndim == 2:
            # If matrix is 2D, we assume it's a single adjacency matrix
            matrix = np.expand_dims(matrix, axis=0)
            
        summary_matrix_continuous = np.max(np.abs(matrix), axis=0)

        # print("Shape of B_taus:", B_taus.shape)
        # print("Summary matrix continuous:", summary_matrix_continuous)

        labels = np.load(label_path)

        # We'll use the f1_max function to find the best threshold for F1 score.
        # Flatten both the summary matrix and the labels for the metric functions.
        y_true_flat = labels.flatten()
        y_pred_flat = summary_matrix_continuous.flatten()
        # print("Shape of y_true_flat:", y_true_flat.shape)
        # print("Shape of y_pred_flat:", y_pred_flat.shape)

        if prune:
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
        else:
            summary_matrix_pruned = np.where(summary_matrix_continuous != 0, 1, 0)


        # Now, calculate the final metrics for this pruned binary matrix
        final_f1_score = self.f1_max(y_true_flat, summary_matrix_pruned.flatten())[1]
        final_accuracy = self.max_accuracy(y_true_flat, summary_matrix_pruned.flatten())[1]
        final_auroc = roc_auc_score(y_true_flat, summary_matrix_pruned.flatten())
        print(f"Final F1 Score after pruning: {final_f1_score:.4f}")
        print(f"Final Accuracy after pruning: {final_accuracy:.4f}")
        print(f"Final AUROC after pruning: {final_auroc:.4f}")



        output_filename = save_path + filename + ".txt"

        # Use a 'with' block to safely open and write to the file
        with open(output_filename, 'w') as f:
            # --- Write scalar values ---
            f.write("--- METRICS AND THRESHOLDS ---\n\n")

            f.write(f"Best Lags: {lags}\n")

            if prune:
                # 1. Best Threshold (from F1-Max)
                f.write(f"Best Threshold (from F1-Max): {best_f1_thresh:.4f}\n")
                
                # 2. Best Threshold (from Accuracy-Max)
                f.write(f"Best Threshold (from Accuracy-Max): {best_acc_thresh:.4f}\n")
            else:
                f.write(f"Lasso Pruning Threshold: {self._pruning_threshold:.4f}\n")
            
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
            if causal_order==None:
                lagged_causal_order = self.get_causal_order_from_lagged_matrix(summary_matrix_pruned)
            else:
                lagged_causal_order = causal_order
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
            
            f.write("--- SUMMARY MATRIX ---\n")
            f.write("(Represents the max causal effect across all lags after pruning)\n\n")
            f.write(summary_matrix_pruned.__str__() + "\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
            # --- Write the summary_matrix_continuous ---
            f.write("--- EFFECT SUMMARY MATRIX ---\n")
            f.write("(Represents the max causal effect across all lags before pruning)\n\n")
            
            # 6. Use numpy.savetxt to write the array to the file handle 'f'
            # 'fmt' controls the number format to keep it clean.
            np.savetxt(f, summary_matrix_continuous, fmt='%.6f', delimiter=',')

        print(f"All results have been successfully saved to '{output_filename}'")


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



if __name__ == "__main__":
    # X = pd.read_csv('../data/varlingam/varlingam_data.csv', header=0)
    # X = pd.read_csv('../data/violated/violated_data.csv', header=0)
    
    # X = pd.read_csv('../data/varlingam/100/data.csv', header=0)

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

    

    # X = pd.read_csv("../causalriver/rivers_ts_flood_shuffled.csv", header=0)
    # X = X.to_numpy()

    X = pd.read_csv("../causalriver/rivers_ts_bavaria_shuffled.csv", header=0)
    X = X.to_numpy()

    # X = pd.read_csv("../data/Web_Activity/shuffled_preprocessed_2.csv", header=0)

    # X = pd.read_csv("../data/Storm_Ingestion_Activity/shuffled_storm_data_normal.csv", header=0)

    # X = pd.read_csv("../data/Middleware_oriented_message_Activity/shuffled_monitoring_metrics_1.csv", header=0)

    # X = pd.read_csv("../data/Antivirus_Activity/shuffled_preprocessed_2.csv", delimiter=',', header=0)
    

    model = VARLiNGAM(lags=5, prune=True)
    model.fit(X)
