import numpy as np
import warnings

def count_pairwise_order_violations(true_order_indices, estimated_order_indices):
    """
    Compares two causal orders (lists of variable indices) and counts how many
    pairwise orderings from the true_order are violated in the estimated_order.

    Parameters:
    - true_order_indices (list or np.ndarray): List of variable indices in true causal order.
    - estimated_order_indices (list or np.ndarray): List of variable indices in estimated causal order.

    Returns:
    - tuple: (lost_orders_count, total_possible_true_pairs)
               Returns (np.nan, np.nan) if inputs are invalid.
    """
    if true_order_indices is None or estimated_order_indices is None:
        warnings.warn("One or both causal orders are None. Cannot compare.")
        return np.nan, np.nan
    
    # Ensure inputs are list-like
    if not (hasattr(true_order_indices, '__len__') and hasattr(estimated_order_indices, '__len__')):
        warnings.warn("One or both causal orders are not list-like. Cannot compare.")
        return np.nan, np.nan

    if len(true_order_indices) != len(estimated_order_indices):
        warnings.warn("True order and estimated order have different lengths. Cannot perform pairwise comparison.")
        return np.nan, np.nan # Or handle error based on context

    n_features = len(true_order_indices)
    if n_features < 2:
        return 0, 0 # No pairs to compare for less than 2 features

    # Create a mapping from variable index to its position in the estimated_order
    try:
        # Ensure indices in estimated_order_indices are integers for the dictionary keys
        est_pos = {int(var_idx): pos for pos, var_idx in enumerate(estimated_order_indices)}
    except (ValueError, TypeError):
        warnings.warn("Failed to create position map from estimated_order_indices. Ensure it's a list/array of numbers.")
        return np.nan, n_features * (n_features - 1) // 2


    # Verify that all variables in true_order_indices are present in the estimated_order_indices's map
    # This ensures that est_pos will not have missing keys for variables from true_order.
    all_vars_mappable = True
    for var_idx in true_order_indices:
        if int(var_idx) not in est_pos: # Convert var_idx to int just in case true_order was mixed type
            all_vars_mappable = False
            warnings.warn(f"Variable index {var_idx} from true_order not found in estimated_order's position map.")
            break
    
    if not all_vars_mappable:
         return np.nan, n_features * (n_features - 1) // 2


    lost_orders_count = 0
    total_possible_true_pairs = 0

    # Iterate through all unique pairs (var_u, var_v) from true_order_indices
    # where var_u appears before var_v.
    for i in range(n_features):
        for j in range(i + 1, n_features):
            var_u = int(true_order_indices[i])  # var_u is before var_v in true_order
            var_v = int(true_order_indices[j])
            
            total_possible_true_pairs += 1

            # Get positions in the estimated order (keys are already int from est_pos creation)
            pos_u_in_est = est_pos[var_u] 
            pos_v_in_est = est_pos[var_v]

            # If var_u is supposed to be before var_v, but in estimated order it's after var_v
            if pos_u_in_est > pos_v_in_est:
                lost_orders_count += 1
                
    return lost_orders_count, total_possible_true_pairs

if __name__ == '__main__':
    # Hardcoded causal orders from your output
    estimated_contemp_order_indices = [34, 15, 22, 16, 14, 28, 25, 39, 29, 24, 33, 3, 35, 38, 17, 7, 23, 6, 13, 4, 11, 2, 19, 12, 0, 27, 26, 9, 5, 10, 18, 40, 30, 41, 31, 21, 20, 32, 36, 37, 8, 1]
    

    true_contemp_order_indices =  [5, 3, 29, 14, 15, 22, 38, 1, 2, 9, 19, 31, 0, 12, 17, 23, 30, 27, 26, 28, 34, 37, 41, 4, 10, 11, 13, 16, 18, 21, 25, 33, 36, 40, 20, 24, 32, 39, 8, 35, 7, 6]

    print("--- Pairwise Causal Order Violation Analysis ---")
    print(f"True Causal Order (first 10): {true_contemp_order_indices[:10]}...")
    print(f"Estimated Causal Order (first 10): {estimated_contemp_order_indices[:10]}...")
    
    lost_pairwise_orders, total_true_pairwise_orders = count_pairwise_order_violations(
        true_contemp_order_indices,
        estimated_contemp_order_indices
    )

    if not (np.isnan(lost_pairwise_orders) or np.isnan(total_true_pairwise_orders)):
        print(f"\nTotal number of ordered pairs in true_causal_order: {total_true_pairwise_orders}")
        print(f"Number of these pairs whose order is violated in estimated_causal_order: {lost_pairwise_orders}")

        if total_true_pairwise_orders > 0:
            accuracy = 1.0 - (lost_pairwise_orders / total_true_pairwise_orders)
            print(f"Pairwise Order Recovery Accuracy: {accuracy:.4f}")
        elif lost_pairwise_orders == 0 and total_true_pairwise_orders == 0: # Should be n_features < 2
             print(f"Pairwise Order Recovery Accuracy: 1.0000 (No pairs to compare)")
        elif lost_pairwise_orders == 0 and total_true_pairwise_orders > 0: # All correct
             print(f"Pairwise Order Recovery Accuracy: 1.0000")
        else: # total_true_pairwise_orders == 0 and lost_pairwise_orders > 0 (should not happen) or other cases
            print(f"Pairwise Order Recovery Accuracy: N/A")
    else:
        print("Could not compute pairwise order violations due to input issues.")

    print("\nAnalysis Finished.")