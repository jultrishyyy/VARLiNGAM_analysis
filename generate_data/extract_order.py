import numpy as np
import warnings




def get_causal_order_from_matrix(matrix):
    n_vars = matrix.shape[0]
    if matrix.shape[1] != n_vars:
        raise ValueError("Input matrix must be square.")

    # Adjacency list: adj[u] = list of nodes v such that u -> v (u causes v)
    adj = [[] for _ in range(n_vars)]
    # In-degree: current_in_degree[v] = number of incoming edges to v from unprocessed nodes
    initial_in_degree = np.zeros(n_vars, dtype=int)

    for r_idx in range(n_vars):  # r_idx is the effect node i
        for c_idx in range(n_vars):  # c_idx is the cause node j
            if matrix[r_idx, c_idx] != 0:
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
                warnings.warn(f"Cycle breaking fallback: selected node {selected_node} from remaining. "
                            f"In-degree was {current_in_degree[selected_node] if selected_node < n_vars else 'N/A'}")
            else:
                candidate_nodes_min_pos_in_degree.sort() # Tie-break by smallest index
                selected_node = candidate_nodes_min_pos_in_degree[0]
                warnings.warn(f"Cycle encountered. Heuristically breaking by selecting node {selected_node} "
                            f"with current min positive in-degree {min_pos_in_degree_val}.")

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


data_array = np.load("../data/Antivirus_Activity/summary_matrix.npy")
order = get_causal_order_from_matrix(data_array)
np.save("../data/Antivirus_Activity/causal_order.npy", order)
print("Causal order from the matrix:")
print(order)