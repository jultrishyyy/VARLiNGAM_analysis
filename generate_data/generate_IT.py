import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# --- Configuration: Set your file paths here ---
# Directory where your input data files (CSV and structure.txt) are located
# DATA_DIR = "../data/Storm_Ingestion_Activity" 
# CSV_FILE_NAME = "storm_data_normal.csv"
# STRUCTURE_FILE_NAME = "storm_structure.txt"

# # Directory where output files (graph image, summary matrix CSV) will be saved
# OUTPUT_DIR = "../data/Storm_Ingestion_Activity"
# # --- End Configuration ---

# CSV_FILE_PATH = os.path.join(DATA_DIR, CSV_FILE_NAME)
# STRUCTURE_FILE_PATH = os.path.join(DATA_DIR, STRUCTURE_FILE_NAME)


def plot_general_causal_graph_indexed(adjacency_df, filename="causal_graph_indexed.png"):
    """
    Plots and saves a causal graph from an adjacency matrix DataFrame.
    Nodes are labeled with their integer indices (0 to N-1).
    Assumes adjacency_df.loc[cause_name, effect_name] == 1 means cause_name -> effect_name.
    """
    node_names_str = adjacency_df.columns.tolist() # Original string names
    n_vars = len(node_names_str)
    
    adj_matrix_np = adjacency_df.values # Convert DataFrame to NumPy array

    G = nx.DiGraph()
    # Nodes in the graph will be integers 0 to n_vars-1
    graph_node_indices = list(range(n_vars))
    G.add_nodes_from(graph_node_indices)

    # Add edges based on the NumPy matrix using integer indices
    # adj_matrix_np[cause_idx, effect_idx] == 1 means node_at_cause_idx -> node_at_effect_idx
    for cause_idx in range(n_vars):
        for effect_idx in range(n_vars):
            if adj_matrix_np[effect_idx, cause_idx] != 0:
                G.add_edge(cause_idx, effect_idx)
    
    plt.figure(figsize=(7, 7)) 

    if n_vars > 15:
        print(f"Plotting graph with {n_vars} nodes (indexed). Using spring_layout.")
        k_val = 0.9 / np.sqrt(max(n_vars, 1)) 
        iterations = 50
        pos = nx.spring_layout(G, k=k_val, iterations=iterations, seed=42, scale=1.5)
        base_node_size = 10000 
        node_size = max(100, base_node_size / n_vars) 
        font_size = max(5, int(100 / np.sqrt(n_vars))) 
        arrow_size = 10
        edge_width = 1.0
    else: 
        # if n_vars <= 5:
        #     k_val = 0.9 
        # elif n_vars <= 10: 
        #     k_val = 0.8 
        # else:
        #     k_val = 0.6
        
        iterations = 100 
        pos = nx.spring_layout(G, k=8, iterations=iterations, scale=2.0) 
        
        node_size = 3000 
        font_size = 35  # Font size for integer labels
        arrow_size = 25
        edge_width = 3.5

        # pos = nx.kamada_kawai_layout(G) # Good for smaller graphs
        # node_size = 3000 # Larger nodes, increased from 1500
        # font_size = 35
        # arrow_size = 25
        # edge_width = 3.5
    
    # nx.draw will use the integer nodes for labels because 'with_labels=True'
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="lightgreen", 
            font_size=font_size, font_weight="bold", arrows=True, arrowsize=arrow_size,
            edge_color='black', width=edge_width, connectionstyle='arc3,rad=0.3')
            
    # plt.title(f"Causal Graph with Indexed Nodes (n={n_vars})", fontsize=18)

    # Optional: Create a mapping for legend or reference if needed later
    # index_to_name_mapping = {i: name for i, name in enumerate(node_names_str)}
    # print("\nNode Index to Name Mapping for plotted graph:")
    # for index, name in index_to_name_mapping.items():
    #     print(f"{index}: {name}")

    try:
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        print(f"Causal graph with indexed nodes saved to {filename}")
    except Exception as e:
        print(f"Error saving indexed graph: {e}")
    plt.close()


if __name__ == "__main__":
    
    # param_data = pd.read_csv("../data/Web_Activity/preprocessed_2.csv", delimiter=',', index_col=0, header=0)

    # param_data = param_data.iloc[:3000]

    # param_data.columns = param_data.columns.str.replace(' ', '_')

    # three_col_format = np.loadtxt("../data/Web_Activity/structure.txt",
    #                               delimiter=' ', dtype=str)

    # param_data = pd.read_csv("../data/Storm_Ingestion_Activity/storm_data_normal.csv", delimiter=',', index_col=0, header=0)

    # param_data.columns = param_data.columns.str.replace(' ', '_')

    # three_col_format = np.loadtxt("../data/Storm_Ingestion_Activity/storm_structure.txt",
    #                               delimiter=' ', dtype=str)

    # param_data = pd.read_csv("../data/Middleware_oriented_message_Activity/monitoring_metrics_1.csv", delimiter=',', index_col=0, header=0)

    # param_data.columns = param_data.columns.str.replace(' ', '_')

    # three_col_format = np.loadtxt("../data/Middleware_oriented_message_Activity/structure.txt", delimiter=' ', dtype=str)


    param_data = pd.read_csv("../data/Antivirus_Activity/preprocessed_2.csv", delimiter=',', index_col=0, header=0)

    param_data.columns = param_data.columns.str.replace(' ', '_')

    three_col_format = np.loadtxt("../data/Antivirus_Activity/structure.txt",
                                  delimiter=' ', dtype=str)

    summary_matrix = pd.DataFrame(np.zeros([param_data.shape[1], param_data.shape[1]]), columns=param_data.columns,
                                  index=param_data.columns)
    for i in range(three_col_format.shape[0]):
        c = three_col_format[i, 0]
        e = three_col_format[i, 1]
        summary_matrix[e].loc[c] = 1

    print("\nSummary adjacency matrix:")
    print(summary_matrix)

    # 5. Plot and save the causal graph using the modified function
    graph_filename = os.path.join("../data/Antivirus_Activity/causal_graph_from_structure.png")
    plot_general_causal_graph_indexed(summary_matrix, filename=graph_filename)

    summary_matrix_np = summary_matrix.values
    summary_matrix_npy_filename = os.path.join("../data/Antivirus_Activity/summary_matrix.npy")
    try:
        np.save(summary_matrix_npy_filename, summary_matrix_np)
        print(f"\nSummary matrix (as NumPy array) saved to '{summary_matrix_npy_filename}'")
    except Exception as e:
        print(f"Error saving summary matrix as .npy: {e}")

    
    print("\nSummary adjacency matrix (as NumPy array):")
    print(summary_matrix_np)