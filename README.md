# VARLiNGAM Testing and Usage Guide

This repository provides tools and scripts for testing and using the VARLiNGAM algorithm. Below, you'll find instructions for running tests, using your own data, and understanding the folder structure and provided utilities.

## 1. Getting Started: Testing VARLiNGAM

To quickly test if VARLiNGAM works on your device, navigate to the root directory of this repository, where a `test.py` script is provided. Sample data and a corresponding summary matrix are included for testing and evaluating VARLiNGAM.

### 1.1 Running the Default Test
Run the following command to execute the test with the provided sample data:
```bash
python test.py
```

### 1.2 Testing with Custom Data
To test VARLiNGAM with your own dataset and labels, use:
```bash
python test.py --data YOUR_DATA_PATH --label YOUR_LABEL_PATH
```
- `YOUR_DATA_PATH`: Path to your dataset.
- `YOUR_LABEL_PATH`: Path to the summary matrix corresponding to your dataset. The matrix must have the same number of variables (rows) as your dataset.

### 1.3 Running Without Evaluation
If you want to run VARLiNGAM without evaluating results (e.g., if no labels are available), set the `evaluate` flag to `0` (default is `1` for evaluation):
```bash
python test.py --data YOUR_DATA_PATH --evaluate 0
```

## 2. Analyzing VARLiNGAM

This section analyzes VARLiNGAM from the following aspects:

* **Time Effect Removal by VAR**: We assess how well VAR removes time effects by comparing the Autocorrelation Function (ACF) and Cross-Correlation Function (CCF) of the original dataset with those of the corresponding residuals after VAR.
* **Performance Comparison (VAR, DirectLiNGAM, VARLiNGAM)**: We evaluate the performance using metrics such as:
    * F1 score
    * Number of Correctly Predicted Edges (True Positives) in the estimated summary matrix
    * Number of Incorrectly Predicted Edges (False Positives)
    * Number of Correct Edges Not Predicted (False Negatives)
    * Number of Wrongly Ordered Cause-Effect Pairs in the obtained causal order (specifically for DirectLiNGAM and VARLiNGAM)
* **Impact of Time Series Order**: We investigate the significance of time series order by shuffling the original data and comparing VARLiNGAM's performance on both the original and shuffled datasets.

---

### Running the Analysis

To generate the results, execute the following scripts:

* **For VAR and ACF/CCF results:**
    ```bash
    python .\analyze\analyze_VAR.py
    ```
* **For DirectLiNGAM results:**
    ```bash
    python .\analyze\analyze_DirctLiNGAM.py
    ```
* **For VARLiNGAM results:**
    ```bash
    python .\analyze\analyze_VARLiNGAM.py
    ```

A portion of the results can be found in the `result` folder.

## 3. Folder Structure

### 3.1 `analyze/`
This folder contains main execution scripts for testing diffeent algorithms.

- **`analyze_VAR.py`**: The script for testing standardalone VAR method. 
- **`analyze_DirectLiNGAM.py`**: The script for testing standardalone Direct LiNGAM method. 
- **`analyze_VARLiNGAM.py`**: The script for testing VARLiNGAM method. 

### 3.2 `data/`
This folder contains datasets for testing VARLiNGAM. Each dataset is stored in a separate subfolder, accompanied by its ground truth data, such as summary matrix.

Among, Antivirus_Activity, Middleware_oriented_message_Activity, Storm_Ingestion_Activity, and Web_Activity are from the IT monitoring data. (Link: https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series)

Flood, Bavaria, East_germany are from CausalRiver datasets. (Link: https://github.com/CausalRivers/causalrivers)

### 3.3 `generate_data/`
This folder includes scripts for generating synthetic datasets and summary matrix for testing and analysis.

- **`generate_synthetic_data.py`**: Generates synthetic datasets that either align with or violate VARLiNGAM assumptions (e.g., non-linearity, cyclic relationships, Gaussian noise) with a lag of 1. Generated datasets, along with ground truth causal order, summary matrix, and summary graph, are saved in `../data/varlingam/`.
- **`generate_IT_summary_matrix.py`**: Generates summary matrix for IT monitoring datasets.
- **`generate_causalriver_summary_matrix.py`**: Generates summary matrix for CausalRiver datasets.
- **`process_causalriver.py`**: Preprocess CausalRiver datasets, including handling NAN data and sample the original dataset with some time interval.
- **`shuffle.py`**: Shuffles time series datasets for preprocessing or testing purposes.

### 3.4 `helper/`
This folder contains utility scripts to support VARLiNGAM analysis operations.

- **`helper_methods.py`**:
This contains several helper methods:
  - `convert_Btaus_to_summary_matrix(Btaus)`: Converts a `B_taus` matrix into a summary matrix.
  - `plot_summary_causal_graph(matrix, filepath)`: Takes a matrix (or matrices) and a file path as input, constructs a causal graph, and saves it as a PNG file at the specified path.
  - `prune_summary_matrix_with_best_f1_threshold`: prune the estimated summary matrix with the threshold that give the best F1 score.
  - `save_results_and_metrics`: save all results to specified path, including estimated summary matrix, F1 score, TP, NP, TF, NF, and number of wrongly ordered cause-effect pairs.

### 3.5 `lingam/`
This folder implements the complete VARLiNGAM algorithm, including the implementation of DirectLiNGAM. It is from the github repository: https://github.com/cdt15/lingam.

### 3.6 `result/`
This is where the algorithm results are saved. Under each sub-folder for each dataset, there are three files:

- **`VAR_result.txt`**: The result of testing standardalone VAR method. 
- **`DirectLiNGAM_result.txt`**: The result of testing standardalone Direct LiNGAM method. 
- **`VARLiNGAM_result.txt`**: The result of testing VARLiNGAM method. 

## 3. Notes
- Ensure that your dataset and summary matrix are compatible (i.e., the summary matrix must have the same number of variables as the dataset).
- For additional details or issues, please refer to the repository's issue tracker or contact the maintainers.
- For the pruning process, you could also use other methods like fixed threshold pruning. When we tested this, our provided pruning method resulted in a very small prune threshold for the `Antivirus_Activity` dataset, leading to all edges being predicted in the final causal structure. Therefore, we set the prune threshold to 0.1 specifically for this dataset.
- The `CausalRiverBavaria` and `CausalRiverEastGermany` datasets are too large to upload here. Please refer to their original GitHub repository mentioned above to access the data if needed.

