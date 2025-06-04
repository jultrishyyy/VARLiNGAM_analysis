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

## 2. Folder Structure

### 2.1 `data/`
This folder contains datasets for testing VARLiNGAM. Each dataset is stored in a separate subfolder, accompanied by its ground truth data, such as summary matrices.

### 2.2 `generate_data/`
This folder includes scripts for generating synthetic datasets and summary matrices for testing and analysis.

- **`generate_data.py`**: Generates synthetic datasets that either align with or violate VARLiNGAM assumptions (e.g., non-linearity, cyclic relationships, Gaussian noise) with a lag of 2. Generated datasets, along with ground truth causal order, summary matrix, and summary graph, are saved in `../data/varlingam/`.
- **`generate_IT_summary_matrix.py`**: Generates summary matrices for IT monitoring datasets.
- **`generate_causalriver_summary_matrix.py`**: Generates summary matrices for CausalRiver datasets.
- **`shuffle.py`**: Shuffles time series datasets for preprocessing or testing purposes.

### 2.3 `utils/`
This folder contains utility scripts to support VARLiNGAM operations.

- **`helper.py`**:
  - `convert_Btaus_to_summary_matrix(Btaus)`: Converts a `B_taus` matrix into a summary matrix.
  - `plot_summary_causal_graph(matrix, filepath)`: Takes a matrix (or matrices) and a file path as input, constructs a causal graph, and saves it as a PNG file at the specified path.

## 3. Notes
- Ensure that your dataset and summary matrix are compatible (i.e., the summary matrix must have the same number of variables as the dataset).
- For additional details or issues, please refer to the repository's issue tracker or contact the maintainers.

