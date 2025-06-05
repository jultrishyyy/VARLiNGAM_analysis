import numpy as np
import pandas as pd
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from helper.helper_methods import get_acf_ccf_ratio_over_lags
import lingam

DATA_PATH = os.path.join(ROOT_DIR, "data", "Middleware_oriented_message_Activity")
OUTPUT_PATH = os.path.join(ROOT_DIR, "result", "Middleware_oriented_message_Activity")
data_filename = DATA_PATH + '/monitoring_metrics_1.csv'

if __name__ == "__main__":
    
    X = pd.read_csv(data_filename, delimiter=',', index_col=0, header=0)
    X = X.to_numpy()

    raw_acf_ccf_ratio = get_acf_ccf_ratio_over_lags(X, max_lag=10)



    



