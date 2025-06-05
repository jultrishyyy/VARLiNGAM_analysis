import pandas as pd
import io
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Modify the paths to your input and output files
DATA_PATH = os.path.join(ROOT_DIR, "data", "Storm_Ingestion_Activity")
input_filename = DATA_PATH + '/storm_data_normal.csv'
output_filename = DATA_PATH + '/shuffled_storm_data_normal.csv'


if __name__ == "__main__":
    
    print(f"\nInput file: {input_filename}")

    # Read the data, setting the first column as the index
    df = pd.read_csv(input_filename, delimiter=',', index_col=0, header=0)

    print("\n--- Original Data ---")
    print(df)


    # Shuffle the DataFrame rows
    # frac=1 means sample 100% of the rows, which shuffles the DataFrame.
    # random_state ensures the shuffle is reproducible. You can remove it for a different shuffle each time.
    shuffled_df = df.sample(frac=1, random_state=42)


    print("\n--- Shuffled Data ---")
    print(shuffled_df)


    # Save the shuffled data to a new CSV file

    shuffled_df.to_csv(output_filename, index=True)

    print(f"\nShuffled data has been saved to '{output_filename}'")