import pandas as pd
import io

# Read the data, setting the first column as the index
df = pd.read_csv("../causalriver/rivers_ts_flood_preprocessed.csv", index_col=0)
# df = pd.read_csv("../causalriver/rivers_ts_flood.csv")

print("--- Original Data ---")
print(df)


# 2. Shuffle the DataFrame rows
# frac=1 means sample 100% of the rows, which shuffles the DataFrame.
# random_state ensures the shuffle is reproducible. You can remove it for a different shuffle each time.
shuffled_df = df.sample(frac=1, random_state=42)


print("\n--- Shuffled Data ---")
print(shuffled_df)


# 3. (Optional) Save the shuffled data to a new CSV file
output_filename = '../causalriver/rivers_ts_flood_shuffled.csv'
shuffled_df.to_csv(output_filename, index=True)
# shuffled_df.to_csv(output_filename)

print(f"\nShuffled data has been saved to '{output_filename}'")