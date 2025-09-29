import pandas as pd
import glob
# Define the folder containing the CSV files
folder_path = "."  # Current directory
# List all CSV files in the folder, excluding files with "merged" or "single" in the name
csv_files = [f for f in glob.glob(f"{folder_path}/*.csv") if "merged" not in f and "single" not in f]
# Initialize an empty list to store DataFrames
dataframes = []
# Process each file
for file in csv_files:
    # Extract the dataset name from the file name
    dataset_name = file.split("/")[-1].replace(".csv", "")
    # Read the CSV file
    df = pd.read_csv(file,index_col=0)
    # Rename the original Index column to task_index
    df.rename(columns={"Index": "task_index"}, inplace=True)
    # Add a column for the dataset name
    df['dataset'] = dataset_name
    # Append the DataFrame to the list
    dataframes.append(df)
# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)
# Add an overall index column
merged_df.insert(0, "index", range(1, len(merged_df) + 1))
# Save the merged DataFrame to a CSV file
output_path = f"{folder_path}/merged_results.csv"
merged_df.to_csv(output_path, index=False)
print(f"Merged CSV saved to: {output_path}")