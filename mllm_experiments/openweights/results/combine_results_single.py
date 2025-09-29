import pandas as pd
import glob

folder_path = "."
csv_files = [f for f in glob.glob(f"{folder_path}/*.csv") if "single" in f and "merged" not in f]

dataframes = []
for file in csv_files:
    dataset_name = file.split("/")[-1].replace("single_results_", "").replace(".csv", "")
    df = pd.read_csv(file)
    df.rename(columns={"Index": "task_index"}, inplace=True)
    df['dataset'] = dataset_name
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
merged_df.insert(0, "index", range(1, len(merged_df) + 1))
merged_df.to_csv(f"{folder_path}/merged_single_results.csv", index=False)
print(f"Merged {len(csv_files)} single result files with {len(merged_df)} total rows")