import os
import re
import glob
import pandas as pd

# Only CSVs with "single" in the name
files = sorted(glob.glob("*single*.csv"))

def extract_meta(filename: str):
    base = os.path.basename(filename)
    prompt, rest = base.split("_results_", 1)
    model = re.sub(r"\.csv$", "", rest)
    return prompt, model

df_list = []
for f in files:
    df = pd.read_csv(f)
    # Expecting columns: Index, Image, Output
    if not {"Image", "Output"}.issubset(df.columns):
        continue
    prompt, model = extract_meta(f)
    out = df[["Image", "Output"]].copy()
    out["source_file"] = f
    out["model"] = model
    out["prompt"] = prompt
    df_list.append(out[["source_file", "Image", "Output", "model", "prompt"]])

if df_list:
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv("collated_single_results.csv", index=False)
    print("Finished writing collated_single_results.csv")
else:
    print("No matching 'single' CSV files found.")