import pandas as pd
import glob
import re
# Find all relevant CSV files
files = [f for f in glob.glob("*.csv") 
         if all(x not in f for x in ["collated", "single"])]
# Function to extract model and prompt from file name
def extract_metadata(filename):
    parts = filename.split("_results_")
    prompt = parts[0]
    model_match = re.search(r"gemini(_\d+_\d+)(?:_(flash(?:_lite)?)?)?", parts[1])
    model = "gemini" + model_match.group(1)
    if model_match.group(2):
        model += "_" + model_match.group(2)

    return prompt, model
# Collect processed DataFrames
df_list = []
for file in files:
    df = pd.read_csv(file)
    df = df[df['Output'].isin(["A", "B"])].copy()
    prompt, model = extract_metadata(file)
    df['source_file'] = file
    df['content']     = df['Output']
    df['a_images']    = df['Image A']
    df['b_images']    = df['Image B']
    df['model']       = model
    df['prompt']      = prompt
    df = df[['source_file','content','a_images','b_images','model','prompt']]
    df_list.append(df)
# Combine and export
final_df = pd.concat(df_list, ignore_index=True)
final_df.to_csv("collated_results.csv", index=False)
print("Finished writing collated_results.csv")