# Merging final data

The scripts in this folder are used to combine the results from the MLLM experiments into the final datasets used in the analyses.

Each script loads the results from all models and merges them with the covariates related to each post. A final dataset is then stored in the `../replication_data` directory.

## Scripts

- `cleaning_and_merging_mllm.R` compiles the dataset for the main results, including the prompt variations and cue modality tests.
- `cleaning_and_merging_mllm_alt.R` compiles the dataset for the robustness check using alternative slurs.	
- `cleaning_and_merging_mllm_single.R` compiles the dataset for the robustness check using a single task conjoint design.

### Note

These scripts require that all files are present. Some files are hosted by GitHub LFS and must be downloaded correctly before it can be run. 