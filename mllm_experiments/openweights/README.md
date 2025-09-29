# Open-weights models experiments

This directory contains Python scripts used to run the open-weights models.

These scripts do not need to be run to reproduce the findings. They only need to be run if you intend to repeat the experiments using a cluster equipped with GPUs.

## GPU resources
These scripts are intended to be run on a high-performance computing cluster equipped with modern graphical processing units (GPUs). The smaller models can be run on a single GPU but most of the larger models require 2-4 GPUs. The analyses were run on the Rutgers University Amarel Cluster, which is equipped with 4 x A100 and 4 x L40S NVIDIA GPUs. 

## Credentials

Some models require authorization using a Huggingface user access token. This can be obtained using the [this website](https://huggingface.co/docs/hub/en/security-tokens) and added to the environment before scripts are run.

## Results
The `results` directory contains CSVs containing the results from each script that were downloaded after the tasks were completed on the server. 

The two Python scripts collate the results, one for the single task replication and one for the other results. The final files are `merged_results.csv` and `merged_single_results.csv`.