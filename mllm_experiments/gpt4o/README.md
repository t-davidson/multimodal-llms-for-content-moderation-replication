# GPT-4o experiments

This directory contains Jupyter notebooks used to run the experiments with GPT-4o using the OpenAI Batch API. Each notebook corresponds to a separate experiment.

These scripts do not need to be run to reproduce the findings. They only need to be run if you intend to repeat the experiments using the OpenAI API.

## Anonymization and required credentials
The code has been anonymized to remove the API key, references to specific objects created by the API, and the URL for the AWS server containing the posts.

In order to replicate the code, it is necessary to setup an AWS S3 server with a public address pointing to a directory containing the posts. It is also necessary to register for an OpenAI API key.

The cost to run each script will vary depending on the current OpenAI API pricing.

## Batching and errors
A single batch request has a maximum of 50,000 items so some queries are split into multiple batches. Additional batches are also ran to repeat any instances where the original batches failed. This appears to happen in a small number of cases in each batch. The final results are then all merged together into JSONL objects that are loaded and combined into final tables using the `parse-batch-results*` scripts.