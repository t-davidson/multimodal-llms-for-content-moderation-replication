# Gemini experiments

This directory contains Python scripts used to run the experiments with Gemini using the Google Developer API. Each script corresponds to a separate experiment.

These scripts do not need to be run to reproduce the findings. They only need to be run if you intend to repeat the experiments using the API.

## Anonymization and required credentials
The code has been anonymized to remove the API key. If you intend to repeat any experiments, replace the key in `key.yaml` with an activate API key.

Unlike the OpenAI experiments, where the posts are stored on an AWS server, these scripts entail uploading local copies of the images to the API.

The cost to run each script will vary depending on the current API pricing.