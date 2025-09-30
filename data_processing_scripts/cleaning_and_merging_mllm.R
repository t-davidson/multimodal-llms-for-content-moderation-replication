library(tidyverse)

data_amarel <- read_csv("../mllm_experiments/openweights/results/merged_results.csv")
data_gpt <- read_csv("../mllm_experiments/gpt4o/results/gpt4o-experiments-results-main.csv")
data_gemini <- read_csv("../mllm_experiments/gemini/results/collated_results.csv")


# Standardizing some variations in model naming
data_amarel <- data_amarel %>%
    mutate(
        dataset_clean = dataset %>%
            str_replace("qwen_(\\d+)", "qwen\\1") %>%   # qwen_72 to qwen72
            str_replace("qwen($|[^0-9])", "qwen7\\1"),
        prompt = str_extract(dataset_clean, "^[^_]+"),  # baseline, debiasing, etc
        model  = dataset_clean %>%
            str_remove("^[^_]+_results_") %>%           # gemma3_27b to
            str_replace("_", "-")                       # gemma3-27b
    )

# Ignoring alt (processed in another script)
data_amarel <- data_amarel %>% filter(prompt != "alt")
data_gemini <- data_gemini %>% filter(prompt != "alt")

# Aligning column names and selecting those required for analysis
data_gpt <- data_gpt %>% select(treatment = source_file, model, prompt, choice = content, post_A = a_images, post_B = b_images)
data_gemini <- data_gemini %>% select(treatment = source_file, model, prompt, choice = content, post_A = a_images, post_B = b_images)
data_amarel <- data_amarel %>% select(treatment = dataset, model, prompt, choice = Output, post_A = "Image A", post_B = "Image B")

# Combining datasets and generating unique row ID
data <- bind_rows(data_gpt, data_gemini, data_amarel)
data$id <- 1:dim(data)[1]


# Transforming from rows representing pairs to rows representing posts
new_data <- data %>%
    pivot_longer(cols = starts_with("post_"),
                 names_to = "post",
                 values_to = "post_id",
                 names_prefix = "post_") %>%
    mutate(
        chosen = ifelse(post == choice, 1, 0) # Outcome
    ) %>%
    arrange(id, post)

new_data_selected <- new_data %>% select(id, post_id, chosen, treatment, model, prompt)

# Loading posts information table
post_info <- read_csv("../posts_generation_scripts_and_data/post-information-main.tsv")
post_info_2 <- post_info %>% select(post_id = ...1, interaction, identity, context, engagement, reply, curse, slur, user_id, scenario_id, engagement_val)

# Merging post table with conjoint info
posts_final <- new_data_selected %>% left_join(post_info_2, by = "post_id")

# Storing final analysis dataset
write_csv(posts_final, "../replication_data/final_mllm_results.csv")
