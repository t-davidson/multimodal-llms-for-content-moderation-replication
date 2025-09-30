library(tidyverse)

data_amarel <- read_csv("../mllm_experiments/openweights/results/merged_results.csv")
data_gpt <- read_csv("../mllm_experiments/gpt4o/results/gpt4o-experiments-results-alt.csv")
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

# Filter to only alt experiments
data_amarel <- data_amarel %>% filter(prompt == "alt")
data_gemini <- data_gemini %>% filter(prompt == "alt")

# Load reference data for mapping images to paths
# This is necessary because the posts containing original and alternative
# slurs were located in separate directories.
# This ensures that each is matched to the relevant post information below
image_indices_ref <- read_csv("../mllm_experiments/gpt4o/image_indices_alt_99750.csv")

# Due to differences in the way the data were recorded (GPT-4o used images on an
# AWS server and others used local data). Each dataset is processed differently.

### PROCESSING GPT DATA (extract post IDs from paths)
data_gpt <- data_gpt %>%
    mutate(
        data_A = ifelse(str_detect(a_paths, "output_alt"), "Alt", "Main"),
        data_B = ifelse(str_detect(b_paths, "output_alt"), "Alt", "Main"),
        # Extract post IDs from after "tweet" in the paths
        post_A = str_extract(a_paths, "(?<=tweet)\\d+"),
        post_B = str_extract(b_paths, "(?<=tweet)\\d+")
    ) %>%
    select(treatment = source_file, choice = content, post_A, post_B, data_A, data_B, model, prompt) %>%
    mutate(
        post_A = as.character(post_A),
        post_B = as.character(post_B)
    )

### PROCESSING AMAREL DATA (use Image columns directly, join by image IDs)
data_amarel <- data_amarel %>%
    # Join by matching image IDs to get path information
    left_join(image_indices_ref, by = c("Image A" = "a_images", "Image B" = "b_images")) %>%
    # Extract the actual post IDs from the paths (not from Image columns)
    mutate(
        post_A = str_extract(a_paths, "(?<=tweet)\\d+"),
        post_B = str_extract(b_paths, "(?<=tweet)\\d+"),
        data_A = ifelse(str_detect(a_paths, "output_alt"), "Alt", "Main"),
        data_B = ifelse(str_detect(b_paths, "output_alt"), "Alt", "Main")
    ) %>%
    select(treatment = dataset, choice = Output, post_A, post_B, data_A, data_B, model, prompt)


### PROCESSING GEMINI DATA (use a_images/b_images columns, join by image IDs)
data_gemini <- data_gemini %>%
    # Join by matching image IDs to get path information
    left_join(image_indices_ref, by = c("a_images", "b_images")) %>%
    # Extract the actual post IDs from the paths (not from a_images/b_images columns)
    mutate(
        post_A = str_extract(a_paths, "(?<=tweet)\\d+"),
        post_B = str_extract(b_paths, "(?<=tweet)\\d+"),
        data_A = ifelse(str_detect(a_paths, "output_alt"), "Alt", "Main"),
        data_B = ifelse(str_detect(b_paths, "output_alt"), "Alt", "Main")
    ) %>%
    select(treatment = source_file, choice = content, post_A, post_B, data_A, data_B, model, prompt)

# Combine all datasets
data <- bind_rows(data_gpt, data_amarel, data_gemini)
data$id <- 1:dim(data)[1] # Unique row ID

# Transforming data format
# We need to handle both post_A/post_B and data_A/data_B columns together
new_data <- data %>%
    pivot_longer(cols = starts_with("post_"),
                 names_to = "post",
                 values_to = "post_id",
                 names_prefix = "post_") %>%
    # Add corresponding data_A/data_B information
    mutate(
        data_type = case_when(
            post == "A" ~ data_A,
            post == "B" ~ data_B,
            TRUE ~ NA_character_
        ),
        chosen = ifelse(post == choice, 1, 0)
    ) %>%
    arrange(id, post)

new_data$post_id <- as.numeric(new_data$post_id)

# Loading posts information table (original)
post_info_main <- read_csv("../posts_generation_scripts_and_data/post-information-main.tsv")
post_info_main <- post_info_main %>% select(post_id = ...1, interaction, identity, context, engagement, reply, curse, slur, user_id, scenario_id, engagement_val)

# Loading posts information table (alternative)
post_info_alt<- read_csv("../posts_generation_scripts_and_data/post-information-alternative.tsv")
post_info_alt <- post_info_alt %>% select(post_id = ...1, interaction, identity, context, engagement, reply, curse, slur, user_id, scenario_id, engagement_val)

# Merging post table with conjoint info based on data_type
main_posts <- new_data %>%
    filter(data_type == "Main") %>%
    left_join(post_info_main, by = "post_id")

alt_posts <- new_data %>%
    filter(data_type == "Alt") %>%
    left_join(post_info_alt, by = "post_id")

posts_final <- bind_rows(main_posts, alt_posts)

write_csv(posts_final, "../replication_data/final_mllm_results_alt.csv")
