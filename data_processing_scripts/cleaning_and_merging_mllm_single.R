library(tidyverse)

### LOADING DATA
data_amarel <- read_csv("../mllm_experiments/openweights/results/merged_single_results.csv")
data_gpt <- read_csv("../mllm_experiments/gpt4o/results/gpt4o-experiments-results-single.csv")
data_gemini <- read_csv("../mllm_experiments/gemini/results/collated_single_results.csv")

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

# Model = GPT-4o unless "mini" in source_file
data_gpt <- data_gpt %>% mutate(model = ifelse(str_detect(source_file, "mini"), "GPT-4o mini", "GPT-4o"))

# Model = Gemini 2.5 Flash unless "Lite" in model
data_gemini <- data_gemini %>% mutate(model = ifelse(str_detect(model, "lite"), "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash"))

# Standardizing
data_amarel <- data_amarel %>% select(id=index, post_id = Image, chosen = Output, model)
data_gpt$id <- 1:dim(data_gpt)[1]
data_gpt <- data_gpt %>% select(id, post_id = image_number, chosen = content, model)
data_gemini$id <- 1:dim(data_gemini)[1]
data_gemini <- data_gemini %>% select(id, post_id = Image, chosen = Output, model)

data_single <- bind_rows(data_amarel, data_gemini, data_gpt)


# Loading posts information table
post_info <- read_csv("../posts_generation_scripts_and_data/post-information-main.tsv")

post_info_2 <- post_info %>% select(post_id = ...1, interaction, identity, context, engagement, reply, curse, slur, user_id, scenario_id, engagement_val)


# Processing
data_single <- data_single %>% mutate(chosen = ifelse(chosen == "Yes", 1,
                                                       ifelse(chosen == "No", 0, NA)))

data_single_final <- data_single %>% left_join(post_info_2, by = "post_id")

write_csv(data_single_final, "../replication_data/final_mllm_results_single.csv")
