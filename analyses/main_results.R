library(cregg)
library(tidyverse)
library(ggplot2)
library(stringr)
library(scico)
library(forcats)
library(patchwork)
library(cowplot)
library(grid)
library(ggh4x)

set.seed(1485008901) # Seed used for published results

# Loading functions to compute bootstrap statistics
source("bootstrap_conjoint_functions.R")

data_machine <- read_csv("../replication_data/final_mllm_results.csv")

data_human <- read_csv("../replication_data/final_human_results.csv") %>% filter(task != 16) # dropping repeat task

# Process machine to distinguish between models and prompts
data_machine <- data_machine %>%
    mutate(
        # Create 'prompt' variable from model column instead of treatment
        prompt = case_when(
            str_detect(prompt, "baseline") ~ "baseline",
            str_detect(prompt, "uniform") ~ "uniform",
            str_detect(prompt, "faceless") ~ "faceless",
            str_detect(prompt, "nameless") ~ "nameless",
            str_detect(prompt, "debiasing|debiased") ~ "context-sensitive",
            TRUE ~ NA_character_
        ),
        prompt = factor(prompt, levels = c("baseline", "uniform", "faceless", "nameless", "context-sensitive")),

        # Recode 'model' to more informative labels
        model = case_when(
            model == "base"     ~ "GPT-4o",
            model == "mini"     ~ "GPT-4o-mini",
            model == "gemini_2_5_flash" ~ "Gemini 2.5 Flash",
            model == "gemini_2_5_flash_lite" ~ "Gemini 2.5 Flash Lite",
            model == "qwen7"    ~ "Qwen2-7B",
            model == "qwen72"   ~ "Qwen2-72B",
            model == "qwen2"    ~ "Qwen2-2B",
            model == "gemma3-4b" ~ "Gemma3-4B",
            model == "gemma3-12b" ~ "Gemma3-12B",
            model == "gemma3-27b" ~ "Gemma3-27B",
            model == "internvl3-2b" ~ "InternVL3-2B",
            model == "internvl3-14b" ~ "InternVL3-14B",
            model == "internvl3-78b" ~ "InternVL3-78B",
            TRUE ~ NA_character_
        ),
        model = factor(model, levels = c("GPT-4o-mini", "GPT-4o",
                                         "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash",
                                         "Qwen2-7B", "Qwen2-72B", "Qwen2-2B",
                                         "Gemma3-4B", "Gemma3-12B", "Gemma3-27B",
                                         "InternVL3-2B", "InternVL3-14B", "InternVL3-78B"))
    )


# Model-prompt counts (verify each has 60k rows)
model_prompt_counts <- data_machine %>%
    count(model, prompt) %>%
    arrange(model, prompt)

# Recoding slurs to avoid using directly in plots
data_machine <- data_machine %>% mutate(slur = replace_na(slur, "No slur")) %>%
    mutate(slur = as.factor(slur))

data_machine$slur <- fct_recode(data_machine$slur, "Racism" = "nigger",
                        "Reclaimed\nslur" = "nigga",
                        "Reverse\nracism" = "cracker",
                        "Homophobia" = "faggot",
                        "Sexism" = "bitch",
                        "Generic\ninsult" = "asshole")

data_human <- data_human %>% mutate(slur = replace_na(slur, "No slur")) %>%
    mutate(slur = as.factor(slur))

data_human$slur <- fct_recode(data_human$slur, "Racism" = "nigger",
                                "Reclaimed\nslur" = "nigga",
                                "Reverse\nracism" = "cracker",
                                "Homophobia" = "faggot",
                                "Sexism" = "bitch",
                                "Generic\ninsult" = "asshole")

# Recoding values
data_machine <- data_machine %>%
    rename(topic = context) %>%
    mutate(interaction = recode(interaction, "H" = "Slur", "B" = "No Slur"),
           identity = recode(identity, "BM" = "Black man", "BF" = "Black woman", "WM" = "White man", "WF" = "White woman", "A" = "Anonymous"),
           topic = str_to_title(topic),
           curse = ifelse(curse, "Curse", "No Curse"))

data_human <- data_human %>%
    rename(topic = context) %>%
    mutate(interaction = recode(interaction, "H" = "Slur", "B" = "No Slur"),
           identity = recode(identity, "BM" = "Black man", "BF" = "Black woman", "WM" = "White man", "WF" = "White woman", "A" = "Anonymous"),
           topic = str_to_title(topic),
           curse = ifelse(curse, "Curse", "No Curse"))

# Transforming variables to factors for analysis
data_machine <- data_machine %>% mutate(
    interaction = as.factor(interaction),
    identity = as.factor(identity),
    topic = as.factor(topic),
    engagement = as.factor(ifelse(engagement == "L", "Low", "High")),
    reply = as.factor(reply),
    slur = as.factor(slur),
    curse = as.factor(curse)
) %>% select(id, chosen, interaction, identity, topic, engagement, reply, curse, slur, model, prompt)

data_human <- data_human %>% mutate(
    interaction = as.factor(interaction),
    identity = as.factor(identity),
    topic = as.factor(topic),
    engagement = as.factor(ifelse(engagement == "L", "Low", "High")),
    reply = as.factor(reply),
    slur = as.factor(slur),
    curse = as.factor(curse)
) %>% select(id, chosen, interaction, identity, topic, engagement, reply, curse, slur)

# Fixing reference categories
data_machine$topic <- relevel(data_machine$topic, ref = "Everyday")
data_machine$reply <- relevel(data_machine$reply, ref = "None")
data_machine$engagement <- relevel(data_machine$engagement, ref = "Low")
data_machine$curse <- relevel(data_machine$curse, ref = "No Curse")

data_human$topic <- relevel(data_human$topic, ref = "Everyday")
data_human$reply <- relevel(data_human$reply, ref = "None")
data_human$engagement <- relevel(data_human$engagement, ref = "Low")
data_human$curse <- relevel(data_human$curse, ref = "No Curse")

# Splitting data_machine by model and prompt
# GPT-4o
data_gpt_baseline <- data_machine %>% filter(model == "GPT-4o", prompt == "baseline")
data_gpt_contextaware <- data_machine %>% filter(model == "GPT-4o", prompt == "context-sensitive")
data_gpt_uniform <- data_machine %>% filter(model == "GPT-4o", prompt == "uniform")
data_gpt_faceless <- data_machine %>% filter(model == "GPT-4o", prompt == "faceless")
data_gpt_nameless <- data_machine %>% filter(model == "GPT-4o", prompt == "nameless")

# GPT-4o-mini
data_gptmini_baseline <- data_machine %>% filter(model == "GPT-4o-mini", prompt == "baseline")
data_gptmini_contextaware <- data_machine %>% filter(model == "GPT-4o-mini", prompt == "context-sensitive")
data_gptmini_uniform <- data_machine %>% filter(model == "GPT-4o-mini", prompt == "uniform")
data_gptmini_faceless <- data_machine %>% filter(model == "GPT-4o-mini", prompt == "faceless")
data_gptmini_nameless <- data_machine %>% filter(model == "GPT-4o-mini", prompt == "nameless")


# Gemini 2.5 Flash
data_flash_baseline <- data_machine %>% filter(model == "Gemini 2.5 Flash", prompt == "baseline")
data_flash_contextaware <- data_machine %>% filter(model == "Gemini 2.5 Flash", prompt == "context-sensitive")
data_flash_uniform <- data_machine %>% filter(model == "Gemini 2.5 Flash", prompt == "uniform")
data_flash_faceless <- data_machine %>% filter(model == "Gemini 2.5 Flash", prompt == "faceless")
data_flash_nameless <- data_machine %>% filter(model == "Gemini 2.5 Flash", prompt == "nameless")

# Gemini 2.5 Flash Lite
data_flashlite_baseline <- data_machine %>% filter(model == "Gemini 2.5 Flash Lite", prompt == "baseline")
data_flashlite_contextaware <- data_machine %>% filter(model == "Gemini 2.5 Flash Lite", prompt == "context-sensitive")
data_flashlite_uniform <- data_machine %>% filter(model == "Gemini 2.5 Flash Lite", prompt == "uniform")
data_flashlite_faceless <- data_machine %>% filter(model == "Gemini 2.5 Flash Lite", prompt == "faceless")
data_flashlite_nameless <- data_machine %>% filter(model == "Gemini 2.5 Flash Lite", prompt == "nameless")

# Qwen2-7B
data_qwen7_baseline <- data_machine %>% filter(model == "Qwen2-7B", prompt == "baseline")
data_qwen7_contextaware <- data_machine %>% filter(model == "Qwen2-7B", prompt == "context-sensitive")
data_qwen7_uniform <- data_machine %>% filter(model == "Qwen2-7B", prompt == "uniform")
data_qwen7_faceless <- data_machine %>% filter(model == "Qwen2-7B", prompt == "faceless")
data_qwen7_nameless <- data_machine %>% filter(model == "Qwen2-7B", prompt == "nameless")

# Qwen2-72B
data_qwen72_baseline <- data_machine %>% filter(model == "Qwen2-72B", prompt == "baseline")
data_qwen72_contextaware <- data_machine %>% filter(model == "Qwen2-72B", prompt == "context-sensitive")
data_qwen72_uniform <- data_machine %>% filter(model == "Qwen2-72B", prompt == "uniform")
data_qwen72_faceless <- data_machine %>% filter(model == "Qwen2-72B", prompt == "faceless")
data_qwen72_nameless <- data_machine %>% filter(model == "Qwen2-72B", prompt == "nameless")

# Qwen2-2B
data_qwen2_baseline <- data_machine %>% filter(model == "Qwen2-2B", prompt == "baseline")
data_qwen2_contextaware <- data_machine %>% filter(model == "Qwen2-2B", prompt == "context-sensitive")
data_qwen2_uniform <- data_machine %>% filter(model == "Qwen2-2B", prompt == "uniform")
data_qwen2_faceless <- data_machine %>% filter(model == "Qwen2-2B", prompt == "faceless")
data_qwen2_nameless <- data_machine %>% filter(model == "Qwen2-2B", prompt == "nameless")

# Gemma3-4B
data_gemma4_baseline <- data_machine %>% filter(model == "Gemma3-4B", prompt == "baseline")
data_gemma4_contextaware <- data_machine %>% filter(model == "Gemma3-4B", prompt == "context-sensitive")
data_gemma4_uniform <- data_machine %>% filter(model == "Gemma3-4B", prompt == "uniform")
data_gemma4_faceless <- data_machine %>% filter(model == "Gemma3-4B", prompt == "faceless")
data_gemma4_nameless <- data_machine %>% filter(model == "Gemma3-4B", prompt == "nameless")

# Gemma3-12B
data_gemma12_baseline <- data_machine %>% filter(model == "Gemma3-12B", prompt == "baseline")
data_gemma12_contextaware <- data_machine %>% filter(model == "Gemma3-12B", prompt == "context-sensitive")
data_gemma12_uniform <- data_machine %>% filter(model == "Gemma3-12B", prompt == "uniform")
data_gemma12_faceless <- data_machine %>% filter(model == "Gemma3-12B", prompt == "faceless")
data_gemma12_nameless <- data_machine %>% filter(model == "Gemma3-12B", prompt == "nameless")

# Gemma3-27B
data_gemma27_baseline <- data_machine %>% filter(model == "Gemma3-27B", prompt == "baseline")
data_gemma27_contextaware <- data_machine %>% filter(model == "Gemma3-27B", prompt == "context-sensitive")
data_gemma27_uniform <- data_machine %>% filter(model == "Gemma3-27B", prompt == "uniform")
data_gemma27_faceless <- data_machine %>% filter(model == "Gemma3-27B", prompt == "faceless")
data_gemma27_nameless <- data_machine %>% filter(model == "Gemma3-27B", prompt == "nameless")

# InternVL3-2B
data_internvl2_baseline <- data_machine %>% filter(model == "InternVL3-2B", prompt == "baseline")
data_internvl2_contextaware <- data_machine %>% filter(model == "InternVL3-2B", prompt == "context-sensitive")
data_internvl2_uniform <- data_machine %>% filter(model == "InternVL3-2B", prompt == "uniform")
data_internvl2_faceless <- data_machine %>% filter(model == "InternVL3-2B", prompt == "faceless")
data_internvl2_nameless <- data_machine %>% filter(model == "InternVL3-2B", prompt == "nameless")

# InternVL3-14B
data_internvl14_baseline <- data_machine %>% filter(model == "InternVL3-14B", prompt == "baseline")
data_internvl14_contextaware <- data_machine %>% filter(model == "InternVL3-14B", prompt == "context-sensitive")
data_internvl14_uniform <- data_machine %>% filter(model == "InternVL3-14B", prompt == "uniform")
data_internvl14_faceless <- data_machine %>% filter(model == "InternVL3-14B", prompt == "faceless")
data_internvl14_nameless <- data_machine %>% filter(model == "InternVL3-14B", prompt == "nameless")

# InternVL3-78B
data_internvl78_baseline <- data_machine %>% filter(model == "InternVL3-78B", prompt == "baseline")
data_internvl78_contextaware <- data_machine %>% filter(model == "InternVL3-78B", prompt == "context-sensitive")
data_internvl78_uniform <- data_machine %>% filter(model == "InternVL3-78B", prompt == "uniform")
data_internvl78_faceless <- data_machine %>% filter(model == "InternVL3-78B", prompt == "faceless")
data_internvl78_nameless <- data_machine %>% filter(model == "InternVL3-78B", prompt == "nameless")



########################################################
# PART I: Main effects, baseline prompt only           #
########################################################

main_formula <- chosen ~ slur + identity + topic + reply + curse + engagement
amces_main_machine_b_gpt <- cj_bsp(data_gpt_baseline, main_formula)
amces_main_machine_b_gptmini <- cj_bsp(data_gptmini_baseline, main_formula)
amces_main_machine_b_gf <- cj_bsp(data_flash_baseline, main_formula)
amces_main_machine_b_gfl <- cj_bsp(data_flashlite_baseline, main_formula)
amces_main_machine_b_qwen2 <- cj_bsp(data_qwen2_baseline, main_formula)
amces_main_machine_b_qwen7 <- cj_bsp(data_qwen7_baseline, main_formula)
amces_main_machine_b_qwen72 <- cj_bsp(data_qwen72_baseline, main_formula)
amces_main_machine_b_gemma4 <- cj_bsp(data_gemma4_baseline, main_formula)
amces_main_machine_b_gemma12 <- cj_bsp(data_gemma12_baseline, main_formula)
amces_main_machine_b_gemma27 <- cj_bsp(data_gemma27_baseline, main_formula)
amces_main_machine_b_internvl2 <- cj_bsp(data_internvl2_baseline, main_formula)
amces_main_machine_b_internvl14 <- cj_bsp(data_internvl14_baseline, main_formula)
amces_main_machine_b_internvl78 <- cj_bsp(data_internvl78_baseline, main_formula)
amces_main_human <- cj(data_human, main_formula, id = ~id)


# Apply consistent feature formatting to all AMCE results
feature_levels <- c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")
format_features <- function(df) {
    df %>% mutate(
        feature = str_to_title(as.character(feature)),
        feature = factor(feature, levels = feature_levels)
    )
}

amces_main_machine_b_gpt <- format_features(amces_main_machine_b_gpt)
amces_main_machine_b_gptmini <- format_features(amces_main_machine_b_gptmini)
amces_main_machine_b_gf <- format_features(amces_main_machine_b_gf)
amces_main_machine_b_gfl <- format_features(amces_main_machine_b_gfl)
amces_main_machine_b_qwen2 <- format_features(amces_main_machine_b_qwen2)
amces_main_machine_b_qwen7 <- format_features(amces_main_machine_b_qwen7)
amces_main_machine_b_qwen72 <- format_features(amces_main_machine_b_qwen72)
amces_main_machine_b_gemma4 <- format_features(amces_main_machine_b_gemma4)
amces_main_machine_b_gemma12 <- format_features(amces_main_machine_b_gemma12)
amces_main_machine_b_gemma27 <- format_features(amces_main_machine_b_gemma27)
amces_main_machine_b_internvl2 <- format_features(amces_main_machine_b_internvl2)
amces_main_machine_b_internvl14 <- format_features(amces_main_machine_b_internvl14)
amces_main_machine_b_internvl78 <- format_features(amces_main_machine_b_internvl78)
amces_main_human <- format_features(amces_main_human)

amces_main_machine_b_gpt$out <- "GPT-4o"
amces_main_machine_b_gptmini$out <- "GPT-4o mini"
amces_main_machine_b_gf$out <- "Gemini 2.5 Flash"
amces_main_machine_b_gfl$out <- "Gemini 2.5 Flash Lite"
amces_main_machine_b_qwen2$out <- "Qwen2 2B"
amces_main_machine_b_qwen7$out <- "Qwen2 7B"
amces_main_machine_b_qwen72$out <- "Qwen2 72B"
amces_main_machine_b_gemma4$out <- "Gemma3 4B"
amces_main_machine_b_gemma12$out <- "Gemma3 12B"
amces_main_machine_b_gemma27$out <- "Gemma3 27B"
amces_main_machine_b_internvl2$out <- "InternVL3 2B"
amces_main_machine_b_internvl14$out <- "InternVL3 14B"
amces_main_machine_b_internvl78$out <- "InternVL3 78B"
amces_main_human$out <- "Human"

amces_main <- bind_rows(amces_main_human, amces_main_machine_b_gpt,
                        amces_main_machine_b_gptmini,
                        amces_main_machine_b_gf,
                        amces_main_machine_b_gfl,
                        amces_main_machine_b_qwen2,
                        amces_main_machine_b_qwen7,
                        amces_main_machine_b_qwen72,
                        amces_main_machine_b_gemma4,
                        amces_main_machine_b_gemma12,
                        amces_main_machine_b_gemma27,
                        amces_main_machine_b_internvl2,
                        amces_main_machine_b_internvl14,
                        amces_main_machine_b_internvl78)

# Figure 1 requires a lot of set-up to compose all elements, starting below

# Color scheme organized by model families
custom_colors <- c(
    # Human baseline
    "Human" = "#2C2C2C",

    # OpenAI GPT family
    "GPT-4o" = "#FF6B35",
    "GPT-4o mini" = "#FFB84D",

    # Google Gemini family
    "Gemini 2.5 Flash" = "#1A73E8",
    "Gemini 2.5 Flash Lite" = "#4285F4",

    # Qwen family
    "Qwen2 2B" = "#81C784",
    "Qwen2 7B" = "#4CAF50",
    "Qwen2 72B" = "#2E7D32",

    # Gemma family
    "Gemma3 4B" = "#9C6ADE",
    "Gemma3 12B" = "#7B47C7",
    "Gemma3 27B" = "#5A2D91",

    # InternVL family
    "InternVL3 2B" = "#80CBC4",
    "InternVL3 14B" = "#26A69A",
    "InternVL3 78B" = "#00695C"
)

amces_main <- amces_main %>%
    mutate(out = factor(out, levels = c(
        "Human",
        "GPT-4o mini",
        "GPT-4o",
        "Gemini 2.5 Flash Lite",
        "Gemini 2.5 Flash",
        "Qwen2 2B",
        "Qwen2 7B",
        "Qwen2 72B",
        "Gemma3 4B",
        "Gemma3 12B",
        "Gemma3 27B",
        "InternVL3 2B",
        "InternVL3 14B",
        "InternVL3 78B"
    )))

# Filter out reference categories
amces_main_filtered <- amces_main %>%
    filter(!(feature == "Slur" & level == "Generic\ninsult"),
           !(feature == "Identity" & level == "Anonymous"),
           !(feature == "Topic" & level == "Everyday"),
           !(feature == "Curse" & level == "No Curse"),
           !(feature == "Reply" & level == "None"),
           !(feature == "Engagement" & level == "Low"))

# Create custom facet labels with reference category information
feature_labels <- c(
    "Slur" = "Slur (ref: Generic insult)",
    "Identity" = "Identity (ref: Anonymous)",
    "Topic" = "Topic (ref: Everyday)",
    "Curse" = "Curse (ref: No curse)",
    "Reply" = "Reply (ref: None)",
    "Engagement" = "Engagement (ref: Low)"
)

# Create shape mapping for each model
model_shapes <- c(
    # Circle
    "Human" = 16,
    # Triangle
    "GPT-4o" = 17,
    "GPT-4o mini" = 17,
    # Square
    "Gemini 2.5 Flash" = 15,
    "Gemini 2.5 Flash Lite" = 15,
    # Diamond
    "Qwen2 2B" = 18,
    "Qwen2 7B" = 18,
    "Qwen2 72B" = 18,
    # Plus
    "Gemma3 4B" = 3,
    "Gemma3 12B" = 3,
    "Gemma3 27B" = 3,
    # Cross
    "InternVL3 2B" = 4,
    "InternVL3 14B" = 4,
    "InternVL3 78B" = 4
)


model_order <- rev(levels(amces_main_filtered$out))
feature_order <- c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")

slur_order <- c(
    "No slur", "Sexism", "Reverse\nracism", "Reclaimed\nslur", "Homophobia", "Racism"
)

# Constructing a dataset containing information for "banding"
amces_with_stripe <- amces_main_filtered %>%
    filter(out %in% model_order) %>%
    mutate(
        out    = factor(out, levels = model_order),
        feature = factor(feature, levels = feature_order),
        # relevel only for Slur
        level  = if_else(feature == "Slur", fct_relevel(level, slur_order), level)
    ) %>%
    group_by(feature) %>%
    mutate(
        # per-facet index that respects the factor order and
        # compresses to 1..K for the levels present in that facet
        level_num = dense_rank(match(level, levels(level))),
        stripe    = level_num %% 2 == 0
    ) %>%
    ungroup()


# Making the main plot
p_combined <- ggplot(
    amces_with_stripe,
    aes(x = estimate, y = level, group = out, color = out, shape = out)
) +
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = level_num - 0.5, ymax = level_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +
    geom_point(position = position_dodge(width = 1), size = 1.5) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 1)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_color_manual(values = custom_colors) +
    scale_shape_manual(values = model_shapes) +
    facet_manual(
        ~feature, design = "ABC\nDEF", scales = "free",
        labeller = labeller(feature = feature_labels),
        respect = TRUE, heights = c(2.2, 0.8)
    ) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(x = "Average Marginal Component Effect", y = NULL,
         colour = "Model", shape = "Model") +
    theme_bw(base_size = 7.8) +
    theme(legend.position = "none", panel.grid.major.y = element_blank())

# Some extra work is needed to make the legend organized by model family

# Define the function to create separate legends for each model family
create_family_legend <- function(models, colors, shapes) {
    df <- data.frame(x = 1, y = seq_along(models), model = factor(models, levels = models))
    ggplot(df, aes(x, y, color = model, shape = model)) +
        geom_point(size = 1.5) +
        scale_color_manual(values = colors[models]) +
        scale_shape_manual(values = shapes[models]) +
        guides(color = guide_legend(ncol = 1), shape = guide_legend(ncol = 1)) +
        theme_void(base_size = 7.8) +
        theme(
            legend.title = element_blank(),
            legend.key.height = unit(0.4, "lines"),
            legend.key.width = unit(0.3, "lines"),
            legend.spacing.y = unit(0.05, "cm"),
            legend.text = element_text(size = 7.8),
            legend.margin = margin(0, 0, 0, 0)
        )
}

# Extract legends for each family
leg_human <- get_legend(create_family_legend("Human", custom_colors, model_shapes))
leg_gpt <- get_legend(create_family_legend(c("GPT-4o", "GPT-4o mini"), custom_colors, model_shapes))
leg_gemini <- get_legend(create_family_legend(c("Gemini 2.5 Flash", "Gemini 2.5 Flash Lite"), custom_colors, model_shapes))
leg_qwen <- get_legend(create_family_legend(c("Qwen2 72B", "Qwen2 7B", "Qwen2 2B"), custom_colors, model_shapes))
leg_internvl <- get_legend(create_family_legend(c("InternVL3 78B", "InternVL3 14B", "InternVL3 2B"), custom_colors, model_shapes))
leg_gemma <- get_legend(create_family_legend(c("Gemma3 27B", "Gemma3 12B", "Gemma3 4B"), custom_colors, model_shapes))

# Combine all legends horizontally
legend_row <- plot_grid(leg_human, leg_gpt, leg_gemini, leg_qwen, leg_gemma, leg_internvl,
                        nrow = 1, rel_widths = c(1, 1, 1, 1, 1, 1))

# Create a centered version, otherwise they are stretched across entire plot
legend_centered <- plot_grid(
    NULL, legend_row, NULL,
    nrow = 1,
    rel_widths = c(0.10, 0.8, 0.10)
)

# Combine plot with legend
combined <- plot_grid(p_combined, legend_centered, ncol = 1, rel_heights = c(1, 0.05))
combined # Show plot

ggsave("../figures/figure2.pdf") #, width = 180, height = 220, units = "mm") # Fig. 1
# Note carefully: Setting dimensions results in extra whitespace between plot
# and legend. So size it in RStudio plot viewer and then save. As long as the
# aspect ratio is fine it will get resized in LaTeX

########################################################
# PART II: Variation in slur perception by "user" race #
########################################################

# Marginal Means
mms_slurs_b_by_speaker_gpt <- mm_diffs_bsp(data_gpt_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o", prompt = "Baseline")
mms_slurs_ca_by_speaker_gpt <- mm_diffs_bsp(data_gpt_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gpt <- mm_diffs_bsp(data_gpt_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o", prompt = "Uniform")

mms_slurs_b_by_speaker_gptmini <- mm_diffs_bsp(data_gptmini_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini", prompt = "Baseline")
mms_slurs_ca_by_speaker_gptmini <- mm_diffs_bsp(data_gptmini_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gptmini <- mm_diffs_bsp(data_gptmini_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini", prompt = "Uniform")

mms_slurs_b_by_speaker_gf <- mm_diffs_bsp(data_flash_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash", prompt = "Baseline")
mms_slurs_ca_by_speaker_gf <- mm_diffs_bsp(data_flash_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gf <- mm_diffs_bsp(data_flash_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash", prompt = "Uniform")

mms_slurs_b_by_speaker_gfl <- mm_diffs_bsp(data_flashlite_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite", prompt = "Baseline")
mms_slurs_ca_by_speaker_gfl <- mm_diffs_bsp(data_flashlite_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gfl <- mm_diffs_bsp(data_flashlite_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite", prompt = "Uniform")

mms_slurs_b_by_speaker_qwen2 <- mm_diffs_bsp(data_qwen2_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B", prompt = "Baseline")
mms_slurs_ca_by_speaker_qwen2 <- mm_diffs_bsp(data_qwen2_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_qwen2 <- mm_diffs_bsp(data_qwen2_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B", prompt = "Uniform")

mms_slurs_b_by_speaker_qwen7 <- mm_diffs_bsp(data_qwen7_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B", prompt = "Baseline")
mms_slurs_ca_by_speaker_qwen7 <- mm_diffs_bsp(data_qwen7_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_qwen7 <- mm_diffs_bsp(data_qwen7_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B", prompt = "Uniform")

mms_slurs_b_by_speaker_qwen72 <- mm_diffs_bsp(data_qwen72_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B", prompt = "Baseline")
mms_slurs_ca_by_speaker_qwen72 <- mm_diffs_bsp(data_qwen72_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_qwen72 <- mm_diffs_bsp(data_qwen72_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B", prompt = "Uniform")

mms_slurs_b_by_speaker_gemma4 <- mm_diffs_bsp(data_gemma4_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B", prompt = "Baseline")
mms_slurs_ca_by_speaker_gemma4 <- mm_diffs_bsp(data_gemma4_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gemma4 <- mm_diffs_bsp(data_gemma4_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B", prompt = "Uniform")

mms_slurs_b_by_speaker_gemma12 <- mm_diffs_bsp(data_gemma12_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B", prompt = "Baseline")
mms_slurs_ca_by_speaker_gemma12 <- mm_diffs_bsp(data_gemma12_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gemma12 <- mm_diffs_bsp(data_gemma12_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B", prompt = "Uniform")

mms_slurs_b_by_speaker_gemma27 <- mm_diffs_bsp(data_gemma27_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B", prompt = "Baseline")
mms_slurs_ca_by_speaker_gemma27 <- mm_diffs_bsp(data_gemma27_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_gemma27 <- mm_diffs_bsp(data_gemma27_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B", prompt = "Uniform")

mms_slurs_b_by_speaker_internvl2 <- mm_diffs_bsp(data_internvl2_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B", prompt = "Baseline")
mms_slurs_ca_by_speaker_internvl2 <- mm_diffs_bsp(data_internvl2_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_internvl2 <- mm_diffs_bsp(data_internvl2_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B", prompt = "Uniform")

mms_slurs_b_by_speaker_internvl14 <- mm_diffs_bsp(data_internvl14_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B", prompt = "Baseline")
mms_slurs_ca_by_speaker_internvl14 <- mm_diffs_bsp(data_internvl14_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_internvl14 <- mm_diffs_bsp(data_internvl14_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B", prompt = "Uniform")

mms_slurs_b_by_speaker_internvl78 <- mm_diffs_bsp(data_internvl78_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B", prompt = "Baseline")
mms_slurs_ca_by_speaker_internvl78 <- mm_diffs_bsp(data_internvl78_contextaware, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B", prompt = "Context-Sensitive")
mms_slurs_ci_by_speaker_internvl78 <- mm_diffs_bsp(data_internvl78_uniform, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B", prompt = "Uniform")

mms_slurs_human_by_speaker <- mm_diffs(data_human, chosen ~ slur, by = ~identity, id = ~id) %>%
    mutate(model = "Human", prompt = "Baseline")

# Step 2: Combine all datasets
mms_slurs_by_speaker <- bind_rows(
    mms_slurs_b_by_speaker_gpt,
    mms_slurs_ca_by_speaker_gpt,
    mms_slurs_ci_by_speaker_gpt,
    mms_slurs_b_by_speaker_gptmini,
    mms_slurs_ca_by_speaker_gptmini,
    mms_slurs_ci_by_speaker_gptmini,
    mms_slurs_b_by_speaker_gf,
    mms_slurs_ca_by_speaker_gf,
    mms_slurs_ci_by_speaker_gf,
    mms_slurs_b_by_speaker_gfl,
    mms_slurs_ca_by_speaker_gfl,
    mms_slurs_ci_by_speaker_gfl,
    mms_slurs_b_by_speaker_qwen2,
    mms_slurs_ca_by_speaker_qwen2,
    mms_slurs_ci_by_speaker_qwen2,
    mms_slurs_b_by_speaker_qwen7,
    mms_slurs_ca_by_speaker_qwen7,
    mms_slurs_ci_by_speaker_qwen7,
    mms_slurs_b_by_speaker_qwen72,
    mms_slurs_ca_by_speaker_qwen72,
    mms_slurs_ci_by_speaker_qwen72,
    mms_slurs_b_by_speaker_gemma4,
    mms_slurs_ca_by_speaker_gemma4,
    mms_slurs_ci_by_speaker_gemma4,
    mms_slurs_b_by_speaker_gemma12,
    mms_slurs_ca_by_speaker_gemma12,
    mms_slurs_ci_by_speaker_gemma12,
    mms_slurs_b_by_speaker_gemma27,
    mms_slurs_ca_by_speaker_gemma27,
    mms_slurs_ci_by_speaker_gemma27,
    mms_slurs_b_by_speaker_internvl2,
    mms_slurs_ca_by_speaker_internvl2,
    mms_slurs_ci_by_speaker_internvl2,
    mms_slurs_b_by_speaker_internvl14,
    mms_slurs_ca_by_speaker_internvl14,
    mms_slurs_ci_by_speaker_internvl14,
    mms_slurs_b_by_speaker_internvl78,
    mms_slurs_ca_by_speaker_internvl78,
    mms_slurs_ci_by_speaker_internvl78,
    mms_slurs_human_by_speaker
)

# Step 3: Adjust factor levels for ordered facets
mms_slurs_by_speaker$model <- factor(mms_slurs_by_speaker$model,
                                     levels = c("Human", "GPT-4o mini", "GPT-4o", "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash",  "Qwen2 2B", "Qwen2 7B", "Qwen2 72B", "Gemma3 4B", "Gemma3 12B", "Gemma3 27B", "InternVL3 2B", "InternVL3 14B", "InternVL3 78B")) # Organized by model family


# Set proper factor level order for slur types (reversed because ggplot displays bottom to top)
mms_slurs_by_speaker$level <- factor(mms_slurs_by_speaker$level,
                                    levels = c("No slur", "Racism", "Reclaimed\nslur", "Reverse\nracism",
                                              "Sexism", "Homophobia", "Generic\ninsult"))

# Calculating sig results by prompt for racialized terms
mms_slurs_by_speaker %>%
    filter(model != "Human") %>%
    filter(level %in% c("Racism", "Reverse\nracism", "Reclaimed\nslur")) %>%
    group_by(prompt) %>%
    summarise(
        n_total = n(),
        n_sig = sum(lower > 0 | upper < 0),
        pct_sig = round(100 * n_sig / n_total, 1),
        mean_eff = round(mean(abs(estimate)),3)
    )

# Modify levels to replace newline with spaces
# e.g. "Reclaimed\nslur" becomes "Reclaimed slur"
mms_slurs_by_speaker$level_mod <- gsub("\n", " ", mms_slurs_by_speaker$level)

# Define the model order once
model_levels <- rev(unique(mms_slurs_by_speaker$model))
model_levels <- c(setdiff(model_levels, "Human"), "Human")

# Prepare plot data and assign stripe bands by model within each facet
mms_striped <- mms_slurs_by_speaker %>%
    filter(
        level %in% c("Racism", "Reclaimed\nslur", "Reverse\nracism", "Sexism", "No slur")) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()



# Plot with facet-specific banding
ggplot(mms_striped %>% filter(prompt == "Baseline"),
       aes(x = estimate, y = model, color = identity)) +
    # banding within each facet
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +
    geom_point(position = position_dodge(width = 0.8), size = 1) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.8)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_y_discrete(limits = model_levels) +
    facet_grid(~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        caption = "Reference group: Anonymous"
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing = unit(0.1, "cm"),
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.height = unit(0.3, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/figure3.pdf", width = 180, height = 220, units = "mm") # Figure 3


mms_striped <- mms_slurs_by_speaker %>%
    filter(level %in% c("Generic\ninsult", "Homophobia")) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(model_num = dense_rank(model),
        stripe = model_num %% 2 == 0) %>%
    ungroup()

# Plot with facet-specific banding
ggplot(mms_striped %>% filter(prompt == "Baseline"),
       aes(x = estimate, y = model, color = identity)) +
    # banding within each facet
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +
    geom_point(position = position_dodge(width = 0.8), size = 1) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.8)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_y_discrete(limits = model_levels) +
    facet_grid(~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        caption = "Reference group: Anonymous"
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "right",
        legend.box = "vertical",
        legend.spacing = unit(0.1, "cm"),
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.height = unit(0.3, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/extended_data_1.pdf", width = 100, height = 220, units = "mm")

########################################################################
# PART III: Variation in slur perception by "user" race across prompts #
########################################################################

# Define the model order once
model_levels <- rev(unique(mms_slurs_by_speaker$model))
model_levels <- c(setdiff(model_levels, "Human"), "Human")

# Prepare plot data and assign stripe bands by model within each facet
mms_striped <- mms_slurs_by_speaker %>%
    filter((grepl("4o", model) | grepl("Flash", model) | grepl("Human", model))) %>%
    mutate(model = factor(model, levels = model_levels)
           ) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()

slur_order <- c(
    "No slur",
    "Generic insult",
    "Sexism",
    "Homophobia",
    "Reverse racism",
    "Reclaimed slur",
    "Racism"
)

mms_striped <- mms_striped %>%
    mutate(level_mod = fct_relevel(level_mod, slur_order))


# Plot with facet-specific banding
ggplot(mms_striped,
       aes(x = estimate, y = model, color = identity, shape = prompt, alpha = prompt)) +

    # Banding layer — facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.8), size = 1.25) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.8)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_y_discrete(limits = model_levels) +
    scale_alpha_manual(values = c("Baseline" = 1, "Context-Sensitive" = 0.8, "Uniform" = 0.8), guide = "none") + # Alpha rules
    scale_shape_manual(values = c("Baseline" = 16, "Context-Sensitive" = 15, "Uniform" = 17)) +
    facet_grid(~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        shape = "Prompt",
        caption = NULL
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing = unit(0.1, "cm"),
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.height = unit(0.3, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/extended_data_2.pdf", width = 180, height = 180, units = "mm")

mms_striped <- mms_slurs_by_speaker %>%
    filter((grepl("B", model) | grepl("Human", model))) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()

mms_striped <- mms_striped %>%
    mutate(level_mod = fct_relevel(level_mod, slur_order))


# Plot with facet-specific banding
ggplot(mms_striped,
       aes(x = estimate, y = model, color = identity, shape = prompt, alpha = prompt)) +

    # Banding layer — facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.8), size = 1.25) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.8)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_y_discrete(limits = model_levels) +
    scale_alpha_manual(values = c("Baseline" = 1, "Context-Sensitive" = 0.8, "Uniform" = 0.8), guide = "none") + # Alpha rules
    scale_shape_manual(values = c("Baseline" = 16, "Context-Sensitive" = 15, "Uniform" = 17)) +
    facet_grid(~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        shape = "Prompt",
        caption = NULL
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing = unit(0.1, "cm"),
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.height = unit(0.3, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/extended_data_3.pdf", width = 180, height = 230, units = "mm")

ggplot(mms_slurs_by_speaker %>% filter(model %in% c("Gemini 2.5 Flash", "Qwen2 72B", "Gemma3 12B"),
                                       level %in% c("Racism", "Reclaimed\nslur", "Reverse\nracism")),
       aes(x = estimate, y = model, color = identity, shape = prompt)) +

    geom_point(position = position_dodge(width = 0.8), size = 1) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.8)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_y_discrete(limits = model_levels) +
    scale_shape_manual(values = c("Baseline" = 16, "Context-Sensitive" = 15, "Uniform" = 17)) +
    facet_grid(model ~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        shape = "Prompt",
        caption = "Reference group: Anonymous"
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing = unit(0.1, "cm"),
        legend.spacing.y = unit(0.1, "cm"),
        legend.key.height = unit(0.3, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/figure4.pdf", width = 115, height = 85, units = "mm")
# Note: Set to maximum dimensions allowed for Research Briefing


########################################################################
# PART IV: Variation across visual and textual identity cues           #
########################################################################

amces_main_fl_gpt <- cj_bsp(data_gpt_faceless, main_formula)
amces_main_nl_gpt <- cj_bsp(data_gpt_nameless, main_formula)
amces_main_fl_gptmini <- cj_bsp(data_gptmini_faceless, main_formula)
amces_main_nl_gptmini <- cj_bsp(data_gptmini_nameless, main_formula)
amces_main_fl_gf <- cj_bsp(data_flash_faceless, main_formula)
amces_main_nl_gf <- cj_bsp(data_flash_nameless, main_formula)
amces_main_fl_gfl <- cj_bsp(data_flashlite_faceless, main_formula)
amces_main_nl_gfl <- cj_bsp(data_flashlite_nameless, main_formula)
amces_main_fl_qwen2 <- cj_bsp(data_qwen2_faceless, main_formula)
amces_main_nl_qwen2 <- cj_bsp(data_qwen2_nameless, main_formula)
amces_main_fl_qwen7 <- cj_bsp(data_qwen7_faceless, main_formula)
amces_main_nl_qwen7 <- cj_bsp(data_qwen7_nameless, main_formula)
amces_main_fl_qwen72 <- cj_bsp(data_qwen72_faceless, main_formula)
amces_main_nl_qwen72 <- cj_bsp(data_qwen72_nameless, main_formula)
amces_main_fl_gemma4 <- cj_bsp(data_gemma4_faceless, main_formula)
amces_main_nl_gemma4 <- cj_bsp(data_gemma4_nameless, main_formula)
amces_main_fl_gemma12 <- cj_bsp(data_gemma12_faceless, main_formula)
amces_main_nl_gemma12 <- cj_bsp(data_gemma12_nameless, main_formula)
amces_main_fl_gemma27 <- cj_bsp(data_gemma27_faceless, main_formula)
amces_main_nl_gemma27 <- cj_bsp(data_gemma27_nameless, main_formula)
amces_main_fl_internvl2 <- cj_bsp(data_internvl2_faceless, main_formula)
amces_main_nl_internvl2 <- cj_bsp(data_internvl2_nameless, main_formula)
amces_main_fl_internvl14 <- cj_bsp(data_internvl14_faceless, main_formula)
amces_main_nl_internvl14 <- cj_bsp(data_internvl14_nameless, main_formula)
amces_main_fl_internvl78 <- cj_bsp(data_internvl78_faceless, main_formula)
amces_main_nl_internvl78 <- cj_bsp(data_internvl78_nameless, main_formula)


# Apply consistent feature formatting to all faceless/nameless AMCE results
amces_main_fl_gpt <- format_features(amces_main_fl_gpt)
amces_main_nl_gpt <- format_features(amces_main_nl_gpt)
amces_main_fl_gptmini <- format_features(amces_main_fl_gptmini)
amces_main_nl_gptmini <- format_features(amces_main_nl_gptmini)
amces_main_fl_gf <- format_features(amces_main_fl_gf)
amces_main_nl_gf <- format_features(amces_main_nl_gf)
amces_main_fl_gfl <- format_features(amces_main_fl_gfl)
amces_main_nl_gfl <- format_features(amces_main_nl_gfl)
amces_main_fl_qwen2 <- format_features(amces_main_fl_qwen2)
amces_main_nl_qwen2 <- format_features(amces_main_nl_qwen2)
amces_main_fl_qwen7 <- format_features(amces_main_fl_qwen7)
amces_main_nl_qwen7 <- format_features(amces_main_nl_qwen7)
amces_main_fl_qwen72 <- format_features(amces_main_fl_qwen72)
amces_main_nl_qwen72 <- format_features(amces_main_nl_qwen72)
amces_main_fl_gemma4 <- format_features(amces_main_fl_gemma4)
amces_main_nl_gemma4 <- format_features(amces_main_nl_gemma4)
amces_main_fl_gemma12 <- format_features(amces_main_fl_gemma12)
amces_main_nl_gemma12 <- format_features(amces_main_nl_gemma12)
amces_main_fl_gemma27 <- format_features(amces_main_fl_gemma27)
amces_main_nl_gemma27 <- format_features(amces_main_nl_gemma27)
amces_main_fl_internvl2 <- format_features(amces_main_fl_internvl2)
amces_main_nl_internvl2 <- format_features(amces_main_nl_internvl2)
amces_main_fl_internvl14 <- format_features(amces_main_fl_internvl14)
amces_main_nl_internvl14 <- format_features(amces_main_nl_internvl14)
amces_main_fl_internvl78 <- format_features(amces_main_fl_internvl78)
amces_main_nl_internvl78 <- format_features(amces_main_nl_internvl78)

amces_main_fl_gpt$out <- "GPT-4o baseline (neutral face)"
amces_main_nl_gpt$out <- "GPT-4o baseline (neutral name)"
amces_main_fl_gptmini$out <- "GPT-4o mini baseline (neutral face)"
amces_main_nl_gptmini$out <- "GPT-4o mini baseline (neutral name)"
amces_main_fl_gf$out <- "Gemini 2.5 Flash baseline (neutral face)"
amces_main_nl_gf$out <- "Gemini 2.5 Flash baseline (neutral name)"
amces_main_fl_gfl$out <- "Gemini 2.5 Flash Lite baseline (neutral face)"
amces_main_nl_gfl$out <- "Gemini 2.5 Flash Lite baseline (neutral name)"
amces_main_fl_qwen2$out <- "Qwen-2 2B baseline (neutral face)"
amces_main_nl_qwen2$out <- "Qwen-2 2B baseline (neutral name)"
amces_main_fl_qwen7$out <- "Qwen-2 7B baseline (neutral face)"
amces_main_nl_qwen7$out <- "Qwen-2 7B baseline (neutral name)"
amces_main_fl_qwen72$out <- "Qwen-2 72B baseline (neutral face)"
amces_main_nl_qwen72$out <- "Qwen-2 72B baseline (neutral name)"
amces_main_fl_gemma4$out <- "Gemma3 4B baseline (neutral face)"
amces_main_nl_gemma4$out <- "Gemma3 4B baseline (neutral name)"
amces_main_fl_gemma12$out <- "Gemma3 12B baseline (neutral face)"
amces_main_nl_gemma12$out <- "Gemma3 12B baseline (neutral name)"
amces_main_fl_gemma27$out <- "Gemma3 27B baseline (neutral face)"
amces_main_nl_gemma27$out <- "Gemma3 27B baseline (neutral name)"
amces_main_fl_internvl2$out <- "InternVL3 2B baseline (neutral face)"
amces_main_nl_internvl2$out <- "InternVL3 2B baseline (neutral name)"
amces_main_fl_internvl14$out <- "InternVL3 14B baseline (neutral face)"
amces_main_nl_internvl14$out <- "InternVL3 14B baseline (neutral name)"
amces_main_fl_internvl78$out <- "InternVL3 78B baseline (neutral face)"
amces_main_nl_internvl78$out <- "InternVL3 78B baseline (neutral name)"

amces_facename <- bind_rows(amces_main_machine_b_gpt, amces_main_fl_gpt, amces_main_nl_gpt,
                            amces_main_machine_b_gptmini, amces_main_fl_gptmini, amces_main_nl_gptmini,
                            amces_main_machine_b_gf, amces_main_fl_gf, amces_main_nl_gf,
                            amces_main_machine_b_gfl, amces_main_fl_gfl, amces_main_nl_gfl,
                            amces_main_machine_b_qwen2, amces_main_fl_qwen2, amces_main_nl_qwen2,
                            amces_main_machine_b_qwen7, amces_main_fl_qwen7, amces_main_nl_qwen7,
                            amces_main_machine_b_qwen72, amces_main_fl_qwen72, amces_main_nl_qwen72,
                            amces_main_machine_b_gemma4, amces_main_fl_gemma4, amces_main_nl_gemma4,
                            amces_main_machine_b_gemma12, amces_main_fl_gemma12, amces_main_nl_gemma12,
                            amces_main_machine_b_gemma27, amces_main_fl_gemma27, amces_main_nl_gemma27,
                            amces_main_machine_b_internvl2, amces_main_fl_internvl2, amces_main_nl_internvl2,
                            amces_main_machine_b_internvl14, amces_main_fl_internvl14, amces_main_nl_internvl14,
                            amces_main_machine_b_internvl78, amces_main_fl_internvl78, amces_main_nl_internvl78)

# Filtering to outcome of interest
amces_facename_identity <- amces_facename %>% filter(feature == "Identity")

# Relabeling variables for plot
amces_facename_identity <- amces_facename_identity %>%
    mutate(
        model = case_when(
            str_detect(out, "GPT-4o mini") ~ "GPT-4o mini",
            str_detect(out, "GPT-4o") ~ "GPT-4o",
            str_detect(out, "Gemini 2.5 Flash Lite") ~ "Gemini 2.5 Flash Lite",
            str_detect(out, "Gemini 2.5 Flash") ~ "Gemini 2.5 Flash",
            str_detect(out, "InternVL3 78B") ~ "InternVL3 78B",
            str_detect(out, "InternVL3 14B") ~ "InternVL3 14B",
            str_detect(out, "InternVL3 2B") ~ "InternVL3 2B",
            str_detect(out, "Gemma3 27B") ~ "Gemma3 27B",
            str_detect(out, "Gemma3 12B") ~ "Gemma3 12B",
            str_detect(out, "Gemma3 4B") ~ "Gemma3 4B",
            str_detect(out, "72B") ~ "Qwen2 72B",
            str_detect(out, "7B") ~ "Qwen2 7B",
            TRUE ~ "Qwen2 2B" # Note: This needs to go last to avoid matching Qwen-2 7 to Qwen-2 72
        ),
        type = case_when(
            str_detect(out, "neutral face") ~ "Name only",
            str_detect(out, "neutral name") ~ "Face only",
            TRUE ~ "Name and face"
        )
    )

# Reordering
amces_facename_identity <- amces_facename_identity %>%
    mutate(
        type = factor(type, levels = c("Name and face", "Name only", "Face only")),
        model = factor(model, levels = c("GPT-4o mini", "GPT-4o", "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash", "Qwen2 2B", "Qwen2 7B", "Qwen2 72B", "Gemma3 4B", "Gemma3 12B", "Gemma3 27B", "InternVL3 2B", "InternVL3 14B", "InternVL3 78B")),
        level = factor(level, levels = c("White woman", "White man", "Black woman", "Black man",  "Anonymous"))
    )

# Improved color scheme for face/name conditions
custom_colors_facename <- c(
    "Name and face" = "#2C2C2C",    # Baseline
    "Face only" = "#1B82C7",        # Visual cue only
    "Name only" = "#C7511B"         # Textual cue only
)


amces_facename_identity_ <- amces_facename_identity %>%
    filter(level != "Anonymous") %>%
    mutate(model = factor(model, levels = model_levels))

ggplot(amces_facename_identity_,
       aes(x = estimate, y = model,
           shape = type)) +
    geom_point(position = position_dodge(width = 0.7), size = 1.5) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0, position = position_dodge(width = 0.7)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    #scale_color_manual(values = custom_colors_facename) + # Custom colors for each model
    scale_shape_manual(values = c("Name and face" = 13, "Face only" = 1, "Name only" = 4)) +
    facet_wrap(~level, nrow = 1) +
    labs(#color = "Vignette type",
        shape = "Identity cue", # Same name for both to ensure mutual legend
        x = "Average Marginal Component Effect", y = NULL,
        caption = "Reference group: Anonymous") +
    theme_bw(base_size = 7.8) +
    guides(shape  = guide_legend(nrow = 3, byrow = FALSE,
                                 keyheight = unit(0.5, "lines"),
                                 label.vjust = 0.5)) +
    theme(legend.position = "bottom",
          strip.text = element_text(size = 6.5),
          axis.text.x = element_text(angle = 90))
ggsave("../figures/figure5.pdf", width = 120, height = 200, units = "mm")


## Repeating slur-identity analysis with faceless-nameless data

mms_slurs_by_speaker_gpt_nameless <- mm_diffs_bsp(data_gpt_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o", type = "Face only")
mms_slurs_by_speaker_gpt_faceless <- mm_diffs_bsp(data_gpt_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o", type = "Name only")


mms_slurs_by_speaker_gptmini_nameless <- mm_diffs_bsp(data_gptmini_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini", type = "Face only")
mms_slurs_by_speaker_gptmini_faceless <- mm_diffs_bsp(data_gptmini_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini", type = "Name only")

mms_slurs_by_speaker_gf_nameless <- mm_diffs_bsp(data_flash_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash", type = "Face only")
mms_slurs_by_speaker_gf_faceless <- mm_diffs_bsp(data_flash_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash", type = "Name only")

mms_slurs_by_speaker_gfl_nameless <- mm_diffs_bsp(data_flashlite_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite", type = "Face only")
mms_slurs_by_speaker_gfl_faceless <- mm_diffs_bsp(data_flashlite_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite", type = "Name only")

mms_slurs_by_speaker_qwen2_nameless <- mm_diffs_bsp(data_qwen2_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B", type = "Face only")
mms_slurs_by_speaker_qwen2_faceless <- mm_diffs_bsp(data_qwen2_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B", type = "Name only")

mms_slurs_by_speaker_qwen7_nameless <- mm_diffs_bsp(data_qwen7_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B", type = "Face only")
mms_slurs_by_speaker_qwen7_faceless <- mm_diffs_bsp(data_qwen7_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B", type = "Name only")

mms_slurs_by_speaker_qwen72_nameless <- mm_diffs_bsp(data_qwen72_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B", type = "Face only")
mms_slurs_by_speaker_qwen72_faceless <- mm_diffs_bsp(data_qwen72_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B", type = "Name only")

mms_slurs_by_speaker_gemma4_nameless <- mm_diffs_bsp(data_gemma4_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B", type = "Face only")
mms_slurs_by_speaker_gemma4_faceless <- mm_diffs_bsp(data_gemma4_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B", type = "Name only")

mms_slurs_by_speaker_gemma12_nameless <- mm_diffs_bsp(data_gemma12_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B", type = "Face only")
mms_slurs_by_speaker_gemma12_faceless <- mm_diffs_bsp(data_gemma12_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B", type = "Name only")

mms_slurs_by_speaker_gemma27_nameless <- mm_diffs_bsp(data_gemma27_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B", type = "Face only")
mms_slurs_by_speaker_gemma27_faceless <- mm_diffs_bsp(data_gemma27_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B", type = "Name only")

mms_slurs_by_speaker_internvl2_nameless <- mm_diffs_bsp(data_internvl2_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B", type = "Face only")
mms_slurs_by_speaker_internvl2_faceless <- mm_diffs_bsp(data_internvl2_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B", type = "Name only")

mms_slurs_by_speaker_internvl14_nameless <- mm_diffs_bsp(data_internvl14_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B", type = "Face only")
mms_slurs_by_speaker_internvl14_faceless <- mm_diffs_bsp(data_internvl14_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B", type = "Name only")

mms_slurs_by_speaker_internvl78_nameless <- mm_diffs_bsp(data_internvl78_nameless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B", type = "Face only")
mms_slurs_by_speaker_internvl78_faceless <- mm_diffs_bsp(data_internvl78_faceless, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B", type = "Name only")


# Step 2: Combine all datasets
mms_slurs_by_speaker2 <- bind_rows(
    mms_slurs_b_by_speaker_gpt,
    mms_slurs_by_speaker_gpt_nameless,
    mms_slurs_by_speaker_gpt_faceless,
    mms_slurs_b_by_speaker_gptmini,
    mms_slurs_by_speaker_gptmini_nameless,
    mms_slurs_by_speaker_gptmini_faceless,
    mms_slurs_b_by_speaker_gf,
    mms_slurs_by_speaker_gf_nameless,
    mms_slurs_by_speaker_gf_faceless,
    mms_slurs_b_by_speaker_gfl,
    mms_slurs_by_speaker_gfl_nameless,
    mms_slurs_by_speaker_gfl_faceless,
    mms_slurs_b_by_speaker_qwen2,
    mms_slurs_by_speaker_qwen2_nameless,
    mms_slurs_by_speaker_qwen2_faceless,
    mms_slurs_b_by_speaker_qwen7,
    mms_slurs_by_speaker_qwen7_nameless,
    mms_slurs_by_speaker_qwen7_faceless,
    mms_slurs_b_by_speaker_qwen72,
    mms_slurs_by_speaker_qwen72_nameless,
    mms_slurs_by_speaker_qwen72_faceless,
    mms_slurs_b_by_speaker_gemma4,
    mms_slurs_by_speaker_gemma4_nameless,
    mms_slurs_by_speaker_gemma4_faceless,
    mms_slurs_b_by_speaker_gemma12,
    mms_slurs_by_speaker_gemma12_nameless,
    mms_slurs_by_speaker_gemma12_faceless,
    mms_slurs_b_by_speaker_gemma27,
    mms_slurs_by_speaker_gemma27_nameless,
    mms_slurs_by_speaker_gemma27_faceless,
    mms_slurs_b_by_speaker_internvl2,
    mms_slurs_by_speaker_internvl2_nameless,
    mms_slurs_by_speaker_internvl2_faceless,
    mms_slurs_b_by_speaker_internvl14,
    mms_slurs_by_speaker_internvl14_nameless,
    mms_slurs_by_speaker_internvl14_faceless,
    mms_slurs_b_by_speaker_internvl78,
    mms_slurs_by_speaker_internvl78_nameless,
    mms_slurs_by_speaker_internvl78_faceless
)

# Set it to "Face and name" for the baselines
mms_slurs_by_speaker2 <- mms_slurs_by_speaker2 %>% mutate(type = ifelse(is.na(type), "Name and face", type))

# Step 3: Adjust factor levels for ordered facets
mms_slurs_by_speaker2$model <- factor(mms_slurs_by_speaker2$model,
                                      levels = c("GPT-4o", "GPT-4o mini", "Gemini 2.5 Flash", "Gemini 2.5 Flash Lite", "Qwen2 2B", "Qwen2 7B", "Qwen2 72B", "Gemma3 4B", "Gemma3 12B", "Gemma3 27B", "InternVL3 2B", "InternVL3 14B", "InternVL3 78B")) # Organized by model family

# Reverse the levels of the `level` variable in the dataset
mms_slurs_by_speaker2$level <- forcats::fct_rev(mms_slurs_by_speaker2$level)

# Calculating overall amount of significant variation.
mms_slurs_by_speaker2 %>%
    filter(level %in% c("Racism", "Reverse\nracism", "Reclaimed\nslur")) %>%
    group_by(type) %>%
    summarise(
        n_total = n(),
        n_sig = sum(lower > 0 | upper < 0),
        pct_sig = round(100 * n_sig / n_total, 1),
        mean_eff = round(mean(abs(estimate)),3)
    )



# Alternative version
# Define the model order once
model_levels <- rev(unique(mms_slurs_by_speaker2$model))

mms_slurs_by_speaker2$level_mod <- gsub("\n", " ", mms_slurs_by_speaker2$level)

# Prepare plot data and assign stripe bands by model within each facet
mms_striped <- mms_slurs_by_speaker2 %>%
    filter(
        level %in% c("Racism", "Reclaimed\nslur", "Reverse\nracism", "Sexism", "Generic\ninsult")) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()


# There is too much information for one plot so splitting across closed and open
# Focusing only on the racialized slurs

mms_slurs_by_speaker2 <- mms_slurs_by_speaker2 %>%
    filter(level_mod %in% c("Racism", "Reclaimed slur", "Reverse racism"))

# CLOSED ONLY
mms_striped <- mms_slurs_by_speaker2 %>%
    filter((grepl("4o", model) | grepl("Flash", model) | grepl("Human", model))) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()

# Plot with facet-specific banding
ggplot(mms_striped,
       aes(x = estimate, y = model, color = identity, shape = type)) +

    # Banding layer — facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.9), size = 1.2) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.9)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_y_discrete(limits = model_levels) +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_shape_manual(values = c("Name and face" = 13, "Face only" = 1, "Name only" = 4)) +
    facet_grid(~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        shape = "Prompt",
        caption = "Reference group: Anonymous"
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing = unit(0, "cm"),
        legend.spacing.y = unit(0, "cm"),
        legend.key.height = unit(0, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/extended_data_4.pdf", width = 180, height = 180, units = "mm")


# OPEN ONLY
mms_striped <- mms_slurs_by_speaker2 %>%
    filter((grepl("B", model))) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()

# Plot with facet-specific banding
ggplot(mms_striped,
       aes(x = estimate, y = model, color = identity, shape = type)) +

    # Banding layer — facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.9), size = 1.2) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0,
                   position = position_dodge(width = 0.9)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_y_discrete(limits = model_levels) +
    scale_color_scico_d(palette = "managua", begin = 0.05, end = 0.95) +
    scale_shape_manual(values = c("Name and face" = 13, "Face only" = 1, "Name only" = 4)) +
    facet_grid(~ level_mod, scales = "free_y", space = "free") +
    xlim(min(mms_striped$lower) - 0.01, max(mms_striped$upper) + 0.01) +
    labs(
        x = "Difference in Marginal Means",
        y = NULL,
        color = "Identity",
        shape = "Prompt",
        caption = "Reference group: Anonymous"
    ) +
    theme_bw(base_size = 7.8) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing = unit(0, "cm"),
        legend.spacing.y = unit(0, "cm"),
        legend.key.height = unit(0, "cm"),
        axis.text.x = element_text(angle = 90),
        strip.text.y = element_blank(),
        strip.text.x = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
    )
ggsave("../figures/extended_data_5.pdf", width = 180, height = 220, units = "mm")

