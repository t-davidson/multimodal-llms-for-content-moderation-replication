library(cregg)
library(tidyverse)
library(ggplot2)
library(stringr)
library(scico)
library(forcats)
library(grid)

set.seed(1485008901) # Seed used for published results

source("bootstrap_conjoint_functions.R")

data_machine <- read_csv("../replication_data/final_mllm_results_alt.csv")

data_human <- read_csv("../replication_data/final_human_results.csv") %>% filter(task != 16) # dropping repeat task

# Process machine to recode model names
data_machine <- data_machine %>%
    mutate(
        model = case_when(
            model == "base" ~ "GPT-4o",
            model == "mini" ~ "GPT-4o-mini",
            model == "gemini_2_5_flash" ~ "Gemini 2.5 Flash",
            model == "gemini_2_5_flash_lite" ~ "Gemini 2.5 Flash Lite",
            model == "qwen7" ~ "Qwen2-7B",
            model == "qwen72" ~ "Qwen2-72B",
            model == "qwen2" ~ "Qwen2-2B",
            model == "gemma3-4b" ~ "Gemma3-4B",
            model == "gemma3-12b" ~ "Gemma3-12B",
            model == "gemma3-27b" ~ "Gemma3-27B",
            model == "internvl3-2b" ~ "InternVL3-2B",
            model == "internvl3-14b" ~ "InternVL3-14B",
            model == "internvl3-78b" ~ "InternVL3-78B",
            TRUE ~ NA_character_
        ),
        model = factor(model, levels = c("GPT-4o", "GPT-4o-mini", "Gemini 2.5 Flash", "Gemini 2.5 Flash Lite", "Qwen2-7B", "Qwen2-72B", "Qwen2-2B", "Gemma3-4B", "Gemma3-12B", "Gemma3-27B", "InternVL3-2B", "InternVL3-14B", "InternVL3-78B"))
    )


# Model-prompt counts (verify each has 60k rows)
model_prompt_counts <- data_machine %>%
    count(model, prompt) %>%
    arrange(model, prompt)

# Recode slurs to codes
data_machine <- data_machine %>%
    mutate(slur = recode(slur,
                         "nigger" = "AB-1",
                         "nigga" = "AB-2",
                         "coon" = "AB-3",
                         "spook" = "AB-4",
                         "chink" = "AA-1",
                         "wetback" = "AH-1",
                         "cracker" = "AW-1",
                         "redneck" = "AW-2",
                         "honkey" = "AW-3",
                         "bitch" = "S-1",
                         "slut" = "S-2",
                         "cunt" = "S-3",
                         "faggot" = "H-1",
                         "fag" = "H-2",
                         "homo" = "H-3",
                         "asshole" = "G-1",
                         "bastard" = "G-2",
                         "motherfucker" = "G-3"))

data_human <- data_human %>%
    mutate(slur = recode(slur,
                         "nigger" = "AB-1",
                         "nigga" = "AB-2",
                         "coon" = "AB-3",
                         "spook" = "AB-4",
                         "chink" = "AA-1",
                         "wetback" = "AH-1",
                         "cracker" = "AW-1",
                         "redneck" = "AW-2",
                         "honkey" = "AW-3",
                         "bitch" = "S-1",
                         "slut" = "S-2",
                         "cunt" = "S-3",
                         "faggot" = "H-1",
                         "fag" = "H-2",
                         "homo" = "H-3",
                         "asshole" = "G-1",
                         "bastard" = "G-2",
                         "motherfucker" = "G-3"))

data_machine <- data_machine %>% mutate(slur_type =
        ifelse(slur %in% c("AB-1", "AB-2", "AB-4", "AB-3"), "Anti-Black racism",
               ifelse(slur %in% c("AH-1", "AA-1"), "Other racism",
               ifelse(slur %in% c("AW-1", "AW-3", "AW-2"), "Reverse racism",
                      ifelse(slur %in% c("H-1", "H-2", "H-3"), "Homophobia",
                             ifelse(slur %in% c("S-1","S-2", "S-3"), "Sexism",
                                    ifelse(slur %in% c("G-1", "G-3", "G-2"), "Generic", "No slur")))))))

# Add benign such that these are a single variable
data_machine <- data_machine %>%
    mutate(slur = replace_na(slur, "No slur"))
data_human <- data_human %>%
    mutate(slur = replace_na(slur, "No slur"))

data_machine$slur <- as.factor(data_machine$slur)
data_human$slur <- as.factor(data_human$slur)

# Recoding values
data_machine <- data_machine %>%
    rename(topic = context) %>%
    mutate(identity = recode(identity, "BM" = "Black man", "BF" = "Black woman", "WM" = "White man", "WF" = "White woman", "A" = "Anonymous"),
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
    identity = as.factor(identity),
    topic = as.factor(topic),
    engagement = as.factor(ifelse(engagement == "L", "Low engagement", "High engagement")),
    reply = as.factor(reply),
    slur = as.factor(slur),
    curse = as.factor(curse)
) %>% select(id, chosen, interaction, identity, topic, engagement, reply, curse, slur, model)

data_human <- data_human %>% mutate(
    identity = as.factor(identity),
    topic = as.factor(topic),
    engagement = as.factor(ifelse(engagement == "L", "Low engagement", "High engagement")),
    reply = as.factor(reply),
    slur = as.factor(slur),
    curse = as.factor(curse)
) %>% select(id, chosen, interaction, identity, topic, engagement, reply, curse, slur)

# Fixing reference categories
data_machine$topic <- relevel(data_machine$topic, ref = "Everyday")
data_machine$reply <- relevel(data_machine$reply, ref = "None")
data_machine$engagement <- relevel(data_machine$engagement, ref = "Low engagement")
data_machine$curse <- relevel(data_machine$curse, ref = "No Curse")
data_machine$slur <- relevel(data_machine$slur, ref = "No slur")

data_human$topic <- relevel(data_human$topic, ref = "Everyday")
data_human$reply <- relevel(data_human$reply, ref = "None")
data_human$engagement <- relevel(data_human$engagement, ref = "Low engagement")
data_human$curse <- relevel(data_human$curse, ref = "No Curse")
data_human$slur <- relevel(data_human$slur, ref = "No slur")

# Splitting data_machine by model
# GPT-4o
data_gpt_baseline <- data_machine %>% filter(model == "GPT-4o")

# GPT-4o-mini
data_gptmini_baseline <- data_machine %>% filter(model == "GPT-4o-mini")

# Gemini 2.5 Flash
data_flash_baseline <- data_machine %>% filter(model == "Gemini 2.5 Flash")

# Gemini 2.5 Flash Lite
data_flashlite_baseline <- data_machine %>% filter(model == "Gemini 2.5 Flash Lite")

# Qwen2-7B
data_qwen7_baseline <- data_machine %>% filter(model == "Qwen2-7B")

# Qwen2-72B
data_qwen72_baseline <- data_machine %>% filter(model == "Qwen2-72B")

# Qwen2-2B
data_qwen2_baseline <- data_machine %>% filter(model == "Qwen2-2B")

# Gemma3-4B
data_gemma4_baseline <- data_machine %>% filter(model == "Gemma3-4B")

# Gemma3-12B
data_gemma12_baseline <- data_machine %>% filter(model == "Gemma3-12B")

# Gemma3-27B
data_gemma27_baseline <- data_machine %>% filter(model == "Gemma3-27B")

# InternVL3-2B
data_internvl2_baseline <- data_machine %>% filter(model == "InternVL3-2B")

# InternVL3-14B
data_internvl14_baseline <- data_machine %>% filter(model == "InternVL3-14B")

# InternVL3-78B
data_internvl78_baseline <- data_machine %>% filter(model == "InternVL3-78B")



##############################################
# PART I: Main effects of alternative slurs  #
##############################################

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


amces_main_machine_b_gpt <- amces_main_machine_b_gpt %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_gptmini <- amces_main_machine_b_gptmini %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_gf <- amces_main_machine_b_gf %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_gfl <- amces_main_machine_b_gfl %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_qwen2 <- amces_main_machine_b_qwen2 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_qwen7 <- amces_main_machine_b_qwen7 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_qwen72 <- amces_main_machine_b_qwen72 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_gemma4 <- amces_main_machine_b_gemma4 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_gemma12 <- amces_main_machine_b_gemma12 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_gemma27 <- amces_main_machine_b_gemma27 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_internvl2 <- amces_main_machine_b_internvl2 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_internvl14 <- amces_main_machine_b_internvl14 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_machine_b_internvl78 <- amces_main_machine_b_internvl78 %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

amces_main_human <- amces_main_human %>%
    mutate(feature = str_to_title(as.character(feature)),
           feature = factor(feature, levels = c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")),
    )

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
        "InternVL3 78B",
        "InternVL3 14B",
        "InternVL3 2B",
        "Gemma3 27B",
        "Gemma3 12B",
        "Gemma3 4B",
        "Qwen2 72B",
        "Qwen2 7B",
        "Qwen2 2B",
        "Gemini 2.5 Flash",
        "Gemini 2.5 Flash Lite",
        "GPT-4o",
        "GPT-4o mini",
        "Human"
    )))

amces_main <- amces_main %>% mutate(slur_type =
                                            ifelse(level %in% c("AB-1", "AB-2", "AB-4", "AB-3"), "Anti-Black racism",
                                                   ifelse(level %in% c("AH-1", "AA-1"), "Other racism",
                                                   ifelse(level %in% c("AW-1", "AW-3", "AW-2"), "Reverse racism",
                                                          ifelse(level %in% c("H-1", "H-2", "H-3"), "Homophobia",
                                                                 ifelse(level %in% c("S-1","S-2", "S-3"), "Sexism",
                                                                        ifelse(level %in% c("G-1", "G-3", "G-2"), "Generic",
                                                                               ifelse(level %in% c("No slur"), "No slur", NA))))))))

# Reorder levels within each slur category for better display
amces_main <- amces_main %>%
    mutate(level = factor(level, levels = c(
        # Anti-Black racism terms in numerical order
        "AB-1", "AB-2", "AB-3", "AB-4",
        # Other racism terms in numerical order
        "AA-1", "AH-1",
        # Reverse racism terms in numerical order
        "AW-1", "AW-2", "AW-3",
        # Homophobia terms in numerical order
        "H-1", "H-2", "H-3",
        # Sexism terms in numerical order
        "S-1", "S-2", "S-3",
        # Generic terms in numerical order
        "G-1", "G-2", "G-3",
        # No slur
        "No slur"
    )))

# Also reorder within slur_type groups for the grouped plot
amces_main <- amces_main %>%
    mutate(level = fct_relevel(level,
        # Anti-Black racism terms in numerical order
        "AB-1", "AB-2", "AB-3", "AB-4",
        # Other racism terms in numerical order
        "AA-1", "AH-1",
        # Reverse racism terms in numerical order
        "AW-1", "AW-2", "AW-3",
        # Homophobia terms in numerical order
        "H-1", "H-2", "H-3",
        # Sexism terms in numerical order
        "S-1", "S-2", "S-3",
        # Generic terms in numerical order
        "G-1", "G-2", "G-3",
        "No slur"
    ))

amces_main <- amces_main %>% mutate(slur_type = factor(slur_type, levels = c("Anti-Black racism", "Other racism", "Reverse racism", "Homophobia", "Sexism", "Generic", "No slur")))

# Create shape mapping for each model (matching the family structure)
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

# Create custom facet labels with reference category information
feature_labels <- c(
    "Slur" = "Slur (ref: Generic insult)",
    "Identity" = "Identity (ref: Anonymous)",
    "Topic" = "Topic (ref: Everyday)",
    "Curse" = "Curse (ref: No curse)",
    "Reply" = "Reply (ref: None)",
    "Engagement" = "Engagement (ref: Low)"
)

# Filter out reference categories
amces_main_filtered <- amces_main %>%
    filter(!(feature == "Slur" & level == "No slur"),
           !(feature == "Identity" & level == "Anonymous"),
           !(feature == "Topic" & level == "Everyday"),
           !(feature == "Curse" & level == "No Curse"),
           !(feature == "Reply" & level == "None"),
           !(feature == "Engagement" & level == "Low engagement"))

# Prepare data for the plot. We need to add a rank for levels within each facet.
slur_plot_data_striped <- amces_main %>%
    filter(feature == "Slur" & level != "No slur") %>%
    group_by(slur_type) %>%
    # Create a numeric rank for each level within the facet.
    # This will be used for the y-coordinates of the striping rectangles.
    mutate(level_rank = dense_rank(level)) %>%
    ungroup()

slur_plot_data_striped <- slur_plot_data_striped %>% #Setting order for plot
    mutate(
        slur_type = fct_relevel(
            slur_type,
            "Anti-Black racism",
            "Other racism",
            "Homophobia",
            "Reverse racism",
            "Sexism",
            "Generic",
            "No slur"
        )
    )

# Full plot
ggplot(slur_plot_data_striped, aes(x = estimate, y = level, group = out, color = out, shape = out)) +
    # Add Striping using geom_rect.
    # The data is filtered to only include odd-ranked levels for the stripes.
    geom_rect(
        data = . %>% filter(level_rank %% 2 == 1),
        aes(ymin = level_rank - 0.5, ymax = level_rank + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94",
        color = NA,
        inherit.aes = FALSE
    ) +
    geom_point(position = position_dodge(width = 1), size = 1.2) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0, position = position_dodge(width = 1)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.2, color = "black") +
    scale_y_discrete(limits = rev) +
    scale_color_manual(values = custom_colors) +
    scale_shape_manual(values = model_shapes) +
    facet_wrap(~slur_type, scales = "free_y", ncol = 1) +
    labs(colour = "Model", shape = "Model", x = "Average Marginal Component Effect", y = NULL) +
    guides(color = guide_legend(reverse = FALSE), shape = guide_legend(reverse = FALSE)) +
    theme_minimal(base_size = 7) +
    theme(legend.position = "right",
          strip.text.x = element_text(size = 8)
          )
ggsave("../figures/supplementary_fig_2.pdf", width = 180, height = 240, units = "mm")

#########################################################
# PART II: Variation in slur perception by "user" race. #
#########################################################

# Since alt script only has baseline, we'll use that for all models

mms_b_by_speaker_gpt <- mm_diffs_bsp(data_gpt_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o")
mms_b_by_speaker_gptmini <- mm_diffs_bsp(data_gptmini_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini")
mms_b_by_speaker_gf <- mm_diffs_bsp(data_flash_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash")
mms_b_by_speaker_gfl <- mm_diffs_bsp(data_flashlite_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite")
mms_b_by_speaker_qwen2 <- mm_diffs_bsp(data_qwen2_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B")
mms_b_by_speaker_qwen7 <- mm_diffs_bsp(data_qwen7_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B")
mms_b_by_speaker_qwen72 <- mm_diffs_bsp(data_qwen72_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B")
mms_b_by_speaker_gemma4 <- mm_diffs_bsp(data_gemma4_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B")
mms_b_by_speaker_gemma12 <- mm_diffs_bsp(data_gemma12_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B")
mms_b_by_speaker_gemma27 <- mm_diffs_bsp(data_gemma27_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B")
mms_b_by_speaker_internvl2 <- mm_diffs_bsp(data_internvl2_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B")
mms_b_by_speaker_internvl14 <- mm_diffs_bsp(data_internvl14_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B")
mms_b_by_speaker_internvl78 <- mm_diffs_bsp(data_internvl78_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B")

mms_human_by_speaker <- mm_diffs(data_human, chosen ~ slur, by = ~identity, id = ~id) %>%
    mutate(model = "Human")

# Step 2: Combine all datasets
mms_by_speaker <- bind_rows(
    mms_b_by_speaker_gpt,
    mms_b_by_speaker_gptmini,
    mms_b_by_speaker_gf,
    mms_b_by_speaker_gfl,
    mms_b_by_speaker_qwen2,
    mms_b_by_speaker_qwen7,
    mms_b_by_speaker_qwen72,
    mms_b_by_speaker_gemma4,
    mms_b_by_speaker_gemma12,
    mms_b_by_speaker_gemma27,
    mms_b_by_speaker_internvl2,
    mms_b_by_speaker_internvl14,
    mms_b_by_speaker_internvl78,
    mms_human_by_speaker
)

# Step 3: Adjust factor levels for ordered facets
mms_by_speaker$model <- factor(mms_by_speaker$model,
                                     levels = c("Human", "GPT-4o mini", "GPT-4o", "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash", "Qwen2 2B", "Qwen2 7B", "Qwen2 72B", "Gemma3 4B", "Gemma3 12B", "Gemma3 27B", "InternVL3 2B", "InternVL3 14B", "InternVL3 78B"))

# Set up factor levels for individual slur terms to match alt dataset
mms_by_speaker$level <- factor(mms_by_speaker$level, levels = c(
    # Anti-Black racism terms in numerical order
    "AB-1", "AB-2", "AB-3", "AB-4",
    # Other racism terms in numerical order
    "AA-1", "AH-1",
    # Reverse racism terms in numerical order
    "AW-1", "AW-2", "AW-3",
    # Homophobia terms in numerical order
    "H-1", "H-2", "H-3",
    # Sexism terms in numerical order
    "S-1", "S-2", "S-3",
    # Generic terms in numerical order
    "G-1", "G-2", "G-3",
    # No slur
    "No slur"
))

# Add slur_type grouping to mms_by_speaker for filtering
mms_by_speaker <- mms_by_speaker %>%
    mutate(slur_type = case_when(
        level %in% c("AB-1", "AB-2", "AB-3", "AB-4", "AA-1", "AH-1") ~ "Racism",
        level %in% c("AW-1", "AW-2", "AW-3") ~ "Reverse racism",
        level %in% c("H-1", "H-2", "H-3") ~ "Homophobia",
        level %in% c("S-1", "S-2", "S-3") ~ "Sexism",
        level %in% c("G-1", "G-2", "G-3") ~ "Generic",
        level == "No slur" ~ "No slur",
        TRUE ~ "Other"
    ))

# Modify levels to replace newline with spaces for cleaner display
mms_by_speaker$level_mod <- gsub("\\n", " ", mms_by_speaker$level)

# Define the model order
model_levels <- rev(unique(mms_by_speaker$model))
model_levels <- c(setdiff(model_levels, "Human"), "Human")

# Prepare plot data and assign stripe bands by model within each facet
mms_striped <- mms_by_speaker %>%
    filter(
        level %in% c("AB-1", "AB-2", "AB-3", "AB-4", "AA-1", "AH-1", "AW-1", "AW-2", "AW-3", "S-1", "S-2", "S-3")
    ) %>%
    mutate(
        model = factor(model, levels = model_levels),
        # Set specific order for slur terms in numerical order
        level_mod = factor(level_mod, levels = c("AB-1", "AB-2", "AB-3", "AB-4", "AA-1", "AH-1", "AW-1", "AW-2", "AW-3", "S-1", "S-2", "S-3"))
       ) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()

# Racism
ggplot(mms_striped %>% filter(level_mod %in% c("AB-1", "AB-3", "AB-4", "AA-1", "AH-1")),
       aes(x = estimate, y = model, color = identity)) +

    # Striping layer, facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.8), size = 1.5) +
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
    theme_bw(base_size = 7) +
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
ggsave("../figures/supplementary_fig_3.pdf", width = 180, height = 180, units = "mm")

# Reverse racism
ggplot(mms_striped %>% filter(level_mod %in% c("AW-1", "AW-2", "AW-3")),
       aes(x = estimate, y = model, color = identity)) +

    # Striping layer, facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.8), size = 1.5) +
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
    theme_bw(base_size = 7) +
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
ggsave("../figures/supplementary_fig_4.pdf", width = 180, height = 180, units = "mm")

# Sexism
ggplot(mms_striped %>% filter(level_mod %in% c("S-1", "S-2", "S-3")),
       aes(x = estimate, y = model, color = identity)) +

    # Striping layer, facet-aware
    geom_rect(
        data = function(d) d %>% filter(stripe),
        aes(ymin = model_num - 0.5, ymax = model_num + 0.5),
        xmin = -Inf, xmax = Inf,
        fill = "grey94", color = NA,
        inherit.aes = FALSE
    ) +

    geom_point(position = position_dodge(width = 0.8), size = 1.5) +
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
    theme_bw(base_size = 7) +
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
ggsave("../figures/supplementary_fig_5.pdf", width = 180, height = 180, units = "mm")
