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

source("bootstrap_conjoint_functions.R")

data_machine <- read_csv("../replication_data/final_mllm_results_single.csv")

# Process machine data
data_machine <- data_machine %>%
    mutate(
        # Recode only the necessary model labels
        model = case_when(
            model == "qwen7"       ~ "Qwen2-7B",
            model == "qwen72"      ~ "Qwen2-72B",
            model == "qwen2"       ~ "Qwen2-2B",
            model == "gemma3-4b"   ~ "Gemma3-4B",
            model == "gemma3-12b"  ~ "Gemma3-12B",
            model == "gemma3-27b"  ~ "Gemma3-27B",
            model == "internvl3-2b" ~ "InternVL3-2B",
            model == "internvl3-14b" ~ "InternVL3-14B",
            model == "internvl3-78b" ~ "InternVL3-78B",
            TRUE ~ model  # leave as-is for already-correct labels
        ),
        model = factor(model, levels = c("GPT-4o mini", "GPT-4o",
                                         "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash",
                                         "Qwen2-7B", "Qwen2-72B", "Qwen2-2B",
                                         "Gemma3-4B", "Gemma3-12B", "Gemma3-27B",
                                         "InternVL3-2B", "InternVL3-14B", "InternVL3-78B"))
    )



# Recoding slurs
data_machine <- data_machine %>% mutate(slur = replace_na(slur, "No slur")) %>%
    mutate(slur = as.factor(slur))

data_machine$slur <- fct_recode(data_machine$slur, "Racism" = "nigger",
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


# Transforming variables to factors for analysis
data_machine <- data_machine %>% mutate(
    interaction = as.factor(interaction),
    identity = as.factor(identity),
    topic = as.factor(topic),
    engagement = as.factor(ifelse(engagement == "L", "Low", "High")),
    reply = as.factor(reply),
    slur = as.factor(slur),
    curse = as.factor(curse)
) %>% select(id, chosen, interaction, identity, topic, engagement, reply, curse, slur, model)


# Fixing reference categories
data_machine$topic <- relevel(data_machine$topic, ref = "Everyday")
data_machine$reply <- relevel(data_machine$reply, ref = "None")
data_machine$engagement <- relevel(data_machine$engagement, ref = "Low")
data_machine$curse <- relevel(data_machine$curse, ref = "No Curse")


# Splitting data_machine by model
# GPT-4o
data_gpt_baseline <- data_machine %>% filter(model == "GPT-4o")

# GPT-4o-mini
data_gptmini_baseline <- data_machine %>% filter(model == "GPT-4o mini")

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

amces_main <- bind_rows(amces_main_machine_b_gpt,
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




model_order <- rev(levels(amces_main_filtered$out))
feature_order <- c("Slur", "Identity", "Topic", "Curse", "Reply", "Engagement")

slur_order <- c(
    "No slur", "Sexism", "Reverse\nracism", "Reclaimed\nslur", "Homophobia", "Racism"
)

# Constructing a dataset containing information for "striping"
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
leg_gpt <- get_legend(create_family_legend(c("GPT-4o", "GPT-4o mini"), custom_colors, model_shapes))
leg_gemini <- get_legend(create_family_legend(c("Gemini 2.5 Flash", "Gemini 2.5 Flash Lite"), custom_colors, model_shapes))
leg_qwen <- get_legend(create_family_legend(c("Qwen2 72B", "Qwen2 7B", "Qwen2 2B"), custom_colors, model_shapes))
leg_internvl <- get_legend(create_family_legend(c("InternVL3 78B", "InternVL3 14B", "InternVL3 2B"), custom_colors, model_shapes))
leg_gemma <- get_legend(create_family_legend(c("Gemma3 27B", "Gemma3 12B", "Gemma3 4B"), custom_colors, model_shapes))

# Combine all legends horizontally
legend_row <- plot_grid(leg_gpt, leg_gemini, leg_qwen, leg_gemma, leg_internvl,
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

ggsave("../figures/supplementary_fig_6.pdf")#, width = 180, height = 220, units = "mm")


########################################################
# PART II: Variation in slur perception by "user" race #
########################################################


mms_slurs_b_by_speaker_gpt <- mm_diffs_bsp(data_gpt_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o")

mms_slurs_b_by_speaker_gptmini <- mm_diffs_bsp(data_gptmini_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "GPT-4o mini")

mms_slurs_b_by_speaker_gf <- mm_diffs_bsp(data_flash_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash")

mms_slurs_b_by_speaker_gfl <- mm_diffs_bsp(data_flashlite_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemini 2.5 Flash Lite")

mms_slurs_b_by_speaker_qwen2 <- mm_diffs_bsp(data_qwen2_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 2B")

mms_slurs_b_by_speaker_qwen7 <- mm_diffs_bsp(data_qwen7_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 7B")

mms_slurs_b_by_speaker_qwen72 <- mm_diffs_bsp(data_qwen72_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Qwen2 72B")

mms_slurs_b_by_speaker_gemma4 <- mm_diffs_bsp(data_gemma4_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 4B")

mms_slurs_b_by_speaker_gemma12 <- mm_diffs_bsp(data_gemma12_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 12B")

mms_slurs_b_by_speaker_gemma27 <- mm_diffs_bsp(data_gemma27_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "Gemma3 27B")

mms_slurs_b_by_speaker_internvl2 <- mm_diffs_bsp(data_internvl2_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 2B")

mms_slurs_b_by_speaker_internvl14 <- mm_diffs_bsp(data_internvl14_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 14B")

mms_slurs_b_by_speaker_internvl78 <- mm_diffs_bsp(data_internvl78_baseline, chosen ~ slur, by = ~identity) %>%
    mutate(model = "InternVL3 78B")

# Step 2: Combine all datasets
mms_slurs_by_speaker <- bind_rows(
    mms_slurs_b_by_speaker_gpt,
    mms_slurs_b_by_speaker_gptmini,
    mms_slurs_b_by_speaker_gf,
    mms_slurs_b_by_speaker_gfl,
    mms_slurs_b_by_speaker_qwen2,
    mms_slurs_b_by_speaker_qwen7,
    mms_slurs_b_by_speaker_qwen72,
    mms_slurs_b_by_speaker_gemma4,
    mms_slurs_b_by_speaker_gemma12,
    mms_slurs_b_by_speaker_gemma27,
    mms_slurs_b_by_speaker_internvl2,
    mms_slurs_b_by_speaker_internvl14,
    mms_slurs_b_by_speaker_internvl78
)

# Step 3: Adjust factor levels for ordered facets
mms_slurs_by_speaker$model <- factor(mms_slurs_by_speaker$model,
                                     levels = c("GPT-4o mini", "GPT-4o", "Gemini 2.5 Flash Lite", "Gemini 2.5 Flash",  "Qwen2 2B", "Qwen2 7B", "Qwen2 72B", "Gemma3 4B", "Gemma3 12B", "Gemma3 27B", "InternVL3 2B", "InternVL3 14B", "InternVL3 78B")) # Organized by model family


# Set proper factor level order for slur types (reversed because ggplot displays bottom to top)
mms_slurs_by_speaker$level <- factor(mms_slurs_by_speaker$level,
                                    levels = c("No slur", "Racism", "Reclaimed\nslur", "Reverse\nracism",
                                              "Sexism", "Homophobia", "Generic\ninsult"))


# Modify levels to replace newline with spaces
# e.g. "Reclaimed\nslur" becomes "Reclaimed slur"
mms_slurs_by_speaker$level_mod <- gsub("\n", " ", mms_slurs_by_speaker$level)

# Define the model order once
model_levels <- rev(unique(mms_slurs_by_speaker$model))
#model_levels <- c(setdiff(model_levels, "Human"), "Human")

# Prepare plot data and assign stripe bands by model within each facet
mms_striped <- mms_slurs_by_speaker %>%
    filter(
        level %in% c("Racism", "Reclaimed\nslur", "Reverse\nracism", "Sexism", "No slur")
    ) %>%
    mutate(model = factor(model, levels = model_levels)) %>%
    group_by(level_mod) %>%
    mutate(
        model_num = dense_rank(model),
        stripe = model_num %% 2 == 0
    ) %>%
    ungroup()

# Plot with facet-specific zebra striping
ggplot(mms_striped,
       aes(x = estimate, y = model, color = identity)) +

    # Zebra striping layer — facet-aware
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
ggsave("../figures/supplementary_fig_7.pdf", width = 180, height = 220, units = "mm")

# Why null effects for some? Slurs always chosen
# Notably, models almost always select certain slurs
data_gpt_baseline %>% filter(slur == "Racism") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_flash_baseline %>% filter(slur == "Racism") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_gpt_baseline %>% filter(slur == "Reclaimed\nslur") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_flash_baseline %>% filter(slur == "Reclaimed\nslur") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_gpt_baseline %>% filter(slur == "Reverse\nracism") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_flash_baseline %>% filter(slur == "Reverse\nracism") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_gpt_baseline %>% filter(slur == "Sexism") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())
data_flash_baseline %>% filter(slur == "Sexism") %>% summarize(pct_chosen = 100*sum(chosen)/n(), n_chosen = sum(chosen), n = n())

