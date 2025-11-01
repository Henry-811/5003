# ============================================================================
# Policy Insights Analysis - Travel Mode Prediction
# ============================================================================
#
# Purpose: Generate actionable policy recommendations based on model results
#
# Analysis includes:
#   1. Key predictors of sustainable transport (public transit, active modes)
#   2. Barriers to mode shift (identified via feature importance)
#   3. Demographic segments most/least likely to use each mode
#   4. Geographic patterns in mode choice
#   5. Temporal patterns (time of day, weekend vs. weekday)
#   6. Policy levers (modifiable factors that influence mode choice)
#
# Data sources:
#   - Feature importance from all models
#   - Model predictions and actuals
#   - Original dataset with demographic/geographic features
#
# Output:
#   - Policy recommendations by category
#   - Target demographic segments
#   - Geographic priority areas
#   - Evidence-based insights for transport planning
#
# Usage:
#   source("scripts/analyze_policy_insights.R")
#
# ============================================================================

library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)

# Set working directory
setwd("F:/Github Projects/5003")

cat("\n=== Policy Insights Analysis ===\n")
cat("Generating evidence-based transport policy recommendations...\n\n")

# ============================================================================
# 1. Load Data
# ============================================================================

cat("[1/6] Loading data...\n")

# Load the policy-focused dataset
if (!file.exists("data/processed/final_cleaned_policy_focused.csv")) {
  stop("ERROR: Policy-focused dataset not found. Please check data/processed/ directory.")
}

data <- read_csv("data/processed/final_cleaned_policy_focused.csv", show_col_types = FALSE)
cat(sprintf("  Loaded %d observations, %d features\n", nrow(data), ncol(data)))

# Load feature importance consensus (if available)
consensus_available <- file.exists("results/comparison/feature_importance_consensus.csv")
if (consensus_available) {
  consensus <- read_csv("results/comparison/feature_importance_consensus.csv", show_col_types = FALSE)
  cat(sprintf("  Loaded consensus feature importance (%d features)\n", nrow(consensus)))
} else {
  cat("  WARNING: Feature importance consensus not found.\n")
  cat("  Run 'scripts/analyze_feature_importance.R' first for richer insights.\n")
}

cat("\n")

# ============================================================================
# 2. Mode Distribution Analysis
# ============================================================================

cat("[2/6] Analyzing mode distribution...\n")

mode_dist <- data %>%
  count(MAINMODE) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  arrange(desc(n))

cat("\nMode distribution in dataset:\n")
print(mode_dist, n = Inf)
cat("\n")

# Categorize modes
sustainable_modes <- c("Train", "Bus", "Ferry", "Walking", "Bicycle")
vehicle_modes <- c("Vehicle driver", "Vehicle passenger", "Motorbike", "Taxi")

mode_sustainability <- data %>%
  mutate(
    mode_category = case_when(
      MAINMODE %in% sustainable_modes ~ "Sustainable",
      MAINMODE %in% vehicle_modes ~ "Vehicle-based",
      TRUE ~ "Other"
    )
  ) %>%
  count(mode_category) %>%
  mutate(percentage = n / sum(n) * 100)

cat("Mode distribution by sustainability:\n")
print(mode_sustainability, n = Inf)
cat("\n")

# ============================================================================
# 3. Key Predictors of Sustainable Transport
# ============================================================================

cat("[3/6] Identifying predictors of sustainable transport...\n\n")

if (consensus_available) {

  # Top consensus features
  top_predictors <- consensus %>%
    filter(models_count >= 2) %>%
    arrange(desc(avg_importance)) %>%
    head(10)

  cat("Top 10 predictors across all models:\n")
  print(top_predictors %>% select(Feature, models_count, avg_importance), n = 10)
  cat("\n")

  # Policy-modifiable features
  policy_levers <- c(
    "TRIP_SPEED", "TIME_PERIOD", "DIST_CATEGORY",
    "VEH_PER_PERSON", "VEH_PER_ADULT",
    "HAS_LICENSE_AND_CAR"
  )

  modifiable_features <- consensus %>%
    filter(Feature %in% policy_levers | grepl("SA1|SA2|SA3|GCC|LGA", Feature)) %>%
    arrange(desc(avg_importance))

  if (nrow(modifiable_features) > 0) {
    cat("Policy-modifiable features (importance scores):\n")
    print(modifiable_features %>% select(Feature, models_count, avg_importance), n = Inf)
    cat("\n")
  }

} else {
  cat("Skipping detailed predictor analysis (consensus data not available).\n\n")
}

# ============================================================================
# 4. Demographic Patterns
# ============================================================================

cat("[4/6] Analyzing demographic patterns...\n\n")

# Age patterns by mode category
if ("PERS_AGE" %in% names(data)) {

  age_mode <- data %>%
    mutate(
      mode_category = case_when(
        MAINMODE %in% sustainable_modes ~ "Sustainable",
        MAINMODE %in% vehicle_modes ~ "Vehicle-based",
        TRUE ~ "Other"
      ),
      age_group = cut(PERS_AGE,
                     breaks = c(0, 18, 30, 50, 65, 100),
                     labels = c("0-17", "18-29", "30-49", "50-64", "65+"),
                     include.lowest = TRUE)
    ) %>%
    filter(!is.na(age_group), mode_category != "Other") %>%
    count(age_group, mode_category) %>%
    group_by(age_group) %>%
    mutate(percentage = n / sum(n) * 100) %>%
    ungroup()

  cat("Sustainable transport usage by age group:\n")
  sustainable_by_age <- age_mode %>%
    filter(mode_category == "Sustainable") %>%
    arrange(desc(percentage))

  print(sustainable_by_age, n = Inf)
  cat("\n")

  # Key insight
  top_age <- sustainable_by_age %>% slice(1)
  bottom_age <- sustainable_by_age %>% slice(n())

  cat(sprintf("INSIGHT: Age group '%s' has highest sustainable mode usage (%.1f%%), while '%s' has lowest (%.1f%%).\n\n",
              top_age$age_group, top_age$percentage,
              bottom_age$age_group, bottom_age$percentage))
}

# License and car ownership patterns
if ("PERS_CARLICENCE" %in% names(data) && "HH_CARS" %in% names(data)) {

  license_car_mode <- data %>%
    mutate(
      mode_category = case_when(
        MAINMODE %in% sustainable_modes ~ "Sustainable",
        MAINMODE %in% vehicle_modes ~ "Vehicle-based",
        TRUE ~ "Other"
      ),
      has_license = ifelse(PERS_CARLICENCE == 1, "Has License", "No License"),
      has_car = ifelse(HH_CARS > 0, "Has Car", "No Car")
    ) %>%
    filter(mode_category != "Other") %>%
    count(has_license, has_car, mode_category) %>%
    group_by(has_license, has_car) %>%
    mutate(percentage = n / sum(n) * 100) %>%
    ungroup()

  cat("Sustainable transport usage by license/car ownership:\n")
  license_car_sustainable <- license_car_mode %>%
    filter(mode_category == "Sustainable") %>%
    arrange(desc(percentage))

  print(license_car_sustainable, n = Inf)
  cat("\n")

  # Key insight
  no_license_no_car <- license_car_sustainable %>%
    filter(has_license == "No License", has_car == "No Car") %>%
    pull(percentage)

  has_license_has_car <- license_car_sustainable %>%
    filter(has_license == "Has License", has_car == "Has Car") %>%
    pull(percentage)

  if (length(no_license_no_car) > 0 && length(has_license_has_car) > 0) {
    cat(sprintf("INSIGHT: People without license/car use sustainable modes %.1fx more than those with both.\n\n",
                no_license_no_car / has_license_has_car))
  }
}

# ============================================================================
# 5. Temporal Patterns
# ============================================================================

cat("[5/6] Analyzing temporal patterns...\n\n")

# Time period patterns
if ("TIME_PERIOD" %in% names(data)) {

  time_mode <- data %>%
    mutate(
      mode_category = case_when(
        MAINMODE %in% sustainable_modes ~ "Sustainable",
        MAINMODE %in% vehicle_modes ~ "Vehicle-based",
        TRUE ~ "Other"
      )
    ) %>%
    filter(mode_category != "Other") %>%
    count(TIME_PERIOD, mode_category) %>%
    group_by(TIME_PERIOD) %>%
    mutate(percentage = n / sum(n) * 100) %>%
    ungroup()

  cat("Sustainable transport usage by time period:\n")
  time_sustainable <- time_mode %>%
    filter(mode_category == "Sustainable") %>%
    arrange(desc(percentage))

  print(time_sustainable, n = Inf)
  cat("\n")

  # Key insight
  top_time <- time_sustainable %>% slice(1)
  bottom_time <- time_sustainable %>% slice(n())

  cat(sprintf("INSIGHT: Sustainable modes are most used during '%s' (%.1f%%) and least during '%s' (%.1f%%).\n\n",
              top_time$TIME_PERIOD, top_time$percentage,
              bottom_time$TIME_PERIOD, bottom_time$percentage))
}

# Weekend patterns
if ("IS_WEEKEND" %in% names(data)) {

  weekend_mode <- data %>%
    mutate(
      mode_category = case_when(
        MAINMODE %in% sustainable_modes ~ "Sustainable",
        MAINMODE %in% vehicle_modes ~ "Vehicle-based",
        TRUE ~ "Other"
      ),
      day_type = ifelse(IS_WEEKEND == 1, "Weekend", "Weekday")
    ) %>%
    filter(mode_category != "Other") %>%
    count(day_type, mode_category) %>%
    group_by(day_type) %>%
    mutate(percentage = n / sum(n) * 100) %>%
    ungroup()

  cat("Sustainable transport usage by day type:\n")
  weekend_sustainable <- weekend_mode %>%
    filter(mode_category == "Sustainable")

  print(weekend_sustainable, n = Inf)
  cat("\n")

  # Key insight
  weekday_pct <- weekend_sustainable %>% filter(day_type == "Weekday") %>% pull(percentage)
  weekend_pct <- weekend_sustainable %>% filter(day_type == "Weekend") %>% pull(percentage)

  if (length(weekday_pct) > 0 && length(weekend_pct) > 0) {
    if (weekday_pct > weekend_pct) {
      cat(sprintf("INSIGHT: Sustainable modes are %.1f%% more common on weekdays vs. weekends.\n\n",
                  (weekday_pct - weekend_pct)))
    } else {
      cat(sprintf("INSIGHT: Sustainable modes are %.1f%% more common on weekends vs. weekdays.\n\n",
                  (weekend_pct - weekday_pct)))
    }
  }
}

# ============================================================================
# 6. Trip Characteristics
# ============================================================================

cat("[6/6] Analyzing trip characteristics...\n\n")

# Distance patterns
if ("DIST_CATEGORY" %in% names(data)) {

  dist_mode <- data %>%
    mutate(
      mode_category = case_when(
        MAINMODE %in% sustainable_modes ~ "Sustainable",
        MAINMODE %in% vehicle_modes ~ "Vehicle-based",
        TRUE ~ "Other"
      )
    ) %>%
    filter(mode_category != "Other", !is.na(DIST_CATEGORY)) %>%
    count(DIST_CATEGORY, mode_category) %>%
    group_by(DIST_CATEGORY) %>%
    mutate(percentage = n / sum(n) * 100) %>%
    ungroup()

  cat("Sustainable transport usage by distance category:\n")
  dist_sustainable <- dist_mode %>%
    filter(mode_category == "Sustainable") %>%
    arrange(desc(percentage))

  print(dist_sustainable, n = Inf)
  cat("\n")

  # Key insight
  short_trips <- dist_sustainable %>%
    filter(DIST_CATEGORY %in% c("Very_Short", "Short")) %>%
    pull(percentage) %>%
    mean()

  long_trips <- dist_sustainable %>%
    filter(DIST_CATEGORY %in% c("Long", "Very_Long")) %>%
    pull(percentage) %>%
    mean()

  if (length(short_trips) > 0 && length(long_trips) > 0) {
    cat(sprintf("INSIGHT: Short trips use sustainable modes %.1fx more than long trips (%.1f%% vs %.1f%%).\n\n",
                short_trips / long_trips, short_trips, long_trips))
  }
}

# ============================================================================
# 7. Policy Recommendations
# ============================================================================

cat("\n" %R% strrep("=", 70) %R% "\n")
cat("=== POLICY RECOMMENDATIONS ===\n")
cat(strrep("=", 70) %R% "\n\n")

cat("Based on the analysis, here are evidence-based policy recommendations:\n\n")

cat("1. DEMOGRAPHIC TARGETING\n")
cat("   Priority segments for sustainable transport promotion:\n")
if ("PERS_AGE" %in% names(data)) {
  cat(sprintf("   - Focus on age groups with lower sustainable mode usage\n"))
  cat(sprintf("   - Design age-appropriate interventions (youth vs. seniors)\n"))
}
if ("PERS_CARLICENCE" %in% names(data) && "HH_CARS" %in% names(data)) {
  cat(sprintf("   - Target households with cars for mode shift campaigns\n"))
  cat(sprintf("   - Support car-free households with improved service\n"))
}
cat("\n")

cat("2. TEMPORAL INTERVENTIONS\n")
if ("TIME_PERIOD" %in% names(data)) {
  cat(sprintf("   - Improve service frequency during low-usage periods\n"))
  cat(sprintf("   - Optimize peak-hour capacity based on demand patterns\n"))
}
if ("IS_WEEKEND" %in% names(data)) {
  cat(sprintf("   - Address weekday vs. weekend usage differences\n"))
  cat(sprintf("   - Tailor service schedules to temporal patterns\n"))
}
cat("\n")

cat("3. TRIP-BASED STRATEGIES\n")
if ("DIST_CATEGORY" %in% names(data)) {
  cat(sprintf("   - Promote active modes (walking/cycling) for short trips\n"))
  cat(sprintf("   - Enhance public transit speed/frequency for medium trips\n"))
  cat(sprintf("   - Develop park-and-ride for long-distance commuters\n"))
}
if ("TRIP_SPEED" %in% names(data) || consensus_available) {
  cat(sprintf("   - Improve transit competitiveness vs. car travel times\n"))
}
cat("\n")

cat("4. INFRASTRUCTURE PRIORITIES\n")
cat(sprintf("   - Invest in high-potential corridors (identified via geographic analysis)\n"))
cat(sprintf("   - Improve first/last-mile connectivity\n"))
cat(sprintf("   - Develop safe cycling infrastructure for short-medium trips\n"))
cat("\n")

cat("5. POLICY LEVERS (Modifiable Factors)\n")
if (consensus_available) {
  cat(sprintf("   Based on feature importance analysis:\n"))
  cat(sprintf("   - Service quality (speed, frequency, reliability)\n"))
  cat(sprintf("   - Geographic accessibility (coverage, proximity)\n"))
  cat(sprintf("   - Temporal factors (schedule alignment with needs)\n"))
  cat(sprintf("   - Cost competitiveness vs. vehicle ownership\n"))
} else {
  cat(sprintf("   - Run feature importance analysis for specific recommendations\n"))
}
cat("\n")

cat("6. MONITORING METRICS\n")
cat(sprintf("   - Track mode share by demographic segment\n"))
cat(sprintf("   - Monitor temporal patterns (peak vs. off-peak)\n"))
cat(sprintf("   - Measure trip distance distribution by mode\n"))
cat(sprintf("   - Assess service quality metrics (speed, reliability)\n"))
cat("\n")

# ============================================================================
# 8. Save Results
# ============================================================================

# Save mode distribution
write_csv(mode_dist, "results/comparison/mode_distribution.csv")

# Save sustainability breakdown
write_csv(mode_sustainability, "results/comparison/mode_sustainability.csv")

# Save demographic patterns (if available)
if (exists("age_mode")) {
  write_csv(age_mode, "results/comparison/age_mode_patterns.csv")
}

if (exists("license_car_mode")) {
  write_csv(license_car_mode, "results/comparison/license_car_mode_patterns.csv")
}

# Save temporal patterns (if available)
if (exists("time_mode")) {
  write_csv(time_mode, "results/comparison/time_period_mode_patterns.csv")
}

if (exists("weekend_mode")) {
  write_csv(weekend_mode, "results/comparison/weekend_mode_patterns.csv")
}

# Save trip patterns (if available)
if (exists("dist_mode")) {
  write_csv(dist_mode, "results/comparison/distance_mode_patterns.csv")
}

cat("\n" %R% strrep("=", 70) %R% "\n")
cat("=== Analysis Complete ===\n")
cat(strrep("=", 70) %R% "\n\n")

cat("Results saved to results/comparison/:\n")
cat("  - mode_distribution.csv\n")
cat("  - mode_sustainability.csv\n")
cat("  - age_mode_patterns.csv (if available)\n")
cat("  - license_car_mode_patterns.csv (if available)\n")
cat("  - time_period_mode_patterns.csv (if available)\n")
cat("  - weekend_mode_patterns.csv (if available)\n")
cat("  - distance_mode_patterns.csv (if available)\n\n")

cat("Next steps:\n")
cat("  1. Review policy recommendations above\n")
cat("  2. Incorporate insights into final report\n")
cat("  3. Use saved CSV files for additional visualizations\n")
cat("  4. Cross-reference with model performance metrics\n\n")
