# ==============================================================================
# Shared Preprocessing Script for All 5 Models
# ==============================================================================
# This script creates identical train/test splits for fair model comparison
# Each model can then apply its own specific transformations
# ==============================================================================

library(dplyr)
library(tidyr)
library(caret)

# Set seed for reproducibility - CRITICAL for fair comparison
set.seed(5003)

cat("Starting shared preprocessing...\n")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

data_path <- "final_decoded.csv"
if (!file.exists(data_path)) {
  stop("Error: final_decoded.csv not found. Please ensure the file is in the working directory.")
}

df <- read.csv(data_path,
               header = TRUE,
               fileEncoding = "UTF-8",
               na.strings = c("", "NA", "N/A"))

cat(sprintf("Loaded data: %d rows, %d columns\n", nrow(df), ncol(df)))

# ==============================================================================
# 2. IDENTIFY TARGET VARIABLE
# ==============================================================================

target_col <- "STOP_MAINMODE"
if (!target_col %in% names(df)) {
  stop("Error: Target variable 'STOP_MAINMODE' not found in dataset")
}

cat(sprintf("\nTarget variable: %s\n", target_col))
cat("Class distribution:\n")
print(table(df[[target_col]]))

# ==============================================================================
# 3. REMOVE LEAKY/POSTERIOR FEATURES
# ==============================================================================
# Based on Week 7 analysis - these features are either:
# - Synonymous with target (TRIP_LINKMODE, TRIP_MODE1)
# - Posterior/leaky features (collected AFTER the trip)

leaky_features <- c(
  "TRIP_LINKMODE",     # Synonymous with target
  "TRIP_MODE1",        # Synonymous with target
  "STOPID"             # ID column, not predictive
)

# Remove leaky features if they exist
leaky_present <- leaky_features[leaky_features %in% names(df)]
if (length(leaky_present) > 0) {
  cat("\nRemoving leaky features:", paste(leaky_present, collapse = ", "), "\n")
  df <- df %>% select(-all_of(leaky_present))
}

# Note: Review VEH_VEHYEAR and STOP_NETWORK_DIST for potential leakage
# For now, we keep them but can remove later if needed

# ==============================================================================
# 4. HANDLE MISSING VALUES
# ==============================================================================

cat("\nHandling missing values...\n")

# Calculate missing rates before imputation
missing_summary <- df %>%
  summarise(across(everything(), ~sum(is.na(.))/n()*100)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_pct") %>%
  arrange(desc(missing_pct))

cat(sprintf("Variables with >50%% missing: %d\n",
            sum(missing_summary$missing_pct > 50)))

# Strategy:
# - Numeric: Impute with median
# - Categorical: Impute with mode or create "Missing" category
# - High missingness (>80%): Consider dropping

# Identify numeric and categorical columns
numeric_cols <- names(df)[sapply(df, is.numeric)]
categorical_cols <- names(df)[!sapply(df, is.numeric)]

# Remove target from imputation lists
numeric_cols <- setdiff(numeric_cols, target_col)
categorical_cols <- setdiff(categorical_cols, target_col)

# Impute numeric with median
for (col in numeric_cols) {
  if (sum(is.na(df[[col]])) > 0) {
    median_val <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- median_val
  }
}

# Impute categorical with mode or "Missing"
for (col in categorical_cols) {
  if (sum(is.na(df[[col]])) > 0) {
    # Create "Missing" category for high missingness
    df[[col]][is.na(df[[col]])] <- "Missing"
    df[[col]] <- as.factor(df[[col]])
  }
}

cat("Missing value imputation complete.\n")

# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================

cat("\nEngineering features...\n")

# Create new features based on domain knowledge
df <- df %>%
  mutate(
    # Trip characteristics
    TRIP_SPEED_RATIO = ifelse(TRIP_TOTTRIPTIME > 0,
                               TRIP_NETWORK_DIST / TRIP_TOTTRIPTIME, 0),

    # Household vehicle ratios
    VEH_PER_PERSON = ifelse(HH_HHSIZE > 0, HH_TOTALVEHS / HH_HHSIZE, 0),

    # Time of day categories
    TIME_CATEGORY = case_when(
      STOP_STARTHR >= 6 & STOP_STARTHR < 9 ~ "Morning_Peak",
      STOP_STARTHR >= 9 & STOP_STARTHR < 16 ~ "Midday",
      STOP_STARTHR >= 16 & STOP_STARTHR < 19 ~ "Evening_Peak",
      STOP_STARTHR >= 19 | STOP_STARTHR < 6 ~ "Off_Peak",
      TRUE ~ "Unknown"
    ),

    # Age-income interaction (if these columns exist)
    # AGE_INCOME_GROUP = paste(PERS_AGEGROUP, PERS_PERSINC, sep = "_")
  )

# Handle any Inf or NaN values from divisions
df <- df %>%
  mutate(across(where(is.numeric), ~replace(., is.infinite(.) | is.nan(.), 0)))

cat(sprintf("Feature engineering complete. Total features: %d\n", ncol(df)))

# ==============================================================================
# 6. REMOVE ROWS WITH MISSING TARGET
# ==============================================================================

before_rows <- nrow(df)
df <- df %>% filter(!is.na(!!sym(target_col)))
after_rows <- nrow(df)

if (before_rows != after_rows) {
  cat(sprintf("Removed %d rows with missing target variable\n",
              before_rows - after_rows))
}

# ==============================================================================
# 7. TRAIN/TEST SPLIT - STRATIFIED
# ==============================================================================
# CRITICAL: This must be identical for all 5 models!

cat("\nCreating stratified train/test split...\n")

# 80% train, 20% test (stratified by target)
train_index <- createDataPartition(df[[target_col]],
                                   p = 0.8,
                                   list = FALSE,
                                   times = 1)

train_base <- df[train_index, ]
test_base <- df[-train_index, ]

cat(sprintf("Train set: %d rows\n", nrow(train_base)))
cat(sprintf("Test set: %d rows\n", nrow(test_base)))

# Verify class distribution is preserved
cat("\nTrain set class distribution:\n")
print(prop.table(table(train_base[[target_col]])))
cat("\nTest set class distribution:\n")
print(prop.table(table(test_base[[target_col]])))

# ==============================================================================
# 8. SAVE PROCESSED DATA
# ==============================================================================

# Create data directory if it doesn't exist
if (!dir.exists("data")) {
  dir.create("data")
}

saveRDS(train_base, "data/train_base.rds")
saveRDS(test_base, "data/test_base.rds")

cat("\nSaved processed data to:\n")
cat("  - data/train_base.rds\n")
cat("  - data/test_base.rds\n")

# Also save feature names for reference
feature_names <- setdiff(names(df), target_col)
saveRDS(list(
  target = target_col,
  features = feature_names,
  numeric_features = intersect(feature_names, numeric_cols),
  categorical_features = intersect(feature_names, categorical_cols)
), "data/feature_metadata.rds")

cat("\nShared preprocessing complete!\n")
cat("All 5 models should now use data/train_base.rds and data/test_base.rds\n")
cat("Each model can apply its own specific transformations (scaling, encoding, etc.)\n")
