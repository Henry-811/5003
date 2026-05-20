# ==============================================================================
# Enhanced Preprocessing Script
# ==============================================================================
# Improves upon Week 7 preprocessing (final_decoded_clean.csv)
# Implements advanced techniques for XGBoost optimization
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(caret)
  library(recipes)  # For advanced preprocessing
})

# Optional: For advanced imputation
if (requireNamespace("mice", quietly = TRUE)) {
  library(mice)
  has_mice <- TRUE
} else {
  cat("Note: 'mice' package not available. Using simpler imputation.\n")
  has_mice <- FALSE
}

set.seed(5003)  # For reproducibility

cat("==============================================================================\n")
cat("ENHANCED PREPROCESSING FOR XGBOOST\n")
cat("==============================================================================\n\n")

# ==============================================================================
# CONFIGURATION - SOURCE DATASET PATH
# ==============================================================================
# Set the path to the cleaned dataset you want to use:
# This can be set from main.R or directly here
if (!exists("INPUT_DATA_PATH")) {
  INPUT_DATA_PATH <- "data/processed/final_cleaned_policy_focused.csv"  # Default: policy-focused (91 features)
  # INPUT_DATA_PATH <- "data/raw/data_test.csv"                         # Alternative: original (130+ features)
}

cat("Dataset Configuration:\n")
cat(sprintf("  Input: %s\n\n", INPUT_DATA_PATH))

# Create directories
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("results/preprocessing")) dir.create("results/preprocessing", recursive = TRUE)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

cat("Step 1: Loading data...\n")

# Check if file exists
if (!file.exists(INPUT_DATA_PATH)) {
  stop(sprintf("Error: Dataset not found at %s", INPUT_DATA_PATH))
}

# Determine dataset type from filename
if (grepl("policy_focused", INPUT_DATA_PATH)) {
  dataset_type <- "policy_focused"
  cat("✓ Using policy-focused dataset (vehicle proxies removed)\n")
} else if (grepl("data_test", INPUT_DATA_PATH) || grepl("final_cleaned_data", INPUT_DATA_PATH)) {
  dataset_type <- "original"
  cat("✓ Using original dataset (with vehicle proxies)\n")
} else {
  dataset_type <- "unknown"
  cat("✓ Using custom dataset\n")
}

input_file <- INPUT_DATA_PATH

df_original <- read.csv(input_file,
                        header = TRUE,
                        fileEncoding = "UTF-8",
                        na.strings = c("", "NA", "N/A", "NULL"))

cat(sprintf("✓ Loaded: %d rows × %d columns (%s)\n\n",
            nrow(df_original), ncol(df_original), dataset_type))

# Keep copy for before/after comparison
df <- df_original

# ==============================================================================
# 2. IDENTIFY COLUMNS
# ==============================================================================

cat("Step 2: Identifying column types...\n")

target_col <- "STOP_MAINMODE"
if (!target_col %in% names(df)) {
  target_col <- names(df)[grep("MODE", names(df), ignore.case = TRUE)][1]
  cat(sprintf("Using '%s' as target\n", target_col))
}

numeric_cols <- names(df)[sapply(df, is.numeric)]
categorical_cols <- names(df)[!sapply(df, is.numeric)]

# Remove target from feature lists
numeric_cols <- setdiff(numeric_cols, target_col)
categorical_cols <- setdiff(categorical_cols, target_col)

cat(sprintf("Target: %s\n", target_col))
cat(sprintf("Numeric features: %d\n", length(numeric_cols)))
cat(sprintf("Categorical features: %d\n\n", length(categorical_cols)))

# ==============================================================================
# 3. REMOVE ID COLUMNS AND LEAKY/POSTERIOR FEATURES
# ==============================================================================

cat("Step 3: Removing ID columns and leaky features...\n")

# ID columns should not be used as features (no predictive value, cause overfitting)
id_columns <- c("STOPID", "PERSID", "TRIPID", "VEHID", "HHID", "VEH_HHID")
id_columns_present <- intersect(id_columns, names(df))

if (length(id_columns_present) > 0) {
  cat(sprintf("Removing %d ID columns:\n", length(id_columns_present)))
  print(id_columns_present)
  df <- df %>% select(-all_of(id_columns_present))
  cat("\n")
} else {
  cat("No ID columns found\n\n")
}

# Features that are collected AFTER mode choice or are synonymous with target
# DATA LEAKAGE PREVENTION:
# These features either:
# 1. Are the same as target (FULLMODE ≈ MAINMODE at different granularity)
# 2. Are determined AFTER mode choice (FARETYPE, TICKETTYPE chosen after selecting transport)
# 3. Contain the target encoded in different form (MODE1-9, LINKMODE)
# Removing these ensures model learns from legitimate predictors only
leaky_patterns <- c("LINKMODE", "MODE1", "MODE2", "TRIPMODE", "FULLMODE",
                    "FARETYPE", "TICKETTYPE")
leaky_features <- names(df)[sapply(names(df), function(x) {
  any(sapply(leaky_patterns, function(pat) grepl(pat, x, ignore.case = TRUE)))
})]

# Keep target but remove other leaky features
leaky_features <- setdiff(leaky_features, target_col)

if (length(leaky_features) > 0) {
  cat(sprintf("Removing %d leaky features:\n", length(leaky_features)))
  print(leaky_features)
  df <- df %>% select(-all_of(leaky_features))
  cat("\n")
} else {
  cat("No leaky features detected\n\n")
}

# ==============================================================================
# 4. HANDLE MISSING VALUES - INTELLIGENT STRATEGY
# ==============================================================================

cat("Step 4: Handling missing values...\n")

# Calculate missing percentages
missing_pct <- sapply(df, function(x) sum(is.na(x))/length(x)*100)

# Strategy:
# - >80% missing: DROP
# - 20-80% missing: Predictive imputation OR create indicator + simple imputation
# - <20% missing: Simple imputation (median/mode)

# Drop columns with >80% missing
high_missing_cols <- names(missing_pct)[missing_pct > 80 & names(missing_pct) != target_col]
if (length(high_missing_cols) > 0) {
  cat(sprintf("Dropping %d columns with >80%% missing:\n", length(high_missing_cols)))
  print(high_missing_cols)
  df <- df %>% select(-all_of(high_missing_cols))
  cat("\n")
}

# Update column lists
numeric_cols <- setdiff(numeric_cols, high_missing_cols)
categorical_cols <- setdiff(categorical_cols, high_missing_cols)

# Create missingness indicators for important features with moderate missing
moderate_missing_cols <- names(missing_pct)[missing_pct > 10 & missing_pct <= 80]
moderate_missing_cols <- intersect(moderate_missing_cols, names(df))

if (length(moderate_missing_cols) > 0) {
  cat(sprintf("Creating missingness indicators for %d columns...\n", length(moderate_missing_cols)))
  for (col in moderate_missing_cols) {
    indicator_col <- paste0(col, "_missing")
    df[[indicator_col]] <- as.integer(is.na(df[[col]]))
  }
  cat("\n")
}

# Simple imputation for remaining missing values
cat("Imputing remaining missing values...\n")

# Numeric: median imputation
for (col in numeric_cols) {
  if (col %in% names(df) && sum(is.na(df[[col]])) > 0) {
    median_val <- median(df[[col]], na.rm = TRUE)
    if (!is.na(median_val)) {
      df[[col]][is.na(df[[col]])] <- median_val
    }
  }
}

# Categorical: mode imputation or "Missing" category
for (col in categorical_cols) {
  if (col %in% names(df) && sum(is.na(df[[col]])) > 0) {
    # Get mode
    mode_val <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
    if (!is.na(mode_val) && mode_val != "") {
      df[[col]][is.na(df[[col]])] <- mode_val
    } else {
      df[[col]][is.na(df[[col]])] <- "Missing"
    }
    df[[col]] <- as.factor(df[[col]])
  }
}

cat("✓ Missing value handling complete\n\n")

# ==============================================================================
# 5. REMOVE ZERO/NEAR-ZERO VARIANCE FEATURES
# ==============================================================================

cat("Step 5: Removing zero/near-zero variance features...\n")

# Check numeric columns
numeric_in_df <- intersect(numeric_cols, names(df))
zero_var_cols <- character()

for (col in numeric_in_df) {
  var_val <- var(df[[col]], na.rm = TRUE)
  if (is.na(var_val) || var_val == 0) {
    zero_var_cols <- c(zero_var_cols, col)
  }
}

# Check categorical columns for single-level factors
cat_in_df <- intersect(categorical_cols, names(df))
for (col in cat_in_df) {
  if (length(unique(df[[col]][!is.na(df[[col]])])) <= 1) {
    zero_var_cols <- c(zero_var_cols, col)
  }
}

if (length(zero_var_cols) > 0) {
  cat(sprintf("Removing %d zero-variance columns:\n", length(zero_var_cols)))
  print(zero_var_cols)
  df <- df %>% select(-all_of(zero_var_cols))
  numeric_cols <- setdiff(numeric_cols, zero_var_cols)
  categorical_cols <- setdiff(categorical_cols, zero_var_cols)
  cat("\n")
} else {
  cat("No zero-variance columns found\n\n")
}

# ==============================================================================
# 6. ADVANCED FEATURE ENGINEERING
# ==============================================================================

cat("Step 6: Engineering new features...\n")

# Update column lists to reflect current state
numeric_in_df <- intersect(numeric_cols, names(df))
cat_in_df <- intersect(categorical_cols, names(df))

df <- df %>%
  mutate(
    # Vehicle availability features
    VEH_PER_PERSON = if ("HH_TOTALVEHS" %in% names(.) && "HH_HHSIZE" %in% names(.)) {
      ifelse(HH_HHSIZE > 0, HH_TOTALVEHS / HH_HHSIZE, 0)
    } else 0,

    VEH_PER_ADULT = if ("HH_CARS" %in% names(.) && "HH_HHSIZE" %in% names(.)) {
      ifelse(HH_HHSIZE > 0, HH_CARS / HH_HHSIZE, 0)
    } else 0,

    # Trip characteristics
    TRIP_SPEED = if ("TRIP_NETWORK_DIST" %in% names(.) && "TRIP_TOTTRIPTIME" %in% names(.)) {
      ifelse(TRIP_TOTTRIPTIME > 0, TRIP_NETWORK_DIST / (TRIP_TOTTRIPTIME / 60), 0)
    } else 0,

    # Temporal features
    TIME_PERIOD = if ("STOP_STARTHR" %in% names(.)) {
      case_when(
        STOP_STARTHR >= 6 & STOP_STARTHR < 9 ~ "Morning_Peak",
        STOP_STARTHR >= 9 & STOP_STARTHR < 16 ~ "Midday",
        STOP_STARTHR >= 16 & STOP_STARTHR < 19 ~ "Evening_Peak",
        STOP_STARTHR >= 19 | STOP_STARTHR < 6 ~ "Off_Peak",
        TRUE ~ "Unknown"
      )
    } else "Unknown",

    IS_WEEKEND = if ("HH_TRAVDOW" %in% names(.)) {
      as.integer(HH_TRAVDOW %in% c("Saturday", "Sunday"))
    } else 0,

    # Distance categories
    DIST_CATEGORY = if ("TRIP_NETWORK_DIST" %in% names(.)) {
      case_when(
        TRIP_NETWORK_DIST < 2 ~ "Very_Short",
        TRIP_NETWORK_DIST < 5 ~ "Short",
        TRIP_NETWORK_DIST < 15 ~ "Medium",
        TRIP_NETWORK_DIST < 30 ~ "Long",
        TRUE ~ "Very_Long"
      )
    } else "Unknown"
  )

# Interaction features (if columns exist)
if ("PERS_CARLICENCE" %in% names(df) && "HH_CARS" %in% names(df)) {
  df <- df %>%
    mutate(HAS_LICENSE_AND_CAR = as.integer(PERS_CARLICENCE == "Yes" & HH_CARS > 0))
}

# Handle any Inf/NaN from calculations
df <- df %>%
  mutate(across(where(is.numeric), ~replace(., is.infinite(.) | is.nan(.), 0)))

cat("✓ Created new engineered features\n\n")

# ==============================================================================
# 7. IDENTIFY HIGH CARDINALITY CATEGORICAL FEATURES (for model encoding)
# ==============================================================================

cat("Step 7: Identifying high cardinality categorical features...\n")

# Identify high cardinality features (>50 unique values)
# Note: These will be handled with target encoding in the XGBoost script
cat_in_df <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
cat_in_df <- setdiff(cat_in_df, target_col)

high_card_cols <- cat_in_df[sapply(df[cat_in_df], function(x) {
  length(unique(x[!is.na(x)])) > 50
})]

if (length(high_card_cols) > 0) {
  cat(sprintf("Found %d high cardinality columns (>50 levels):\n", length(high_card_cols)))
  print(high_card_cols)
  cat("Note: These will be target-encoded in the XGBoost script\n\n")
} else {
  cat("No high cardinality columns found\n\n")
}

# Create detailed encoding analysis report
cat("Generating encoding analysis report...\n")

# Calculate cardinality for all categorical columns
all_cat_cardinality <- sapply(df[cat_in_df], function(x) length(unique(x[!is.na(x)])))

# Create encoding decisions table
encoding_decisions <- data.frame(
  Column_Name = names(all_cat_cardinality),
  Unique_Levels = all_cat_cardinality,
  Encoding_Method = ifelse(all_cat_cardinality > 50, "Target Encoding", "One-Hot Encoding"),
  Features_Created = ifelse(all_cat_cardinality > 50, 1, all_cat_cardinality),
  Rationale = ifelse(
    all_cat_cardinality > 50,
    paste0(">50 levels, reduces dimensionality (", all_cat_cardinality, " → 1)"),
    paste0("Low/medium cardinality, preserves distinct categories")
  ),
  stringsAsFactors = FALSE
)

# Sort by unique levels (descending)
encoding_decisions <- encoding_decisions[order(-encoding_decisions$Unique_Levels), ]

# Save encoding decisions CSV
write.csv(encoding_decisions, "results/preprocessing/encoding_decisions.csv", row.names = FALSE)
cat("✓ Saved: results/preprocessing/encoding_decisions.csv\n")

# Generate encoding analysis text report
sink("results/preprocessing/encoding_analysis.txt")
cat("CATEGORICAL ENCODING ANALYSIS\n")
cat("=============================\n\n")
cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("Total Categorical Columns:", length(cat_in_df), "\n\n")

cat("ENCODING STRATEGY:\n")
cat("- High Cardinality (>50 levels): Target Encoding\n")
cat("- Low/Medium Cardinality (≤50 levels): One-Hot Encoding\n\n")

cat("WHY THIS THRESHOLD?\n")
cat("- One-hot with >50 levels creates sparse, high-dimensional data\n")
cat("- Target encoding captures relationship with target in 1 column\n")
cat("- Prevents curse of dimensionality and overfitting\n")
cat("- Empirically validated threshold for tree-based models\n\n")

# Cardinality breakdown
cat("BREAKDOWN BY CARDINALITY:\n")
very_low <- sum(all_cat_cardinality >= 2 & all_cat_cardinality <= 5)
low <- sum(all_cat_cardinality > 5 & all_cat_cardinality <= 10)
medium <- sum(all_cat_cardinality > 10 & all_cat_cardinality <= 50)
high <- sum(all_cat_cardinality > 50)

# Estimate one-hot features
very_low_feats <- sum(all_cat_cardinality[all_cat_cardinality >= 2 & all_cat_cardinality <= 5])
low_feats <- sum(all_cat_cardinality[all_cat_cardinality > 5 & all_cat_cardinality <= 10])
medium_feats <- sum(all_cat_cardinality[all_cat_cardinality > 10 & all_cat_cardinality <= 50])

cat(sprintf("Very Low (2-5 levels): %d columns → ~%d one-hot features\n", very_low, very_low_feats))
cat(sprintf("Low (6-10 levels): %d columns → ~%d one-hot features\n", low, low_feats))
cat(sprintf("Medium (11-50 levels): %d columns → ~%d one-hot features\n", medium, medium_feats))
cat(sprintf("High (>50 levels): %d columns → %d target-encoded features\n\n", high, high))

# Calculate total features
numeric_in_df_count <- sum(sapply(df, is.numeric))
target_encoded_count <- high
onehot_estimated <- very_low_feats + low_feats + medium_feats

cat("ESTIMATED FEATURES AFTER ENCODING:\n")
cat(sprintf("- Numeric: %d\n", numeric_in_df_count))
cat(sprintf("- Target Encoded: %d\n", target_encoded_count))
cat(sprintf("- One-Hot Encoded: ~%d\n", onehot_estimated))
cat(sprintf("- TOTAL: ~%d features\n\n", numeric_in_df_count + target_encoded_count + onehot_estimated))

# Comparison with naive one-hot
naive_onehot <- sum(all_cat_cardinality)
cat("COMPARISON:\n")
cat(sprintf("- Naive one-hot encoding: %d features\n", naive_onehot + numeric_in_df_count))
cat(sprintf("- Mixed encoding (our approach): ~%d features\n", numeric_in_df_count + target_encoded_count + onehot_estimated))
reduction_pct <- round((1 - (numeric_in_df_count + target_encoded_count + onehot_estimated) / (naive_onehot + numeric_in_df_count)) * 100, 1)
cat(sprintf("- Reduction: %.1f%%\n\n", reduction_pct))

cat("TOP 10 HIGHEST CARDINALITY COLUMNS:\n")
cat("===================================\n")
top_10 <- head(encoding_decisions, 10)
for (i in 1:nrow(top_10)) {
  cat(sprintf("%d. %s: %d levels → %s\n",
              i,
              top_10$Column_Name[i],
              top_10$Unique_Levels[i],
              top_10$Encoding_Method[i]))
}

sink()
cat("✓ Saved: results/preprocessing/encoding_analysis.txt\n\n")

# ==============================================================================
# 8. REMOVE HIGHLY CORRELATED FEATURES
# ==============================================================================

cat("Step 8: Removing highly correlated features...\n")

numeric_in_df <- names(df)[sapply(df, is.numeric)]
numeric_in_df <- setdiff(numeric_in_df, target_col)

if (length(numeric_in_df) > 1) {
  # Remove zero-variance numeric columns before correlation analysis
  numeric_vars_to_check <- numeric_in_df[sapply(df[numeric_in_df], function(x) {
    var_val <- var(x, na.rm = TRUE)
    !is.na(var_val) && var_val > 0
  })]

  if (length(numeric_vars_to_check) > 1) {
    # Calculate correlation matrix
    cor_matrix <- cor(df[numeric_vars_to_check], use = "pairwise.complete.obs")

    # Check if correlation matrix has any NA/NaN values
    if (any(is.na(cor_matrix)) || any(is.nan(cor_matrix))) {
      # Replace NA/NaN with 0 (no correlation)
      cor_matrix[is.na(cor_matrix) | is.nan(cor_matrix)] <- 0
    }

    # Find highly correlated pairs (|r| > 0.95)
    tryCatch({
      high_cor <- findCorrelation(cor_matrix, cutoff = 0.95, names = TRUE, verbose = FALSE)

      if (length(high_cor) > 0) {
        cat(sprintf("Removing %d highly correlated features (|r| > 0.95):\n", length(high_cor)))
        print(high_cor)
        df <- df %>% select(-all_of(high_cor))
        cat("\n")
      } else {
        cat("No highly correlated features to remove\n\n")
      }
    }, error = function(e) {
      cat("Warning: Correlation analysis failed, skipping this step\n")
      cat("Error message:", e$message, "\n\n")
    })
  } else {
    cat("Not enough non-zero-variance numeric features for correlation analysis\n\n")
  }
} else {
  cat("Not enough numeric features for correlation analysis\n\n")
}

# ==============================================================================
# 9. OUTLIER TREATMENT - DOMAIN AWARE
# ==============================================================================

cat("Step 9: Domain-aware outlier treatment...\n")

# Only cap features where extreme values are unlikely to be valid
# DO NOT cap: distance, time (long trips are valid)
# DO cap: speed (>200 km/h is unrealistic), ratios

if ("TRIP_SPEED" %in% names(df)) {
  # Cap speed at 99th percentile or 150 km/h
  speed_cap <- min(quantile(df$TRIP_SPEED, 0.99, na.rm = TRUE), 150)
  n_capped <- sum(df$TRIP_SPEED > speed_cap, na.rm = TRUE)
  if (n_capped > 0) {
    cat(sprintf("Capping TRIP_SPEED at %.1f km/h (%d values)\n", speed_cap, n_capped))
    df$TRIP_SPEED[df$TRIP_SPEED > speed_cap] <- speed_cap
  }
}

cat("✓ Outlier treatment complete\n\n")

# ==============================================================================
# 10. CALCULATE CLASS WEIGHTS FOR IMBALANCED DATA
# ==============================================================================

cat("Step 10: Calculating class weights for imbalanced data...\n")

if (target_col %in% names(df)) {
  class_counts <- table(df[[target_col]])
  total_samples <- sum(class_counts)
  n_classes <- length(class_counts)

  # Calculate class weights (inverse frequency)
  class_weights <- total_samples / (n_classes * class_counts)

  cat("Class distribution:\n")
  print(class_counts)
  cat("\nClass weights (for XGBoost):\n")
  print(class_weights)
  cat("\n")

  # Save class weights
  saveRDS(list(
    class_counts = class_counts,
    class_weights = class_weights,
    weight_formula = "total_samples / (n_classes * class_count)"
  ), "data/processed/class_weights.rds")

  cat("✓ Saved class weights to data/processed/class_weights.rds\n\n")
}

# ==============================================================================
# 11. STRATIFIED TRAIN/TEST SPLIT
# ==============================================================================

cat("Step 11: Saving final cleaned dataset before split...\n")

# Remove rows with missing target
df <- df %>% filter(!is.na(!!sym(target_col)))

cat(sprintf("Dataset after removing missing targets: %d rows\n", nrow(df)))

# Save the complete cleaned dataset BEFORE splitting
saveRDS(df, "data/processed/final_cleaned_data.rds")
cat("✓ Saved: data/processed/final_cleaned_data.rds\n")

# Also save as CSV for easy inspection
write.csv(df, "data/processed/final_cleaned_data.csv", row.names = FALSE)
cat(sprintf("✓ Saved: data/processed/final_cleaned_data.csv (%d rows × %d columns)\n\n",
            nrow(df), ncol(df)))

cat("Step 12: Creating stratified train/test split...\n")

# 80/20 stratified split
train_index <- createDataPartition(df[[target_col]],
                                   p = 0.8,
                                   list = FALSE,
                                   times = 1)

train_base <- df[train_index, ]
test_base <- df[-train_index, ]

cat(sprintf("\nTrain set: %d rows\n", nrow(train_base)))
cat(sprintf("Test set: %d rows\n\n", nrow(test_base)))

# Verify stratification
cat("Train set class distribution:\n")
print(prop.table(table(train_base[[target_col]])))
cat("\nTest set class distribution:\n")
print(prop.table(table(test_base[[target_col]])))
cat("\n")

# ==============================================================================
# 13. SAVE PROCESSED DATA AND METADATA
# ==============================================================================

cat("Step 13: Saving train/test splits...\n")

# Save train/test splits (always to same location)
saveRDS(train_base, "data/processed/train_base.rds")
saveRDS(test_base, "data/processed/test_base.rds")

cat("✓ Saved: data/processed/train_base.rds\n")
cat("✓ Saved: data/processed/test_base.rds\n")

# Save preprocessing metadata
feature_names <- setdiff(names(df), target_col)
numeric_features <- names(df)[sapply(df, is.numeric)]
categorical_features <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]

metadata <- list(
  dataset_type = dataset_type,
  source_file = input_file,
  target_variable = target_col,
  n_original_features = ncol(df_original) - 1,
  n_final_features = length(feature_names),
  features_removed = setdiff(names(df_original), names(df)),
  features_added = setdiff(names(df), names(df_original)),
  numeric_features = intersect(numeric_features, feature_names),
  categorical_features = intersect(categorical_features, feature_names),
  high_cardinality_features = high_card_cols,
  train_size = nrow(train_base),
  test_size = nrow(test_base),
  random_seed = 5003,
  preprocessing_date = Sys.time()
)

saveRDS(metadata, "data/processed/preprocessing_metadata.rds")
cat("✓ Saved: data/processed/preprocessing_metadata.rds\n\n")

# ==============================================================================
# 14. GENERATE BEFORE/AFTER COMPARISON
# ==============================================================================

cat("Step 14: Generating before/after comparison...\n")

comparison <- data.frame(
  Metric = c(
    "Total Rows",
    "Total Columns",
    "Numeric Features",
    "Categorical Features",
    "Missing Values",
    "Zero-Variance Features",
    "High Cardinality Features"
  ),
  Before = c(
    nrow(df_original),
    ncol(df_original),
    length(names(df_original)[sapply(df_original, is.numeric)]),
    length(names(df_original)[!sapply(df_original, is.numeric)]),
    sum(is.na(df_original)),
    "N/A",
    "N/A"
  ),
  After = c(
    nrow(df),
    ncol(df),
    length(numeric_features),
    length(categorical_features),
    sum(is.na(df)),
    length(zero_var_cols),
    length(high_card_cols)
  )
)

write.csv(comparison, "results/preprocessing/before_after_comparison.csv", row.names = FALSE)
cat("✓ Saved: results/preprocessing/before_after_comparison.csv\n\n")

# ==============================================================================
# 15. FINAL SUMMARY
# ==============================================================================

cat("==============================================================================\n")
cat("ENHANCED PREPROCESSING COMPLETE\n")
cat("==============================================================================\n\n")

cat("Summary of improvements:\n")
cat(sprintf("  ✓ Removed %d ID columns\n", length(id_columns_present)))
cat(sprintf("  ✓ Removed %d leaky features\n", length(leaky_features)))
cat(sprintf("  ✓ Dropped %d columns with >80%% missing\n", length(high_missing_cols)))
cat(sprintf("  ✓ Created %d missingness indicators\n", length(moderate_missing_cols)))
cat(sprintf("  ✓ Removed %d zero-variance features\n", length(zero_var_cols)))
cat(sprintf("  ✓ Identified %d high-cardinality features (for target encoding)\n", length(high_card_cols)))
if (exists("high_cor")) {
  cat(sprintf("  ✓ Removed %d highly correlated features\n", length(high_cor)))
}
cat(sprintf("  ✓ Created engineered features (ratios, interactions, temporal)\n"))
cat(sprintf("  ✓ Calculated class weights for imbalanced learning\n"))
cat(sprintf("  ✓ Stratified 80/20 train/test split\n\n"))

cat("Final dataset:\n")
cat(sprintf("  - Features: %d (was %d)\n", length(feature_names), ncol(df_original) - 1))
cat(sprintf("  - Train samples: %d\n", nrow(train_base)))
cat(sprintf("  - Test samples: %d\n", nrow(test_base)))
cat(sprintf("  - Missing values: %d (was %d)\n\n", sum(is.na(df)), sum(is.na(df_original))))

cat("Next steps:\n")
cat("  1. Review: results/preprocessing/before_after_comparison.csv\n")
cat("  2. Run XGBoost: source('src/models/xgboost_model.R')\n\n")
