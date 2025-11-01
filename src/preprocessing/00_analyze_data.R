# ==============================================================================
# Data Diagnostic Analysis Script
# ==============================================================================
# Analyzes final_decoded_clean.csv to identify preprocessing deficiencies
# and areas for improvement before XGBoost modeling
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(reshape2)
  library(corrplot)
  library(knitr)
})

cat("==============================================================================\n")
cat("DATA DIAGNOSTIC ANALYSIS\n")
cat("==============================================================================\n\n")

# Create results directory
if (!dir.exists("results")) dir.create("results")
if (!dir.exists("results/preprocessing")) dir.create("results/preprocessing")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

cat("Loading data from final_decoded_clean.csv...\n")

if (!file.exists("data/interim/final_decoded_clean.csv")) {
  stop("Error: data/interim/final_decoded_clean.csv not found. Please ensure file is in data/interim/ directory.")
}

df <- read.csv("data/interim/final_decoded_clean.csv",
               header = TRUE,
               fileEncoding = "UTF-8",
               na.strings = c("", "NA", "N/A", "NULL"))

cat(sprintf("✓ Loaded: %d rows × %d columns\n\n", nrow(df), ncol(df)))

# ==============================================================================
# 2. BASIC STATISTICS
# ==============================================================================

cat("==============================================================================\n")
cat("BASIC DATA SUMMARY\n")
cat("==============================================================================\n\n")

# Identify target variable
target_col <- "STOP_MAINMODE"
if (!target_col %in% names(df)) {
  warning("Target variable 'STOP_MAINMODE' not found! Searching for alternative...")
  target_col <- names(df)[grep("MODE", names(df), ignore.case = TRUE)][1]
  cat(sprintf("Using '%s' as target variable\n", target_col))
}

# Separate numeric and categorical
numeric_cols <- names(df)[sapply(df, is.numeric)]
categorical_cols <- names(df)[!sapply(df, is.numeric)]

cat(sprintf("Numeric columns: %d\n", length(numeric_cols)))
cat(sprintf("Categorical columns: %d\n", length(categorical_cols)))
cat(sprintf("Target variable: %s\n\n", target_col))

# ==============================================================================
# 3. MISSING VALUE ANALYSIS
# ==============================================================================

cat("==============================================================================\n")
cat("MISSING VALUE ANALYSIS\n")
cat("==============================================================================\n\n")

# Calculate missing percentages
missing_summary <- df %>%
  summarise(across(everything(), ~sum(is.na(.))/n()*100)) %>%
  pivot_longer(everything(), names_to = "Column", values_to = "Missing_Pct") %>%
  arrange(desc(Missing_Pct))

# Categorize by severity
high_missing <- missing_summary %>% filter(Missing_Pct > 50)
medium_missing <- missing_summary %>% filter(Missing_Pct > 10 & Missing_Pct <= 50)
low_missing <- missing_summary %>% filter(Missing_Pct > 0 & Missing_Pct <= 10)

cat(sprintf("Columns with >50%% missing: %d\n", nrow(high_missing)))
cat(sprintf("Columns with 10-50%% missing: %d\n", nrow(medium_missing)))
cat(sprintf("Columns with <10%% missing: %d\n", nrow(low_missing)))
cat(sprintf("Complete columns: %d\n\n", sum(missing_summary$Missing_Pct == 0)))

if (nrow(high_missing) > 0) {
  cat("⚠ HIGH MISSING (>50%) - Consider dropping:\n")
  print(head(high_missing, 10))
  cat("\n")
}

# Create missing value heatmap for top 30 columns
top_missing <- head(missing_summary, 30)
if (nrow(top_missing) > 0) {
  png("results/preprocessing/missing_values_heatmap.png",
      width = 10, height = 8, units = "in", res = 300)

  p <- ggplot(top_missing, aes(x = 1, y = reorder(Column, Missing_Pct), fill = Missing_Pct)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "lightblue", high = "darkred") +
    geom_text(aes(label = sprintf("%.1f%%", Missing_Pct)), color = "white", size = 3) +
    theme_minimal() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank()) +
    labs(title = "Top 30 Columns by Missing Value Percentage",
         x = "", y = "Column", fill = "Missing %")

  print(p)
  dev.off()

  cat("✓ Saved: results/preprocessing/missing_values_heatmap.png\n\n")
}

# ==============================================================================
# 4. TARGET VARIABLE ANALYSIS
# ==============================================================================

cat("==============================================================================\n")
cat("TARGET VARIABLE ANALYSIS\n")
cat("==============================================================================\n\n")

if (target_col %in% names(df)) {
  target_dist <- table(df[[target_col]], useNA = "ifany")
  target_prop <- prop.table(target_dist) * 100

  cat("Class Distribution:\n")
  print(target_dist)
  cat("\nClass Proportions:\n")
  print(round(target_prop, 2))
  cat("\n")

  # Calculate imbalance ratio
  max_class <- max(target_dist)
  min_class <- min(target_dist[target_dist > 0])
  imbalance_ratio <- max_class / min_class

  cat(sprintf("Imbalance Ratio: %.2f:1\n", imbalance_ratio))
  if (imbalance_ratio > 10) {
    cat("⚠ SEVERE CLASS IMBALANCE detected! Consider:\n")
    cat("  - Stratified sampling\n")
    cat("  - Class weights in XGBoost\n")
    cat("  - SMOTE or oversampling minority classes\n")
  } else if (imbalance_ratio > 3) {
    cat("⚠ MODERATE CLASS IMBALANCE detected! Use stratified sampling.\n")
  }
  cat("\n")

  # Plot class distribution
  png("results/preprocessing/target_distribution.png",
      width = 10, height = 6, units = "in", res = 300)

  dist_df <- data.frame(
    Class = names(target_dist),
    Count = as.numeric(target_dist),
    Percentage = as.numeric(target_prop)
  )

  p <- ggplot(dist_df, aes(x = reorder(Class, -Count), y = Count, fill = Class)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = sprintf("%d (%.1f%%)", Count, Percentage)),
              vjust = -0.5, size = 3) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none") +
    labs(title = "Target Variable (Travel Mode) Distribution",
         x = "Travel Mode", y = "Count")

  print(p)
  dev.off()

  cat("✓ Saved: results/preprocessing/target_distribution.png\n\n")
}

# ==============================================================================
# 5. DATA QUALITY CHECKS
# ==============================================================================

cat("==============================================================================\n")
cat("DATA QUALITY CHECKS\n")
cat("==============================================================================\n\n")

# Check for duplicates
n_duplicates <- sum(duplicated(df))
cat(sprintf("Duplicate rows: %d (%.2f%%)\n", n_duplicates, n_duplicates/nrow(df)*100))

# Check for zero/near-zero variance features
if (length(numeric_cols) > 0) {
  zero_var_cols <- numeric_cols[sapply(df[numeric_cols], function(x) {
    var(x, na.rm = TRUE) == 0 || is.na(var(x, na.rm = TRUE))
  })]

  cat(sprintf("Zero variance numeric columns: %d\n", length(zero_var_cols)))
  if (length(zero_var_cols) > 0) {
    cat("⚠ These columns should be removed:\n")
    print(zero_var_cols)
  }
}
cat("\n")

# Check for high cardinality categorical features
if (length(categorical_cols) > 0) {
  cat("Categorical Feature Cardinality:\n")
  cardinality <- sapply(df[categorical_cols], function(x) length(unique(x[!is.na(x)])))
  cardinality_df <- data.frame(
    Column = names(cardinality),
    Unique_Values = cardinality,
    Cardinality_Level = case_when(
      cardinality > 100 ~ "Very High (>100)",
      cardinality > 50 ~ "High (50-100)",
      cardinality > 10 ~ "Medium (10-50)",
      TRUE ~ "Low (<10)"
    )
  )
  cardinality_df <- cardinality_df %>% arrange(desc(Unique_Values))

  print(head(cardinality_df, 20))
  cat("\n")

  high_card <- cardinality_df %>% filter(Unique_Values > 50)
  if (nrow(high_card) > 0) {
    cat(sprintf("⚠ %d columns with >50 unique values - Consider:\n", nrow(high_card)))
    cat("  - Target encoding\n")
    cat("  - Frequency encoding\n")
    cat("  - Dropping if not informative\n\n")
  }
}

# ==============================================================================
# 6. MULTICOLLINEARITY ANALYSIS
# ==============================================================================

cat("==============================================================================\n")
cat("MULTICOLLINEARITY ANALYSIS\n")
cat("==============================================================================\n\n")

if (length(numeric_cols) > 2) {
  # Calculate correlation matrix
  numeric_df <- df[numeric_cols]
  complete_numeric <- numeric_df[, colSums(is.na(numeric_df)) < nrow(numeric_df) * 0.5]

  if (ncol(complete_numeric) > 1) {
    cor_matrix <- cor(complete_numeric, use = "pairwise.complete.obs")

    # Find highly correlated pairs
    cor_df <- as.data.frame(as.table(cor_matrix))
    names(cor_df) <- c("Var1", "Var2", "Correlation")
    cor_df <- cor_df %>%
      filter(Var1 != Var2) %>%
      filter(abs(Correlation) > 0.8) %>%
      arrange(desc(abs(Correlation)))

    # Remove duplicate pairs
    cor_df <- cor_df[!duplicated(t(apply(cor_df[,1:2], 1, sort))), ]

    cat(sprintf("Highly correlated pairs (|r| > 0.8): %d\n", nrow(cor_df)))
    if (nrow(cor_df) > 0) {
      cat("⚠ Consider removing one from each pair:\n")
      print(head(cor_df, 10))
      cat("\n")
    }

    # Plot correlation matrix (top 30 features)
    if (ncol(complete_numeric) > 2) {
      top_vars <- names(complete_numeric)[1:min(30, ncol(complete_numeric))]
      cor_subset <- cor_matrix[top_vars, top_vars]

      png("results/preprocessing/correlation_matrix.png",
          width = 12, height = 12, units = "in", res = 300)

      corrplot(cor_subset, method = "color", type = "upper",
               tl.col = "black", tl.srt = 45, tl.cex = 0.7,
               title = "Correlation Matrix (Top 30 Numeric Features)",
               mar = c(0,0,2,0))

      dev.off()

      cat("✓ Saved: results/preprocessing/correlation_matrix.png\n\n")
    }
  }
}

# ==============================================================================
# 7. OUTLIER DETECTION
# ==============================================================================

cat("==============================================================================\n")
cat("OUTLIER ANALYSIS\n")
cat("==============================================================================\n\n")

if (length(numeric_cols) > 0) {
  outlier_summary <- lapply(numeric_cols, function(col) {
    x <- df[[col]]
    x <- x[!is.na(x)]

    if (length(x) > 0) {
      Q1 <- quantile(x, 0.25)
      Q3 <- quantile(x, 0.75)
      IQR_val <- Q3 - Q1

      lower_bound <- Q1 - 3 * IQR_val
      upper_bound <- Q3 + 3 * IQR_val

      n_outliers <- sum(x < lower_bound | x > upper_bound)
      pct_outliers <- n_outliers / length(x) * 100

      data.frame(
        Column = col,
        N_Outliers = n_outliers,
        Pct_Outliers = pct_outliers,
        Min = min(x),
        Max = max(x),
        Q1 = Q1,
        Q3 = Q3
      )
    } else {
      NULL
    }
  })

  outlier_df <- do.call(rbind, outlier_summary)
  outlier_df <- outlier_df %>%
    filter(Pct_Outliers > 0) %>%
    arrange(desc(Pct_Outliers))

  cat(sprintf("Columns with outliers (3×IQR rule): %d\n", nrow(outlier_df)))
  if (nrow(outlier_df) > 0) {
    cat("\nTop columns by outlier percentage:\n")
    print(head(outlier_df, 10))
    cat("\n⚠ Review these for domain validity (long trips may be valid, not outliers)\n\n")
  }
}

# ==============================================================================
# 8. POTENTIAL LEAKY FEATURES
# ==============================================================================

cat("==============================================================================\n")
cat("POTENTIAL LEAKY FEATURES CHECK\n")
cat("==============================================================================\n\n")

# Keywords that might indicate posterior/leaky features
leaky_keywords <- c("LINKMODE", "MODE1", "MODE2", "MAINMODE", "TRIPMODE")
potential_leaky <- names(df)[sapply(names(df), function(x) {
  any(sapply(leaky_keywords, function(kw) grepl(kw, x, ignore.case = TRUE)))
})]

# Exclude the target variable
potential_leaky <- setdiff(potential_leaky, target_col)

if (length(potential_leaky) > 0) {
  cat("⚠ POTENTIAL LEAKY FEATURES (verify these are not posterior to target):\n")
  print(potential_leaky)
  cat("\nRecommendation: Remove if these are collected AFTER mode choice\n\n")
} else {
  cat("✓ No obvious leaky features detected\n\n")
}

# ==============================================================================
# 9. GENERATE SUMMARY REPORT
# ==============================================================================

cat("==============================================================================\n")
cat("GENERATING SUMMARY REPORT\n")
cat("==============================================================================\n\n")

# Create summary text file
sink("results/preprocessing/diagnostic_summary.txt")

cat("DATA DIAGNOSTIC SUMMARY REPORT\n")
cat("================================================================================\n")
cat(sprintf("Generated: %s\n", Sys.time()))
cat(sprintf("Dataset: final_decoded_clean.csv\n"))
cat(sprintf("Dimensions: %d rows × %d columns\n\n", nrow(df), ncol(df)))

cat("KEY FINDINGS:\n")
cat("================================================================================\n\n")

cat("1. MISSING VALUES:\n")
cat(sprintf("   - Columns with >50%% missing: %d (consider dropping)\n", nrow(high_missing)))
cat(sprintf("   - Columns with 10-50%% missing: %d (requires imputation)\n", nrow(medium_missing)))
cat(sprintf("   - Columns with <10%% missing: %d (simple imputation OK)\n\n", nrow(low_missing)))

if (exists("imbalance_ratio")) {
  cat("2. CLASS IMBALANCE:\n")
  cat(sprintf("   - Imbalance ratio: %.2f:1\n", imbalance_ratio))
  if (imbalance_ratio > 10) {
    cat("   - SEVERITY: High - Use SMOTE/class weights\n\n")
  } else if (imbalance_ratio > 3) {
    cat("   - SEVERITY: Moderate - Use stratified sampling\n\n")
  } else {
    cat("   - SEVERITY: Low - Standard methods OK\n\n")
  }
}

cat("3. DATA QUALITY:\n")
cat(sprintf("   - Duplicate rows: %d\n", n_duplicates))
if (exists("zero_var_cols")) {
  cat(sprintf("   - Zero variance columns: %d\n", length(zero_var_cols)))
}
if (exists("high_card")) {
  cat(sprintf("   - High cardinality columns (>50 levels): %d\n\n", nrow(high_card)))
}

cat("4. MULTICOLLINEARITY:\n")
if (exists("cor_df") && nrow(cor_df) > 0) {
  cat(sprintf("   - Highly correlated pairs (|r| > 0.8): %d\n", nrow(cor_df)))
  cat("   - Action: Remove one from each pair\n\n")
} else {
  cat("   - No severe multicollinearity detected\n\n")
}

cat("5. OUTLIERS:\n")
if (exists("outlier_df") && nrow(outlier_df) > 0) {
  cat(sprintf("   - Columns with outliers: %d\n", nrow(outlier_df)))
  cat("   - Action: Domain-aware capping (preserve valid long trips)\n\n")
} else {
  cat("   - No severe outliers detected\n\n")
}

cat("6. POTENTIAL LEAKY FEATURES:\n")
if (length(potential_leaky) > 0) {
  cat("   - Found:\n")
  for (feat in potential_leaky) {
    cat(sprintf("     • %s\n", feat))
  }
  cat("\n")
} else {
  cat("   - None detected\n\n")
}

cat("RECOMMENDATIONS FOR PREPROCESSING:\n")
cat("================================================================================\n\n")

cat("1. Drop columns with >80% missing\n")
cat("2. Predictive imputation for important features with 20-80% missing\n")
cat("3. Simple imputation for <20% missing\n")
cat("4. Create interaction features (license × cars, income × hhsize)\n")
cat("5. Target/frequency encoding for high cardinality categoricals\n")
cat("6. Remove highly correlated feature pairs\n")
cat("7. Stratified train/test split to preserve class balance\n")
cat("8. Calculate class weights for XGBoost\n")
cat("9. Remove zero-variance and leaky features\n")
cat("10. Domain-aware outlier capping (preserve valid extremes)\n\n")

cat("NEXT STEPS:\n")
cat("================================================================================\n\n")
cat("1. Review diagnostic plots in results/preprocessing/\n")
cat("2. Implement enhanced preprocessing (scripts/00_preprocessing_enhanced.R)\n")
cat("3. Verify improvements with before/after comparison\n")
cat("4. Proceed with XGBoost modeling\n\n")

sink()

cat("✓ Saved: results/preprocessing/diagnostic_summary.txt\n\n")

# ==============================================================================
# 10. FINAL MESSAGE
# ==============================================================================

cat("==============================================================================\n")
cat("DIAGNOSTIC ANALYSIS COMPLETE\n")
cat("==============================================================================\n\n")

cat("Generated files:\n")
cat("  - results/preprocessing/diagnostic_summary.txt\n")
cat("  - results/preprocessing/missing_values_heatmap.png\n")
cat("  - results/preprocessing/target_distribution.png\n")
cat("  - results/preprocessing/correlation_matrix.png\n\n")

cat("Next: Review the diagnostics, then run:\n")
cat("  source('src/preprocessing/01_preprocess_data.R')\n\n")
