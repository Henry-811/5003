# ==============================================================================
# Main Entry Point - One-Click XGBoost Pipeline
# ==============================================================================
# This script handles EVERYTHING:
# 1. Automatic package installation
# 2. Data diagnostics
# 3. Data preprocessing
# 4. XGBoost model training
# 5. Model evaluation
#
# Usage:
#   source("main.R")  # Uses default (policy-focused dataset)
#
#   Or set dataset before running:
#   INPUT_DATA_PATH <- "data/raw/data_test.csv"; source("main.R")
#   INPUT_DATA_PATH <- "data/processed/final_cleaned_policy_focused.csv"; source("main.R")
# ==============================================================================

cat("\n")
cat("********************************************************************************\n")
cat("*                                                                              *\n")
cat("*                   XGBOOST COMPLETE PIPELINE                                 *\n")
cat("*                   One-Click Execution                                        *\n")
cat("*                                                                              *\n")
cat("********************************************************************************\n")
cat("\n")

# ==============================================================================
# CONFIGURATION - DATASET SELECTION
# ==============================================================================
# Set default dataset if not already specified
if (!exists("INPUT_DATA_PATH")) {
  INPUT_DATA_PATH <- "data/processed/final_cleaned_policy_focused.csv"  # Default: policy-focused
}

cat("Dataset Configuration:\n")
cat("================================================================================\n")
cat(sprintf("Using: %s\n", INPUT_DATA_PATH))
cat("================================================================================\n\n")

# Record start time
main_start <- Sys.time()

# ==============================================================================
# STEP 0: AUTO-INSTALL REQUIRED PACKAGES
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("STEP 0: CHECKING & INSTALLING REQUIRED PACKAGES\n")
cat("================================================================================\n")

step0_start <- Sys.time()

# List of required packages
required_packages <- c(
  "xgboost",      # XGBoost algorithm
  "caret",        # Machine learning utilities, CV
  "ggplot2",      # Visualizations
  "reshape2",     # Data reshaping
  "jsonlite",     # JSON export/import
  "Matrix",       # Sparse matrices
  "dplyr",        # Data manipulation
  "tidyr"         # Data tidying
)

# Optional but recommended packages
optional_packages <- c(
  "ParBayesianOptimization"   # Bayesian hyperparameter optimization
)

# Auto-install required packages
cat("\nInstalling essential packages (if needed)...\n")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE, quiet = TRUE)
  } else {
    cat(sprintf("  ✓ %s already installed\n", pkg))
  }
}

# Auto-install optional packages (non-interactive)
cat("\nInstalling recommended packages (if needed)...\n")
for (pkg in optional_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    tryCatch({
      install.packages(pkg, dependencies = TRUE, quiet = TRUE)
      cat(sprintf("  ✓ %s installed successfully\n", pkg))
    }, error = function(e) {
      cat(sprintf("  ⚠ Could not install %s (optional)\n", pkg))
    })
  } else {
    cat(sprintf("  ✓ %s already installed\n", pkg))
  }
}

step0_time <- as.numeric(difftime(Sys.time(), step0_start, units = "secs"))
cat(sprintf("\n✓ Package setup complete in %.1f seconds\n", step0_time))

# ==============================================================================
# SMART DATA CHECK: Skip preprocessing if data is fresh
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("CHECKING PREPROCESSING DATA FRESHNESS\n")
cat("================================================================================\n")

skip_preprocessing <- FALSE
step1_time <- 0
step2_time <- 0

# Check if processed data exists
if (file.exists("data/processed/train_base.rds") &&
    file.exists("data/processed/test_base.rds") &&
    file.exists("data/processed/preprocessing_metadata.rds")) {

  # Get file modification times
  preprocess_script_time <- file.info("src/preprocessing/01_preprocess_data.R")$mtime
  processed_data_time <- file.info("data/processed/train_base.rds")$mtime

  # Check if preprocessing script was modified after data generation
  if (preprocess_script_time > processed_data_time) {
    cat("⚠️  Preprocessing script modified - will regenerate data\n")
    skip_preprocessing <- FALSE
  } else {
    cat("✓ Processed data is up to date - skipping preprocessing\n")
    cat("  (Saves ~2-5 minutes!)\n")
    skip_preprocessing <- TRUE
  }
} else {
  cat("⚠️  Processed data not found - will generate it now\n")
  skip_preprocessing <- FALSE
}

cat("================================================================================\n")

# ==============================================================================
# STEP 1: DATA DIAGNOSTICS (only if needed)
# ==============================================================================

if (!skip_preprocessing) {
  cat("\n")
  cat("================================================================================\n")
  cat("STEP 1: DATA DIAGNOSTICS\n")
  cat("================================================================================\n")
  cat("Analyzing final_decoded_clean.csv for preprocessing issues...\n\n")

  step1_start <- Sys.time()

  tryCatch({
    source("src/preprocessing/00_analyze_data.R")
    step1_time <- as.numeric(difftime(Sys.time(), step1_start, units = "secs"))
    cat(sprintf("\n✓ Step 1 complete in %.1f seconds\n", step1_time))
  }, error = function(e) {
    cat("\n✗ Step 1 failed:", e$message, "\n")
    cat("Fix errors and try again.\n")
    stop("Pipeline halted at Step 1")
  })

  # ==============================================================================
  # STEP 2: ENHANCED PREPROCESSING
  # ==============================================================================

  cat("\n")
  cat("================================================================================\n")
  cat("STEP 2: ENHANCED PREPROCESSING\n")
  cat("================================================================================\n")
  cat("Improving data quality and creating train/test splits...\n\n")

  step2_start <- Sys.time()

  tryCatch({
    source("src/preprocessing/01_preprocess_data.R")
    step2_time <- as.numeric(difftime(Sys.time(), step2_start, units = "secs"))
    cat(sprintf("\n✓ Step 2 complete in %.1f seconds\n", step2_time))
  }, error = function(e) {
    cat("\n✗ Step 2 failed:", e$message, "\n")
    cat("Check data quality and try again.\n")
    stop("Pipeline halted at Step 2")
  })
} else {
  cat("\n⏩ Skipping Steps 1-2 (using existing processed data)\n")
}

# ==============================================================================
# STEP 3: XGBOOST MODEL TRAINING
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("STEP 3: XGBOOST MODEL TRAINING\n")
cat("================================================================================\n")

# Check if Bayesian Optimization is available
if (requireNamespace("ParBayesianOptimization", quietly = TRUE)) {
  cat("Using Bayesian Optimization for hyperparameter tuning (40 iterations)...\n")
  cat("This will take several minutes but finds better hyperparameters.\n\n")
} else {
  cat("Using random grid search for hyperparameter tuning (20 iterations)...\n\n")
}

step3_start <- Sys.time()

tryCatch({
  source("src/models/xgboost_model.R")
  step3_time <- as.numeric(difftime(Sys.time(), step3_start, units = "secs"))
  cat(sprintf("\n✓ Step 3 complete in %.1f seconds\n", step3_time))
}, error = function(e) {
  cat("\n✗ Step 3 failed:", e$message, "\n")
  cat("Check model configuration and try again.\n")
  stop("Pipeline halted at Step 3")
})

# ==============================================================================
# EXECUTION SUMMARY
# ==============================================================================

main_end <- Sys.time()
total_time <- as.numeric(difftime(main_end, main_start, units = "mins"))

cat("\n")
cat("********************************************************************************\n")
cat("*                                                                              *\n")
cat("*                   PIPELINE COMPLETE!                                         *\n")
cat("*                                                                              *\n")
cat("********************************************************************************\n")
cat("\n")

cat("EXECUTION SUMMARY:\n")
cat("================================================================================\n")
cat(sprintf("Step 0 (Packages):       %.1f seconds\n", step0_time))
cat(sprintf("Step 1 (Diagnostics):    %.1f seconds\n", step1_time))
cat(sprintf("Step 2 (Preprocessing):  %.1f seconds\n", step2_time))
cat(sprintf("Step 3 (XGBoost):        %.1f seconds\n", step3_time))
cat(sprintf("\nTotal pipeline time:     %.2f minutes\n", total_time))
cat("================================================================================\n\n")

cat("OUTPUT LOCATIONS:\n")
cat("================================================================================\n")
cat("Diagnostics:\n")
cat("  - results/preprocessing/diagnostic_summary.txt\n")
cat("  - results/preprocessing/missing_values_heatmap.png\n")
cat("  - results/preprocessing/target_distribution.png\n")
cat("  - results/preprocessing/correlation_matrix.png\n\n")

cat("Preprocessed Data:\n")
cat("  - data/processed/train_base.rds\n")
cat("  - data/processed/test_base.rds\n")
cat("  - data/processed/preprocessing_metadata.rds\n")
cat("  - data/processed/class_weights.rds\n\n")

cat("XGBoost Results:\n")
cat("  - results/xgboost/xgboost_model.rds\n")
cat("  - results/xgboost/best_params.json\n")
cat("  - results/xgboost/metrics.json\n")
cat("  - results/xgboost/predictions.csv\n")
cat("  - results/xgboost/confusion_matrix.png\n")
cat("  - results/xgboost/feature_importance.png\n")
cat("  - results/xgboost/training_curve.png\n")
if (file.exists("results/xgboost/bayesian_optimization.rds")) {
  cat("  - results/xgboost/bayesian_optimization.rds\n")
  cat("  - results/xgboost/bayesian_optimization_progress.png\n")
}
cat("  - results/xgboost/encoding_report.txt\n")
cat("================================================================================\n\n")

cat("NEXT STEPS:\n")
cat("================================================================================\n")
cat("1. Review model performance in results/xgboost/metrics.json\n")
cat("2. Examine feature importance plots\n")
cat("3. Check confusion matrix for misclassification patterns\n")
cat("4. Integrate results into your report\n")
cat("5. Share preprocessed data with teammates\n")
cat("   - data/processed/train_base.rds\n")
cat("   - data/processed/test_base.rds\n")
cat("================================================================================\n\n")

# Load and display key metrics
if (file.exists("results/xgboost/metrics.json")) {
  cat("KEY PERFORMANCE METRICS:\n")
  cat("================================================================================\n")
  metrics <- jsonlite::fromJSON("results/xgboost/metrics.json")
  cat(sprintf("Accuracy:          %.4f (%.2f%%)\n", metrics$accuracy, metrics$accuracy * 100))
  cat(sprintf("Macro F1-Score:    %.4f\n", metrics$macro_f1))
  cat(sprintf("Balanced Accuracy: %.4f\n", metrics$balanced_accuracy))
  cat(sprintf("Training Time:     %.1f seconds\n", metrics$training_time_seconds))
  cat("================================================================================\n\n")
}

cat("\n✨ All done! Happy modeling! ✨\n\n")
