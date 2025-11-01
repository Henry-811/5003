# ==============================================================================
# Complete XGBoost Pipeline Runner
# ==============================================================================
# Runs the entire workflow from diagnostics through model evaluation
# ==============================================================================

cat("\n")
cat("********************************************************************************\n")
cat("*                                                                              *\n")
cat("*                   XGBOOST COMPLETE PIPELINE                                 *\n")
cat("*                   Travel Mode Prediction                                     *\n")
cat("*                                                                              *\n")
cat("********************************************************************************\n")
cat("\n")

# Record start time
pipeline_start <- Sys.time()

# ==============================================================================
# STAGE 1: DATA DIAGNOSTICS
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("STAGE 1: DATA DIAGNOSTICS\n")
cat("================================================================================\n")
cat("Analyzing final_decoded_clean.csv for preprocessing issues...\n\n")

stage1_start <- Sys.time()

tryCatch({
  source("src/preprocessing/00_analyze_data.R")
  stage1_time <- as.numeric(difftime(Sys.time(), stage1_start, units = "secs"))
  cat(sprintf("\n✓ Stage 1 complete in %.1f seconds\n", stage1_time))
  cat("\nPlease review diagnostic outputs in results/preprocessing/\n")
  cat("Press Enter to continue to preprocessing, or Ctrl+C to stop...\n")
  readline()
}, error = function(e) {
  cat("\n✗ Stage 1 failed:", e$message, "\n")
  cat("Fix errors and try again.\n")
  stop("Pipeline halted at Stage 1")
})

# ==============================================================================
# STAGE 2: ENHANCED PREPROCESSING
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("STAGE 2: ENHANCED PREPROCESSING\n")
cat("================================================================================\n")
cat("Improving data quality and creating train/test splits...\n\n")

stage2_start <- Sys.time()

tryCatch({
  source("src/preprocessing/01_preprocess_data.R")
  stage2_time <- as.numeric(difftime(Sys.time(), stage2_start, units = "secs"))
  cat(sprintf("\n✓ Stage 2 complete in %.1f seconds\n", stage2_time))
}, error = function(e) {
  cat("\n✗ Stage 2 failed:", e$message, "\n")
  cat("Check data quality and try again.\n")
  stop("Pipeline halted at Stage 2")
})

# ==============================================================================
# STAGE 3: XGBOOST MODEL TRAINING
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("STAGE 3: XGBOOST MODEL TRAINING\n")
cat("================================================================================\n")

# Check if Bayesian Optimization is available
if (requireNamespace("ParBayesianOptimization", quietly = TRUE)) {
  cat("Using Bayesian Optimization for hyperparameter tuning (40 iterations)...\n")
  cat("This will take several minutes but finds better hyperparameters.\n\n")
} else {
  cat("Using random grid search for hyperparameter tuning (20 iterations)...\n")
  cat("Tip: Install ParBayesianOptimization for better results:\n")
  cat("     source('src/utils/install_packages.R')\n\n")
}

stage3_start <- Sys.time()

tryCatch({
  source("src/models/xgboost_model.R")
  stage3_time <- as.numeric(difftime(Sys.time(), stage3_start, units = "secs"))
  cat(sprintf("\n✓ Stage 3 complete in %.1f seconds\n", stage3_time))
}, error = function(e) {
  cat("\n✗ Stage 3 failed:", e$message, "\n")
  cat("Check model configuration and try again.\n")
  stop("Pipeline halted at Stage 3")
})

# ==============================================================================
# PIPELINE SUMMARY
# ==============================================================================

pipeline_end <- Sys.time()
total_time <- as.numeric(difftime(pipeline_end, pipeline_start, units = "mins"))

cat("\n")
cat("********************************************************************************\n")
cat("*                                                                              *\n")
cat("*                   PIPELINE COMPLETE!                                         *\n")
cat("*                                                                              *\n")
cat("********************************************************************************\n")
cat("\n")

cat("EXECUTION SUMMARY:\n")
cat("================================================================================\n")
cat(sprintf("Stage 1 (Diagnostics):   %.1f seconds\n", stage1_time))
cat(sprintf("Stage 2 (Preprocessing): %.1f seconds\n", stage2_time))
cat(sprintf("Stage 3 (XGBoost):       %.1f seconds\n", stage3_time))
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
  cat("  - results/xgboost/bayesian_optimization.rds (Bayesian opt results)\n")
  cat("  - results/xgboost/bayesian_optimization_progress.png\n")
}
cat("  - results/xgboost/encoding_report.txt\n")
cat("================================================================================\n\n")

cat("NEXT STEPS:\n")
cat("================================================================================\n")
cat("1. Review model performance in results/xgboost/metrics.json\n")
cat("2. Examine feature importance plots\n")
cat("3. Check confusion matrix for misclassification patterns\n")
cat("4. Integrate results into reports/Group_W11_G07.Rmd report\n")
cat("5. Share preprocessed data with teammates (data/processed/train_base.rds, data/processed/test_base.rds)\n")
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
