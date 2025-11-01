# ==============================================================================
# Install Required R Packages for XGBoost Implementation
# ==============================================================================

cat("Installing required R packages for XGBoost model...\n\n")

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

# Optional packages
optional_packages <- c(
  "SHAPforxgboost",           # SHAP value analysis
  "ParBayesianOptimization"   # Bayesian hyperparameter optimization
)

# Install required packages
cat("Installing essential packages...\n")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE)
  } else {
    cat(sprintf("  ✓ %s already installed\n", pkg))
  }
}

cat("\n")

# Ask about optional packages
cat("Optional packages provide additional functionality:\n")
cat("  - SHAPforxgboost: Detailed feature importance analysis\n")
cat("  - ParBayesianOptimization: Smart hyperparameter tuning (RECOMMENDED)\n\n")
cat("Install optional packages? (y/n): ")

response <- tolower(readline())
if (response %in% c("y", "yes")) {
  for (pkg in optional_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("  Installing %s...\n", pkg))
      tryCatch({
        install.packages(pkg, dependencies = TRUE)
      }, error = function(e) {
        cat(sprintf("  Warning: Could not install %s\n", pkg))
        cat(sprintf("  Error: %s\n", e$message))
      })
    } else {
      cat(sprintf("  %s already installed\n", pkg))
    }
  }
}

cat("\n==============================================================================\n")
cat("Package Installation Complete!\n")
cat("==============================================================================\n\n")

# Check which optional packages are installed
if (requireNamespace("ParBayesianOptimization", quietly = TRUE)) {
  cat("✓ Bayesian Optimization: ENABLED\n")
  cat("  Your XGBoost model will use intelligent hyperparameter tuning!\n\n")
} else {
  cat("⚠ Bayesian Optimization: DISABLED\n")
  cat("  Model will use random grid search (less efficient)\n")
  cat("  To enable: Re-run this script and install optional packages\n\n")
}

if (requireNamespace("SHAPforxgboost", quietly = TRUE)) {
  cat("✓ SHAP Analysis: ENABLED\n")
  cat("  Advanced feature importance plots will be generated\n\n")
} else {
  cat("⚠ SHAP Analysis: DISABLED\n")
  cat("  Basic feature importance only\n\n")
}

cat("To run the complete pipeline:\n")
cat("  source('src/run_pipeline.R')\n\n")

cat("To run XGBoost model only:\n")
cat("  source('src/models/xgboost_model.R')\n\n")

