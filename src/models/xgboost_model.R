# ==============================================================================
# XGBoost Model Implementation
# ==============================================================================
# Multi-class classification for STOP_MAINMODE prediction
# With Bayesian hyperparameter optimization and comprehensive evaluation
# ==============================================================================

# Load required packages
suppressPackageStartupMessages({
  library(xgboost)
  library(caret)
  library(ggplot2)
  library(reshape2)
  library(jsonlite)
  library(Matrix)
  library(dplyr)
  library(tidyr)
})

# Setup parallel processing for XGBoost
n_cores_total <- parallel::detectCores() - 1  # Leave 1 core free
n_cv_folds <- 5  # Number of CV folds to run in parallel
n_cores_per_fold <- max(1, floor(n_cores_total / n_cv_folds))  # Cores per fold during CV
n_cores_final <- n_cores_total  # Full cores for final training

cat(sprintf("Setting up hybrid parallel processing with %d total cores...\n", n_cores_total))
cat(sprintf("✓ CV: %d folds in parallel, %d cores each\n", n_cv_folds, n_cores_per_fold))
cat(sprintf("✓ Final training: %d cores\n\n", n_cores_final))

# Load parallel libraries for CV
if (!requireNamespace("foreach", quietly = TRUE)) {
  install.packages("foreach")
}
if (!requireNamespace("doParallel", quietly = TRUE)) {
  install.packages("doParallel")
}
library(foreach)
library(doParallel)

# Setup parallel cluster
cl <- makeCluster(n_cv_folds)
registerDoParallel(cl)
cat(sprintf("✓ Parallel cluster registered with %d workers\n\n", n_cv_folds))

# Optional: For SHAP values
if (requireNamespace("SHAPforxgboost", quietly = TRUE)) {
  library(SHAPforxgboost)
  has_shap <- TRUE
} else {
  cat("Note: SHAPforxgboost not available. SHAP plots will be skipped.\n")
  has_shap <- FALSE
}

# Bayesian Optimization (required)
if (!requireNamespace("ParBayesianOptimization", quietly = TRUE)) {
  stop("ParBayesianOptimization package is required. Install with:\n  install.packages('ParBayesianOptimization')")
}
library(ParBayesianOptimization)
cat("Using Bayesian Optimization for hyperparameter tuning\n")

cat("===============================================\n")
cat("XGBoost Model for Travel Mode Prediction\n")
cat("===============================================\n\n")

# ==============================================================================
# 1. LOAD AND SPLIT DATA FRESH (NO CACHE)
# ==============================================================================

cat("Loading and splitting data fresh (no cached files)...\n")

# Try policy-focused dataset first
if (file.exists("data/processed/final_cleaned_policy_focused.csv")) {
  cat("✓ Loading policy-focused dataset (vehicle proxies removed)\n")
  full_data <- read.csv("data/processed/final_cleaned_policy_focused.csv", stringsAsFactors = FALSE)
  dataset_type <- "policy_focused"
} else if (file.exists("data/processed/final_cleaned_data.csv")) {
  cat("⚠️  Loading base dataset (includes vehicle proxies)\n")
  full_data <- read.csv("data/processed/final_cleaned_data.csv", stringsAsFactors = FALSE)
  dataset_type <- "base"
} else {
  stop("❌ No dataset found! Please run preprocessing first.")
}

cat(sprintf("✓ Loaded: %d rows × %d columns\n", nrow(full_data), ncol(full_data)))

# Set target variable
target_col <- "STOP_MAINMODE"

# Create stratified 80/20 train/test split
cat("\nCreating fresh 80/20 stratified split...\n")
set.seed(5003)  # For reproducibility

# Ensure target is a factor for stratification
full_data[[target_col]] <- as.factor(full_data[[target_col]])

# Create stratified split using caret
train_indices <- createDataPartition(full_data[[target_col]], p = 0.8, list = FALSE)
train_data <- full_data[train_indices, ]
test_data <- full_data[-train_indices, ]

cat(sprintf("✓ Dataset type: %s\n", dataset_type))
cat(sprintf("Train set: %d rows, %d features\n", nrow(train_data), ncol(train_data) - 1))
cat(sprintf("Test set: %d rows\n", nrow(test_data)))

# ==============================================================================
# 2. PREPARE DATA FOR XGBOOST
# ==============================================================================

cat("\nPreparing data for XGBoost...\n")

# Separate features and target
X_train <- train_data %>% select(-all_of(target_col))
y_train <- train_data[[target_col]]
X_test <- test_data %>% select(-all_of(target_col))
y_test <- test_data[[target_col]]

# Create label mapping (0-based for XGBoost)
unique_classes <- sort(unique(y_train))
label_mapping <- setNames(0:(length(unique_classes) - 1), unique_classes)
reverse_mapping <- setNames(names(label_mapping), label_mapping)

y_train_encoded <- label_mapping[as.character(y_train)]
y_test_encoded <- label_mapping[as.character(y_test)]

cat("\nClass mapping:\n")
print(label_mapping)
cat(sprintf("\nNumber of classes: %d\n", length(unique_classes)))

# ==============================================================================
# TARGET ENCODING FUNCTION (with K-Fold to prevent overfitting)
# ==============================================================================

target_encode_cv <- function(train_data, test_data, col_name, target_vector, n_folds = 5) {
  # Create encoded column for train (using CV to avoid leakage)
  train_encoded <- numeric(nrow(train_data))

  # K-fold encoding for train
  set.seed(5003)
  folds <- createFolds(target_vector, k = n_folds, list = TRUE)

  for (fold_idx in seq_along(folds)) {
    val_indices <- folds[[fold_idx]]
    train_indices <- setdiff(1:nrow(train_data), val_indices)

    # Calculate mean target for each category using training fold
    encoding_map <- tapply(target_vector[train_indices],
                          train_data[[col_name]][train_indices],
                          mean, na.rm = TRUE)

    # Apply to validation fold
    train_encoded[val_indices] <- encoding_map[as.character(train_data[[col_name]][val_indices])]
  }

  # For test data, use entire train data
  global_encoding_map <- tapply(target_vector, train_data[[col_name]], mean, na.rm = TRUE)
  test_encoded <- global_encoding_map[as.character(test_data[[col_name]])]

  # Handle unseen categories (use global mean)
  global_mean <- mean(target_vector, na.rm = TRUE)
  train_encoded[is.na(train_encoded)] <- global_mean
  test_encoded[is.na(test_encoded)] <- global_mean

  return(list(train = train_encoded, test = test_encoded))
}

# ==============================================================================
# MIXED ENCODING STRATEGY: TARGET + ONE-HOT
# ==============================================================================

cat("\nProcessing categorical variables with mixed encoding...\n")

# Identify categorical columns
categorical_cols <- names(X_train)[sapply(X_train, function(x) is.character(x) || is.factor(x))]

if (length(categorical_cols) > 0) {
  cat(sprintf("Found %d categorical columns\n", length(categorical_cols)))

  # Calculate cardinality for each categorical column
  cardinality <- sapply(X_train[categorical_cols], function(x) length(unique(x)))

  # Split by cardinality
  high_card_cols <- names(cardinality)[cardinality > 50]
  low_card_cols <- names(cardinality)[cardinality <= 50]

  cat(sprintf("  - High cardinality (>50 levels): %d columns → target encoding\n", length(high_card_cols)))
  cat(sprintf("  - Low/medium cardinality (≤50 levels): %d columns → one-hot encoding\n", length(low_card_cols)))

  # Store encoded features
  encoded_features_list <- list()

  # 1. TARGET ENCODING for high cardinality
  if (length(high_card_cols) > 0) {
    cat("\nApplying target encoding to high-cardinality columns...\n")
    pb_target <- txtProgressBar(min = 0, max = length(high_card_cols), style = 3)

    for (i in seq_along(high_card_cols)) {
      col <- high_card_cols[i]
      encoded <- target_encode_cv(X_train, X_test, col, y_train_encoded, n_folds = 5)

      # Store as matrix columns
      encoded_col_name <- paste0(col, "_target_enc")
      encoded_features_list[[encoded_col_name]] <- list(
        train = matrix(encoded$train, ncol = 1),
        test = matrix(encoded$test, ncol = 1)
      )

      setTxtProgressBar(pb_target, i)
    }
    close(pb_target)
    cat("\n")
  }

  # 2. ONE-HOT ENCODING for low/medium cardinality
  if (length(low_card_cols) > 0) {
    cat("\nApplying one-hot encoding to low/medium-cardinality columns...\n")
    cat(sprintf("  Processing %d columns with %d total unique values...\n",
                length(low_card_cols), sum(cardinality[low_card_cols])))

    # Create subset with only low-cardinality categoricals
    X_train_onehot <- X_train[, low_card_cols, drop = FALSE]
    X_test_onehot <- X_test[, low_card_cols, drop = FALSE]

    formula_str <- paste("~ . -1")

    cat("\n[1/3] Encoding training data (this may take a moment)...\n")
    train_start <- Sys.time()
    X_train_onehot_matrix <- model.matrix(as.formula(formula_str), data = X_train_onehot)
    train_time <- round(as.numeric(difftime(Sys.time(), train_start, units = "secs")), 1)
    cat(sprintf("      ✓ Training data encoded in %.1f seconds (%d features created)\n",
                train_time, ncol(X_train_onehot_matrix)))

    cat("[2/3] Encoding test data...\n")
    test_start <- Sys.time()
    X_test_onehot_matrix <- model.matrix(as.formula(formula_str), data = X_test_onehot)
    test_time <- round(as.numeric(difftime(Sys.time(), test_start, units = "secs")), 1)
    cat(sprintf("      ✓ Test data encoded in %.1f seconds\n", test_time))

    # Align test columns with train
    cat("[3/3] Aligning test columns with training columns...\n")
    missing_cols <- setdiff(colnames(X_train_onehot_matrix), colnames(X_test_onehot_matrix))
    if (length(missing_cols) > 0) {
      for (col in missing_cols) {
        X_test_onehot_matrix <- cbind(X_test_onehot_matrix, 0)
        colnames(X_test_onehot_matrix)[ncol(X_test_onehot_matrix)] <- col
      }
      cat(sprintf("      ✓ Added %d missing columns to test set\n", length(missing_cols)))
    } else {
      cat("      ✓ Columns already aligned\n")
    }
    X_test_onehot_matrix <- X_test_onehot_matrix[, colnames(X_train_onehot_matrix)]

    encoded_features_list[["onehot"]] <- list(
      train = X_train_onehot_matrix,
      test = X_test_onehot_matrix
    )
    cat("\n")
  }

  # 3. COMBINE: Numeric + Target Encoded + One-Hot
  cat("Combining all features...\n")

  # Get numeric columns
  numeric_cols <- names(X_train)[sapply(X_train, is.numeric)]
  X_train_numeric <- as.matrix(X_train[, numeric_cols, drop = FALSE])
  X_test_numeric <- as.matrix(X_test[, numeric_cols, drop = FALSE])

  # Combine all features
  feature_matrices_train <- list(X_train_numeric)
  feature_matrices_test <- list(X_test_numeric)

  for (feat_name in names(encoded_features_list)) {
    feature_matrices_train[[feat_name]] <- encoded_features_list[[feat_name]]$train
    feature_matrices_test[[feat_name]] <- encoded_features_list[[feat_name]]$test
  }

  X_train_matrix <- do.call(cbind, feature_matrices_train)
  X_test_matrix <- do.call(cbind, feature_matrices_test)

  cat(sprintf("✓ Combined features:\n"))
  cat(sprintf("  - Numeric: %d\n", ncol(X_train_numeric)))
  cat(sprintf("  - Target encoded: %d\n", length(high_card_cols)))
  if (length(low_card_cols) > 0) {
    cat(sprintf("  - One-hot encoded: %d\n", ncol(X_train_onehot_matrix)))
  }
  cat(sprintf("  - TOTAL: %d\n\n", ncol(X_train_matrix)))

  # Save encoding report
  cat("Generating encoding report...\n")
  sink("results/xgboost/encoding_report.txt")
  cat("XGBOOST ENCODING REPORT\n")
  cat("=======================\n\n")
  cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

  cat("ENCODING STRATEGY APPLIED:\n")
  cat("- Mixed Encoding (Target + One-Hot)\n")
  cat("- Threshold: 50 unique levels\n\n")

  cat("CATEGORICAL COLUMNS BREAKDOWN:\n")
  cat(sprintf("Total categorical columns: %d\n\n", length(categorical_cols)))

  if (length(high_card_cols) > 0) {
    cat(sprintf("HIGH CARDINALITY (>50 levels) - TARGET ENCODED:\n"))
    cat(sprintf("  Columns: %d\n", length(high_card_cols)))
    cat(sprintf("  Features created: %d\n", length(high_card_cols)))
    cat("  Columns:\n")
    for (col in high_card_cols) {
      cat(sprintf("    - %s (%d levels → 1 feature)\n", col, cardinality[col]))
    }
    cat("\n")
  }

  if (length(low_card_cols) > 0) {
    cat(sprintf("LOW/MEDIUM CARDINALITY (≤50 levels) - ONE-HOT ENCODED:\n"))
    cat(sprintf("  Columns: %d\n", length(low_card_cols)))
    cat(sprintf("  Features created: %d\n", ncol(X_train_onehot_matrix)))
    cat("\n")
  }

  cat("FINAL FEATURE MATRIX:\n")
  cat(sprintf("  - Numeric features: %d\n", ncol(X_train_numeric)))
  cat(sprintf("  - Target-encoded features: %d\n", length(high_card_cols)))
  if (length(low_card_cols) > 0) {
    cat(sprintf("  - One-hot encoded features: %d\n", ncol(X_train_onehot_matrix)))
  }
  cat(sprintf("  - TOTAL FEATURES: %d\n\n", ncol(X_train_matrix)))

  # Calculate what naive one-hot would have created
  naive_features <- ncol(X_train_numeric) + sum(cardinality)
  cat("COMPARISON WITH NAIVE ONE-HOT ENCODING:\n")
  cat(sprintf("  - Naive approach: %d features\n", naive_features))
  cat(sprintf("  - Mixed approach: %d features\n", ncol(X_train_matrix)))
  reduction_pct <- round((1 - ncol(X_train_matrix) / naive_features) * 100, 1)
  cat(sprintf("  - Reduction: %.1f%%\n\n", reduction_pct))

  cat("WHY THIS APPROACH?\n")
  cat("1. Dimensionality Reduction: Prevents curse of dimensionality\n")
  cat("2. Prevents Overfitting: Fewer features = less model complexity\n")
  cat("3. Faster Training: Smaller feature space = faster computation\n")
  cat("4. Better Generalization: Target encoding captures target relationship\n")
  cat("5. Memory Efficiency: Avoids sparse high-dimensional matrices\n\n")

  cat("TARGET ENCODING DETAILS:\n")
  cat("- Method: K-Fold Cross-Validation (k=5)\n")
  cat("- Prevents data leakage by encoding each fold separately\n")
  cat("- Test data encoded using full training statistics\n")
  cat("- Unseen categories handled with global mean\n\n")

  sink()
  cat("✓ Saved: results/xgboost/encoding_report.txt\n\n")

} else {
  # No categorical columns, just use numeric
  X_train_matrix <- as.matrix(X_train)
  X_test_matrix <- as.matrix(X_test)
  cat(sprintf("Final feature matrix: %d features\n\n", ncol(X_train_matrix)))
}

# Validate feature matrix before creating DMatrix
cat("Validating feature matrix...\n")
na_count <- sum(is.na(X_train_matrix))
inf_count <- sum(is.infinite(X_train_matrix))

if (na_count > 0) {
  cat(sprintf("✗ ERROR: Found %d NA values in feature matrix\n", na_count))
  cat("  Replacing NA values with 0...\n")
  X_train_matrix[is.na(X_train_matrix)] <- 0
  X_test_matrix[is.na(X_test_matrix)] <- 0
}

if (inf_count > 0) {
  cat(sprintf("✗ ERROR: Found %d Inf values in feature matrix\n", inf_count))
  cat("  Replacing Inf values with large finite values...\n")
  X_train_matrix[is.infinite(X_train_matrix) & X_train_matrix > 0] <- 1e10
  X_train_matrix[is.infinite(X_train_matrix) & X_train_matrix < 0] <- -1e10
  X_test_matrix[is.infinite(X_test_matrix) & X_test_matrix > 0] <- 1e10
  X_test_matrix[is.infinite(X_test_matrix) & X_test_matrix < 0] <- -1e10
}

cat(sprintf("✓ Feature matrix validated: %d samples x %d features\n",
            nrow(X_train_matrix), ncol(X_train_matrix)))
cat(sprintf("  Range: [%.4f, %.4f]\n\n", min(X_train_matrix), max(X_train_matrix)))

# Create DMatrix objects (XGBoost optimized data structure)
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train_encoded)
dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test_encoded)

# ==============================================================================
# 3. HYPERPARAMETER TUNING
# ==============================================================================

cat("\n===============================================\n")
cat("Starting Hyperparameter Tuning\n")
cat("===============================================\n\n")

# Create stratified folds for CV
set.seed(5003)
folds <- createFolds(y_train_encoded, k = 5, list = TRUE)

# ===========================================================================
# BAYESIAN OPTIMIZATION
# ===========================================================================

  cat("Method: Bayesian Optimization with Gaussian Process\n")
  cat("TESTING MODE: 7 initial + 8 optimization = 15 total iterations\n")
  cat("Each iteration uses 5-fold cross-validation\n\n")

  # Initialize progress tracking
  iteration_counter <- 0
  start_time <- Sys.time()

  # Track best iterations from each Bayesian optimization trial
  iteration_best_nrounds <- list()

  # Create progress log file for real-time monitoring (RStudio buffering workaround)
  progress_file <- "results/xgboost/optimization_progress.log"
  dir.create(dirname(progress_file), showWarnings = FALSE, recursive = TRUE)
  cat(sprintf("=== Bayesian Optimization Progress Log ===\nStarted: %s\n\n",
              format(Sys.time(), "%Y-%m-%d %H:%M:%S")), file = progress_file)

  cat(sprintf("\n📝 Progress is being logged to: %s\n", progress_file))
  cat("   You can open this file in another window to monitor real-time progress!\n\n")

  # Define objective function for Bayesian Optimization
  # Must return list with Score (to maximize) - we'll negate mlogloss
  xgb_cv_bayes <- function(max_depth, eta, subsample, colsample_bytree,
                           min_child_weight, gamma) {

    # Wrap everything in tryCatch to capture exact error location
    tryCatch({

    # Increment and display progress
    iteration_counter <<- iteration_counter + 1
    iteration_start_time <- Sys.time()

    # Log to file (this WILL show in real-time)
    log_msg <- sprintf("\n[%s] Starting Iteration %2d/15...\n",
                       format(Sys.time(), "%H:%M:%S"), iteration_counter)
    cat(log_msg, file = progress_file, append = TRUE)

    # Also try to display in console (may be buffered in RStudio)
    cat(log_msg)
    flush.console()
    Sys.sleep(0)

    # Round discrete parameters
    max_depth <- round(max_depth)
    min_child_weight <- round(min_child_weight)

    # Prepare XGBoost parameters (using reduced cores for parallel CV)
    xgb_params <- list(
      objective = "multi:softprob",
      num_class = length(unique_classes),
      eval_metric = "mlogloss",
      max_depth = max_depth,
      eta = eta,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      min_child_weight = min_child_weight,
      gamma = gamma,
      nthread = n_cores_per_fold  # Cores per fold for parallel CV
    )

    # Log fold start
    cat("  5-Fold CV (parallel): ", file = progress_file, append = TRUE)
    cat("  5-Fold CV (parallel): ")
    flush.console()

    cv_start <- Sys.time()

    # Run cross-validation in parallel
    cv_results <- foreach(fold_idx = seq_along(folds),
                          .packages = c("xgboost"),
                          .combine = rbind,
                          .export = c("folds", "X_train_matrix", "y_train_encoded", "xgb_params", "unique_classes")) %dopar% {
      fold_start <- Sys.time()

      test_indices <- folds[[fold_idx]]
      train_indices <- setdiff(1:nrow(X_train_matrix), test_indices)

      dtrain_cv <- xgb.DMatrix(data = X_train_matrix[train_indices, ],
                               label = y_train_encoded[train_indices])
      dval_cv <- xgb.DMatrix(data = X_train_matrix[test_indices, ],
                            label = y_train_encoded[test_indices])

      # Train model (TESTING: reduced rounds)
      model_cv <- xgb.train(
        params = xgb_params,
        data = dtrain_cv,
        nrounds = 50,                     # TESTING: was 200
        watchlist = list(val = dval_cv),
        early_stopping_rounds = 10,       # TESTING: was 20
        verbose = 0
      )

      # Calculate accuracy and F1 on validation set
      pred_val_probs <- predict(model_cv, dval_cv, reshape = TRUE)
      pred_val_classes <- max.col(pred_val_probs) - 1
      actual_val <- y_train_encoded[test_indices]

      # Calculate macro F1 for validation
      f1_scores_val <- numeric(length(unique_classes))
      for (cls_idx in seq_along(unique_classes)) {
        cls <- cls_idx - 1
        tp <- sum(pred_val_classes == cls & actual_val == cls)
        fp <- sum(pred_val_classes == cls & actual_val != cls)
        fn <- sum(pred_val_classes != cls & actual_val == cls)
        precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
        recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
        f1_scores_val[cls_idx] <- ifelse(precision + recall > 0,
                                          2 * precision * recall / (precision + recall), 0)
      }

      # Training metrics (for overfitting check)
      pred_train_probs <- predict(model_cv, dtrain_cv, reshape = TRUE)
      pred_train_classes <- max.col(pred_train_probs) - 1
      actual_train <- y_train_encoded[train_indices]

      # Calculate train mlogloss (with safety checks)
      train_probs_for_actual <- pred_train_probs[cbind(1:nrow(pred_train_probs), actual_train + 1)]
      train_probs_for_actual <- pmax(train_probs_for_actual, 1e-15)  # Avoid log(0)
      train_probs_for_actual <- pmin(train_probs_for_actual, 1 - 1e-15)  # Avoid log issues

      # Check for any non-numeric or invalid values
      if (any(is.na(train_probs_for_actual)) || any(!is.finite(train_probs_for_actual))) {
        train_mlogloss <- model_cv$best_score  # Fallback to validation mlogloss
      } else {
        train_mlogloss <- -mean(log(train_probs_for_actual))
      }

      # Calculate macro F1 for training
      f1_scores_train <- numeric(length(unique_classes))
      for (cls_idx in seq_along(unique_classes)) {
        cls <- cls_idx - 1
        tp <- sum(pred_train_classes == cls & actual_train == cls)
        fp <- sum(pred_train_classes == cls & actual_train != cls)
        fn <- sum(pred_train_classes != cls & actual_train == cls)
        precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
        recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
        f1_scores_train[cls_idx] <- ifelse(precision + recall > 0,
                                            2 * precision * recall / (precision + recall), 0)
      }
      train_f1 <- mean(f1_scores_train, na.rm = TRUE)

      # Return results as data frame row
      data.frame(
        fold = fold_idx,
        val_mlogloss = model_cv$best_score,
        val_acc = mean(pred_val_classes == actual_val),
        val_f1 = mean(f1_scores_val, na.rm = TRUE),
        train_mlogloss = train_mlogloss,
        train_acc = mean(pred_train_classes == actual_train),
        train_f1 = train_f1,
        best_iteration = model_cv$best_iteration,  # Track best iteration per fold
        val_preds = I(list(pred_val_classes)),
        val_actuals = I(list(actual_val))
      )
    }

    cv_time <- as.numeric(difftime(Sys.time(), cv_start, units = "secs"))
    cat(sprintf("Done (%.1fs)\n", cv_time), file = progress_file, append = TRUE)
    cat(sprintf("Done (%.1fs)\n", cv_time))
    flush.console()

    # Extract results from parallel execution
    cv_scores <- cv_results$val_mlogloss
    cv_val_acc <- cv_results$val_acc
    cv_val_f1 <- cv_results$val_f1
    cv_train_mlogloss <- cv_results$train_mlogloss
    cv_train_acc <- cv_results$train_acc
    cv_train_f1 <- cv_results$train_f1
    cv_best_iterations <- cv_results$best_iteration
    all_val_preds <- cv_results$val_preds
    all_val_actuals <- cv_results$val_actuals

    # Calculate average best iteration across folds for final training
    mean_best_iteration <- round(mean(cv_best_iterations))

    # Store globally for this iteration
    iteration_best_nrounds[[iteration_counter]] <<- mean_best_iteration

    # Calculate mean metrics across folds
    mean_val_mlogloss <- mean(cv_scores)
    mean_val_acc <- mean(cv_val_acc)
    mean_val_f1 <- mean(cv_val_f1)
    mean_train_mlogloss <- mean(cv_train_mlogloss)
    mean_train_acc <- mean(cv_train_acc)
    mean_train_f1 <- mean(cv_train_f1)

    # Calculate overfitting gap (with safety checks)
    if (is.na(mean_train_acc) || is.na(mean_val_acc) || !is.finite(mean_train_acc) || !is.finite(mean_val_acc)) {
      acc_gap <- 0
    } else {
      acc_gap <- (mean_train_acc - mean_val_acc) * 100
    }

    if (is.na(mean_train_f1) || is.na(mean_val_f1) || !is.finite(mean_train_f1) || !is.finite(mean_val_f1)) {
      f1_gap <- 0
    } else {
      f1_gap <- (mean_train_f1 - mean_val_f1) * 100
    }

    # Analyze class distribution (detect majority class bias)
    combined_preds <- unlist(all_val_preds)
    combined_actuals <- unlist(all_val_actuals)

    # Count predictions per class
    pred_counts <- table(combined_preds)
    num_classes_predicted <- length(pred_counts)
    majority_pred_pct <- max(pred_counts) / length(combined_preds) * 100

    # Calculate per-class recall (with safety checks)
    class_recalls <- sapply(0:(length(unique_classes)-1), function(cls) {
      tp <- sum(combined_preds == cls & combined_actuals == cls)
      fn <- sum(combined_preds != cls & combined_actuals == cls)
      ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    })

    # Handle edge cases
    if (length(class_recalls) == 0 || all(is.na(class_recalls))) {
      min_recall <- 0
      max_recall <- 0
      num_classes_with_recall <- 0
    } else {
      min_recall <- min(class_recalls, na.rm = TRUE)
      max_recall <- max(class_recalls, na.rm = TRUE)
      num_classes_with_recall <- sum(class_recalls > 0, na.rm = TRUE)
    }

    # Display iteration summary
    iteration_time <- as.numeric(difftime(Sys.time(), iteration_start_time, units = "secs"))
    total_elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    avg_time <- total_elapsed / iteration_counter
    remaining <- avg_time * (15 - iteration_counter)  # TESTING: was 40

    # Detailed metrics message
    summary_msg <- sprintf(
      paste0(
        "\n  ✓ CV Results (5-fold average):\n",
        "    Validation: mlogloss=%.4f | Acc=%.2f%%%% | MacroF1=%.4f\n",
        "    Training:   mlogloss=%.4f | Acc=%.2f%%%% | MacroF1=%.4f\n",
        "    Overfitting gap: Acc=%.1f%%%% | F1=%.1f%%%%\n",
        "    Class diversity: %d/%d classes predicted | Recall range: [%.2f%%, %.2f%%]\n",
        "    Majority class: %.1f%%%% of predictions\n",
        "  Time: Iter=%.1fs | Total=%s | ETA=%s\n"
      ),
      mean_val_mlogloss, mean_val_acc * 100, mean_val_f1,
      mean_train_mlogloss, mean_train_acc * 100, mean_train_f1,
      acc_gap, f1_gap,
      num_classes_with_recall, length(unique_classes), min_recall * 100, max_recall * 100,
      majority_pred_pct,
      iteration_time,
      sprintf("%02d:%02d", floor(total_elapsed/60), floor(total_elapsed) %% 60),
      sprintf("%02d:%02d", floor(remaining/60), floor(remaining) %% 60)
    )

    # Write to log file (immediate)
    cat(summary_msg, file = progress_file, append = TRUE)

    # Write to console (may be buffered)
    cat(summary_msg)
    flush.console()

    # Return negative mlogloss (Bayesian opt maximizes, we want to minimize mlogloss)
    return(list(Score = -mean_val_mlogloss))

    }, error = function(e) {
      # Detailed error reporting
      error_msg <- sprintf(
        "\n\n✗ ERROR in iteration %d:\n  %s\n\n  Debugging info:\n  max_depth=%d, eta=%.4f, subsample=%.4f\n  colsample_bytree=%.4f, min_child_weight=%d, gamma=%.4f\n\n",
        iteration_counter, e$message,
        round(max_depth), eta, subsample, colsample_bytree, round(min_child_weight), gamma
      )
      cat(error_msg)
      cat(error_msg, file = progress_file, append = TRUE)

      # Return a very bad score so Bayesian opt can continue
      return(list(Score = -999))
    })
  }

  # Define hyperparameter bounds
  bounds <- list(
    max_depth = c(3L, 10L),              # Integer bounds
    eta = c(0.001, 0.3),                 # Learning rate
    subsample = c(0.5, 1.0),             # Row sampling
    colsample_bytree = c(0.5, 1.0),      # Column sampling
    min_child_weight = c(1L, 10L),       # Integer bounds
    gamma = c(0, 1)                      # Minimum loss reduction
  )

  cat("Hyperparameter search space:\n")
  cat("  max_depth: [3, 10]\n")
  cat("  eta: [0.001, 0.3]\n")
  cat("  subsample: [0.5, 1.0]\n")
  cat("  colsample_bytree: [0.5, 1.0]\n")
  cat("  min_child_weight: [1, 10]\n")
  cat("  gamma: [0, 1]\n\n")

  cat("Starting Bayesian Optimization...\n")
  cat("(This will take several minutes)\n")
  cat("Each iteration shows: Val/Train metrics | Accuracy | F1-score | Overfitting gap\n\n")
  flush.console()

  # Run Bayesian Optimization with error handling (TESTING MODE)
  bayes_out <- tryCatch({
    bayesOpt(
      FUN = xgb_cv_bayes,
      bounds = bounds,
      initPoints = 7,       # TESTING: was 10 (must be > 6 parameters)
      iters.n = 8,          # TESTING: was 30 (Bayesian optimization iterations)
      iters.k = 5,          # How many times to sample the acquisition function (must be <= iters.n)
      otherHalting = list(timeLimit = Inf),
      acq = "ucb",          # Upper Confidence Bound acquisition function
      kappa = 2.576,        # Exploration parameter
      verbose = 0           # Disable default verbose (we have custom progress in xgb_cv_bayes)
    )
  }, error = function(e) {
    cat("\n\n✗ Error during Bayesian Optimization:\n")
    cat(sprintf("  Message: %s\n", e$message))
    cat("\n  Debugging information:\n")
    cat(sprintf("  - Iteration counter: %d\n", iteration_counter))
    cat(sprintf("  - Feature matrix dimensions: %d x %d\n", nrow(X_train_matrix), ncol(X_train_matrix)))
    cat(sprintf("  - Target classes: %d unique values\n", length(unique_classes)))
    cat("\n  Please check:\n")
    cat("  1. All features are numeric (no NA or Inf values)\n")
    cat("  2. Target variable is properly encoded\n")
    cat("  3. Cross-validation folds are valid\n\n")
    stop(e)
  })

  cat("\n\nExtracting best parameters from Bayesian optimization results...\n")

  # Extract best parameters (with safety checks)
  best_params_bayes <- getBestPars(bayes_out)
  cat("✓ Got best parameters from bayesOpt\n")

  # Safely extract best score and iteration
  cat("Extracting best score...\n")
  scores <- bayes_out$scoreSummary$Score
  cat(sprintf("Found %d scores\n", length(scores)))
  if (any(!is.finite(scores))) {
    cat("\n⚠️ Warning: Some scores are non-finite. Filtering...\n")
    valid_idx <- which(is.finite(scores))
    best_idx <- valid_idx[which.max(scores[valid_idx])]
    best_score <- -scores[best_idx]
  } else {
    best_idx <- which.max(scores)
    best_score <- -scores[best_idx]
  }

  # Check if best_score is valid
  if (!is.finite(best_score)) {
    cat("\n✗ Error: Could not find valid best score\n")
    best_score <- NA
  }
  cat(sprintf("Best score: %.4f\n", best_score))

  # Extract best nrounds from the best iteration
  best_nrounds <- iteration_best_nrounds[[best_idx]]
  cat(sprintf("Best nrounds (avg from CV): %d\n", best_nrounds))

  # Convert to proper format
  cat("Creating best_params data frame...\n")
  cat("best_params_bayes values:\n")
  print(best_params_bayes)
  cat("Class:", class(best_params_bayes), "\n")
  cat("Length:", length(best_params_bayes), "\n")

  # Handle different return types from getBestPars
  if (is.list(best_params_bayes)) {
    # If it's a list, extract the numeric values
    best_params <- data.frame(
      max_depth = round(as.numeric(best_params_bayes$max_depth)),
      eta = as.numeric(best_params_bayes$eta),
      subsample = as.numeric(best_params_bayes$subsample),
      colsample_bytree = as.numeric(best_params_bayes$colsample_bytree),
      min_child_weight = round(as.numeric(best_params_bayes$min_child_weight)),
      gamma = as.numeric(best_params_bayes$gamma)
    )
  } else {
    # If it's a vector, access by index
    best_params <- data.frame(
      max_depth = round(as.numeric(best_params_bayes[1])),
      eta = as.numeric(best_params_bayes[2]),
      subsample = as.numeric(best_params_bayes[3]),
      colsample_bytree = as.numeric(best_params_bayes[4]),
      min_child_weight = round(as.numeric(best_params_bayes[5])),
      gamma = as.numeric(best_params_bayes[6])
    )
  }
  cat("✓ Created best_params\n")

  cat("\n\n=== Bayesian Optimization Complete ===\n")
  cat(sprintf("Total iterations: %d\n", nrow(bayes_out$scoreSummary)))
  cat(sprintf("Best mlogloss: %.4f\n", best_score))
  cat("\nBest hyperparameters found:\n")
  print(best_params)

  # Save Bayesian optimization results
  saveRDS(bayes_out, "results/xgboost/bayesian_optimization.rds")
  cat("\n✓ Saved Bayesian optimization results to: results/xgboost/bayesian_optimization.rds\n")

  # Save best parameters to JSON for reference
  best_params_to_save <- list(
    max_depth = best_params$max_depth,
    eta = best_params$eta,
    subsample = best_params$subsample,
    colsample_bytree = best_params$colsample_bytree,
    min_child_weight = best_params$min_child_weight,
    gamma = best_params$gamma,
    iteration = which.max(bayes_out$scoreSummary$Score),
    mlogloss = best_score,
    macrof1 = NA,  # Will be updated if available from log
    optimization_date = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
  jsonlite::write_json(best_params_to_save, "results/xgboost/best_params.json",
                       pretty = TRUE, auto_unbox = TRUE)
  cat("✓ Saved best hyperparameters to: results/xgboost/best_params.json\n")

  # Plot optimization progress (with safety checks)
  if (!is.null(bayes_out$scoreSummary)) {
    tryCatch({
      # Filter out non-finite scores for plotting
      plot_scores <- -bayes_out$scoreSummary$Score
      if (any(!is.finite(plot_scores))) {
        cat("\n⚠️ Warning: Filtering non-finite scores for plotting\n")
        plot_scores[!is.finite(plot_scores)] <- NA
      }

      png("results/xgboost/bayesian_optimization_progress.png", width = 10, height = 6, units = "in", res = 300)
      par(mfrow = c(1, 2))

      # Plot 1: Score over iterations
      plot(1:length(plot_scores), plot_scores,
           type = "b", pch = 19, col = "steelblue",
           xlab = "Iteration", ylab = "mlogloss",
           main = "Bayesian Optimization Progress")

      # Add cumulative best (only for finite values)
      finite_scores <- plot_scores[is.finite(plot_scores)]
      if (length(finite_scores) > 0) {
        cummin_scores <- cummin(plot_scores)
        cummin_scores[!is.finite(cummin_scores)] <- NA
        lines(1:length(cummin_scores), cummin_scores, col = "red", lwd = 2)
        legend("topright", legend = c("Trial Score", "Best So Far"),
               col = c("steelblue", "red"), lwd = c(1, 2), pch = c(19, NA))
      }

      # Plot 2: Score distribution (only finite values)
      if (length(finite_scores) > 0) {
        hist(finite_scores, breaks = 20, col = "lightblue",
             border = "white", main = "mlogloss Distribution",
             xlab = "mlogloss", ylab = "Frequency")
        if (is.finite(best_score)) {
          abline(v = best_score, col = "red", lwd = 2, lty = 2)
        }
      }

      dev.off()
      cat("✓ Saved optimization plots to: results/xgboost/bayesian_optimization_progress.png\n")
    }, error = function(e) {
      cat(sprintf("\n⚠️ Warning: Could not create optimization plots: %s\n", e$message))
    })
  }

# ==============================================================================
# 4. TRAIN FINAL MODEL
# ==============================================================================

cat("\n===============================================\n")
cat("Training Final Model\n")
cat("===============================================\n")

# Prepare final parameters (use ALL cores for final training)
final_params <- list(
  objective = "multi:softprob",
  num_class = length(unique_classes),
  eval_metric = "mlogloss",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  nthread = n_cores_final  # Full cores for single final model
)

# Train on full training data for fixed number of rounds (from CV)
# No early stopping - prevents test set leakage
watchlist <- list(train = dtrain)

cat(sprintf("\nTraining XGBoost model for %d rounds (determined by CV)...\n", best_nrounds))
cat("Training on FULL training set without early stopping\n")
cat("(This prevents data leakage from test set)\n\n")
start_time <- Sys.time()

xgb_model <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = best_nrounds,  # Fixed rounds from CV, no early stopping
  watchlist = watchlist,   # Monitor training only, not test
  verbose = 1,
  print_every_n = 50
)

end_time <- Sys.time()
training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("\nTraining completed in %.2f seconds\n", training_time))
cat(sprintf("Trained for %d rounds (from CV)\n", best_nrounds))

# ==============================================================================
# 5. PREDICTIONS
# ==============================================================================

cat("\n===============================================\n")
cat("Making Predictions\n")
cat("===============================================\n")

# Predict probabilities on test set
pred_probs <- predict(xgb_model, dtest, reshape = TRUE)
colnames(pred_probs) <- unique_classes

# Get predicted class (highest probability)
pred_labels_encoded <- max.col(pred_probs) - 1  # 0-based

# Convert back to original labels
pred_class <- reverse_mapping[as.character(pred_labels_encoded)]
true_class <- reverse_mapping[as.character(y_test_encoded)]

# ==============================================================================
# 6. EVALUATION METRICS
# ==============================================================================

cat("\n===============================================\n")
cat("Model Evaluation\n")
cat("===============================================\n")

# Confusion matrix
conf_matrix <- confusionMatrix(
  factor(pred_class, levels = unique_classes),
  factor(true_class, levels = unique_classes)
)

cat("\nConfusion Matrix:\n")
print(conf_matrix$table)

cat("\nOverall Statistics:\n")
print(conf_matrix$overall)

cat("\nPer-Class Statistics:\n")
print(conf_matrix$byClass)

# Calculate additional metrics
accuracy <- as.numeric(conf_matrix$overall['Accuracy'])

# Macro metrics (average across classes)
class_f1 <- conf_matrix$byClass[, 'F1']
class_precision <- conf_matrix$byClass[, 'Precision']
class_recall <- conf_matrix$byClass[, 'Recall']

macro_f1 <- mean(class_f1, na.rm = TRUE)
macro_precision <- mean(class_precision, na.rm = TRUE)
macro_recall <- mean(class_recall, na.rm = TRUE)

# Micro F1 (for multiclass = accuracy)
micro_f1 <- accuracy

# Balanced Accuracy
balanced_acc <- as.numeric(conf_matrix$byClass[1, 'Balanced Accuracy'])
if (is.na(balanced_acc)) {
  # Calculate manually if not available
  sensitivities <- conf_matrix$byClass[, 'Sensitivity']
  balanced_acc <- mean(sensitivities, na.rm = TRUE)
}

cat(sprintf("\n=== Summary Metrics ===\n"))
cat(sprintf("Accuracy: %.4f\n", accuracy))
cat(sprintf("Macro F1-Score: %.4f\n", macro_f1))
cat(sprintf("Macro Precision: %.4f\n", macro_precision))
cat(sprintf("Macro Recall: %.4f\n", macro_recall))
cat(sprintf("Micro F1-Score: %.4f\n", micro_f1))
cat(sprintf("Balanced Accuracy: %.4f\n", balanced_acc))

# ==============================================================================
# 7. SAVE RESULTS
# ==============================================================================

cat("\n===============================================\n")
cat("Saving Results\n")
cat("===============================================\n")

# Create results directory
if (!dir.exists("results")) dir.create("results")
if (!dir.exists("results/xgboost")) dir.create("results/xgboost")

# Save model
saveRDS(xgb_model, "results/xgboost/xgboost_model.rds")
cat("Model saved to: results/xgboost/xgboost_model.rds\n")

# Save best hyperparameters for reference
best_params_list <- list(
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  nrounds = best_nrounds,  # From CV, not early stopping
  cv_score = best_score,
  optimization_date = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
write_json(best_params_list, "results/xgboost/best_params.json", pretty = TRUE, auto_unbox = TRUE)
cat("Best parameters saved to: results/xgboost/best_params.json\n")

# Save predictions
predictions_df <- data.frame(
  true_label = true_class,
  predicted_label = pred_class,
  pred_probs
)
write.csv(predictions_df, "results/xgboost/predictions.csv", row.names = FALSE)
cat("Predictions saved to: results/xgboost/predictions.csv\n")

# Save metrics in standardized format
per_class_df <- as.data.frame(conf_matrix$byClass)
per_class_list <- lapply(1:nrow(per_class_df), function(i) {
  as.list(per_class_df[i, c("Precision", "Recall", "F1")])
})
names(per_class_list) <- rownames(per_class_df)

metrics <- list(
  model_name = "XGBoost",
  accuracy = accuracy,
  macro_f1 = macro_f1,
  macro_precision = macro_precision,
  macro_recall = macro_recall,
  micro_f1 = micro_f1,
  balanced_accuracy = balanced_acc,
  training_time_seconds = training_time,
  nrounds_used = best_nrounds,  # From CV, not early stopping
  per_class_metrics = per_class_list,
  confusion_matrix = unclass(as.matrix(conf_matrix$table))  # Remove table class
)
write_json(metrics, "results/xgboost/metrics.json", pretty = TRUE, auto_unbox = TRUE)
cat("Metrics saved to: results/xgboost/metrics.json\n")

# ==============================================================================
# 8. VISUALIZATIONS
# ==============================================================================

cat("\n===============================================\n")
cat("Creating Visualizations\n")
cat("===============================================\n")

# 8.1 Confusion Matrix Heatmap
conf_df <- as.data.frame(conf_matrix$table)
names(conf_df) <- c("Prediction", "Reference", "Freq")

p1 <- ggplot(conf_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  labs(title = "XGBoost Confusion Matrix",
       x = "True Class",
       y = "Predicted Class",
       fill = "Count")

ggsave("results/xgboost/confusion_matrix.png", p1, width = 10, height = 8, dpi = 300)
cat("Confusion matrix saved to: results/xgboost/confusion_matrix.png\n")

# 8.2 Feature Importance
importance_matrix <- xgb.importance(
  feature_names = colnames(X_train_matrix),
  model = xgb_model
)

# Save importance data
write.csv(importance_matrix, "results/xgboost/feature_importance.csv", row.names = FALSE)

# Plot top 20 features
top_n <- min(20, nrow(importance_matrix))
p2 <- ggplot(importance_matrix[1:top_n, ], aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  labs(title = "Top 20 Feature Importance (Gain)",
       x = "Feature",
       y = "Gain")

ggsave("results/xgboost/feature_importance.png", p2, width = 10, height = 8, dpi = 300)
cat("Feature importance saved to: results/xgboost/feature_importance.png\n")

# 8.3 Training Curve
if (!is.null(xgb_model$evaluation_log)) {
  training_log <- xgb_model$evaluation_log

  # Reshape for plotting
  log_long <- training_log %>%
    select(iter, starts_with("train"), starts_with("test")) %>%
    pivot_longer(cols = -iter, names_to = "metric", values_to = "value")

  p3 <- ggplot(log_long, aes(x = iter, y = value, color = metric)) +
    geom_line(linewidth = 1) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "bottom") +
    labs(title = "XGBoost Training Curve",
         x = "Iteration",
         y = "Log Loss",
         color = "Dataset")

  ggsave("results/xgboost/training_curve.png", p3, width = 10, height = 6, dpi = 300)
  cat("Training curve saved to: results/xgboost/training_curve.png\n")
}

# 8.4 SHAP Values (if available)
if (has_shap && nrow(X_train_matrix) <= 5000) {  # SHAP can be slow for large datasets
  cat("\nCalculating SHAP values (this may take a moment)...\n")

  tryCatch({
    # Use subset for faster computation
    sample_size <- min(1000, nrow(X_train_matrix))
    sample_idx <- sample(1:nrow(X_train_matrix), sample_size)

    shap_values <- shap.values(
      xgb_model = xgb_model,
      X_train = X_train_matrix[sample_idx, ]
    )

    # SHAP summary plot
    png("results/xgboost/shap_summary.png", width = 10, height = 8, units = "in", res = 300)
    shap.plot.summary(shap_values)
    dev.off()

    cat("SHAP summary saved to: results/xgboost/shap_summary.png\n")
  }, error = function(e) {
    cat("SHAP calculation failed:", e$message, "\n")
  })
}

cat("\n===============================================\n")
cat("XGBoost Model Complete!\n")
cat("===============================================\n")
cat("\nAll results saved to: results/xgboost/\n")
cat("\nFiles created:\n")
cat("  - xgboost_model.rds (trained model)\n")
cat("  - best_params.json (hyperparameters)\n")
cat("  - metrics.json (performance metrics)\n")
cat("  - predictions.csv (test set predictions)\n")
cat("  - confusion_matrix.png\n")
cat("  - feature_importance.png\n")
cat("  - training_curve.png\n")
if (has_shap) cat("  - shap_summary.png\n")
cat("\n")

# Cleanup parallel cluster
stopCluster(cl)
cat("✓ Parallel cluster stopped\n")
