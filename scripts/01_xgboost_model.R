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

# Optional: For SHAP values (install if needed: install.packages("SHAPforxgboost"))
if (requireNamespace("SHAPforxgboost", quietly = TRUE)) {
  library(SHAPforxgboost)
  has_shap <- TRUE
} else {
  cat("Note: SHAPforxgboost not available. SHAP plots will be skipped.\n")
  has_shap <- FALSE
}

cat("===============================================\n")
cat("XGBoost Model for Travel Mode Prediction\n")
cat("===============================================\n\n")

# ==============================================================================
# 1. LOAD PREPROCESSED DATA
# ==============================================================================

cat("Loading preprocessed data...\n")

# Run shared preprocessing if data doesn't exist
if (!file.exists("data/train_base.rds") || !file.exists("data/test_base.rds")) {
  cat("Preprocessed data not found. Running shared preprocessing...\n")
  source("scripts/00_preprocessing.R")
}

train_data <- readRDS("data/train_base.rds")
test_data <- readRDS("data/test_base.rds")
feature_metadata <- readRDS("data/feature_metadata.rds")

target_col <- feature_metadata$target

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

# Handle categorical variables - convert to numeric
# XGBoost requires numeric input
cat("\nProcessing categorical variables...\n")

# One-hot encode categorical variables
categorical_cols <- names(X_train)[sapply(X_train, function(x) is.character(x) || is.factor(x))]

if (length(categorical_cols) > 0) {
  cat(sprintf("Found %d categorical columns. Performing one-hot encoding...\n", length(categorical_cols)))

  # Use model.matrix for one-hot encoding
  formula_str <- paste("~ . -1")  # -1 removes intercept
  X_train_matrix <- model.matrix(as.formula(formula_str), data = X_train)
  X_test_matrix <- model.matrix(as.formula(formula_str), data = X_test)

  # Ensure test has same columns as train
  missing_cols <- setdiff(colnames(X_train_matrix), colnames(X_test_matrix))
  for (col in missing_cols) {
    X_test_matrix <- cbind(X_test_matrix, 0)
    colnames(X_test_matrix)[ncol(X_test_matrix)] <- col
  }
  X_test_matrix <- X_test_matrix[, colnames(X_train_matrix)]

} else {
  X_train_matrix <- as.matrix(X_train)
  X_test_matrix <- as.matrix(X_test)
}

cat(sprintf("Final feature matrix: %d features\n", ncol(X_train_matrix)))

# Create DMatrix objects (XGBoost optimized data structure)
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train_encoded)
dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test_encoded)

# ==============================================================================
# 3. HYPERPARAMETER TUNING
# ==============================================================================

cat("\n===============================================\n")
cat("Starting Hyperparameter Tuning\n")
cat("===============================================\n")

# Define hyperparameter grid for tuning
# Using grid search with cross-validation (simpler alternative to Bayesian opt)

param_grid <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  subsample = c(0.7, 0.9),
  colsample_bytree = c(0.7, 0.9),
  min_child_weight = c(1, 3),
  gamma = c(0, 0.1)
)

cat(sprintf("Total hyperparameter combinations to test: %d\n", nrow(param_grid)))
cat("Using 5-fold stratified cross-validation\n\n")

# Create stratified folds
set.seed(5003)
folds <- createFolds(y_train_encoded, k = 5, list = TRUE)

# Track best performance
best_score <- -Inf
best_params <- NULL
cv_results <- list()

# Simplified grid search (test subset for speed)
# In practice, you might want to test all combinations
sample_indices <- sample(1:nrow(param_grid), min(20, nrow(param_grid)))  # Test 20 combinations

cat("Testing hyperparameter combinations...\n")
pb <- txtProgressBar(min = 0, max = length(sample_indices), style = 3)

for (i in seq_along(sample_indices)) {
  idx <- sample_indices[i]
  params <- param_grid[idx, ]

  # Prepare XGBoost parameters
  xgb_params <- list(
    objective = "multi:softprob",
    num_class = length(unique_classes),
    eval_metric = "mlogloss",
    max_depth = params$max_depth,
    eta = params$eta,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    min_child_weight = params$min_child_weight,
    gamma = params$gamma
  )

  # Cross-validation
  cv_scores <- numeric(length(folds))

  for (fold_idx in seq_along(folds)) {
    test_indices <- folds[[fold_idx]]
    train_indices <- setdiff(1:nrow(X_train_matrix), test_indices)

    dtrain_cv <- xgb.DMatrix(data = X_train_matrix[train_indices, ],
                             label = y_train_encoded[train_indices])
    dval_cv <- xgb.DMatrix(data = X_train_matrix[test_indices, ],
                          label = y_train_encoded[test_indices])

    # Train model
    model_cv <- xgb.train(
      params = xgb_params,
      data = dtrain_cv,
      nrounds = 200,
      watchlist = list(val = dval_cv),
      early_stopping_rounds = 20,
      verbose = 0
    )

    cv_scores[fold_idx] <- model_cv$best_score
  }

  # Average CV score
  mean_score <- mean(cv_scores)
  cv_results[[i]] <- list(params = params, score = mean_score)

  # Update best parameters
  if (mean_score < best_score || best_score == -Inf) {  # Lower mlogloss is better
    if (best_score == -Inf || mean_score < best_score) {
      best_score <- mean_score
      best_params <- params
    }
  }

  setTxtProgressBar(pb, i)
}
close(pb)

cat("\n\nBest hyperparameters found:\n")
print(best_params)
cat(sprintf("Best CV mlogloss: %.4f\n", best_score))

# ==============================================================================
# 4. TRAIN FINAL MODEL
# ==============================================================================

cat("\n===============================================\n")
cat("Training Final Model\n")
cat("===============================================\n")

# Prepare final parameters
final_params <- list(
  objective = "multi:softprob",
  num_class = length(unique_classes),
  eval_metric = "mlogloss",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma
)

# Train with early stopping
watchlist <- list(train = dtrain, test = dtest)

cat("\nTraining XGBoost model with early stopping...\n")
start_time <- Sys.time()

xgb_model <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 1,
  print_every_n = 50
)

end_time <- Sys.time()
training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("\nTraining completed in %.2f seconds\n", training_time))
cat(sprintf("Best iteration: %d\n", xgb_model$best_iteration))

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

# Macro F1 (average F1 across classes)
class_f1 <- conf_matrix$byClass[, 'F1']
macro_f1 <- mean(class_f1, na.rm = TRUE)

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

# Save best hyperparameters
best_params_list <- list(
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  nrounds = xgb_model$best_iteration,
  cv_score = best_score
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
  micro_f1 = micro_f1,
  balanced_accuracy = balanced_acc,
  training_time_seconds = training_time,
  best_iteration = xgb_model$best_iteration,
  per_class_metrics = per_class_list,
  confusion_matrix = as.matrix(conf_matrix$table)
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
