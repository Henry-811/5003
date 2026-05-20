# ==============================================================================
# Export Processed Data to CSV
# ==============================================================================
# Converts RDS files to CSV format for easy inspection in Excel/other tools
# ==============================================================================

cat("==============================================================================\n")
cat("EXPORTING PROCESSED DATA TO CSV\n")
cat("==============================================================================\n\n")

# Check if processed data exists
if (!file.exists("data/processed/train_base.rds")) {
  stop("Error: Processed data not found. Run preprocessing first:\n  source('src/preprocessing/01_preprocess_data.R')")
}

# Create export directory
if (!dir.exists("data/processed/csv_export")) {
  dir.create("data/processed/csv_export")
}

# ==============================================================================
# 1. EXPORT TRAIN DATA
# ==============================================================================

cat("Exporting training data...\n")
train_base <- readRDS("data/processed/train_base.rds")
write.csv(train_base, "data/processed/csv_export/train_base.csv", row.names = FALSE)
cat(sprintf("✓ Saved: data/processed/csv_export/train_base.csv (%d rows, %d columns)\n\n",
            nrow(train_base), ncol(train_base)))

# ==============================================================================
# 2. EXPORT TEST DATA
# ==============================================================================

cat("Exporting test data...\n")
test_base <- readRDS("data/processed/test_base.rds")
write.csv(test_base, "data/processed/csv_export/test_base.csv", row.names = FALSE)
cat(sprintf("✓ Saved: data/processed/csv_export/test_base.csv (%d rows, %d columns)\n\n",
            nrow(test_base), ncol(test_base)))

# ==============================================================================
# 3. EXPORT METADATA
# ==============================================================================

cat("Exporting preprocessing metadata...\n")
metadata <- readRDS("data/processed/preprocessing_metadata.rds")

# Convert metadata to readable format
metadata_df <- data.frame(
  Item = c(
    "Target Variable",
    "Original Features",
    "Final Features",
    "Numeric Features",
    "Categorical Features",
    "High Cardinality Features",
    "Train Size",
    "Test Size",
    "Random Seed",
    "Processing Date"
  ),
  Value = c(
    metadata$target_variable,
    metadata$n_original_features,
    metadata$n_final_features,
    length(metadata$numeric_features),
    length(metadata$categorical_features),
    length(metadata$high_cardinality_features),
    metadata$train_size,
    metadata$test_size,
    metadata$random_seed,
    as.character(metadata$preprocessing_date)
  )
)

write.csv(metadata_df, "data/processed/csv_export/metadata_summary.csv", row.names = FALSE)
cat("✓ Saved: data/processed/csv_export/metadata_summary.csv\n\n")

# Export feature lists
cat("Exporting feature lists...\n")

# Features removed
if (length(metadata$features_removed) > 0) {
  removed_df <- data.frame(
    Feature = metadata$features_removed,
    Status = "Removed"
  )
  write.csv(removed_df, "data/processed/csv_export/features_removed.csv", row.names = FALSE)
  cat(sprintf("✓ Saved: features_removed.csv (%d features)\n", nrow(removed_df)))
}

# Features added
if (length(metadata$features_added) > 0) {
  added_df <- data.frame(
    Feature = metadata$features_added,
    Status = "Added (Engineered)"
  )
  write.csv(added_df, "data/processed/csv_export/features_added.csv", row.names = FALSE)
  cat(sprintf("✓ Saved: features_added.csv (%d features)\n", nrow(added_df)))
}

# Final numeric features
numeric_df <- data.frame(
  Feature = metadata$numeric_features,
  Type = "Numeric"
)
write.csv(numeric_df, "data/processed/csv_export/numeric_features.csv", row.names = FALSE)
cat(sprintf("✓ Saved: numeric_features.csv (%d features)\n", length(metadata$numeric_features)))

# Final categorical features
categorical_df <- data.frame(
  Feature = metadata$categorical_features,
  Type = "Categorical"
)
write.csv(categorical_df, "data/processed/csv_export/categorical_features.csv", row.names = FALSE)
cat(sprintf("✓ Saved: categorical_features.csv (%d features)\n", length(metadata$categorical_features)))

cat("\n")

# ==============================================================================
# 4. EXPORT CLASS WEIGHTS
# ==============================================================================

cat("Exporting class weights...\n")
class_weights <- readRDS("data/processed/class_weights.rds")

class_weights_df <- data.frame(
  Class = names(class_weights$class_counts),
  Count = as.numeric(class_weights$class_counts),
  Weight = as.numeric(class_weights$class_weights)
)

write.csv(class_weights_df, "data/processed/csv_export/class_weights.csv", row.names = FALSE)
cat("✓ Saved: data/processed/csv_export/class_weights.csv\n\n")

# ==============================================================================
# 5. CREATE SAMPLE FILES (First 100 rows for quick inspection)
# ==============================================================================

cat("Creating sample files (first 100 rows)...\n")

train_sample <- head(train_base, 100)
write.csv(train_sample, "data/processed/csv_export/train_sample_100.csv", row.names = FALSE)
cat("✓ Saved: train_sample_100.csv (100 rows)\n")

test_sample <- head(test_base, 100)
write.csv(test_sample, "data/processed/csv_export/test_sample_100.csv", row.names = FALSE)
cat("✓ Saved: test_sample_100.csv (100 rows)\n\n")

# ==============================================================================
# 6. SUMMARY
# ==============================================================================

cat("==============================================================================\n")
cat("EXPORT COMPLETE\n")
cat("==============================================================================\n\n")

cat("All files exported to: data/processed/csv_export/\n\n")

cat("Files created:\n")
cat("  Data Files:\n")
cat("    - train_base.csv (full training data)\n")
cat("    - test_base.csv (full test data)\n")
cat("    - train_sample_100.csv (first 100 rows)\n")
cat("    - test_sample_100.csv (first 100 rows)\n\n")

cat("  Metadata Files:\n")
cat("    - metadata_summary.csv (preprocessing info)\n")
cat("    - features_removed.csv (dropped features)\n")
cat("    - features_added.csv (engineered features)\n")
cat("    - numeric_features.csv (final numeric features)\n")
cat("    - categorical_features.csv (final categorical features)\n")
cat("    - class_weights.csv (for imbalanced learning)\n\n")

cat("==============================================================================\n")
cat("TIPS FOR INSPECTION:\n")
cat("==============================================================================\n")
cat("1. Open train_sample_100.csv in Excel for quick inspection\n")
cat("2. Check metadata_summary.csv to see preprocessing statistics\n")
cat("3. Review features_removed.csv to see what was dropped and why\n")
cat("4. Review features_added.csv to see engineered features\n")
cat("5. Check class_weights.csv to see class imbalance handling\n\n")

cat("Note: Full CSV files may be large. Use sample files for quick review.\n\n")
