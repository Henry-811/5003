# ==============================================================================
# Combine Train/Test to Show Final Cleaned Dataset Before Split
# ==============================================================================

cat("Combining train and test datasets...\n\n")

# Load both datasets
train <- readRDS("data/processed/train_base.rds")
test <- readRDS("data/processed/test_base.rds")

cat(sprintf("Train: %d rows\n", nrow(train)))
cat(sprintf("Test:  %d rows\n", nrow(test)))

# Combine them back together (this recreates the dataset before split)
combined <- rbind(train, test)

cat(sprintf("\nCombined: %d rows × %d columns\n\n", nrow(combined), ncol(combined)))

# Save as CSV
write.csv(combined, "data/processed/csv_export/final_cleaned_before_split.csv", row.names = FALSE)
cat("✓ Saved: data/processed/csv_export/final_cleaned_before_split.csv\n")

# Create a sample (first 200 rows)
combined_sample <- head(combined, 200)
write.csv(combined_sample, "data/processed/csv_export/final_cleaned_sample_200.csv", row.names = FALSE)
cat("✓ Saved: data/processed/csv_export/final_cleaned_sample_200.csv\n")

cat("\nDone! You now have the complete cleaned dataset before train/test split.\n")
