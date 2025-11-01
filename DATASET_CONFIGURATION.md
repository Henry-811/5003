# Dataset Configuration Guide

## Quick Start

Change the `DATASET_CHOICE` parameter in `src/preprocessing/01_preprocess_data.R` to select which dataset to use.

## Configuration Options

### Option 1: Policy-Focused (Recommended for Week 11+)

```r
# In src/preprocessing/01_preprocess_data.R (line 38)
DATASET_CHOICE <- "policy_focused"
```

**What it does:**
- Uses `data/processed/final_cleaned_policy_focused.csv`
- Vehicle proxies removed (91 features)
- Better for interpretable policy insights
- Addresses Week 7 feedback about removing "trivial predictors"

**Then run:**
```r
source('src/preprocessing/01_preprocess_data.R')  # Creates train/test splits
source('src/models/xgboost_model.R')              # Trains model
```

---

### Option 2: Original (For Comparison)

```r
# In src/preprocessing/01_preprocess_data.R (line 38)
DATASET_CHOICE <- "original"
```

**What it does:**
- Uses `data/raw/data_test.csv` (or `data/processed/final_cleaned_data.csv`)
- Contains all vehicle proxies (130+ features)
- Higher accuracy but includes data leakage
- Useful for baseline comparison

**Then run:**
```r
source('src/preprocessing/01_preprocess_data.R')  # Creates train/test splits
source('src/models/xgboost_model.R')              # Trains model
```

---

### Option 3: Auto-Detect (Default)

```r
# In src/preprocessing/01_preprocess_data.R (line 38)
DATASET_CHOICE <- "auto"
```

**What it does:**
- Automatically selects the best available dataset
- **Preference order:**
  1. Policy-focused (if exists)
  2. Original raw data
  3. Original cleaned data
  4. Week 7 data

---

## Complete Workflow Examples

### Example 1: Train with Policy-Focused Data

```r
# Step 1: Create policy-focused dataset (if not already done)
source('src/preprocessing/02_remove_vehicle_proxies.R')

# Step 2: Set configuration to use policy-focused
# Edit src/preprocessing/01_preprocess_data.R line 38:
DATASET_CHOICE <- "policy_focused"

# Step 3: Preprocess and train
source('src/preprocessing/01_preprocess_data.R')
source('src/models/xgboost_model.R')
```

**Output:**
- `data/processed/train_base.rds` - Train split (policy-focused, 91 features)
- `data/processed/test_base.rds` - Test split (policy-focused, 91 features)
- `results/xgboost/xgboost_model.rds` - Trained model
- `results/xgboost/best_params.json` - Optimized hyperparameters

---

### Example 2: Train with Original Data (Baseline)

```r
# Step 1: Set configuration to use original
# Edit src/preprocessing/01_preprocess_data.R line 38:
DATASET_CHOICE <- "original"

# Step 2: Preprocess and train
source('src/preprocessing/01_preprocess_data.R')
source('src/models/xgboost_model.R')
```

**Output:**
- `data/processed/train_base.rds` - Train split (original, 130+ features)
- `data/processed/test_base.rds` - Test split (original, 130+ features)
- `results/xgboost/xgboost_model.rds` - Trained model (high accuracy but leakage)
- `results/xgboost/best_params.json` - Optimized hyperparameters

---

### Example 3: Compare Both Models

```r
# Method 1: Use the comparison script (recommended)
source('scripts/compare_models.R')
# This handles everything automatically

# Method 2: Manual comparison
# Train with original
DATASET_CHOICE <- "original"
source('src/preprocessing/01_preprocess_data.R')
source('src/models/xgboost_model.R')
# Save results to results/xgboost_original/

# Train with policy-focused
DATASET_CHOICE <- "policy_focused"
source('src/preprocessing/01_preprocess_data.R')
source('src/models/xgboost_model.R')
# Save results to results/xgboost_policy/
```

---

## What Happens Behind the Scenes

### When you set `DATASET_CHOICE <- "policy_focused"`:

1. **Preprocessing** (`01_preprocess_data.R`):
   - Loads `data/processed/final_cleaned_policy_focused.csv` (92 columns)
   - Removes zero-variance features (3 features)
   - Creates train/test splits (80/20)
   - Saves to `data/processed/train_base.rds` (18,074 rows, 91 features)
   - Saves metadata: `dataset_type = "policy_focused"`

2. **Model Training** (`xgboost_model.R`):
   - Loads `data/processed/train_base.rds`
   - Runs Bayesian Optimization (15 iterations, ~5-15 min)
   - Trains final model with optimized hyperparameters
   - Expected accuracy: ~75-85% (lower than original, but more interpretable)

### When you set `DATASET_CHOICE <- "original"`:

1. **Preprocessing** (`01_preprocess_data.R`):
   - Loads `data/raw/data_test.csv` (111 columns)
   - Removes zero-variance features
   - Creates train/test splits (80/20)
   - Saves to `data/processed/train_base.rds` (18,074 rows, 110+ features)
   - Saves metadata: `dataset_type = "original"`

2. **Model Training** (`xgboost_model.R`):
   - Loads `data/processed/train_base.rds`
   - Runs Bayesian Optimization (15 iterations, ~5-15 min)
   - Trains final model with optimized hyperparameters
   - Expected accuracy: ~92-93% (high but includes data leakage)

---

## File Locations

### Input Files (Source Data)
```
data/raw/data_test.csv                          # Original with proxies (111 cols)
data/processed/final_cleaned_policy_focused.csv # Policy-focused (92 cols)
data/processed/final_cleaned_data.csv           # Original cleaned (optional)
```

### Intermediate Files (Always Overwritten)
```
data/processed/train_base.rds                   # Train split (changes based on DATASET_CHOICE)
data/processed/test_base.rds                    # Test split (changes based on DATASET_CHOICE)
data/processed/preprocessing_metadata.rds       # Metadata (includes dataset_type)
```

### Output Files (Model Results)
```
results/xgboost/xgboost_model.rds               # Trained model
results/xgboost/best_params.json                # Optimized hyperparameters
results/xgboost/metrics.json                    # Accuracy, F1, etc.
results/xgboost/confusion_matrix.png            # Confusion matrix heatmap
results/xgboost/feature_importance.csv          # Feature importance scores
```

---

## Tips

1. **For final submission**: Use `DATASET_CHOICE <- "policy_focused"` to address Week 7 feedback

2. **For debugging/baseline**: Use `DATASET_CHOICE <- "original"` to verify high accuracy is achievable

3. **For comparison**: Run both and compare metrics using `scripts/compare_models.R`

4. **Always check the console output** to confirm which dataset was loaded:
   ```
   Dataset Selection Configuration:
     Choice: policy_focused

   Step 1: Loading data...
   ✓ Using policy-focused dataset (vehicle proxies removed)
   ✓ Dataset type: policy_focused
   ✓ Source file: data/processed/final_cleaned_policy_focused.csv
   ```

---

## Troubleshooting

**Error: "Policy-focused dataset not found"**
```r
# Solution: Create it first
source('src/preprocessing/02_remove_vehicle_proxies.R')
```

**Error: "Original dataset not found"**
```r
# Solution: Check that data/raw/data_test.csv exists
# Or use auto mode: DATASET_CHOICE <- "auto"
```

**Model results seem wrong**
```r
# Check which dataset was actually loaded
meta <- readRDS("data/processed/preprocessing_metadata.rds")
print(meta$dataset_type)  # Should match your DATASET_CHOICE
print(meta$source_file)   # Shows actual file used
```

---

**Version:** 1.0
**Last Updated:** 2025-10-22
**Author:** STAT5003 Group G07
