# Detailed Usage Guide

This guide provides detailed instructions for using the STAT5003 Group Project codebase.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Running All 5 Models](#running-all-5-models)
3. [Viewing Model Results](#viewing-model-results)
4. [Running Preprocessing Documentation](#running-preprocessing-documentation)
5. [Understanding the Dataset](#understanding-the-dataset)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Quick Reference

### Option 1: Run All 5 Models (Recommended)

```r
# Open the main models file
file.edit("src/models/Team Models.Rmd")

# Click "Knit" button in RStudio, or run:
rmarkdown::render("src/models/Team Models.Rmd")
```

**Runtime:** ~20-40 minutes (all 5 models with hyperparameter tuning)

### Option 2: Generate Preprocessing Documentation

```r
# Open the preprocessing documentation
file.edit("docs/shared_preprocessing.Rmd")

# Click "Knit" button in RStudio, or run:
rmarkdown::render("docs/shared_preprocessing.Rmd")
```

**Runtime:** ~2-3 minutes

---

## Running All 5 Models

### What This Does

The `Team Models.Rmd` file contains all 5 classification models:
1. **Random Forest** - Grid search optimization
2. **GLM-NET** - Bayesian optimization (16 iterations)
3. **SVM** - Grid search with 5-fold CV
4. **Decision Tree** - Bayesian optimization (22 iterations)
5. **XGBoost** - Bayesian optimization (22 iterations)

### Step-by-Step

1. **Set Working Directory**
   ```r
   setwd("F:/Github Projects/5003")
   ```

2. **Open the File**
   ```r
   file.edit("src/models/Team Models.Rmd")
   ```

3. **Run the Entire Document**
   - **In RStudio:** Click the "Knit" button at the top
   - **From Console:**
     ```r
     rmarkdown::render("src/models/Team Models.Rmd")
     ```

### What Gets Created

After running, you'll find:

**Optimization Logs** (in `results/logs/`):
```
results/logs/
├── xgboost_optimization.log
├── random_forest_optimization.log
├── glmnet_optimization.log
└── decision_tree_optimization.log
```

**Model-Specific Results**:
```
results/
├── xgboost/              # XGBoost predictions, metrics
├── random_forest/        # Random Forest predictions, metrics
├── glmnet_sparse/        # GLM-NET predictions, metrics
├── decision_tree/        # Decision Tree predictions, metrics
└── comparison/           # Model comparison results
```

---

## Viewing Model Results

### Optimization Logs

Each model creates a detailed optimization log showing:
- Iteration-by-iteration progress
- Best hyperparameters found
- CV Macro F1 scores
- Test set performance
- Training time

**View XGBoost Log:**
```r
file.show("results/logs/xgboost_optimization.log")
```

**View Random Forest Log:**
```r
file.show("results/logs/random_forest_optimization.log")
```

**View GLM-NET Log:**
```r
file.show("results/logs/glmnet_optimization.log")
```

**View Decision Tree Log:**
```r
file.show("results/logs/decision_tree_optimization.log")
```

**View SVM Log:**
```r
file.show("results/logs/svm_optimization.log")
```

### Example Log Output

From `xgboost_optimization.log`:
```
=== OPTIMIZATION COMPLETE ===
Best iteration: 10/22
Best Macro F1: 0.7717
Best params: depth=15, eta=0.2954, subsample=1.00, colsample=1.00, child_wt=1.0, gamma=0.00

=== TEST SET PERFORMANCE ===
Test Accuracy: 0.9399
Test Precision (Macro): 0.8718
Test Recall (Macro): 0.8527
Test Macro F1: 0.8476
Optimization time: 1292.6 sec (21.5 min)
```

---

## Running Preprocessing Documentation

### What This Does

The `shared_preprocessing.Rmd` file documents the complete transformation:
- **Input:** `data/interim/final_decoded_clean.csv` (163 columns, Week 7 baseline)
- **Output:** `data/processed/final_cleaned_policy_focused.csv` (92 columns, policy-focused)

### Step-by-Step

1. **Open the File**
   ```r
   file.edit("docs/shared_preprocessing.Rmd")
   ```

2. **Run the Document**
   - **In RStudio:** Click the "Knit" button
   - **From Console:**
     ```r
     rmarkdown::render("docs/shared_preprocessing.Rmd")
     ```

### What Gets Created

After running, you'll get:
- **HTML Report:** `docs/shared_preprocessing.html`
  - Complete documentation of preprocessing steps
  - Shows all 81 variables removed and why
  - Shows all 9 features created
  - Validation that output matches reference dataset

---

## Understanding the Dataset

### The Policy-Focused Dataset

**File:** `data/processed/final_cleaned_policy_focused.csv`

**Key Facts:**
- **92 columns** (down from 163 in Week 7 baseline)
- **Removed:** 81 variables (vehicle proxies, administrative vars, low-value vars)
- **Added:** 9 new features (engineered features + missing indicators)
- **Used by:** All 5 models (shared dataset)

### What Was Removed?

**1. Vehicle Proxies (25 variables)**
- `VEH_VEHTYPE`, `STOP_VEHOCCUP`, etc.
- **Why:** Only exist when mode = vehicle → 100% data leakage

**2. Administrative Variables (54 variables)**
- Survey metadata, IDs, timestamps
- `STOPID`, `PERSID`, `HHID`, `TRIPID`, `VEHID`
- `STOP_ARRTIME`, `TRIP_DEPTIME`, etc.
- Survey weights: `PERS_PERSWGT14`, `STOP_STOPWGT14`, etc.

**3. Low-Value Variables (2 variables)**
- `PERS_REASONCODE`, `PERS_MRTDOW` (94% same value)

### What Was Added?

**9 New Features:**
- `VEH_PER_PERSON`, `VEH_PER_ADULT` (household vehicle availability ratios)
- `TRIP_SPEED` (calculated from distance/time)
- `TIME_PERIOD` (Morning_Peak/Midday/Evening_Peak/Off_Peak)
- `IS_WEEKEND` (binary indicator)
- `DIST_CATEGORY` (Very_Short/Short/Medium/Long/Very_Long)
- `HAS_LICENSE_AND_CAR` (interaction feature)
- `PERS_ADDITIONALTRAVEL_missing`, `PERS_MRTDOW_missing` (missing indicators)

### Model-Specific Preprocessing

Each model applies its own preprocessing as needed:

| Model | Scaling/Normalization | Why |
|-------|----------------------|-----|
| Random Forest | ❌ No | Tree-based: scale-invariant |
| GLM-NET | ✅ Yes (`step_normalize`) | Distance-based: requires normalization |
| SVM | ✅ Yes (manual `scale()`) | Distance-based: requires normalization |
| Decision Tree | ❌ No | Tree-based: scale-invariant |
| XGBoost | ❌ No | Tree-based: scale-invariant |

**Key Point:** The shared dataset provides **raw features**. Each model applies its own scaling/encoding as needed in `Team Models.Rmd`.

---

## Troubleshooting

### Error: Data file not found

**Problem:**
```
Error: Cannot find file 'data/processed/final_cleaned_policy_focused.csv'
```

**Solution:**
```r
# Check working directory
getwd()  # Should be: F:/Github Projects/5003

# If not, set it:
setwd("F:/Github Projects/5003")

# Verify file exists
file.exists("data/processed/final_cleaned_policy_focused.csv")
```

### Error: Package not found

**Problem:**
```
Error: there is no package called 'tidymodels'
```

**Solution:**
The RMarkdown documents auto-install packages, but you can manually install:
```r
install.packages(c(
  "tidymodels", "ranger", "xgboost", "ParBayesianOptimization",
  "rpart", "rpart.plot", "caret", "e1071", "glmnet",
  "doParallel", "dplyr", "readr"
))
```

### Models take too long

**Problem:** Training all 5 models takes 30-40 minutes

**Solutions:**

**Option 1: Run models individually**
- Open `Team Models.Rmd` in RStudio
- Click "Run" on specific code chunks (not "Knit")
- Run only the model you want (e.g., just XGBoost)

**Option 2: Reduce optimization iterations**
Edit the RMarkdown file:
- **XGBoost:** Change `max_steps = 22` to `max_steps = 10` (line ~1220)
- **GLM-NET:** Change `iterations = 16` to `iterations = 8` (line ~370)
- **Decision Tree:** Change `max_steps = 22` to `max_steps = 10` (line ~950)

**Option 3: Reduce CV folds**
Change from 5-fold to 3-fold CV:
- Find `folds <- vfold_cv(train_data, v = 5, strata = MAINMODE)`
- Change to `v = 3`

### Out of Memory

**Problem:**
```
Error: cannot allocate vector of size X GB
```

**Solutions:**

1. **Close other R sessions**
2. **Increase memory limit (Windows):**
   ```r
   memory.limit(size = 16000)  # 16GB
   ```
3. **Reduce parallel processing:**
   - Find `registerDoParallel(cores = detectCores() - 1)`
   - Change to `registerDoParallel(cores = 2)`

### Knit fails but code runs

**Problem:** RMarkdown knits fail but running chunks individually works

**Solution:**
```r
# Run from console instead of knitting:
rmarkdown::render("src/models/Team Models.Rmd",
                  output_file = "Team_Models_Output.html")
```

---

## Advanced Usage

### Running Individual Models

Instead of running all 5 models, you can run individual models:

1. **Open the file:**
   ```r
   file.edit("src/models/Team Models.Rmd")
   ```

2. **In RStudio:**
   - Click the green "Run" button on specific code chunks
   - Run only the sections you need

3. **Key Sections:**
   - **Setup:** Lines 1-48 (always run this first)
   - **Random Forest:** Lines 49-260
   - **GLM-NET:** Lines 261-623
   - **SVM:** Lines 628-883
   - **Decision Tree:** Lines 888-1158
   - **XGBoost:** Lines 1162-1643

### Customizing Hyperparameter Search

Each model has a hyperparameter search space defined in the code.

**Example: XGBoost**
```r
# Find this in Team Models.Rmd (around line 1213):
bounds <- list(
  max_depth = c(3L, 15L),
  eta = c(0.01, 0.3),
  subsample = c(0.5, 1.0),
  colsample_bytree = c(0.5, 1.0),
  min_child_weight = c(1L, 10L),
  gamma = c(0, 0.5)
)

# Modify to narrow the search:
bounds <- list(
  max_depth = c(10L, 15L),      # Focus on deeper trees
  eta = c(0.1, 0.3),            # Higher learning rates
  subsample = c(0.8, 1.0),      # More data per tree
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1L, 5L),
  gamma = c(0, 0.2)
)
```

### Using Different Train/Test Splits

All models use an 80/20 stratified split with seed=5003.

To change:
```r
# Find this in Team Models.Rmd (appears in each model section):
set.seed(5003)
split <- initial_split(final_data, prop = 0.80, strata = MAINMODE)

# Change to 70/30:
split <- initial_split(final_data, prop = 0.70, strata = MAINMODE)

# Change seed (must change in ALL 5 models for fair comparison):
set.seed(12345)
```

### Comparing Model Performance

After running all 5 models, compare their logs:

```r
# Read all logs
xgb_log <- readLines("results/logs/xgboost_optimization.log")
rf_log <- readLines("results/logs/random_forest_optimization.log")
glm_log <- readLines("results/logs/glmnet_optimization.log")
dt_log <- readLines("results/logs/decision_tree_optimization.log")
svm_log <- readLines("results/logs/svm_optimization.log")

# Extract test set performance
# Look for lines containing "Test Macro F1:"
grep("Test Macro F1", xgb_log, value = TRUE)
grep("Test Macro F1", rf_log, value = TRUE)
grep("Test Macro F1", glm_log, value = TRUE)
grep("Test Macro F1", dt_log, value = TRUE)
grep("Test Macro F1", svm_log, value = TRUE)
```

---

## For Report Writing

### Citing the Preprocessing

```markdown
We transformed the Week 7 baseline dataset (163 columns) into a policy-focused
dataset (92 columns) by removing 81 variables that cause data leakage or provide
no policy value, while adding 9 engineered features. This addresses the instructor
feedback to "remove variables that are direct proxies for the outcome."

See `docs/shared_preprocessing.Rmd` for complete documentation.
```

### Citing Model Comparison

```markdown
All 5 models use the same dataset (`final_cleaned_policy_focused.csv`) and
train/test split (80/20, stratified, seed=5003), ensuring a fair apples-to-apples
comparison.

Model optimization results are available in:
- `results/logs/xgboost_optimization.log`
- `results/logs/random_forest_optimization.log`
- `results/logs/glmnet_optimization.log`
- `results/logs/decision_tree_optimization.log`
- `results/logs/svm_optimization.log`
```

### Explaining Model-Specific Preprocessing

```markdown
While all models share the same input dataset (92 columns), each applies
preprocessing appropriate to its algorithm:

- Tree-based models (Random Forest, Decision Tree, XGBoost) do not require
  feature scaling, as decision trees are scale-invariant.

- Distance-based models (GLM-NET, SVM) apply normalization to numeric features,
  as their distance calculations are sensitive to feature scales.

This model-specific preprocessing is implemented in `src/models/Team Models.Rmd`.
```

---

## System Requirements

- **R:** Version 4.0 or higher
- **Memory:** 8GB RAM recommended (16GB for faster parallel processing)
- **Disk Space:** ~500MB for data and results
- **Time:**
  - All 5 models: ~20-40 minutes
  - Single model: ~5-10 minutes
  - Preprocessing doc: ~2-3 minutes

**Required R Packages** (auto-installed when running RMarkdown):
- `tidymodels`, `ranger`, `doParallel`, `dplyr`, `readr`
- `rpart`, `rpart.plot`, `caret`, `xgboost`
- `ParBayesianOptimization`, `glmnet`, `e1071`

---

## Questions?

1. Check `README.md` for overview and quick start
2. Review `docs/shared_preprocessing.Rmd` for preprocessing details
3. Check optimization logs in `results/logs/` for model training progress
4. Verify working directory: `getwd()` should be `F:/Github Projects/5003`

**Happy Modeling!**
