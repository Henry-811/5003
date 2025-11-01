# STAT5003 Group Project - Travel Mode Prediction

**Group:** Week 11, Group 07
**Task:** Multi-class classification for travel mode choice prediction
**Dataset:** Gold Coast Household Travel Survey data (2015)

---

## ⭐ Key Deliverables

This repository contains the complete implementation for our team's travel mode prediction project:

### 1. **Team Models** (`src/models/Team Models.Rmd`)
All 5 classification models in a single executable RMarkdown document:
- Random Forest
- GLM-NET (Elastic Net Logistic Regression)
- SVM (Support Vector Machine)
- Decision Tree
- XGBoost

**Usage:**
```r
# Open in RStudio and click "Knit" or run:
rmarkdown::render("src/models/Team Models.Rmd")
```

### 2. **Shared Preprocessing Documentation** (`docs/shared_preprocessing.Rmd`)
Complete documentation of the preprocessing pipeline that transforms the Week 7 baseline dataset into the policy-focused dataset.

**Transformation:**
- Input: `data/interim/final_decoded_clean.csv` (163 columns, Week 7 baseline)
- Output: `data/processed/final_cleaned_policy_focused.csv` (92 columns, policy-focused)
- Removes: 81 variables (vehicle proxies, administrative vars, low-value vars)
- Adds: 9 new features (engineered features + missing indicators)

**Usage:**
```r
# Open in RStudio and click "Knit" or run:
rmarkdown::render("docs/shared_preprocessing.Rmd")
```

### 3. **Policy-Focused Dataset** (`data/processed/final_cleaned_policy_focused.csv`)
The shared dataset (92 columns) used by all 5 models:
- ✅ Vehicle proxies removed (addresses Week 7 instructor feedback)
- ✅ Feature engineering applied (VEH_PER_PERSON, TRIP_SPEED, TIME_PERIOD, etc.)
- ✅ Ready for model training (all 5 models use this same dataset)

---

## 📁 Project Structure

```
5003/
├── data/
│   ├── interim/
│   │   └── final_decoded_clean.csv          # Week 7 baseline (163 cols)
│   ├── processed/
│   │   └── final_cleaned_policy_focused.csv # ⭐ MAIN DATASET (92 cols)
│   └── raw/
│       └── data_test.csv                     # Original raw data
│
├── docs/
│   ├── shared_preprocessing.Rmd              # ⭐ PREPROCESSING DOC
│   └── references/
│       └── STAT5003_Group_Project_Instruction.html
│
├── src/
│   ├── models/
│   │   └── Team Models.Rmd                   # ⭐ ALL 5 MODELS
│   ├── preprocessing/                        # Preprocessing R scripts
│   │   ├── 00_analyze_data.R
│   │   ├── 01_preprocess_data.R
│   │   └── 02_remove_vehicle_proxies.R
│   └── run_pipeline.R                        # Legacy pipeline runner
│
├── results/
│   ├── logs/                                 # Model optimization logs
│   │   ├── xgboost_optimization.log
│   │   ├── random_forest_optimization.log
│   │   ├── glmnet_optimization.log
│   │   └── decision_tree_optimization.log
│   ├── xgboost/                              # XGBoost outputs
│   ├── decision_tree/                        # Decision Tree outputs
│   ├── glmnet_sparse/                        # GLM-NET outputs
│   ├── preprocessing/                        # Preprocessing diagnostics
│   └── comparison/                           # Model comparison results
│
├── reports/                                  # Final report files
│   └── Group_W11_G07.Rmd
│
├── README.md                                 # This file
└── USAGE.md                                  # Detailed usage guide
```

---

## 🚀 Quick Start

### Option 1: Run All 5 Models (Recommended)

```r
# Open in RStudio
file.edit("src/models/Team Models.Rmd")

# Click "Knit" button or run:
rmarkdown::render("src/models/Team Models.Rmd")
```

**What this does:**
- Loads the shared policy-focused dataset (92 columns)
- Trains all 5 models with Bayesian/Grid search optimization
- Generates model logs in `results/logs/`
- Outputs predictions and metrics for each model
- Creates optimization progress reports

**Runtime:** ~20-40 minutes (all 5 models with hyperparameter tuning)

### Option 2: Run Preprocessing Documentation

```r
# Generate preprocessing documentation
rmarkdown::render("docs/shared_preprocessing.Rmd")
```

**What this does:**
- Documents the complete transformation from Week 7 → policy-focused dataset
- Shows all 81 variables removed and why
- Shows all 9 features created
- Validates the output matches the reference dataset

---

## 📊 Model Performance

After running `Team Models.Rmd`, check the optimization logs:

```r
# View XGBoost optimization progress
file.show("results/logs/xgboost_optimization.log")

# View Random Forest optimization progress
file.show("results/logs/random_forest_optimization.log")

# View GLM-NET optimization progress
file.show("results/logs/glmnet_optimization.log")

# View Decision Tree optimization progress
file.show("results/logs/decision_tree_optimization.log")

# View SVM optimization progress
file.show("results/logs/svm_optimization.log")
```

Each log contains:
- Iteration-by-iteration optimization progress
- Best hyperparameters found
- CV Macro F1 scores
- Test set performance metrics
- Training time

---

## 🔍 What Makes This Dataset "Policy-Focused"?

### Week 7 Instructor Feedback:
> "Consider removing variables that are direct proxies for the outcome (such as vehicle attributes)"

### Our Solution:

**❌ Removed (81 variables):**
1. **Vehicle Proxies (25 vars):** `VEH_VEHTYPE`, `STOP_VEHOCCUP`, etc.
   - Only exist when mode = vehicle → 100% data leakage
   - Example: If `VEH_VEHTYPE` exists, person definitely used vehicle

2. **Administrative Variables (54 vars):** Survey metadata, IDs, timestamps
   - `STOPID`, `PERSID`, `HHID`, `TRIPID`, `VEHID`
   - `STOP_ARRTIME`, `TRIP_DEPTIME`, etc.
   - Survey weights: `PERS_PERSWGT14`, `STOP_STOPWGT14`, etc.

3. **Low-Value Variables (2 vars):** 94% same value
   - `PERS_REASONCODE`, `PERS_MRTDOW`

**✅ Kept:**
- `HH_CARS` (household vehicle **availability**, not usage)
- `PERS_CARLICENCE` (capability, not behavior)
- Demographics, trip characteristics, geographic features

**➕ Added (9 new features):**
- `VEH_PER_PERSON`, `VEH_PER_ADULT` (household vehicle availability ratios)
- `TRIP_SPEED` (calculated from distance/time)
- `TIME_PERIOD` (Morning_Peak/Midday/Evening_Peak/Off_Peak)
- `IS_WEEKEND` (binary indicator)
- `DIST_CATEGORY` (Very_Short/Short/Medium/Long/Very_Long)
- `HAS_LICENSE_AND_CAR` (interaction feature)
- `PERS_ADDITIONALTRAVEL_missing`, `PERS_MRTDOW_missing` (missing indicators)

---

## 📋 Model-Specific Preprocessing

Each model applies its own preprocessing as needed:

| Model | Scaling/Normalization | Why |
|-------|----------------------|-----|
| Random Forest | ❌ No | Tree-based: scale-invariant |
| GLM-NET | ✅ Yes (`step_normalize`) | Distance-based: requires normalization |
| SVM | ✅ Yes (manual `scale()`) | Distance-based: requires normalization |
| Decision Tree | ❌ No | Tree-based: scale-invariant |
| XGBoost | ❌ No | Tree-based: scale-invariant |

**Key Point:** The shared dataset provides **raw features**. Each model applies its own scaling/encoding as needed.

---

## 🎯 For Report Writing

### Preprocessing Section:
```markdown
We transformed the Week 7 baseline dataset (163 columns) into a policy-focused
dataset (92 columns) by removing 81 variables that cause data leakage or provide
no policy value, while adding 9 engineered features. This addresses the instructor
feedback to "remove variables that are direct proxies for the outcome."

See `docs/shared_preprocessing.Rmd` for complete documentation.
```

### Model Comparison Section:
```r
# All 5 models use the same dataset and train/test split
# Fair comparison - apples to apples

# Results available in:
# - results/logs/xgboost_optimization.log
# - results/logs/random_forest_optimization.log
# - results/logs/glmnet_optimization.log
# - results/logs/decision_tree_optimization.log
# - results/logs/svm_optimization.log
```

---

## 💡 Key Design Decisions

### 1. Shared Preprocessing vs. Model-Specific
- **Shared:** Variable removal, feature engineering → `final_cleaned_policy_focused.csv`
- **Model-specific:** Train/test split, encoding, scaling → in each model's script

**Why:** Ensures fair comparison while allowing model-specific optimizations

### 2. No Cached Preprocessing Data
- `Team Models.Rmd` loads the CSV directly each time
- Each model does its own train/test split (80/20, stratified, seed=5003)
- Ensures reproducibility

### 3. Optimization Methods
- **XGBoost, Decision Tree:** Bayesian Optimization (22 iterations)
- **Random Forest:** Grid Search (multiple mtry/min_n/trees combinations)
- **GLM-NET:** Bayesian Optimization (16 iterations)
- **SVM:** Grid Search (cost/gamma combinations, 5-fold CV)

---

## 🔧 System Requirements

- **R:** Version 4.0 or higher
- **Memory:** 8GB RAM recommended
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

## 📚 Documentation Files

- `README.md` (this file) - Overview and quick start
- `USAGE.md` - Detailed usage instructions
- `docs/shared_preprocessing.Rmd` - Preprocessing documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `REFACTORING_SUMMARY.md` - Code improvements made
- `DATASET_CONFIGURATION.md` - Dataset information

---

## 🐛 Troubleshooting

### Error: Data file not found
```r
# Check working directory
getwd()  # Should be: F:/Github Projects/5003

# If not, set it:
setwd("F:/Github Projects/5003")
```

### Error: Package not found
The RMarkdown documents auto-install packages, but you can manually install:
```r
install.packages(c("tidymodels", "ranger", "xgboost", "ParBayesianOptimization"))
```

### Models take too long
- Reduce optimization iterations in the RMarkdown code chunks
- Run models individually instead of all 5 at once
- Use fewer CV folds (change from 5 to 3)

---

## 📝 Credits

**Group:** Week 11, Group 07
**Course:** STAT5003 - Computational Statistical Methods
**Institution:** University of Sydney

---

## ❓ Questions?

1. Check `USAGE.md` for detailed instructions
2. Review `docs/shared_preprocessing.Rmd` for preprocessing details
3. Check optimization logs in `results/logs/` for model training progress
4. Verify working directory: `getwd()` should be `F:/Github Projects/5003`

**Happy Modeling!** 🚀
