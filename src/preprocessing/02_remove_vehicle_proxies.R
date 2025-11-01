# ==============================================================================
# Remove Vehicle Proxies - Addressing Week 7 Feedback
# ==============================================================================
# Creates policy-focused dataset by removing:
# 1. Trip-specific vehicle attributes (data leakage)
# 2. Low-value variables (94% same value)
# 3. Administrative/redundant variables
#
# Keeps interpretable features for policy insights:
# - Demographics (age, sex, income, occupation)
# - Trip characteristics (distance, time, purpose)
# - Household availability (cars/bikes owned, not used)
# - Engineered features (ratios, categories)
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

cat("==============================================================================\n")
cat("CREATING POLICY-FOCUSED DATASET\n")
cat("Removing vehicle proxies and low-value features\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

cat("Step 1: Loading cleaned data...\n")

if (!file.exists("data/processed/final_cleaned_data.csv")) {
  stop("Error: data/processed/final_cleaned_data.csv not found")
}

df_original <- read.csv("data/processed/final_cleaned_data.csv",
                        header = TRUE,
                        fileEncoding = "UTF-8",
                        stringsAsFactors = FALSE)

cat(sprintf("✓ Loaded: %d rows × %d columns\n\n", nrow(df_original), ncol(df_original)))

# Keep copy for comparison
df <- df_original

# ==============================================================================
# 2. DEFINE VARIABLES TO REMOVE
# ==============================================================================

cat("Step 2: Identifying variables to remove...\n\n")

# -----------------------------------------------------------------------------
# A. TRIP-SPECIFIC VEHICLE PROXIES (DATA LEAKAGE)
# -----------------------------------------------------------------------------
# These variables are only populated when mode = vehicle
# They provide 100% certainty about the target → trivial prediction

vehicle_proxies <- c(
  # Stop-level vehicle attributes (only exist if used vehicle)
  "STOP_VEHOCCUP",          # Vehicle occupancy - only filled if drove
  "STOP_VEHONORANGEFORM",   # Vehicle number on form - only filled if drove
  "STOP_VEHNUM",            # Vehicle number - only filled if drove
  "STOP_VEHPARKED",         # Where parked - only filled if drove
  "STOP_VEHWALKTIME",       # Walk time to vehicle - only filled if drove

  # Vehicle-level attributes (only exist if used vehicle)
  "VEH_VEHTYPE",            # Car/Ute/Van - 100% leakage: if exists → drove
  "VEH_VEHYEAR",            # Vehicle year - only for vehicle users
  "VEH_VEHAGE",             # Vehicle age - only for vehicle users
  "VEH_PETROL",             # Petrol fuel type - only for vehicle users
  "VEH_DIESEL",             # Diesel fuel type - only for vehicle users
  "VEH_GAS",                # Gas fuel type - only for vehicle users
  "VEH_ELECTRIC",           # Electric - only for vehicle users
  "VEH_HYBRID",             # Hybrid - only for vehicle users
  "VEH_RUNNINGCOST",        # Running cost - only for vehicle users
  "VEH_HHWGT14"             # Duplicate weight variable
)

cat("A. Trip-Specific Vehicle Proxies (Data Leakage):\n")
vehicle_proxies_present <- intersect(vehicle_proxies, names(df))
cat(sprintf("   - %d variables identified\n", length(vehicle_proxies_present)))
for (var in vehicle_proxies_present) {
  cat(sprintf("     • %s\n", var))
}
cat("\n")

# -----------------------------------------------------------------------------
# B. LOW-VALUE VARIABLES (94% same value)
# -----------------------------------------------------------------------------
# These have extremely low variance and minimal predictive power
# We'll replace with binary missingness indicators

low_value_vars <- c(
  "PERS_REASONCODE",        # 94% = "N/A (some stops were made)"
  "PERS_MRTDOW"             # 94% = "N/A (some stops were made)"
)

cat("B. Low-Value Variables (94% same value):\n")
cat("   - Will be replaced with binary missingness indicators\n")
for (var in low_value_vars) {
  if (var %in% names(df)) {
    cat(sprintf("     • %s\n", var))
  }
}
cat("\n")

# Create binary indicators before removing
cat("   Creating binary missingness indicators...\n")
if ("PERS_REASONCODE" %in% names(df)) {
  df$PERS_REASONCODE_missing <- as.integer(
    df$PERS_REASONCODE == "N/A (some stops were made)"
  )
  cat("     ✓ Created PERS_REASONCODE_missing\n")
}

if ("PERS_MRTDOW" %in% names(df)) {
  df$PERS_MRTDOW_missing <- as.integer(
    df$PERS_MRTDOW == "N/A (some stops were made)"
  )
  cat("     ✓ Created PERS_MRTDOW_missing\n")
}
cat("\n")

# -----------------------------------------------------------------------------
# C. ADMINISTRATIVE/REDUNDANT VARIABLES
# -----------------------------------------------------------------------------
# Survey metadata and overly specific features not useful for policy insights

administrative_vars <- c(
  # Survey administration metadata
  "PERS_WHOFILLED",         # Who filled the survey - not predictive
  "PERS_PROXY",             # Proxy respondent - not predictive
  "PERS_FILLDOW",           # Day survey filled - not predictive
  "PERS_FILLAG",            # Age of survey - not predictive
  "PERS_FURTHERRESEARCH",   # Consent for research - not predictive
  "PERS_NEWSTOP",           # Administrative flag

  # Cycling-specific (too granular, not broadly relevant)
  "PERS_CYCLEDWORK",        # Cycled to work in past week
  "PERS_CYCLEDSHOPPING",    # Cycled shopping in past week
  "PERS_CYCLEDSCHOOL",      # Cycled to school in past week
  "PERS_CYCLEDEXERCISE",    # Cycled for exercise in past week
  "PERS_CYCLEDOTHER",       # Cycled other in past week
  "PERS_NOCYCLED",          # Did not cycle in past week
  "PERS_CYCLEDBIKEPARK",    # Access to bike parking - too specific
  "PERS_CYCLEDSHOWERS",     # Access to showers - too specific

  # Technology metadata (not policy-relevant)
  "PERS_ISSMARTPHONE",      # Has smartphone - weak predictor
  "PERS_SMARTPHONEBRAND",   # Phone brand - not relevant

  # Trip metadata
  "TRIP_TRIPSTAGES",        # Internal trip structure - redundant
  "TRIP_TRIPNO",            # Trip number - just ordering
  "TRIP_TIME1",             # Duplicate of TRIP_TRAVTIME
  "TRIP_TRIPBASEWEIGHT",    # Duplicate weight

  # Stop metadata
  "STOP_MORESTOPS",         # Boolean - captured in PERS_NUMSTOPS

  # Household metadata
  "HH_TRAVDATE",            # Specific date - temporal already captured
  "HH_TRAVMONTH"            # Month - not needed (IS_WEEKEND captures pattern)
)

cat("C. Administrative/Redundant Variables:\n")
administrative_vars_present <- intersect(administrative_vars, names(df))
cat(sprintf("   - %d variables identified\n", length(administrative_vars_present)))
cat("   Categories:\n")
cat("     • Survey metadata (7)\n")
cat("     • Cycling details (8)\n")
cat("     • Technology metadata (2)\n")
cat("     • Trip/Stop metadata (5)\n")
cat("     • Household metadata (2)\n")
cat("\n")

# -----------------------------------------------------------------------------
# COMBINE ALL REMOVALS
# -----------------------------------------------------------------------------

vars_to_remove <- c(
  vehicle_proxies_present,
  low_value_vars,
  administrative_vars_present
)

# Remove duplicates
vars_to_remove <- unique(vars_to_remove)

cat("SUMMARY:\n")
cat(sprintf("Total variables to remove: %d\n", length(vars_to_remove)))
cat(sprintf("Starting columns: %d\n", ncol(df_original)))
cat(sprintf("Ending columns: %d\n", ncol(df_original) - length(vars_to_remove) + 2))  # +2 for new indicators
cat("\n")

# ==============================================================================
# 3. REMOVE VARIABLES
# ==============================================================================

cat("Step 3: Removing variables...\n")

df_clean <- df %>%
  select(-any_of(vars_to_remove))

cat(sprintf("✓ Removed %d variables\n", ncol(df) - ncol(df_clean)))
cat(sprintf("✓ Retained %d variables (including target)\n\n", ncol(df_clean)))

# ==============================================================================
# 4. VERIFY KEPT VARIABLES
# ==============================================================================

cat("Step 4: Verifying policy-relevant features retained...\n\n")

# Check key categories are present
demographic_vars <- grep("^PERS_(AGEGROUP|SEX|PERSINC|RELATIONSHIP|MAINACT|ANZSCO|ANZSIC)",
                        names(df_clean), value = TRUE)
trip_vars <- grep("^(STOP|TRIP)_(NETWORK_DIST|TRAVTIME|DURATION|ORIGPURP|DESTPURP|SPEED)",
                 names(df_clean), value = TRUE)
household_vars <- grep("^HH_(HHSIZE|HHINC|HHSTRUCTURE|DWELLTYPE|OWNDWELL|CARS|TOTALVEHS|TOTALBIKES)",
                      names(df_clean), value = TRUE)
engineered_vars <- c("VEH_PER_PERSON", "VEH_PER_ADULT", "TRIP_SPEED", "TIME_PERIOD",
                     "IS_WEEKEND", "DIST_CATEGORY", "HAS_LICENSE_AND_CAR")
engineered_vars_present <- intersect(engineered_vars, names(df_clean))

cat("✓ Policy-Relevant Features Retained:\n")
cat(sprintf("   • Demographics: %d variables\n", length(demographic_vars)))
cat(sprintf("   • Trip characteristics: %d variables\n", length(trip_vars)))
cat(sprintf("   • Household availability: %d variables\n", length(household_vars)))
cat(sprintf("   • Engineered features: %d variables\n", length(engineered_vars_present)))
cat("\n")

# Verify no vehicle proxies remain
remaining_vehicle_vars <- grep("VEH_(VEHTYPE|VEHYEAR|PETROL|DIESEL|GAS|ELECTRIC|HYBRID|RUNNINGCOST)",
                              names(df_clean), value = TRUE)
remaining_stop_veh_vars <- grep("STOP_VEH", names(df_clean), value = TRUE)

if (length(remaining_vehicle_vars) > 0 || length(remaining_stop_veh_vars) > 0) {
  cat("⚠ WARNING: Some vehicle proxies may still be present:\n")
  print(c(remaining_vehicle_vars, remaining_stop_veh_vars))
  cat("\n")
} else {
  cat("✓ Confirmed: All trip-specific vehicle proxies removed\n\n")
}

# ==============================================================================
# 5. SAVE POLICY-FOCUSED DATASET
# ==============================================================================

cat("Step 5: Saving policy-focused dataset...\n")

output_file <- "data/processed/final_cleaned_policy_focused.csv"
write.csv(df_clean, output_file, row.names = FALSE)

cat(sprintf("✓ Saved: %s\n", output_file))
cat(sprintf("   - %d rows\n", nrow(df_clean)))
cat(sprintf("   - %d columns\n\n", ncol(df_clean)))

# ==============================================================================
# 6. GENERATE COMPARISON REPORT
# ==============================================================================

cat("Step 6: Generating comparison report...\n")

# Create results directory
if (!dir.exists("results/comparison")) {
  dir.create("results/comparison", recursive = TRUE)
}

# Generate summary report
sink("results/comparison/variable_removal_summary.txt")

cat("VARIABLE REMOVAL SUMMARY REPORT\n")
cat("================================================================================\n")
cat(sprintf("Generated: %s\n", Sys.time()))
cat(sprintf("Purpose: Address Week 7 feedback by removing trivial predictors\n\n"))

cat("DATASET COMPARISON:\n")
cat("================================================================================\n")
cat(sprintf("Original dataset:       %d rows × %d columns\n",
            nrow(df_original), ncol(df_original)))
cat(sprintf("Policy-focused dataset: %d rows × %d columns\n",
            nrow(df_clean), ncol(df_clean)))
cat(sprintf("Variables removed:      %d\n", ncol(df_original) - ncol(df_clean)))
cat(sprintf("Reduction:              %.1f%%\n\n",
            (ncol(df_original) - ncol(df_clean)) / ncol(df_original) * 100))

cat("VARIABLES REMOVED BY CATEGORY:\n")
cat("================================================================================\n\n")

cat("1. TRIP-SPECIFIC VEHICLE PROXIES (Data Leakage)\n")
cat(sprintf("   Count: %d\n", length(vehicle_proxies_present)))
cat("   Reason: Only populated when mode = vehicle → 100% prediction accuracy\n")
cat("   Impact: Prevents trivial predictions, forces model to learn behavior\n")
cat("   Variables:\n")
for (var in vehicle_proxies_present) {
  cat(sprintf("     • %s\n", var))
}
cat("\n")

cat("2. LOW-VALUE VARIABLES (94% same value)\n")
cat(sprintf("   Count: %d\n", length(low_value_vars)))
cat("   Reason: Extremely low variance, minimal predictive power\n")
cat("   Impact: Reduces noise, replaced with binary indicators\n")
cat("   Variables:\n")
for (var in low_value_vars) {
  cat(sprintf("     • %s → %s_missing (binary)\n", var, var))
}
cat("\n")

cat("3. ADMINISTRATIVE/REDUNDANT VARIABLES\n")
cat(sprintf("   Count: %d\n", length(administrative_vars_present)))
cat("   Reason: Survey metadata, overly specific, or redundant information\n")
cat("   Impact: Simplifies model, focuses on policy-relevant features\n")
cat("   Subcategories:\n")
cat("     • Survey metadata: Who filled, when, how\n")
cat("     • Cycling details: Too granular for general mode choice\n")
cat("     • Technology: Phone type not policy-relevant\n")
cat("     • Redundant trip/stop info: Captured by other variables\n\n")

cat("POLICY-RELEVANT FEATURES RETAINED:\n")
cat("================================================================================\n\n")

cat("1. DEMOGRAPHIC FEATURES\n")
cat(sprintf("   Count: %d\n", length(demographic_vars)))
cat("   Purpose: Understand WHO chooses each mode\n")
cat("   Examples: Age, sex, income, occupation, household structure\n\n")

cat("2. TRIP CHARACTERISTICS\n")
cat(sprintf("   Count: %d\n", length(trip_vars)))
cat("   Purpose: Understand WHEN and WHY mode is chosen\n")
cat("   Examples: Distance, duration, purpose, origin/destination\n\n")

cat("3. HOUSEHOLD AVAILABILITY\n")
cat(sprintf("   Count: %d\n", length(household_vars)))
cat("   Purpose: Measure CAPABILITY to use modes (not actual usage)\n")
cat("   Examples: Cars owned, bikes owned, household size\n")
cat("   Note: Ownership ≠ usage → still requires behavioral prediction\n\n")

cat("4. ENGINEERED FEATURES\n")
cat(sprintf("   Count: %d\n", length(engineered_vars_present)))
cat("   Purpose: Capture non-linear relationships and interactions\n")
cat("   Examples:\n")
for (var in engineered_vars_present) {
  cat(sprintf("     • %s\n", var))
}
cat("\n")

cat("ADDRESSING WEEK 7 FEEDBACK:\n")
cat("================================================================================\n\n")

cat("Feedback 1: 'Remove variables that are direct proxies for the outcome'\n")
cat("  ✓ ADDRESSED: Removed all 15 trip-specific vehicle attributes\n")
cat("  • VEH_VEHTYPE, STOP_VEHOCCUP, etc. → Gone\n")
cat("  • Kept household vehicle availability (HH_CARS, VEH_PER_PERSON)\n")
cat("  • Ownership ≠ usage → model must learn behavioral patterns\n\n")

cat("Feedback 2: 'Explore demographic and trip-related features'\n")
cat("  ✓ ADDRESSED: Retained comprehensive demographic & trip variables\n")
cat(sprintf("  • %d demographic features maintained\n", length(demographic_vars)))
cat(sprintf("  • %d trip characteristic features maintained\n", length(trip_vars)))
cat("  • These will now dominate feature importance rankings\n\n")

cat("Feedback 3: 'Focus on explaining relative influence of socio-economic variables'\n")
cat("  ✓ ADDRESSED: Removed trivial predictors that overshadowed true influences\n")
cat("  • Without vehicle proxies, model learns from demographics & context\n")
cat("  • Can now interpret: 'High income + long distance → car choice'\n")
cat("  • Can identify: 'Young adults + short trips → walking/cycling'\n\n")

cat("Feedback 4: 'Highlight variables with behavioral/policy implications'\n")
cat("  ✓ ADDRESSED: Focused dataset on policy-actionable features\n")
cat("  • Demographics: Target interventions to specific groups\n")
cat("  • Trip characteristics: Identify mode-switching thresholds\n")
cat("  • Household availability: Understand infrastructure needs\n")
cat("  • Temporal patterns: Optimize service schedules\n\n")

cat("EXPECTED MODEL IMPACT:\n")
cat("================================================================================\n\n")

cat("Accuracy:\n")
cat("  • Expected to DROP by 5-15%\n")
cat("  • This is GOOD - we removed trivial predictors\n")
cat("  • Remaining accuracy reflects true behavioral prediction\n\n")

cat("Feature Importance:\n")
cat("  • Demographics/trip features will rise to top 10\n")
cat("  • Can now interpret which factors truly drive mode choice\n")
cat("  • Policy-relevant insights become visible\n\n")

cat("Interpretability:\n")
cat("  • Can explain predictions using socio-economic factors\n")
cat("  • Findings generalizable to policy contexts\n")
cat("  • Model tells behavioral story, not trivial vehicle detection\n\n")

cat("NEXT STEPS:\n")
cat("================================================================================\n\n")

cat("1. Train XGBoost on policy-focused dataset\n")
cat("2. Compare feature importance: original vs. policy-focused\n")
cat("3. Generate SHAP plots to interpret demographic influences\n")
cat("4. Extract policy insights (e.g., distance thresholds, demographic patterns)\n")
cat("5. Document findings in report with focus on behavioral interpretation\n\n")

cat("FILES GENERATED:\n")
cat("================================================================================\n")
cat("• data/processed/final_cleaned_policy_focused.csv\n")
cat("• results/comparison/variable_removal_summary.txt (this file)\n\n")

sink()

cat(sprintf("✓ Saved: results/comparison/variable_removal_summary.txt\n\n"))

# ==============================================================================
# 7. FINAL SUMMARY
# ==============================================================================

cat("==============================================================================\n")
cat("POLICY-FOCUSED DATASET CREATION COMPLETE\n")
cat("==============================================================================\n\n")

cat("✓ Removed 3 categories of variables:\n")
cat(sprintf("  • %d trip-specific vehicle proxies (data leakage)\n",
            length(vehicle_proxies_present)))
cat(sprintf("  • %d low-value variables (replaced with indicators)\n",
            length(low_value_vars)))
cat(sprintf("  • %d administrative/redundant variables\n",
            length(administrative_vars_present)))
cat(sprintf("  Total removed: %d variables\n\n", length(vars_to_remove)))

cat("✓ Retained policy-relevant features:\n")
cat(sprintf("  • %d demographic variables\n", length(demographic_vars)))
cat(sprintf("  • %d trip characteristic variables\n", length(trip_vars)))
cat(sprintf("  • %d household availability variables\n", length(household_vars)))
cat(sprintf("  • %d engineered features\n", length(engineered_vars_present)))
cat(sprintf("  Total retained: %d variables\n\n", ncol(df_clean)))

cat("Next: Run XGBoost model with policy-focused dataset\n")
cat("  source('src/models/xgboost_model.R')\n\n")

cat("For comparison analysis:\n")
cat("  source('scripts/compare_models.R')\n\n")
