import pandas as pd
import os

# --------------------------------------------------------
# Auto-detect project root & define data folder
# --------------------------------------------------------
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Project root = one folder up from "analysis"
project_root = os.path.dirname(script_dir)

# Data folder inside project root
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# --------------------------------------------------------
# Load dataset
# --------------------------------------------------------
raw_file = os.path.join(data_dir, "master_dataset_cleaned.xlsx")

df = pd.read_excel(raw_file)

# --------------------------------------------------------
# Data cleaning
# --------------------------------------------------------
# Convert year
df["year"] = df["year"].astype(int)

# Interpolate selected WDI indicators
wdi_numeric_cols = [
    "health_exp_per_capita",
    "female_secondary_edu_pct",
    "urban_pop_pct",
    "fertility_rate",
    "internet_users_pct",
    "school_enrollment_secondary_pct"
]

for col in wdi_numeric_cols:
    if col in df.columns:
        df[col] = df.groupby("iso3c")[col].transform(
            lambda x: x.interpolate(limit_direction="both")
        )

# --------------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------------
cleaned_file = os.path.join(data_dir, "master_dataset_ready.xlsx")
df.to_excel(cleaned_file, index=False)

print(f"✅ Cleaned dataset saved at: {cleaned_file}")
