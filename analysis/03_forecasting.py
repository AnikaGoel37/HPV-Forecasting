import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Auto-detect project root & define data folder
# --------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

# --------------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------------
cleaned_file = os.path.join(data_dir, "master_dataset_ready.xlsx")
df = pd.read_excel(cleaned_file)
df["year"] = df["year"].astype(int)

# --------------------------------------------------------
# Forecasting
# --------------------------------------------------------
countries = df["iso3c"].dropna().unique().tolist()
results = []

for c in countries:
    sub = df[df["iso3c"] == c].sort_values("year")
    series = sub.set_index("year")["HPVc_Females"].dropna()
    
    if series.empty:
        continue
    
    last_year = int(series.index.max())
    target_end = 2030
    steps = target_end - last_year
    forecast_index = list(range(last_year+1, target_end+1)) if steps > 0 else []
    
    try:
        if len(series) >= 4 and steps > 0:
            model = ARIMA(series, order=(1,1,1)).fit()
            fc_res = model.get_forecast(steps=steps)
            fc_mean = fc_res.predicted_mean
            fc_ci = fc_res.conf_int(alpha=0.05)

            # Observed values
            for y, val in series.items():
                results.append([c, sub["country"].iloc[0], int(y), float(val), np.nan, np.nan, np.nan, "ARIMA"])
            
            # Forecasted values
            for i, y in enumerate(forecast_index):
                results.append([c, sub["country"].iloc[0], int(y), np.nan, float(fc_mean.iloc[i]),
                                float(fc_ci.iloc[i,0]), float(fc_ci.iloc[i,1]), "ARIMA"])
        else:
            last_val = float(series.iloc[-1])
            # Observed
            for y, val in series.items():
                results.append([c, sub["country"].iloc[0], int(y), float(val), np.nan, np.nan, np.nan, "Naive"])
            # Forecasts (Naïve)
            for y in forecast_index:
                results.append([c, sub["country"].iloc[0], int(y), np.nan, last_val, np.nan, np.nan, "Naive"])
    except:
        continue

# --------------------------------------------------------
# Save forecast dataset
# --------------------------------------------------------
forecast_df = pd.DataFrame(results, columns=[
    "iso3c","country","year","observed_HPVc","forecast_HPVc","lower_ci","upper_ci","model_used"
])

forecast_file = os.path.join(data_dir, "HPVc_ARIMA_forecasts_2030.xlsx")
forecast_df.to_excel(forecast_file, index=False)

print(f"✅ Forecasts saved at: {forecast_file}")
