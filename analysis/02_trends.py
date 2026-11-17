import pandas as pd
import matplotlib.pyplot as plt
import os

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

# --------------------------------------------------------
# Global trends plot
# --------------------------------------------------------
global_trends = df.groupby("year")[["HPV1_Females", "HPVc_Females"]].mean()

plt.figure(figsize=(10, 6))
plt.plot(global_trends.index, global_trends["HPV1_Females"], label="HPV1 Females (≥1 dose)", marker="o")
plt.plot(global_trends.index, global_trends["HPVc_Females"], label="HPVc Females (Complete)", marker="o")

plt.title("Global HPV Vaccination Coverage (Females)")
plt.xlabel("Year")
plt.ylabel("Coverage (%)")
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# --------------------------------------------------------
# Save plot
# --------------------------------------------------------
plot_file = os.path.join(data_dir, "global_trends.png")
plt.savefig(plot_file, dpi=300)
plt.close()

print(f"✅ Global trends plot saved at: {plot_file}")
