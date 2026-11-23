import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --------------------------- Page config ---------------------------
st.set_page_config(layout="wide", page_title="HPV Dashboard")

# --------------------- Auto-detect project root & data ---------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

# --------------------------- Load observed dataset ---------------------------
hpv_file = os.path.join(data_dir, "master_dataset_ready.xlsx")
if os.path.exists(hpv_file):
    hpv = pd.read_excel(hpv_file)
    st.sidebar.success(f"Loaded dataset: {os.path.basename(hpv_file)}")
else:
    st.sidebar.error("❌ master_dataset_ready.xlsx not found in /data/")
    st.stop()

# --------------------------- Load forecast dataset ---------------------------
forecast_file = os.path.join(data_dir, "HPVc_ARIMA_forecasts_2030.xlsx")
if os.path.exists(forecast_file):
    forecast = pd.read_excel(forecast_file)
    if "year" in forecast.columns:
        forecast["year"] = forecast["year"].astype(int)
    st.sidebar.success(f"Loaded forecast: {os.path.basename(forecast_file)}")
else:
    st.sidebar.error("❌ HPVc forecast file not found in /data/")
    st.stop()

# --------------------------- Ensure & normalize columns ---------------------------
if "year" in hpv.columns:
    hpv["year"] = hpv["year"].astype(int)

# If user used 'cervical_mortality_rate' earlier, standardize
hpv = hpv.rename(columns={"cervical_mortality_rate": "cervical_cancer_death_rate"})

latest_year = int(hpv["year"].max())
cervical_available = "cervical_cancer_death_rate" in hpv.columns

# --------------------------- Colors & matplotlib style ---------------------------
colors = {
    "HPV1": "#1f77b4",
    "HPVc": "#2ca02c",
    "Dropout": "#ff7f0e",
    "Target": "#d62728",
    "CI": "#c7c7c7",
    "GDP": "#9467bd",
    "HealthExp": "#8c564b",
    "Education": "#17becf",
    "Internet": "#e377c2",
    "MortalityLow": "#fff5eb",
    "MortalityMid": "#fd8d3c",
    "MortalityHigh": "#800026"
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--"
})

# --------------------------- Ribbon helper ---------------------------
ribbon_path = os.path.join(project_root, "assets", "cervical_ribbon.png")


def page_heading(title: str):
    st.subheader(title)


# --------------------------- Global UI Theme Styling ---------------------------
st.markdown(
    """
    <style>
    :root {
    --hpv-blue: #0056b3;
    --hpv-blue-light: #e3f2fd;
    --hpv-blue-mid: #1976d2;
    --hpv-blue-dark: #003c82;
}

/* DASHBOARD BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        180deg,
        #8fd9d0 0%,   /* teal ribbon shade */
        #d2f6f1 45%,
        white 95%
    ) !important;
}

/* SPACE AT TOP */
.block-container {
    padding-top: 1rem;
}

/* MAIN TITLE */
.page-title {
    font-size: 80px;
    font-weight: 900;
    color: var(--hpv-blue-dark);
    text-align: center;
    margin: 0.5rem 0 0.1rem 0;
    letter-spacing: 0.02em;
}

/* RIBBON FILTER */
.header-ribbon img {
    filter: drop-shadow(0 0 10px rgba(0, 86, 179, 0.55));
}

/* TAB BAR WRAPPER */
div[role="tablist"] {
    justify-content: center;
    border-bottom: 1px solid #d0e2ff;
    margin-bottom: 0.5rem;
}

/* TAB BUTTON STYLE */
div[role="tab"] {
    font-size: 36px;
    font-weight: 700;
    padding: 14px 28px;
    margin: 0 6px;
    border-radius: 40px;
    
    background-color: #dbe9ff;      /* light blue pill */
    color: #003c82;                  /* dark blue text */
    border: 2px solid #003c82;
    
    transition: all 0.25s ease-in-out;
}

/* TAB HOVER EFFECT */
div[role="tab"]:hover {
    background-color: #003c82;
    color: white;
    box-shadow: 0 0 18px rgba(0, 86, 179, 0.65);
    transform: scale(1.03);
}

/* ACTIVE SELECTED TAB */
div[role="tab"][aria-selected="true"] {
    background-color: #0056b3;
    color: white;
    border-color: #003c82;
    box-shadow: 0 0 25px rgba(0, 86, 179, 0.85);
}

/* BODY & MARKDOWN TEXT SIZE */
p, span {
    font-size: 24px !important;
}

/* SUBHEADINGS */
h1, h2, h3, h4 {
    font-size: 36px !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Header + Tabs ---------------------------
col_l, col_r = st.columns([0.86, 0.14])
with col_l:
    st.markdown('<div class="page-title">HPV ForestX: Country Level Forecasting and Insights</div>',
                unsafe_allow_html=True)

with col_r:
    if os.path.exists(ribbon_path):
        st.markdown('<div class="header-ribbon">', unsafe_allow_html=True)
        st.image(ribbon_path, width=500)
        st.markdown('</div>', unsafe_allow_html=True)

tabs = st.tabs(
    [
        "Forecasting (ARIMA)",
        "Overview & Country Profiles",
        "Correlates",
        "Cervical Cancer Impact",
        "Regional & Income",
        "Methods & Data Sources",
        "Why? (Explainability)",  # [XAI] added by xai_module
    ]
)
# [XAI] tab_xai appended; original 6-way unpack otherwise unchanged
tab_forecast, tab_overview, tab_corr, tab_impact, tab_region, tab_methods, tab_xai = tabs

# --------------------------- Tab 1: Forecasting (ARIMA) ---------------------------
with tab_forecast:
    page_heading("Country HPV Forecasting (ARIMA)")

    if "country" in forecast.columns:
        country = st.selectbox("Select Country",
                               sorted(forecast["country"].dropna().unique()),
                               key="country_forecast")
    else:
        st.warning("No 'country' column in forecast dataset.")
        country = None

    if country:
        fc_data = forecast[forecast["country"] == country].sort_values("year")

        if not fc_data.empty:
            latest_val = None
            if "observed_HPVc" in fc_data.columns:
                latest_val = fc_data[fc_data["year"] == fc_data["year"].max()]["observed_HPVc"].mean()
            else:
                tmp = hpv[(hpv["country"] == country) & (hpv["year"] == hpv["year"].max())]
                if not tmp.empty and "HPVc_Females" in tmp.columns:
                    latest_val = tmp["HPVc_Females"].mean()

            fc_2030 = fc_data[fc_data["year"] == 2030]["forecast_HPVc"].mean() \
                if "forecast_HPVc" in fc_data.columns else None

            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Observed HPVc",
                        f"{latest_val:.1f}%" if latest_val is not None and not pd.isna(latest_val) else "N/A")
            col2.metric("Forecast HPVc (2030)",
                        f"{fc_2030:.1f}%" if fc_2030 is not None and not pd.isna(fc_2030) else "N/A")
            if fc_2030 is not None and not pd.isna(fc_2030):
                col3.metric("Gap to WHO 90% Target", f"{(90 - fc_2030):.1f}%")
            else:
                col3.metric("Gap to WHO 90% Target", "N/A")

            y_cols = []
            if "observed_HPVc" in fc_data.columns:
                y_cols.append("observed_HPVc")
            if "forecast_HPVc" in fc_data.columns:
                y_cols.append("forecast_HPVc")

            if y_cols:
                fig = px.line(
                    fc_data,
                    x="year",
                    y=y_cols,
                    title=f"{country}: Observed vs Forecasted HPVc (ARIMA to 2030)",
                    labels={"value": "HPVc (%)", "year": "Year", "variable": "Series"},
                )
                fig.add_hline(
                    y=90,
                    line_dash="dash",
                    line_color=colors["Target"],
                    annotation_text="WHO 90% Target",
                )
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Forecast dataset for this country lacks 'observed_HPVc' and 'forecast_HPVc' columns.")

            st.caption(
                "📌 Forecasts are based on **ARIMA**, chosen because it had the lowest RMSE (26.56) "
                "and MAE (41.96) compared to Naïve, Holt-Winters, and LSTM models."
            )
        else:
            st.warning(f"No forecast data available for {country}")
    st.info("ℹ️ HPV1 = Human papillomavirus vaccine first dose | HPVc = Human papillomavirus vaccine complete dose")

# --------------------------- Tab 2: Overview & Country Profiles ---------------------------
with tab_overview:
    page_heading("Global HPV Vaccination Overview and Country Profiles")

    st.subheader("Global Overview")
    if all(c in hpv.columns for c in ["HPV1_Females", "HPVc_Females"]):
        global_trend = hpv.groupby("year")[["HPV1_Females", "HPVc_Females"]].mean().reset_index()
        fig = px.line(
            global_trend,
            x="year",
            y=["HPV1_Females", "HPVc_Females"],
            color_discrete_map={"HPV1_Females": colors["HPV1"], "HPVc_Females": colors["HPVc"]},
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("HPV1_Females or HPVc_Females columns not found in observed dataset.")

    st.subheader("Country Explorer")
    if "country" in hpv.columns:
        country_overview = st.selectbox(
            "Select Country", sorted(hpv["country"].dropna().unique()), key="country_overview"
        )
        c_data = hpv[hpv["country"] == country_overview].sort_values("year")
        if not c_data.empty and all(c in c_data.columns for c in ["HPV1_Females", "HPVc_Females"]):
            fig2 = px.line(
                c_data,
                x="year",
                y=["HPV1_Females", "HPVc_Females"],
                color_discrete_map={"HPV1_Females": colors["HPV1"], "HPVc_Females": colors["HPVc"]},
            )
            fig2.update_yaxes(range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

            c_data = c_data.copy()
            c_data["dropout"] = c_data["HPV1_Females"] - c_data["HPVc_Females"]
            fig3 = px.line(c_data, x="year", y="dropout", color_discrete_sequence=[colors["Dropout"]])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Insufficient HPV series for this country in observed dataset.")
    else:
        st.warning("No 'country' column in observed dataset.")

    st.info("ℹ️ HPV1 = Human papillomavirus vaccine first dose | HPVc = Human papillomavirus vaccine complete dose")

# --------------------------- Tab 3: Correlates ---------------------------
with tab_corr:
    page_heading("Correlation with Socioeconomic and Health Factors")

    corr_cols = [
        "HPVc_Females",
        "health_exp_per_capita",
        "female_secondary_edu_pct",
        "internet_users_pct",
        "fertility_rate",
    ]
    existing_cols = [c for c in corr_cols if c in hpv.columns]
    if len(existing_cols) >= 2:
        corr_data = hpv[existing_cols].dropna()
        corr = corr_data.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Scatter Plots")
        factor_options = [
            c
            for c in [
                "health_exp_per_capita",
                "female_secondary_edu_pct",
                "internet_users_pct",
                "fertility_rate",
            ]
            if c in hpv.columns
        ]
        if factor_options:
            factor = st.selectbox("Select Factor", factor_options)
            fig2 = px.scatter(
                hpv,
                x=factor,
                y="HPVc_Females" if "HPVc_Females" in hpv.columns else None,
                color="Income group" if "Income group" in hpv.columns else None,
                trendline="ols",
                title=f"HPVc vs {factor}",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No factor columns available for scatter plots.")
    else:
        st.warning("Not enough socioeconomic columns available for correlation analysis.")

    st.info("ℹ️ HPV1 = Human papillomavirus vaccine first dose | HPVc = Human papillomavirus vaccine complete dose")

# --------------------------- Tab 4: Cervical Cancer Impact ---------------------------
with tab_impact:
    page_heading("Cervical Cancer Impact")

    if not cervical_available:
        st.warning("Cervical cancer mortality column not found in dataset.")
    else:
        merged = hpv.dropna(subset=["cervical_cancer_death_rate"])
        if not merged.empty:
            fig = px.scatter(
                merged,
                x="HPVc_Females" if "HPVc_Females" in merged.columns else None,
                y="cervical_cancer_death_rate",
                color="Income group" if "Income group" in merged.columns else None,
                hover_name="country" if "country" in merged.columns else None,
                trendline="ols",
            )
            st.plotly_chart(fig, use_container_width=True)

            latest_cervical = merged[merged["year"] == latest_year]
            if not latest_cervical.empty:
                fig2 = px.choropleth(
                    latest_cervical,
                    locations="iso3c" if "iso3c" in latest_cervical.columns else None,
                    color="cervical_cancer_death_rate",
                    hover_name="country" if "country" in latest_cervical.columns else None,
                    color_continuous_scale=[
                        colors["MortalityLow"],
                        colors["MortalityMid"],
                        colors["MortalityHigh"],
                    ],
                    title=f"Cervical Cancer Mortality in {latest_year}",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No cervical mortality data for the latest year to map.")
        else:
            st.warning("No cervical cancer mortality records available after dropping NA.")

    st.info("ℹ️ HPV1 = Human papillomavirus vaccine first dose | HPVc = Human papillomavirus vaccine complete dose")

# --------------------------- Tab 5: Regional & Income ---------------------------
with tab_region:
    page_heading("Regional and Income Group Comparisons")

    if "Income group" in hpv.columns and "HPVc_Females" in hpv.columns:
        income_trend = (
            hpv.groupby(["year", "Income group"])[["HPVc_Females"]].mean().reset_index()
        )
        fig = px.line(income_trend, x="year", y="HPVc_Females", color="Income group")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Income group or HPVc_Females column missing for income group comparisons.")

    if "region" in hpv.columns and "HPVc_Females" in hpv.columns:
        region_trend = hpv.groupby(["year", "region"])[["HPVc_Females"]].mean().reset_index()
        fig2 = px.line(region_trend, x="year", y="HPVc_Females", color="region")
        fig2.update_yaxes(range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Dropout Heatmap by Region")
        hpv_copy = hpv.copy()
        if all(c in hpv_copy.columns for c in ["HPV1_Females", "HPVc_Females"]):
            hpv_copy["dropout"] = hpv_copy["HPV1_Females"] - hpv_copy["HPVc_Females"]
            heatmap = hpv_copy.pivot_table(values="dropout", index="region", columns="year")
            fig3, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(heatmap, cmap="RdYlGn_r", ax=ax)
            st.pyplot(fig3)
        else:
            st.info("HPV1_Females or HPVc_Females missing; cannot compute dropout heatmap.")
    else:
        st.info("Region column or HPVc_Females missing; skipping regional heatmap.")

    st.info("ℹ️ HPV1 = Human papillomavirus vaccine first dose | HPVc = Human papillomavirus vaccine complete dose")

# --------------------------- Tab 6: Methods & Data Sources ---------------------------
with tab_methods:
    page_heading("Methods & Data Sources")

    st.markdown(
        """
    ### Datasets
    - WHO/UNICEF (HPV1 and HPVc coverage, WUENIC + administrative data).
    - World Bank (World Development Indicators: GDP per capita, health spending, education, fertility, internet, urbanization, etc.).
    - Cervical cancer mortality (WHO Mortality Database / GLOBOCAN).
    
    ### Forecasting Models Compared
    We compared multiple time-series forecasting approaches for HPVc coverage:
    
    - **Naïve model:** Assumes last observed value continues.
    - **Holt-Winters:** Captures trend and seasonal effects in coverage.
    - **ARIMA:** Autoregressive integrated moving average.
    - **LSTM:** Deep learning recurrent neural network.
    
    ### Model Performance (RMSE / MAE)
    - Naïve: RMSE = 26.67 | MAE = 42.25
    - Holt-Winters: RMSE = 32.33 | MAE = 47.42
    - ARIMA: RMSE = 26.56 | MAE = 41.96
    - LSTM: RMSE = 27.51 | MAE = 43.88
    
    ### Why ARIMA Was Chosen
    - ARIMA had the **lowest RMSE (26.56) and MAE (41.96)**, indicating the best predictive accuracy.  
    - It balances simplicity with interpretability, unlike LSTM which requires more data and tuning.  
    - ARIMA effectively models country-level HPVc trajectories with relatively small datasets, making it more robust for global applications.
    """
    )

    st.info("ℹ️ HPV1 = Human papillomavirus vaccine first dose | HPVc = Human papillomavirus vaccine complete dose")

# --------------------------- Tab 7: Why? (Explainability) ---------------------------
# [XAI] Entire block added by xai_module. Delete this block to revert.
with tab_xai:
    import sys

    _xai_dir = os.path.join(project_root, "xai_module")
    if _xai_dir not in sys.path:
        sys.path.insert(0, _xai_dir)
    try:
        import panel as xai_panel

        xai_panel.render(st)
    except Exception as _exc:  # never let the add-on break the existing app
        st.error(f"Explainability module unavailable: {_exc}")
# [XAI] end
