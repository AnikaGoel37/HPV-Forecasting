"""
Renders the "Why? (Explainability)" tab inside HPV ForestX.

Imported by dashboard.py via a single call: render(st). Everything loads from
precomputed artifacts, so the tab adds no training cost at app startup. If the
artifacts are missing the tab shows an instruction instead of breaking the app.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")

BLUE, ORANGE, RED = "#0056b3", "#ff7f0e", "#c1121f"
FEATURES = ["health_exp_per_capita", "female_secondary_edu_pct", "fertility_rate",
            "internet_users_pct", "urban_pop_pct", "inc", "HPVc_Females"]
PRETTY = {
    "health_exp_per_capita": "Health expenditure per capita (USD)",
    "female_secondary_edu_pct": "Female secondary education (%)",
    "fertility_rate": "Fertility rate (births/woman)",
    "internet_users_pct": "Internet users (%)",
    "urban_pop_pct": "Urban population (%)",
    "inc": "World Bank income group",
    "HPVc_Females": "HPVc vaccination coverage (%)",
}
LEVERS = [f for f in FEATURES if f != "inc"]


def _ok():
    return os.path.exists(os.path.join(ARTIFACTS, "country_attribution.csv"))


def _load():
    at = pd.read_csv(os.path.join(ARTIFACTS, "country_attribution.csv"))
    imp = pd.read_csv(os.path.join(ARTIFACTS, "mortality_importance.csv"))
    with open(os.path.join(ARTIFACTS, "metrics.json")) as fh:
        met = json.load(fh)
    return at, imp, met


def render(st):
    st.subheader("Why? Explaining what drives cervical cancer burden")

    if not _ok():
        st.warning("Explainability artifacts not found. Run:\n\n"
                   "`python xai_module/build_xai.py`")
        return

    at, imp, met = _load()
    A, B = met["model_a_mortality"], met["model_b_coverage_null"]
    base = A["shap_base_value"]

    st.caption(
        f"Random-forest attribution model for cervical cancer mortality — "
        f"grouped cross-validation by country (no country appears in both train "
        f"and test), R² = {A['rf_r2_grouped']:.2f}, MAE = {A['rf_mae_grouped']:.2f} "
        f"deaths per 100,000, across {A['n_countries']} countries. "
        f"Contributions are SHAP values relative to the global baseline of "
        f"{base:.2f}. **Associational, not causal.**"
    )

    # ---------------- The headline finding ----------------
    st.info(
        f"**Key finding.** These socioeconomic determinants explain cervical cancer "
        f"*burden* (R² = {A['rf_r2_grouped']:.2f}) but **not** vaccination *coverage* "
        f"(R² = {B['rf_r2_grouped']:.2f} — no better than chance). Coverage appears to be "
        f"driven by programmatic factors (whether a national programme exists, delivery "
        f"platform, supply, political commitment) rather than by a country's level of "
        f"development."
    )

    tab_country, tab_global, tab_method = st.tabs(
        ["Country attribution", "Global drivers", "Model validity"])

    # ---------------- Country attribution ----------------
    with tab_country:
        names = at.sort_values("country")["country"].tolist()
        choice = st.selectbox("Select country", names, key="xai_country")
        row = at[at["country"] == choice].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Observed mortality", f"{row['observed_mortality']:.1f}",
                  help="deaths per 100,000 women")
        c2.metric("Model prediction", f"{row['predicted_mortality']:.1f}",
                  f"{row['predicted_mortality'] - base:+.1f} vs baseline")
        c3.metric("Dominant risk driver", row["dominant_risk_driver"].split(" (")[0],
                  f"{row['dominant_risk_shap']:+.2f}")

        s = pd.Series({PRETTY[f]: row[f"shap_{f}"] for f in FEATURES}).sort_values()
        fig = go.Figure(go.Bar(
            x=s.values, y=s.index, orientation="h",
            marker_color=[RED if v >= 0 else BLUE for v in s.values],
            text=[f"{v:+.2f}" for v in s.values], textposition="outside"))
        fig.update_layout(
            title=f"What moves {choice} away from the global baseline of {base:.2f}",
            xaxis_title="Contribution to predicted mortality (deaths per 100,000)",
            height=430, margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.add_vline(x=0, line_color="black", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Red raises predicted mortality, blue lowers it. {choice}'s largest "
            f"risk driver is **{row['dominant_risk_driver']}** "
            f"({row['dominant_risk_shap']:+.2f}); its strongest protective factor is "
            f"**{row['dominant_protective']}** ({row['dominant_protective_shap']:+.2f})."
        )

        st.markdown("#### Where each driver binds")
        mp = px.choropleth(
            at, locations="iso3c", color="dominant_risk_driver", hover_name="country",
            hover_data={"predicted_mortality": ":.1f", "iso3c": False},
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Dominant driver of cervical cancer mortality risk, by country")
        mp.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0),
                         legend_title="Dominant driver",
                         geo=dict(bgcolor="rgba(0,0,0,0)"),
                         paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(mp, use_container_width=True)
        st.caption(f"Coverage shown for the {len(at)} countries with reported "
                   "cervical cancer mortality data.")

    # ---------------- Global drivers + scenario ----------------
    with tab_global:
        d = imp.sort_values("mean_abs_shap")
        g = go.Figure(go.Bar(
            x=d["mean_abs_shap"], y=d["pretty"], orientation="h",
            marker_color=[RED if f == "HPVc_Females" else BLUE for f in d["feature"]],
            text=[f"{v:.2f}" for v in d["mean_abs_shap"]], textposition="outside"))
        g.update_layout(
            title="Average contribution of each determinant",
            xaxis_title="mean |SHAP| (deaths per 100,000)",
            height=400, margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g, use_container_width=True)

        hpvc_imp = float(imp[imp.feature == "HPVc_Females"]["mean_abs_shap"].iloc[0])
        st.warning(
            f"**HPVc coverage contributes only {hpvc_imp:.2f} deaths per 100,000** "
            f"once development level is adjusted for — 5th of 7 determinants. This is "
            f"biologically expected: HPV vaccination began around 2010, and vaccinated "
            f"cohorts have not yet reached the age of cervical cancer incidence. "
            f"Bivariate coverage–mortality associations are therefore confounded by "
            f"development and should not be read as vaccine impact."
        )

        st.markdown("#### Scenario explorer")
        st.caption("Model-implied response only. The determinants are correlated and "
                   "the model is not causal, so treat these as directional planning "
                   "estimates, not predicted effects of an intervention.")

        mpath = os.path.join(ARTIFACTS, "rf_model.joblib")
        if not os.path.exists(mpath):
            st.warning("Model file missing — rerun `python xai_module/build_xai.py`.")
        else:
            import joblib
            bundle = joblib.load(mpath)
            model, feats = bundle["model"], bundle["features"]

            names = at.sort_values("country")["country"].tolist()
            sc_country = st.selectbox("Country", names, key="xai_sc_country")
            r = at[at["country"] == sc_country].iloc[0]
            lever = st.selectbox("Lever", LEVERS, format_func=lambda f: PRETTY[f],
                                 key="xai_lever")

            cur = float(r[f"val_{lever}"])
            lo = float(at[f"val_{lever}"].quantile(0.01))
            hi = float(at[f"val_{lever}"].quantile(0.99))
            if hi <= lo:
                hi = lo + 1.0
            tgt = st.slider(PRETTY[lever], float(round(lo, 1)), float(round(hi, 1)),
                            float(round(min(max(cur, lo), hi), 1)), key="xai_slider")

            x = pd.DataFrame([{f: r[f"val_{f}"] for f in feats}])[feats]
            p0 = float(model.predict(x)[0])
            x2 = x.copy(); x2[lever] = tgt
            p1 = float(model.predict(x2)[0])

            e1, e2, e3 = st.columns(3)
            e1.metric("Current", f"{cur:,.1f}")
            e2.metric("Scenario", f"{tgt:,.1f}", f"{tgt - cur:+,.1f}")
            e3.metric("Model-implied mortality", f"{p1:.2f}", f"{p1 - p0:+.2f}")

    # ---------------- Model validity ----------------
    with tab_method:
        st.markdown("#### Why the model is validated by country, not by row")
        st.write(
            "The dataset is a country-year panel: each country contributes up to 15 "
            "rows. Randomly splitting rows puts the same country in both training and "
            "test folds, so the model can memorise country identity instead of "
            "learning from the determinants. Grouped cross-validation keeps every "
            "country wholly in one fold."
        )

        comp = pd.DataFrame({
            "Model": ["Cervical mortality (A)", "Cervical mortality (A)",
                      "HPVc coverage (B)", "HPVc coverage (B)"],
            "Validation": ["Random CV over country-years", "Grouped CV by country",
                           "Random CV over country-years", "Grouped CV by country"],
            "R²": [A["rf_r2_leaky_reference"], A["rf_r2_grouped"],
                   B["rf_r2_leaky_reference"], B["rf_r2_grouped"]],
        })
        f = px.bar(comp, x="Model", y="R²", color="Validation", barmode="group",
                   color_discrete_sequence=["#8c8c8c", BLUE], text_auto=".2f")
        f.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        f.add_hline(y=0, line_color="black", line_width=1)
        st.plotly_chart(f, use_container_width=True)

        st.caption(
            f"Random CV overstates R² by {A['leakage_inflation']:.2f} for mortality and "
            f"{B['leakage_inflation']:.2f} for coverage. Under honest validation, "
            f"coverage is not predictable from these determinants at all."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Random forest R²", f"{A['rf_r2_grouped']:.3f}")
        m2.metric("Linear model R²", f"{A['linear_r2_grouped']:.3f}",
                  help="Linear is competitive, indicating the determinant effects "
                       "are largely monotonic.")
        m3.metric("SHAP vs permutation ρ", f"{A['shap_perm_rank_corr']:.2f}",
                  help="Rank agreement between two independent importance measures.")

        st.caption(
            f"SHAP additivity error < {A['shap_additivity_max_error']:.0e}. "
            f"A linear model reaches R² = {A['linear_r2_grouped']:.2f} versus "
            f"{A['rf_r2_grouped']:.2f} for the random forest, so the dominant "
            f"relationships are close to monotonic; SHAP is used here for per-country "
            f"decomposition rather than to claim superior fit."
        )
