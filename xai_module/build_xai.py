"""
Builds the explainability layer for the HPV ForestX study.

Two models, deliberately:

  Model A (explained)  cervical cancer mortality ~ socioeconomic determinants
                       + HPVc coverage.  Grouped CV R2 ~ 0.25 -- real signal,
                       so SHAP attributions are meaningful.

  Model B (null)       HPVc coverage ~ socioeconomic determinants.
                       Honest CV R2 ~ 0 -- reported as a negative result, with
                       the leakage inflation quantified.

Validation is GroupKFold by country throughout. Random CV over the country-year
panel puts the same country in train and test, which inflates R2 by ~0.5 and is
the reason the pooled analysis looks stronger than it is.

Run diagnostics.py first to see the evidence. Writes only inside xai_module/.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)

import shap

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
DATA = os.path.join(r"C:\Users\goela\Downloads\hpv_dashboard", "data")
os.makedirs(ARTIFACTS, exist_ok=True)

INCOME = {"Low income": 0, "Lower middle income": 1,
          "Upper middle income": 2, "High income": 3}

DETERMINANTS = ["health_exp_per_capita", "female_secondary_edu_pct",
                "fertility_rate", "internet_users_pct", "urban_pop_pct", "inc"]
MORTALITY_FEATURES = DETERMINANTS + ["HPVc_Females"]
TARGET_A = "cervical_mortality_rate"
TARGET_B = "HPVc_Females"

PRETTY = {
    "health_exp_per_capita": "Health expenditure per capita (USD)",
    "female_secondary_edu_pct": "Female secondary education (%)",
    "fertility_rate": "Fertility rate (births/woman)",
    "internet_users_pct": "Internet users (%)",
    "urban_pop_pct": "Urban population (%)",
    "inc": "World Bank income group",
    "HPVc_Females": "HPVc vaccination coverage (%)",
}


def log(m):
    print(f"[xai] {m}", flush=True)


def load():
    d = pd.read_excel(os.path.join(DATA, "master_dataset_ready.xlsx"))
    d["year"] = d["year"].astype(int)
    d["inc"] = d["Income group"].map(INCOME)
    return d


def impute(df, cols):
    return df[cols].apply(lambda c: c.fillna(c.median()))


# ============================ Model A: mortality ============================
def build_mortality_model(d):
    log("--- Model A: cervical cancer mortality ---")
    m = d[d[TARGET_A].notna()].copy()
    X = impute(m, MORTALITY_FEATURES)
    y = m[TARGET_A]
    groups = m["iso3c"]
    log(f"{len(m)} country-years, {m.iso3c.nunique()} countries")

    rf = RandomForestRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=-1)

    # Honest: no country spans train and test.
    oof = cross_val_predict(rf, X, y, cv=GroupKFold(5), groups=groups)
    honest_r2 = r2_score(y, oof)
    honest_mae = mean_absolute_error(y, oof)

    # Leaky reference, to quantify what pooling would have claimed.
    leaky = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=42))
    leaky_r2 = r2_score(y, leaky)

    log(f"grouped CV  R2={honest_r2:+.3f}  MAE={honest_mae:.2f}")
    log(f"leaky CV    R2={leaky_r2:+.3f}   (inflation {leaky_r2 - honest_r2:+.3f})")

    # Linear baseline with standardized coefficients, same folds.
    mu, sd = X.mean(), X.std().replace(0, 1)
    Xs = (X - mu) / sd
    lin_oof = cross_val_predict(LinearRegression(), Xs, y, cv=GroupKFold(5), groups=groups)
    lin_r2 = r2_score(y, lin_oof)
    std_betas = pd.Series(LinearRegression().fit(Xs, y).coef_, index=MORTALITY_FEATURES)
    log(f"linear      R2={lin_r2:+.3f}")

    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X)
    base = float(np.array(explainer.expected_value).ravel()[0])
    additivity = float(np.abs(base + sv.sum(axis=1) - rf.predict(X)).max())
    log(f"SHAP {sv.shape}  base={base:.2f}  additivity_err={additivity:.1e}")

    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({
        "feature": MORTALITY_FEATURES,
        "pretty": [PRETTY[f] for f in MORTALITY_FEATURES],
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "perm_importance": perm.importances_mean,
        "perm_std": perm.importances_std,
        "std_beta": std_betas.reindex(MORTALITY_FEATURES).values,
    }).sort_values("mean_abs_shap", ascending=False)

    rank_corr = float(importance.mean_abs_shap.corr(
        importance.perm_importance, method="spearman"))
    log(f"SHAP vs permutation Spearman rho = {rank_corr:.3f}")

    metrics = {
        "model": "cervical_mortality_rate ~ socioeconomic determinants + HPVc",
        "n_rows": int(len(m)), "n_countries": int(m.iso3c.nunique()),
        "cv": "GroupKFold(5) by country",
        "rf_r2_grouped": honest_r2, "rf_mae_grouped": honest_mae,
        "rf_rmse_grouped": float(np.sqrt(mean_squared_error(y, oof))),
        "rf_r2_leaky_reference": leaky_r2,
        "leakage_inflation": leaky_r2 - honest_r2,
        "linear_r2_grouped": lin_r2,
        "shap_base_value": base,
        "shap_additivity_max_error": additivity,
        "shap_perm_rank_corr": rank_corr,
    }

    shap_df = pd.DataFrame(sv, columns=MORTALITY_FEATURES, index=X.index)
    return rf, m, X, shap_df, importance, metrics


def country_attribution(m, X, shap_df, rf, base):
    """Latest year per country, with its dominant driver of mortality risk."""
    idx = m.groupby("iso3c")["year"].idxmax()
    rows, sv = m.loc[idx], shap_df.loc[idx]

    out = pd.DataFrame({
        "iso3c": rows.iso3c.values, "country": rows.country.values,
        "year": rows.year.values,
        "observed_mortality": rows[TARGET_A].values,
        "predicted_mortality": rf.predict(X.loc[idx]),
        "Region": rows.Region.values, "Income group": rows["Income group"].values,
    })
    for f in MORTALITY_FEATURES:
        out[f"shap_{f}"] = sv[f].values
        out[f"val_{f}"] = X.loc[idx, f].values

    # Positive SHAP = raises mortality = the binding risk driver.
    out["dominant_risk_driver"] = [PRETTY[c] for c in sv[MORTALITY_FEATURES].idxmax(axis=1)]
    out["dominant_risk_shap"] = sv[MORTALITY_FEATURES].max(axis=1).values
    out["dominant_protective"] = [PRETTY[c] for c in sv[MORTALITY_FEATURES].idxmin(axis=1)]
    out["dominant_protective_shap"] = sv[MORTALITY_FEATURES].min(axis=1).values
    out["base_value"] = base
    log(f"country attributions: {len(out)} countries")
    return out


# ============================ Model B: the null ============================
def build_coverage_null(d):
    log("--- Model B: HPVc coverage (null result) ---")
    v = d[d[TARGET_B].notna()].copy()
    rf = RandomForestRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=-1)

    X, y, g = impute(v, DETERMINANTS), v[TARGET_B], v.iso3c
    grouped = r2_score(y, cross_val_predict(rf, X, y, cv=GroupKFold(5), groups=g))
    leaky = r2_score(y, cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=42)))

    cm = v.groupby("iso3c").agg({**{f: "mean" for f in DETERMINANTS},
                                 TARGET_B: "mean"}).reset_index()
    Xc, yc = impute(cm, DETERMINANTS), cm[TARGET_B]
    country_r2 = r2_score(yc, cross_val_predict(
        rf, Xc, yc, cv=KFold(5, shuffle=True, random_state=42)))

    latest = v.sort_values("year").groupby("iso3c").tail(1)
    aucs = {}
    for thr in (30, 50, 70):
        yb = (latest[TARGET_B] >= thr).astype(int)
        p = cross_val_predict(
            RandomForestClassifier(500, min_samples_leaf=2, random_state=42, n_jobs=-1),
            impute(latest, DETERMINANTS), yb,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            method="predict_proba")[:, 1]
        aucs[f"auc_ge_{thr}"] = float(roc_auc_score(yb, p))

    log(f"coverage grouped R2={grouped:+.3f}  country-level R2={country_r2:+.3f}  "
        f"leaky R2={leaky:+.3f}")
    log(f"classification AUCs: {aucs}")

    return {
        "model": "HPVc_Females ~ socioeconomic determinants",
        "n_rows": int(len(v)), "n_countries": int(v.iso3c.nunique()),
        "rf_r2_grouped": grouped,
        "rf_r2_country_level": country_r2,
        "rf_r2_leaky_reference": leaky,
        "leakage_inflation": leaky - grouped,
        **aucs,
    }


def main():
    d = load()
    rf, m, X, shap_df, importance, metrics = build_mortality_model(d)
    attrib = country_attribution(m, X, shap_df, rf, metrics["shap_base_value"])
    null = build_coverage_null(d)

    keep = ["iso3c", "country", "year", TARGET_A, "Region", "Income group"]
    pd.concat([m[keep].reset_index(drop=True), X.reset_index(drop=True)], axis=1) \
        .to_csv(os.path.join(ARTIFACTS, "mortality_features.csv"), index=False)
    shap_df.to_csv(os.path.join(ARTIFACTS, "mortality_shap.csv"), index=False)
    importance.to_csv(os.path.join(ARTIFACTS, "mortality_importance.csv"), index=False)
    attrib.to_csv(os.path.join(ARTIFACTS, "country_attribution.csv"), index=False)

    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as fh:
        json.dump({"model_a_mortality": metrics, "model_b_coverage_null": null},
                  fh, indent=2)
    joblib.dump({"model": rf, "features": MORTALITY_FEATURES, "pretty": PRETTY},
                os.path.join(ARTIFACTS, "rf_model.joblib"))

    log("artifacts written")
    print()
    print(importance[["pretty", "mean_abs_shap", "perm_importance", "std_beta"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
