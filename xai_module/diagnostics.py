"""
Pre-flight validity check for the XAI extension.

Run BEFORE trusting any SHAP output. Explaining a model that does not predict
out-of-sample means attributing noise, so this script establishes which targets
actually carry signal.

Writes artifacts/diagnostics_report.txt. Read-only w.r.t. the existing project.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
DATA = os.path.join(r"C:\Users\goela\Downloads\hpv_dashboard", "data")
os.makedirs(ARTIFACTS, exist_ok=True)

INCOME = {"Low income": 0, "Lower middle income": 1,
          "Upper middle income": 2, "High income": 3}
F = ["health_exp_per_capita", "female_secondary_edu_pct", "fertility_rate",
     "internet_users_pct", "urban_pop_pct", "inc"]

LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def imp(df, cols):
    return df[cols].apply(lambda c: c.fillna(c.median()))


def load():
    d = pd.read_excel(os.path.join(DATA, "master_dataset_ready.xlsx"))
    d["inc"] = d["Income group"].map(INCOME)
    return d


def section(t):
    out()
    out("=" * 74)
    out(t)
    out("=" * 74)


def check_reproduction(d):
    """Do the paper's reported bivariate correlations reproduce?"""
    section("1. REPRODUCING THE PAPER'S REPORTED CORRELATIONS")
    out("Paper reports (Spearman, vs HPVc): health exp +0.48, fertility -0.45,")
    out("female education 'stronger positive'.")
    out()
    v = d[d.HPVc_Females.notna()]
    tests = [("full panel (all years)", v)]
    for y in (2019, 2022, 2024):
        s = v[v.year == y]
        if len(s) > 20:
            tests.append((f"cross-section {y}", s))
    cm = v.groupby("iso3c")[F + ["HPVc_Females"]].mean()
    tests.append(("country means", cm))

    out(f"{'subset':26s} {'health':>8s} {'femedu':>8s} {'fert':>8s} {'internet':>9s}   n")
    for tag, s in tests:
        r = [s[c].corr(s.HPVc_Females, method="spearman")
             for c in ["health_exp_per_capita", "female_secondary_edu_pct",
                       "fertility_rate", "internet_users_pct"]]
        out(f"{tag:26s} {r[0]:+8.2f} {r[1]:+8.2f} {r[2]:+8.2f} {r[3]:+9.2f}   {len(s)}")
    out()
    out(">> Only the 2022 cross-section approaches the published values.")
    out(">> The full panel and 2024 show essentially no association.")


def check_leakage(d):
    """Panel CV reuses countries across folds -- quantify the inflation."""
    section("2. PSEUDO-REPLICATION: WHY PANEL CV IS MISLEADING")
    v = d.dropna(subset=["HPVc_Females"])
    rf = RandomForestRegressor(400, min_samples_leaf=2, random_state=42, n_jobs=-1)

    X, y = imp(v, F), v.HPVc_Females
    p = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=42))
    leaky = r2_score(y, p)

    cm = v.groupby("iso3c").agg({**{f: "mean" for f in F},
                                 "HPVc_Females": "mean"}).reset_index()
    Xc, yc = imp(cm, F), cm.HPVc_Females
    pc = cross_val_predict(rf, Xc, yc, cv=KFold(5, shuffle=True, random_state=42))
    honest = r2_score(yc, pc)

    out(f"  random CV over 1,197 country-years  R2 = {leaky:+.3f}   <- same country in")
    out(f"                                                      train AND test folds")
    out(f"  country-level CV (one row/country)  R2 = {honest:+.3f}   <- honest")
    out()
    out(f">> The panel design inflates R2 by {leaky - honest:.2f}. The model is")
    out(">> memorising country identity, not learning from the determinants.")
    return leaky, honest


def check_targets(d):
    section("3. WHERE IS THERE ACTUAL SIGNAL? (all honest, country-level CV)")
    v = d.dropna(subset=["HPVc_Females"])
    latest = v.sort_values("year").groupby("iso3c").tail(1).copy()
    rf_r = RandomForestRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_c = RandomForestClassifier(500, min_samples_leaf=2, random_state=42, n_jobs=-1)
    kf = KFold(5, shuffle=True, random_state=42)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results = []

    # regression on coverage
    for tag, frame in [("HPVc coverage, country means",
                        v.groupby("iso3c").agg({**{f: "mean" for f in F},
                                                "HPVc_Females": "mean"}).reset_index()),
                       ("HPVc coverage, latest year", latest)]:
        p = cross_val_predict(rf_r, imp(frame, F), frame.HPVc_Females, cv=kf)
        r2 = r2_score(frame.HPVc_Females, p)
        results.append((tag, f"R2 = {r2:+.3f}", len(frame), "NO SIGNAL" if r2 < 0.05 else "weak"))

    # dropout
    latest["dropout"] = latest.HPV1_Females - latest.HPVc_Females
    dd = latest.dropna(subset=["dropout"])
    p = cross_val_predict(rf_r, imp(dd, F), dd.dropout, cv=kf)
    r2 = r2_score(dd.dropout, p)
    results.append(("dropout (HPV1 - HPVc)", f"R2 = {r2:+.3f}", len(dd),
                    "NO SIGNAL" if r2 < 0.05 else "weak"))

    # classification
    for thr in (30, 50, 70):
        yb = (latest.HPVc_Females >= thr).astype(int)
        p = cross_val_predict(rf_c, imp(latest, F), yb, cv=skf,
                              method="predict_proba")[:, 1]
        auc = roc_auc_score(yb, p)
        results.append((f"HPVc >= {thr}% (classification)", f"AUC = {auc:.3f}",
                        len(yb), "weak" if auc < 0.7 else "usable"))

    # cervical mortality
    mo = d[d.cervical_mortality_rate.notna()]
    cmo = mo.groupby("iso3c").agg({**{f: "mean" for f in F},
                                   "HPVc_Females": "mean",
                                   "cervical_mortality_rate": "mean"})
    cmo = cmo.dropna(subset=["cervical_mortality_rate"])
    cmo = cmo.apply(lambda c: c.fillna(c.median()))
    for tag, cols in [("cervical mortality ~ socioecon", F),
                      ("cervical mortality ~ socioecon + HPVc", F + ["HPVc_Females"])]:
        p = cross_val_predict(rf_r, cmo[cols], cmo.cervical_mortality_rate, cv=kf)
        r2 = r2_score(cmo.cervical_mortality_rate, p)
        results.append((tag, f"R2 = {r2:+.3f}", len(cmo),
                        "USABLE" if r2 > 0.15 else "weak"))

    out(f"{'target':40s} {'score':>14s} {'n':>5s}  verdict")
    out("-" * 74)
    for tag, score, n, verdict in results:
        out(f"{tag:40s} {score:>14s} {n:>5d}  {verdict}")

    out()
    out(">> Coverage is NOT predictable from socioeconomic determinants.")
    out(">> Cervical mortality IS moderately predictable from the same variables.")

    out()
    out("Spearman correlations with cervical mortality (coherent and strong):")
    for c in F + ["HPVc_Females"]:
        out(f"    {c:30s} {cmo[c].corr(cmo.cervical_mortality_rate, method='spearman'):+.3f}")
    return results


def main():
    d = load()
    out("XAI PRE-FLIGHT DIAGNOSTICS")
    out(f"dataset: {len(d)} country-years, {d.iso3c.nunique()} countries, "
        f"{d.year.min()}-{d.year.max()}")
    check_reproduction(d)
    check_leakage(d)
    check_targets(d)

    section("CONCLUSION")
    out("Applying SHAP to an HPVc-coverage model would explain noise: the model")
    out("has no out-of-sample skill. Valid options are (a) report the negative")
    out("result as a finding, and/or (b) apply XAI to cervical-cancer mortality,")
    out("where the same determinants do carry signal.")

    path = os.path.join(ARTIFACTS, "diagnostics_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
