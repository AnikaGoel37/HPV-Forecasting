"""
Publication figure suite for the XAI extension.

Reads artifacts/ from build_xai.py. 300-dpi PNG + vector PDF at Sage
double-column width (~6.9in). Pure matplotlib, so no browser export engine
is required; the interactive choropleth lives in the dashboard panel instead.

Figure numbering continues from the existing manuscript (Figures 1-14).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import shap

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

FEATURES = ["health_exp_per_capita", "female_secondary_edu_pct", "fertility_rate",
            "internet_users_pct", "urban_pop_pct", "inc", "HPVc_Females"]
PRETTY = {
    "health_exp_per_capita": "Health expenditure\nper capita (USD)",
    "female_secondary_edu_pct": "Female secondary\neducation (%)",
    "fertility_rate": "Fertility rate\n(births/woman)",
    "internet_users_pct": "Internet users (%)",
    "urban_pop_pct": "Urban population (%)",
    "inc": "Income group",
    "HPVc_Females": "HPVc vaccination\ncoverage (%)",
}
FLAT = {k: v.replace("\n", " ") for k, v in PRETTY.items()}

BLUE, ORANGE, RED, GREY = "#0056b3", "#ff7f0e", "#c1121f", "#8c8c8c"
W = 6.9

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linestyle": "--",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def log(m):
    print(f"[fig] {m}", flush=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    log(f"wrote {name}")


def load():
    X = pd.read_csv(os.path.join(ARTIFACTS, "mortality_features.csv"))
    sv = pd.read_csv(os.path.join(ARTIFACTS, "mortality_shap.csv"))
    imp = pd.read_csv(os.path.join(ARTIFACTS, "mortality_importance.csv"))
    at = pd.read_csv(os.path.join(ARTIFACTS, "country_attribution.csv"))
    with open(os.path.join(ARTIFACTS, "metrics.json")) as fh:
        met = json.load(fh)
    return X, sv, imp, at, met


# ---------------- Fig 15: beeswarm ----------------
def fig_beeswarm(X, sv):
    e = shap.Explanation(values=sv[FEATURES].values, data=X[FEATURES].values,
                         feature_names=[FLAT[f] for f in FEATURES])
    plt.figure(figsize=(W, 4.0))
    shap.plots.beeswarm(e, max_display=len(FEATURES), show=False)
    plt.xlabel("SHAP value (deaths per 100,000 women)")
    plt.title("Determinant contributions to cervical cancer mortality")
    save(plt.gcf(), "Figure_15_SHAP_beeswarm_mortality")


# ---------------- Fig 16: SHAP vs linear beta ----------------
def fig_shap_vs_linear(imp, met):
    d = imp.sort_values("mean_abs_shap")
    y = np.arange(len(d))
    labels = [FLAT[f] for f in d.feature]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.4), sharey=True)

    ax[0].barh(y, d.mean_abs_shap, color=BLUE, height=0.64)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].set_xlabel("mean |SHAP| (deaths per 100,000)")
    ax[0].set_title("(a) Random forest + SHAP", loc="left", fontweight="bold")

    ax[1].barh(y, d.std_beta, color=[BLUE if v >= 0 else ORANGE for v in d.std_beta],
               height=0.64)
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_xlabel("standardized $\\beta$")
    ax[1].set_title("(b) Linear regression", loc="left", fontweight="bold")
    ax[1].legend(handles=[Line2D([0], [0], color=BLUE, lw=6, label="raises mortality"),
                          Line2D([0], [0], color=ORANGE, lw=6, label="lowers mortality")],
                 loc="lower right", frameon=False, fontsize=7)

    a = met["model_a_mortality"]
    fig.suptitle(
        f"Determinant importance for cervical cancer mortality\n"
        f"grouped-CV R² — random forest {a['rf_r2_grouped']:.2f}, "
        f"linear {a['linear_r2_grouped']:.2f}",
        y=1.06, fontsize=9.5)
    save(fig, "Figure_16_SHAP_vs_linear_beta")


# ---------------- Fig 17: dependence ----------------
def fig_dependence(X, sv, imp):
    top = list(imp.sort_values("mean_abs_shap", ascending=False).feature)[:3]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9))
    for ax, f in zip(axes, top):
        xv, yv = X[f].values, sv[f].values
        sc = ax.scatter(xv, yv, c=X["inc"].values, cmap="viridis", s=10,
                        alpha=0.6, linewidths=0)
        ax.axhline(0, color="black", lw=0.7)
        ok = ~np.isnan(xv)
        if ok.sum() > 30:
            qs = np.unique(np.quantile(xv[ok], np.linspace(0, 1, 11)))
            cx, cy = [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (xv >= lo) & (xv <= hi)
                if m.sum() >= 5:
                    cx.append(np.median(xv[m])); cy.append(np.median(yv[m]))
            ax.plot(cx, cy, color=ORANGE, lw=2.0, zorder=5)
        # wrapped + smaller so adjacent panel labels cannot collide
        ax.set_xlabel(PRETTY[f], fontsize=7.5)
        if f == top[0]:
            ax.set_ylabel("SHAP (deaths/100k)")
        if f == "health_exp_per_capita":
            ax.set_xscale("symlog")
    fig.subplots_adjust(wspace=0.34)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["Low", "L-Mid", "U-Mid", "High"], fontsize=7)
    cb.set_label("Income group", fontsize=8)
    fig.suptitle("Nonlinear dependence of cervical cancer mortality on its leading determinants",
                 y=1.04, fontsize=9.5)
    save(fig, "Figure_17_SHAP_dependence")


# ---------------- Fig 18: country waterfalls ----------------
def fig_waterfalls(at, met):
    base = met["model_a_mortality"]["shap_base_value"]
    a = at.dropna(subset=["predicted_mortality"])
    picks = [("Highest predicted burden", a.loc[a.predicted_mortality.idxmax()]),
             ("Lowest predicted burden", a.loc[a.predicted_mortality.idxmin()])]
    # Income groups vary by data availability -- no low-income country reports
    # cervical mortality, so build exemplars only from groups actually present.
    chosen = {picks[0][1].iso3c, picks[1][1].iso3c}
    for grp in ["Lower middle income", "Upper middle income", "High income"]:
        s = a[(a["Income group"] == grp) & (~a.iso3c.isin(chosen))]
        if len(s) and len(picks) < 4:
            r = s.loc[s.predicted_mortality.idxmax()]
            picks.append((f"{grp} exemplar", r))
            chosen.add(r.iso3c)

    fig, axes = plt.subplots(2, 2, figsize=(W, 5.4))
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    for ax, (tag, r) in zip(axes.ravel(), picks):
        v = np.array([r[f"shap_{f}"] for f in FEATURES])
        o = np.argsort(np.abs(v))
        ax.barh(np.arange(len(v)), v[o],
                color=[BLUE if x >= 0 else ORANGE for x in v[o]], height=0.62)
        ax.set_yticks(np.arange(len(v)))
        ax.set_yticklabels([FLAT[FEATURES[i]] for i in o], fontsize=6.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{r['country']} ({r['iso3c']}), {int(r['year'])} — {tag}\n"
                     f"predicted {r['predicted_mortality']:.1f} vs baseline {base:.1f} "
                     f"deaths/100k", fontsize=7.6, loc="left")
        ax.set_xlabel("contribution (deaths per 100,000)", fontsize=7.5)
    fig.suptitle("Country-level attribution of cervical cancer mortality risk",
                 y=1.01, fontsize=9.5)
    fig.tight_layout()
    save(fig, "Figure_18_country_waterfalls")


# ---------------- Fig 19: dominant risk driver ----------------
def fig_drivers(at):
    order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    d = at[at["Income group"].isin(order)]
    ct = pd.crosstab(d["Income group"], d["dominant_risk_driver"]).reindex(order).fillna(0)
    ct = ct[ct.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(1, 2, figsize=(W, 3.2),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    cmap = plt.get_cmap("tab10")
    bottom = np.zeros(len(ct))
    for i, c in enumerate(ct.columns):
        ax[0].barh(np.arange(len(ct)), ct[c], left=bottom, color=cmap(i % 10),
                   height=0.6, label=c)
        bottom += ct[c].values
    ax[0].set_yticks(np.arange(len(ct)))
    ax[0].set_yticklabels([s.replace(" income", "") for s in ct.index], fontsize=8)
    ax[0].set_xlabel("number of countries")
    ax[0].set_title("(a) By income group", loc="left", fontweight="bold")
    ax[0].legend(fontsize=6.2, loc="lower right")

    tot = at.dominant_risk_driver.value_counts().sort_values()
    ax[1].barh(np.arange(len(tot)), tot.values, color=BLUE, height=0.6)
    ax[1].set_yticks(np.arange(len(tot)))
    ax[1].set_yticklabels(tot.index, fontsize=7)
    ax[1].set_xlabel("number of countries")
    ax[1].set_title("(b) Overall", loc="left", fontweight="bold")

    fig.suptitle("Dominant driver of cervical cancer mortality risk, by country",
                 y=1.03, fontsize=9.5)
    fig.tight_layout()
    save(fig, "Figure_19_dominant_risk_driver")


# ---------------- Fig 20: the null result + leakage ----------------
def fig_null_result(met):
    a, b = met["model_a_mortality"], met["model_b_coverage_null"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.3),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    labels = ["HPVc coverage\n(Model B)", "Cervical mortality\n(Model A)"]
    honest = [b["rf_r2_grouped"], a["rf_r2_grouped"]]
    leaky = [b["rf_r2_leaky_reference"], a["rf_r2_leaky_reference"]]
    x = np.arange(2); w = 0.36
    ax[0].bar(x - w / 2, leaky, w, color=GREY, label="random CV over country-years\n(same country in both folds)")
    ax[0].bar(x + w / 2, honest, w, color=BLUE, label="grouped CV by country\n(honest)")
    ax[0].axhline(0, color="black", lw=0.9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("out-of-sample R²")
    ax[0].set_title("(a) Pseudo-replication inflates apparent skill",
                    loc="left", fontweight="bold", fontsize=9)
    # headroom so the legend clears the tallest bar and its value label
    ax[0].set_ylim(min(honest) - 0.12, 1.25)
    ax[0].legend(fontsize=6.2, loc="upper left", framealpha=0.95)
    for xi, (l, h) in enumerate(zip(leaky, honest)):
        ax[0].text(xi - w / 2, l + 0.02, f"{l:+.2f}", ha="center", fontsize=7.5)
        ax[0].text(xi + w / 2, h + 0.02, f"{h:+.2f}", ha="center", fontsize=7.5)

    aucs = [b["auc_ge_30"], b["auc_ge_50"], b["auc_ge_70"]]
    ax[1].bar(np.arange(3), aucs, 0.55, color=ORANGE)
    ax[1].axhline(0.5, color=RED, ls="--", lw=1.2)
    ax[1].text(2.45, 0.508, "chance", color=RED, fontsize=7, ha="right")
    ax[1].set_xticks(np.arange(3))
    ax[1].set_xticklabels(["≥30%", "≥50%", "≥70%"], fontsize=8)
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_ylabel("AUC")
    ax[1].set_xlabel("coverage threshold")
    ax[1].set_title("(b) Coverage is barely\nseparable from determinants",
                    loc="left", fontweight="bold", fontsize=9)
    for i, v in enumerate(aucs):
        ax[1].text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)

    fig.suptitle("Socioeconomic determinants explain disease burden, not vaccination coverage",
                 y=1.04, fontsize=9.5)
    fig.tight_layout()
    save(fig, "Figure_20_null_result_and_leakage")


# ---------------- Fig 21: HPVc contribution is negligible ----------------
def fig_hpvc_confounding(X, sv, imp):
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.1),
                           gridspec_kw={"width_ratios": [1, 1.1]})

    d = imp.sort_values("mean_abs_shap", ascending=True)
    cols = [RED if f == "HPVc_Females" else BLUE for f in d.feature]
    ax[0].barh(np.arange(len(d)), d.mean_abs_shap, color=cols, height=0.62)
    ax[0].set_yticks(np.arange(len(d)))
    ax[0].set_yticklabels([FLAT[f] for f in d.feature], fontsize=7)
    ax[0].set_xlabel("mean |SHAP| (deaths per 100,000)")
    ax[0].set_title("(a) Adjusted contribution", loc="left", fontweight="bold", fontsize=9)

    ax[1].scatter(X["HPVc_Females"], sv["HPVc_Females"], s=11, alpha=0.55,
                  color=RED, linewidths=0)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xlabel("HPVc vaccination coverage (%)")
    ax[1].set_ylabel("SHAP contribution\n(deaths per 100,000)")
    ax[1].set_title("(b) Effect on predicted mortality", loc="left",
                    fontweight="bold", fontsize=9)
    span = float(np.abs(sv["HPVc_Females"]).max())
    ax[1].text(0.03, 0.05, f"max |contribution| = {span:.2f} deaths/100k",
               transform=ax[1].transAxes, fontsize=7.5,
               bbox=dict(fc="white", ec="#cccccc"))

    fig.suptitle("After adjusting for development, vaccination coverage adds little to\n"
                 "current mortality — vaccinated cohorts have not yet reached cancer age",
                 y=1.10, fontsize=9)
    fig.tight_layout()
    save(fig, "Figure_21_HPVc_confounding")


# ---------------- Fig 22: robustness ----------------
def fig_robustness(imp, met):
    m = met["model_a_mortality"]
    fig, ax = plt.subplots(figsize=(W * 0.6, 3.2))
    ax.scatter(imp.mean_abs_shap, imp.perm_importance, s=46, color=BLUE, zorder=3)
    for _, r in imp.iterrows():
        ax.annotate(FLAT[r.feature], (r.mean_abs_shap, r.perm_importance),
                    fontsize=6.6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("mean |SHAP|")
    ax.set_ylabel("permutation importance")
    ax.set_title(f"Attribution robustness\nSpearman $\\rho$ = {m['shap_perm_rank_corr']:.2f}, "
                 f"additivity error < {m['shap_additivity_max_error']:.0e}", fontsize=9)
    save(fig, "Figure_22_robustness_check")


def main():
    X, sv, imp, at, met = load()
    fig_beeswarm(X, sv)
    fig_shap_vs_linear(imp, met)
    fig_dependence(X, sv, imp)
    fig_waterfalls(at, met)
    fig_drivers(at)
    fig_null_result(met)
    fig_hpvc_confounding(X, sv, imp)
    fig_robustness(imp, met)
    log(f"all figures in {FIGURES}")


if __name__ == "__main__":
    main()
