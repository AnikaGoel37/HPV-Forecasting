# XAI Module — HPV ForestX

Explainability extension for the HPV ForestX study. Self-contained: delete this
folder and revert the two `# [XAI]` hunks in `dashboard.py` to undo everything.
See [CHANGELOG.md](CHANGELOG.md).

## Run order

```bash
PY="C:/Users/goela/Downloads/hpv_dashboard/venv/Scripts/python.exe"
$PY xai_module/diagnostics.py    # validity check -- run first
$PY xai_module/build_xai.py      # trains models, computes SHAP
$PY xai_module/make_figures.py   # writes figures/
```

| File | Purpose |
|---|---|
| `diagnostics.py` | Pre-flight validity check. Establishes which targets carry signal |
| `build_xai.py` | Fits Model A (mortality) + Model B (coverage null), computes SHAP |
| `make_figures.py` | Figures 15–22, 300-dpi PNG + vector PDF |
| `panel.py` | Renders the dashboard's "Why?" tab |
| `artifacts/` | Model, SHAP values, per-country attributions, metrics |
| `figures/` | Publication output |

## What changed and why

The original plan was SHAP on a model predicting **HPVc coverage**. That model
has no out-of-sample skill, so its attributions would be noise. The target moved
to **cervical cancer mortality**, where the same determinants do carry signal,
and the coverage failure is reported as a finding in its own right.

## Findings

**1. Determinants explain disease burden, not vaccination coverage.**

| Model | Grouped CV R² | Random CV R² (leaky) |
|---|---|---|
| Cervical mortality | **+0.247** | +0.748 |
| HPVc coverage | **−0.267** | +0.599 |

Coverage classification is barely above chance (AUC 0.64–0.66). The implication:
coverage is driven by *programmatic* factors — whether a national programme
exists, delivery platform, supply, political commitment — not by development
level. This explains why the WHO target is missed for reasons money and
schooling alone do not capture.

**2. Panel cross-validation inflates apparent skill by ~0.5 R².**

The data is a country-year panel; each country contributes up to 15 rows. Random
row splits put the same country in train and test, letting the model memorise
country identity. There are **141 countries, not 1,197 independent
observations**. Any pooled-panel regression on this data will overstate
significance through pseudo-replication.

**3. Health expenditure acts as a threshold, not a gradient.** (Figure 17)

Below roughly USD 1,500–2,000 per capita the contribution is +2 to +2.5
deaths/100,000; above it, −2 and flat. A step function, which a linear
coefficient cannot represent. Fertility breaks similarly near 2.2 births/woman.

**4. HPVc coverage contributes almost nothing to current mortality.** (Figure 21)

mean|SHAP| = 0.09 deaths/100,000 — 5th of 7 determinants. Biologically expected:
HPV vaccination began around 2010 and vaccinated cohorts have not reached
cervical cancer age. **Bivariate coverage–mortality associations are confounded
by development level and should not be read as vaccine impact.**

## Limitations to state in the manuscript

- **Selection bias in the mortality sample.** Only 73 of 144 countries report
  cervical mortality, skewed wealthy: 45 high income, 24 upper-middle, 4
  lower-middle, **zero low income**. The mortality model therefore says little
  about the LMICs the study is most concerned with.
- **Associational, not causal.** SHAP decomposes a predictive model. Scenario
  outputs are model-implied responses, not intervention effects.
- **A linear model is competitive** (R² 0.359 vs 0.247), so the dominant
  relationships are largely monotonic. SHAP is used here for per-country
  decomposition, not to claim superior fit.
- **Median imputation** fills feature gaps; `female_secondary_edu_pct` and
  `school_enrollment_secondary_pct` correlate at 0.62, so the latter was dropped
  to avoid collinearity sign-flipping.
- **Ecological inference** — country-level associations do not transfer to
  individuals.

## Unresolved: the published correlations do not reproduce

The manuscript reports Spearman correlations with HPVc of health expenditure
**+0.48** and fertility **−0.45**. From `master_dataset_ready.xlsx`:

| Subset | health exp | fertility |
|---|---|---|
| Full panel | +0.17 | −0.06 |
| Cross-section 2022 | +0.42 | −0.30 |
| Cross-section 2024 | −0.00 | −0.02 |
| Country means | +0.16 | −0.03 |

Only 2022 approaches the published values. Either the published statistics come
from a single year, or from a different merged dataset than the one the
dashboard loads. **Resolve this before submission** — the Influenza repo
contains several other candidates (`cleaned_hpv_actuals.csv`,
`merged_dataset*.xlsx`). This module draws no conclusion about which is correct.

## Figures

| Figure | Content |
|---|---|
| 15 | SHAP beeswarm — global determinant contributions |
| 16 | mean\|SHAP\| vs standardized β |
| 17 | Dependence plots — threshold effects |
| 18 | Country waterfalls — four archetypes |
| 19 | Dominant risk driver by income group |
| 20 | The coverage null result + leakage inflation |
| 21 | HPVc contribution is negligible after adjustment |
| 22 | Robustness — SHAP vs permutation importance (ρ = 0.96) |
