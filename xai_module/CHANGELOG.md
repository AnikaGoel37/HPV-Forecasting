# Change Log — XAI Module

Every edit made **outside** `xai_module/`. Revert these to restore the original state.

## Environment
| # | Change | Revert |
|---|--------|--------|
| E1 | `pip install shap scikit-learn` into `hpv_dashboard/venv` | `pip uninstall shap scikit-learn` |

`hpv_dashboard/requirements.txt` was **not** modified.

## Site code
| # | File | Change | Revert |
|---|------|--------|--------|
| S1 | `hpv_dashboard/app/dashboard.py` | Added 7th tab `"Why? (Explainability)"` — 2 hunks, both marked `# [XAI]` | Restore `dashboard.py.orig_backup` (same folder), or delete the `# [XAI]` blocks |

Backup of the original file: `hpv_dashboard/app/dashboard.py.orig_backup`

## New files (self-contained, safe to delete)
- `Influenza/xai_module/` — all XAI code, figures, artifacts
- `Influenza/local_site/serve.ps1` — localhost:8080 launcher

## Full revert
```
copy hpv_dashboard\app\dashboard.py.orig_backup hpv_dashboard\app\dashboard.py
rmdir /s Influenza\xai_module
rmdir /s Influenza\local_site
```
