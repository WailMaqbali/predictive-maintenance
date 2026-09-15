# Build Log

Running log of decisions, blockers, and anything that didn't go as planned while building this project autonomously.

## 2026-09-15

- Environment: Windows 11, Python 3.13.3, git 2.55.0, gh CLI 2.100.0 (authenticated as WailMaqbali).
- Created repo structure: `src/`, `notebooks/`, `figures/`, `tests/`, `data/`, `models/`.
- Downloaded dataset from the confirmed-working raw GitHub URL into `data/predictive_maintenance.csv` (10,000 rows incl. header — 9999 data rows, matches the documented ~10,000-row dataset).
- Created a local `.venv` and installed `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imbalanced-learn, joblib, fastapi, uvicorn, pydantic, pytest). All imports verified working.
- `git init` done in this folder. Global git identity already configured (Wail Al Maqbali), so no local override needed.
- `gh auth status` confirms an authenticated GitHub CLI session — plan is to `gh repo create` and push once the local build is complete and committed.
