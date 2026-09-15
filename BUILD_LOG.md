# Build Log

Running log of decisions, blockers, and anything that didn't go as planned while building this project autonomously.

## 2026-09-15

- Environment: Windows 11, Python 3.13.3, git 2.55.0, gh CLI 2.100.0 (authenticated as WailMaqbali).
- Created repo structure: `src/`, `notebooks/`, `figures/`, `tests/`, `data/`, `models/`.
- Downloaded dataset from the confirmed-working raw GitHub URL into `data/predictive_maintenance.csv` (10,000 rows incl. header — 9999 data rows, matches the documented ~10,000-row dataset).
- Created a local `.venv` and installed `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imbalanced-learn, joblib, fastapi, uvicorn, pydantic, pytest). All imports verified working.
- `git init` done in this folder. Global git identity already configured (Wail Al Maqbali), so no local override needed.
- `gh auth status` confirms an authenticated GitHub CLI session — plan is to `gh repo create` and push once the local build is complete and committed.
- Inspected the raw CSV directly: confirmed the two documented AI4I quirks — 18 rows with
  `Failure Type == "Random Failures"` but `Target == 0`, and 9 rows with `Target == 1` but
  `Failure Type == "No Failure"` (27/10000 rows, 0.27%). Kept these rows (not dropped) and
  added an explicit `Label inconsistent` flag column in `src/data.py` rather than silently
  cleaning them away, since the dataset's own documentation acknowledges them as injected
  noise rather than data-entry errors.
- Trained 4 models end-to-end (`python -m src.train`): Logistic Regression, Random Forest,
  Random Forest + SMOTE, XGBoost. All real numbers, no fabrication — see README.md /
  MODEL_CARD.md for the full table. Headline: XGBoost wins on PR-AUC (0.864) and was
  selected as the served model; Random Forest has notably higher precision (0.917) at
  slightly lower recall, a legitimate alternative depending on operational priorities.
- Real, noteworthy finding: SMOTE underperformed plain `class_weight="balanced"` on Random
  Forest here (PR-AUC 0.829 vs 0.858). Documented in the README as a deliberate,
  non-cherry-picked result — with only ~270 real positive training examples, SMOTE's
  synthetic neighbors appear to add more noise than signal on this dataset. This directly
  addresses the "note the SMOTE connection" instruction without overstating SMOTE as a
  universal win.
- Real, noteworthy limitation: Tool Wear Failure recall is poor across every model tested
  (XGBoost 10%, Random Forest 10%, RF+SMOTE 20%, Logistic Regression 40% but only via a
  precision-destroying low threshold-equivalent). Root cause is data volume: only 45 Tool
  Wear Failure rows exist in the full 10,000-row dataset (10 in the test split). Documented
  explicitly, with numbers, in MODEL_CARD.md's Known Limitations section rather than
  glossed over.
- Random Failures (18 rows total, all with `Target == 0`) never appear in any test split's
  true-failure set, so no model's performance on that failure mode could be measured at
  all — noted as a limitation rather than silently omitted.
- No blockers encountered. All planned components (EDA, preprocessing, 4 trained models,
  evaluation, FastAPI serving, tests, notebook, docs) built successfully, with 7 incremental
  commits (scaffold -> EDA -> preprocessing/models -> serving -> tests -> notebook -> docs).
- `gh repo create` had a transient GraphQL error on the first attempt but the repo was
  actually created server-side despite the error message; the retry correctly reported
  "Name already exists". The local `--remote=origin` step never ran because the first call
  errored before reaching it, so `git remote -v` was empty afterward -- fixed by manually
  running `git remote add origin https://github.com/WailMaqbali/predictive-maintenance.git`
  and `git push -u origin master`, which succeeded. Repo is public at
  https://github.com/WailMaqbali/predictive-maintenance.
