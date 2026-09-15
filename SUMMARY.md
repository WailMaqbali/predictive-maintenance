# Summary

Built autonomously end to end. Repo is live and pushed: **https://github.com/WailMaqbali/predictive-maintenance**

## What got built

A complete failure-risk classification project on the AI4I 2020 predictive maintenance
dataset (10,000 rows, 3.4% failure rate):

- EDA (`notebooks/01_eda.ipynb`, executed with real outputs, plus `src/eda.py`) covering
  class balance, feature distributions, correlation matrix, failure-type breakdown, and
  failure rate by product quality type.
- Feature engineering: `Power [W]` (torque × angular velocity) and `Temp difference [K]`,
  both tied directly to the dataset's own failure-mode definitions.
- Four trained models with explicit class-imbalance handling: Logistic Regression
  (class-weighted), Random Forest (class-weighted), Random Forest + SMOTE, and XGBoost
  (scale_pos_weight). All four are real, comparable results — nothing cherry-picked.
- Full evaluation: precision/recall/F1/ROC-AUC/PR-AUC, confusion matrices, PR/ROC curve
  comparisons, and a per-failure-type recall breakdown, all saved as figures.
- A working FastAPI `/predict` endpoint (`src/serve.py`) — manually tested against both a
  low-risk and a high-stress synthetic reading, correctly separated (probability ~0.00001
  vs ~0.9997).
- 25 pytest tests, all real assertions (not smoke tests), covering feature-engineering
  physics, label-inconsistency detection, stratified-split correctness, preprocessor output
  shape, metric edge cases, and the API's request/response contract.
- README.md, MODEL_CARD.md, BUILD_LOG.md, this SUMMARY.md.
- 7 incremental git commits, pushed to a public GitHub repo.

## Headline real results

Held-out test set: 2,000 rows, 68 true failures.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.178 | 0.868 | 0.295 | 0.466 |
| Random Forest | 0.917 | 0.809 | 0.859 | 0.858 |
| Random Forest + SMOTE | 0.636 | 0.824 | 0.718 | 0.829 |
| **XGBoost (selected)** | 0.800 | 0.824 | 0.812 | **0.864** |

XGBoost was selected by PR-AUC and is the model behind the served `/predict` endpoint.
Two things worth knowing going in:

1. **SMOTE underperformed simple class-weighting** on Random Forest here (0.829 vs 0.858
   PR-AUC) — a real, non-obvious result worth remembering, not a universal endorsement of
   SMOTE.
2. **Tool Wear Failure recall is genuinely weak** (10% for XGBoost) because only 45 examples
   of that failure type exist in the whole 10,000-row dataset. This is called out explicitly,
   with numbers, in MODEL_CARD.md — it's a data problem, not a modeling bug.

## What to look at first

1. `README.md` — 60-second overview, results table, embedded figures.
2. `figures/pr_curves.png` and `figures/model_comparison.png` — the clearest single view of
   how the four models actually compare.
3. `MODEL_CARD.md` — known limitations section, especially the Tool Wear Failure and
   Random Failures gaps.
4. `src/serve.py` + try the `/predict` endpoint locally (`uvicorn src.serve:app --reload`).

## What's still rough

- No CI (GitHub Actions) wired up yet to run `pytest` automatically on push.
- No hyperparameter search — models use sensible defaults/light manual tuning, so there's
  probably a bit more performance available.
- No probability calibration check.
- Random split (not time-aware) — correct for this snapshot dataset, but worth remembering
  if this pipeline is ever pointed at real time-series fleet data instead.
- Full detail on every rough edge is in the README "What I'd improve" section and
  MODEL_CARD.md "Known limitations" — nothing here is hidden or glossed over.

No blockers were hit during the build; see `BUILD_LOG.md` for the one transient `gh`
API hiccup (repo still got created and pushed fine after a manual remote-add).
