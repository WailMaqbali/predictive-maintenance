# Model Card: Predictive Maintenance Failure-Risk Classifier

Follows the general structure of Mitchell et al., "Model Cards for Model Reporting" (2019).

## Model description

- **Task:** Binary classification — predict whether a milling machine, given a snapshot of
  sensor readings, is in a failure state (`Target = 1`) or not (`Target = 0`).
- **Best model:** XGBoost classifier (`scale_pos_weight` set to the training-set
  negative:positive ratio, ~28.6:1), selected by PR-AUC on a held-out test set among four
  candidates. Full comparison in `README.md`.
- **Inputs:** Product quality type (L/M/H), air temperature (K), process temperature (K),
  rotational speed (rpm), torque (Nm), tool wear (min). Two engineered features are derived
  at inference time: `Power [W]` (torque × angular velocity) and `Temp difference [K]`
  (process − air temperature).
- **Output:** A probability in [0, 1] that the machine is currently in a failure state, a
  thresholded label (0.5 cutoff), and a coarse risk tier (low < 0.2, medium 0.2–0.6,
  high ≥ 0.6).
- **Architecture:** Gradient-boosted decision trees (`xgboost.XGBClassifier`,
  `n_estimators=300`, `max_depth=5`, `learning_rate=0.1`), fed by a `ColumnTransformer`
  (standard-scaled numeric features, one-hot encoded `Type`).

## Intended use

Built as a **portfolio and educational project** demonstrating an end-to-end predictive
maintenance ML pipeline: EDA, imbalance-aware modeling, honest evaluation, and a served
endpoint. It is **not a validated safety-critical deployment**. Before any real operational
use (e.g. flagging real mining or manufacturing equipment for maintenance), it would need:
real fleet data (this dataset is synthesised to statistically resemble real sensor data,
per the original paper, not collected from an actual production line), domain-expert
validation of the failure-type breakdown below, calibration checking, and a
deployment-specific decision threshold set against real cost tradeoffs (see README
"What I'd improve").

**Out of scope:** remaining-useful-life-in-hours estimation. This dataset contains
independent snapshots, not run-to-failure time series, so it cannot support that kind of
prediction — treat any such claim about this model as a misuse of what the data supports.

## Training data

AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020; UCI Machine Learning Repository),
10,000 rows. 8,000 rows (80%) used for training via a stratified split (random_state=42),
2,000 (20%) held out for evaluation, never seen during training or model selection.

Class balance: 9,661 negative (96.6%) / 339 positive (3.4%) machine-failure rows.

Failure type breakdown across the full dataset (before the train/test split):

| Failure type | Count | % of all rows |
|---|---|---|
| No Failure | 9,652 | 96.52% |
| Heat Dissipation Failure | 112 | 1.12% |
| Power Failure | 95 | 0.95% |
| Overstrain Failure | 78 | 0.78% |
| Tool Wear Failure | 45 | 0.45% |
| Random Failures | 18 | 0.18% |

**Known label inconsistencies** (documented in the dataset, handled explicitly rather than
silently dropped — see `src/data.py::flag_label_inconsistencies`): 18 of the "Random
Failures" rows have `Target = 0` (not counted as a failure despite the failure-type label),
and 9 rows have `Target = 1` with `Failure Type = "No Failure"`. Together, 27 of 10,000 rows
(0.27%) have a Target/Failure Type disagreement.

## Evaluation results

Held-out test set (2,000 rows, 68 true failures, 3.4% — matches the full dataset's rate by
construction of the stratified split). Exact output of `python -m src.train`
(`models/metrics.json`), not rounded favorably:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (class_weight=balanced) | 0.178 | 0.868 | 0.295 | 0.934 | 0.466 |
| Random Forest (class_weight=balanced) | 0.917 | 0.809 | 0.859 | 0.975 | 0.858 |
| Random Forest + SMOTE | 0.636 | 0.824 | 0.718 | 0.972 | 0.829 |
| **XGBoost (scale_pos_weight, selected)** | 0.800 | 0.824 | 0.812 | 0.980 | 0.864 |

XGBoost confusion matrix on the test set: 1,918 true negatives, 14 false positives,
12 false negatives, 56 true positives.

### Recall by failure type (XGBoost, on the 68 true failures in the test set)

| Failure type | Test support | Recall |
|---|---|---|
| Heat Dissipation Failure | 28 | 0.964 |
| Overstrain Failure | 15 | 1.000 |
| Power Failure | 13 | 1.000 |
| Tool Wear Failure | 10 | **0.100** |
| Label-inconsistent rows (Target=1, Failure Type="No Failure") | 2 | 0.000 |

## Known limitations

- **Tool Wear Failure is not reliably detected.** Only 45 examples exist in the entire
  10,000-row dataset (10 in this test split); XGBoost recalls just 1 of those 10 in the
  test set (10% recall). Every other model tested does similarly poorly on this failure
  type (Random Forest 10%, Random Forest+SMOTE 20%, Logistic Regression 40% — LogReg's
  higher recall here comes at the cost of its overall 0.18 precision, i.e. it flags
  almost everything). **Do not rely on this model to catch tool-wear-driven failures
  specifically** — this is a data volume problem, not something fixable by better modeling
  on this dataset.
- **Random Failures (18 total rows) were not evaluable at all** — none appeared in this
  test split's true-failure set, so no model's performance on this mode could even be
  measured. The dataset's own documentation describes these as failures the simulated
  model attributes to unpredictable causes outside the five other mechanisms.
- **The 27 label-inconsistent rows are a real, small source of noise** in both training and
  evaluation. The 2 test-set rows with `Target=1, Failure Type="No Failure"` were recalled
  0% of the time by every tree-based model — plausibly because they look, on their sensor
  readings, like genuine non-failures, and the model is right to score them low.
- **Precision/recall tradeoff varies a lot by model** and the "best" model depends on
  operational priorities: XGBoost was chosen by PR-AUC, but Random Forest has notably
  higher precision (0.917 vs 0.800) at the cost of some recall (0.809 vs 0.824) — a
  maintenance team more worried about alert fatigue than a missed failure might prefer
  Random Forest instead.
- **Synthetic, not field, data.** Per the original paper, this dataset is synthesised to
  reflect real predictive maintenance data statistically — it is not sensor data collected
  from an actual production line. Real deployment data will differ, likely substantially,
  from this distribution.
- **No calibration check performed.** Predicted probabilities are used for thresholding and
  a coarse risk tier, not validated as calibrated probabilities (see README).

## Ethical / safety considerations

This is a maintenance-risk flagging tool, not a safety interlock. The two error types have
asymmetric real-world consequences:

- **False negative** (model says "no failure" when a failure is actually occurring or
  imminent): the practical implication is a **missed maintenance opportunity** — equipment
  proceeds to run in a failure or pre-failure state undetected, which in a real deployment
  could mean unplanned downtime, damaged equipment, or (in a heavy-industry context like
  the mining sector this project is aimed at) a safety incident if a mechanical failure
  occurs during operation. This model's overall recall is 0.82, meaning roughly 1 in 5
  failures in this dataset's distribution would be missed at the default 0.5 threshold —
  and recall on Tool Wear Failure specifically is far worse (0.10, see above).
- **False positive** (model flags a failure that isn't happening): costs unnecessary
  inspection/maintenance time and, at scale, alert fatigue that could cause real alerts to
  be deprioritized or ignored.

Given the asymmetry and the model's known Tool Wear Failure blind spot, this model should
never be the sole basis for a maintenance or safety decision in a real deployment — it is
a triage aid, and any operational use would need the additional validation and calibration
work noted above, plus human review of any flagged (and, ideally, periodic audit of
unflagged) equipment.
