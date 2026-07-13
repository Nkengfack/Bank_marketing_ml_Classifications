# Bank_marketing_ml_Classification
# Bank Term Deposit Subscription Prediction

A comparative performance analysis of machine learning classifiers on the UCI Bank Marketing dataset, built as the MSc Data Science final project (Module 7PAM2002, University of Hertfordshire).

**Research question:** which supervised machine learning model best predicts whether a bank client will subscribe to a term deposit, using only information available *before* the marketing call is made, and how large is the performance cost of enforcing that constraint?

The constraint matters. The strongest predictor in this dataset, call duration, does not exist until the call has ended, so a model that uses it cannot decide who to ring. Every model here is trained twice, with and without duration, to measure exactly what that leakage is worth.

## Headline results

| | |
|---|---|
| Final model | XGBoost, without duration, decision threshold 0.20 |
| Test F1 | 0.530 (95% bootstrap CI 0.503 to 0.555) |
| Test PR-AUC | 0.487 (95% CI 0.452 to 0.523), against a 0.113 no-skill baseline |
| Test ROC-AUC | 0.81 (the published benchmark, Moro et al. 2014, is ~0.80) |
| Cost of removing duration | 0.14 ROC-AUC and up to 0.19 PR-AUC, for every model family |
| Business reading | Call 14% of the client list, reach 60% of the subscribers in it (4.2x lift) |

Six models were compared through one identical pipeline: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost and LightGBM. Decision Tree and SVM were dropped on baseline evidence; the remaining four were tuned with randomised search under 5-fold stratified cross-validation, scored on average precision.

## Repository contents

| File | What it is |
|---|---|
| `bank_marketing_ML_test.ipynb` | The main notebook. Part 1 covers loading, cleaning and EDA; Part 2 covers preprocessing, the six-model baseline, tuning with and without duration, threshold selection, bootstrap confidence intervals, permutation importance and a saved-model prediction demo. Markdown cells explain each decision. |

| `bank_marketing_dataset.csv` | The UCI Bank Marketing data (bank-additional-full): 41,188 rows, 21 columns, semicolon separated. |
| `LICENSE`|
| `README.md` | This file. |

The notebook also writes outputs when run: result tables as CSV (`baseline_results.csv`, `tuned_with_duration.csv`, `tuned_without_duration.csv`, `duration_impact.csv`), all figures as PNG, and the deployable artefacts `final_model.joblib` and `final_threshold.joblib`.

## How to run

**Easiest: Google Colab.** Open `bank_marketing_ML_test.ipynb` in Colab and run all cells. The notebook loads the dataset directly from this repository over HTTPS, so no upload is needed. The full run, including both tuning passes, takes roughly 30 to 60 minutes on a standard Colab instance; everything before Section 4.6 runs in a few minutes.

**Locally:**

```bash
git clone https://github.com/Nkengfack/Bank_marketing_ml_Classifications.git
cd Bank_marketing_ml_Classifications
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
jupyter notebook bank_marketing_ML_test.ipynb   # or: python bank_marketing_ml_test.py
```

Tested with Python 3.10+. No GPU is required.

**Using the saved model:**

```python
import joblib
model = joblib.load('final_model.joblib')       # full pipeline: preprocessing + XGBoost
thr   = joblib.load('final_threshold.joblib')   # 0.20

proba = model.predict_proba(new_clients)[:, 1]  # new_clients: DataFrame with the raw
decision = proba >= thr                         # pre-call columns (no 'duration')
```

The pipeline contains the whole preprocessing stage (RobustScaler, ordinal and one-hot encoders), so it accepts raw feature values, not encoded ones. It expects the engineered columns `contact_date_known`, `days_since_contact` and `contact_intensity` in place of `pdays`; Section 3.18 to 3.19 of the notebook shows the three lines that create them.

## Data notes for anyone reusing this code

Three properties of this dataset will silently break a naive analysis:

1. **The file is semicolon separated.** Read it with `sep=';'` or all 21 columns collapse into one.
2. **Missing values are hidden as the text `unknown`** (12,718 entries across six columns, worst is `default` at 20.9%). They are invisible to `isnull()`. This project keeps `unknown` as its own category rather than dropping or imputing.
3. **`pdays = 999` does not always mean "never contacted".** 4,110 rows carry 999 alongside `previous > 0`, so for them the code means the contact date was unrecorded. The project replaces the column with a binary flag and a recoded day count.

Class balance is 88.7% no to 11.3% yes, so accuracy is close to meaningless; the code evaluates with F1, PR-AUC, ROC-AUC and MCC.

## Reproducibility

All splits, searches and the bootstrap use fixed seeds (`random_state=42` throughout, `numpy.random.default_rng(42)` for the bootstrap), and the train/test split is made once and reused by every experiment. An assertion on the dataset shape stops the notebook if the source file ever changes.

## Dataset licence and citation

The dataset was created by Moro, Cortez and Rita and is distributed through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) under a CC BY 4.0 licence. If you use it, cite:

> Moro, S., Cortez, P. and Rita, P. (2014) 'A data-driven approach to predict the success of bank telemarketing', *Decision Support Systems*, 62, pp. 22-31. https://doi.org/10.1016/j.dss.2014.03.001

## Author

Nicolas Nkengfack, MSc Data Science, University of Hertfordshire.
