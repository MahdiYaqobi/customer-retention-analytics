# Customer Retention Analytics — Telco Customer Churn Prediction

Predicting which customers are likely to churn, using the IBM Telco Customer Churn dataset. The project covers the full workflow: data understanding, cleaning, EDA, feature engineering, model comparison, threshold selection, and a reusable inference script for scoring new customers.

## Final Result

- **Model:** Logistic Regression (scikit-learn pipeline: preprocessing + classifier)
- **Decision threshold:** 0.30 (chosen over the default 0.50 to prioritize Recall)
- **Test set performance:** Recall 0.751 · Precision 0.525 · F1 0.618 · ROC-AUC 0.842 · Accuracy 0.754
- Of 374 actual churners in the test set, the model correctly flags 281, while 254 loyal customers were incorrectly flagged.

## Business Problem

Missing an actual churner (a false negative) is assumed to be more costly to the business than flagging a customer who ultimately stays (a false positive). This asymmetry is why **Recall** is treated as the primary evaluation metric, and why the final classification threshold was deliberately lowered below the default 0.50.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 columns covering demographics, account details, and subscribed services. Target: `Churn` (Yes/No), moderately imbalanced (~73% stayed, ~27% churned).

## Project Structure

```
Customer-retention-analytics/
├── data/
│   ├── raw/                     # Telco-Customer-Churn.csv, new_customers.csv (sample for inference)
│   └── processed/                # Cleaned data, train/test splits, churn_predictions.csv output
├── models/
│   ├── preprocessor.joblib       # Fitted ColumnTransformer (scaling + encoding)
│   ├── churn_model_logreg.pkl    # Final trained pipeline (preprocessing + model)
│   └── model_metadata.txt        # Selected threshold and final test metrics
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_Exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_preprocessing.ipynb
│   └── 05_model_training_and_evaluation.ipynb
├── src/
│   └── predict.py                # Inference script: raw customer data → churn prediction
└── README.md
```

## Workflow

**1. Data Understanding** — First pass over the raw dataset: shape, types, summary stats. Flagged early that `TotalCharges` was stored as text, which usually means hidden blanks rather than real missing values.

**2. EDA** — Explored churn against each feature using visualizations and statistical tests (Chi-square for categorical relationships, Mann-Whitney U for numerical ones, since tenure and monthly charges aren't normally distributed). Key patterns: month-to-month contracts, Fiber optic internet, and Electronic check payments are all strongly associated with higher churn; tenure is the sharpest divide (median ~10 months for churners vs. ~38 for retained customers).

**3. Data Cleaning** — No duplicates, no explicit missing values, but `TotalCharges` had blank strings for customers with zero tenure. Filled with 0, since these are new customers who haven't been billed yet, not truly missing data. Column names standardized to snake_case.

**4. Preprocessing & Feature Engineering** — Added five engineered features: `total_services`, `is_new_customer`, `tenure_group` (New/Developing/Established/Loyal), `streaming_services`, `has_security_services`. Split 80/20, stratified by churn. Built and saved a `ColumnTransformer` (one-hot encoding + scaling).

**5. Model Training & Evaluation** — Compared three baseline models:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.801 | 0.656 | 0.524 | 0.582 | 0.842 |
| Decision Tree | 0.735 | 0.501 | 0.505 | 0.503 | 0.661 |
| Random Forest | 0.774 | 0.596 | 0.465 | 0.523 | 0.821 |

5-fold cross-validation confirmed the same ranking and gave a more stable estimate:

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.532 ± 0.036 | 0.675 ± 0.022 | 0.595 ± 0.025 | 0.846 ± 0.012 |
| Decision Tree | 0.496 ± 0.028 | 0.489 ± 0.025 | 0.493 ± 0.025 | 0.655 ± 0.018 |
| Random Forest | 0.485 ± 0.013 | 0.638 ± 0.037 | 0.551 ± 0.020 | 0.822 ± 0.011 |

Random Forest was tuned with `GridSearchCV` (5-fold CV, scoring on Recall), which raised its Recall (CV: 0.485 → 0.499, test: 0.465 → 0.495) but still left it below Logistic Regression. **Logistic Regression was selected as the final model** — it had the highest Recall and ROC-AUC of the three, without the added complexity or reduced interpretability of an ensemble, which also matters for a business audience that will want to understand *why* the model flags a given customer.

**Threshold selection.** An exploratory pass on the test set (thresholds 0.50 → 0.30) suggested lowering the threshold would raise Recall substantially — but selecting a threshold using the test set would compromise it as an honest final evaluation. That exploratory result was kept as a reference but not used to make the decision. Instead, threshold selection was redone using `cross_val_predict` on the **training set only**, producing out-of-fold probabilities:

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.50 | 0.675 | 0.532 | 0.595 |
| 0.45 | 0.632 | 0.591 | 0.611 |
| 0.40 | 0.602 | 0.652 | 0.626 |
| 0.35 | 0.572 | 0.711 | **0.634** |
| 0.30 | 0.537 | 0.753 | 0.627 |

0.35 gives the best F1 balance, but **0.30 was chosen**: the Recall gain over 0.35 (+4.2 points) was judged worth the smaller Precision loss (-3.5 points), given that a missed churner is more costly to the business than an unnecessary retention contact. The final threshold was applied once to the untouched test set — Recall landed at 0.751, close to the 0.753 cross-validated estimate, indicating the choice generalizes rather than overfitting to one split.

### Multicollinearity in Service Features

Several service-related columns (`online_security`, `tech_support`, `streaming_tv`, etc.) include a `"No internet service"` category that is fully redundant with `internet_service = No` — a customer without internet automatically falls into that category for every dependent service. This produced identical, diluted coefficients (-0.258) across seven columns in the initial model, since the true effect had nowhere unambiguous to attach to. Merging `"No internet service"` into `"No"` before encoding removed the redundancy: `internet_service_No` then showed its actual coefficient (-1.076), and overall performance was unaffected. This was an interpretability fix, not a performance fix.

### Reading the Coefficients

Logistic Regression coefficients indicate the direction and relative strength of each feature's association with predicted churn, within this model — they are not proof of causation, and a coefficient's magnitude reflects its role in this specific linear model rather than an absolute measure of real-world importance. With that caveat, the largest coefficients were:

| Feature | Coefficient | Direction |
|---|---:|---|
| Fiber optic internet | +0.949 | associated with higher churn |
| Month-to-month contract | +0.671 | associated with higher churn |
| Tenure | -1.117 | associated with lower churn |
| No internet service | -1.076 | associated with lower churn |
| Two-year contract | -0.859 | associated with lower churn |

## Business Recommendations

These are framed as hypotheses to investigate or levers to test, not confirmed causes:

1. **Fiber optic service** Fiber optic internet has the strongest positive association with churn among the listed risk factors. Worth investigating whether this points to service reliability, support quality, or pricing relative to competitors.
2. **Month-to-month contracts** are the second-strongest association. Worth checking whether this reflects price sensitivity or simply the low switching cost of short-term plans.
3. **Incentivize longer contracts.** Tenure and two-year contracts show the strongest association with retention. Targeted incentives to move at-risk month-to-month customers onto longer terms could be tested.
4. **Promote support/security add-ons for at-risk segments.** Customers without tech support or online security show a modest association with churn. Bundling a trial for customers who already show other risk factors (e.g. Fiber optic + month-to-month) would target this more efficiently than a blanket offer.
5. **Score customers regularly rather than relying on segment-level rules.** A customer might carry one risk factor (Fiber optic) but be offset by others (long tenure, two-year contract). The trained model captures this combination directly — the recommendation is to run it on the active customer base periodically and prioritize outreach by predicted probability.

## Making Predictions

`src/predict.py` takes raw customer data (same schema as the original Telco CSV) and outputs a churn probability and label for each row. It applies the same cleaning and feature engineering as the notebooks internally, so no manual preprocessing is required before running it.

```bash
python src/predict.py --input data/raw/new_customers.csv --output data/processed/churn_predictions.csv
```

If `--output` is omitted, predictions are written to `data/processed/churn_predictions.csv` by default. The output CSV contains the original customer data plus `churn_probability`, `churn_prediction` (0/1), and `churn_label` (Stay/Churn), using the 0.30 threshold.

## Model Artifacts

- **`models/churn_model_logreg.pkl`** — the complete pipeline used by `predict.py` for preprocessing and Logistic Regression inference.
- **`models/preprocessor.joblib`** — the standalone preprocessing step used when the pipeline was originally built in notebook `05`. Not called separately at inference time; `churn_model_logreg.pkl` already includes it.
- **`models/model_metadata.txt`** — records the selected threshold (0.30) and final test metrics for reference.

## Tech Stack

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Jupyter.

## Limitations & Next Steps

- Coefficients describe associations within this model, not proven causes — the recommendations above are starting points for investigation, not conclusions.
- Only Logistic Regression, Decision Tree, and Random Forest were evaluated; gradient boosting methods were not explored.
- The 0.30 threshold reflects a stated but unquantified cost assumption (missed churner > unnecessary contact); a formal cost-benefit analysis, if retention costs and customer lifetime value were known, could refine it further.
- A reusable inference script (`src/predict.py`) exists for batch scoring from a CSV, but there is no API, web interface, or scheduled scoring job — production deployment has not been implemented.
