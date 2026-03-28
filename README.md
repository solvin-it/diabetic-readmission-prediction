# Predicting 30-Day Hospital Readmission for Diabetic Patients

A completed machine learning capstone project focused on predicting whether diabetic patients will be readmitted to the hospital within 30 days of discharge. The repository covers the full ML lifecycle from problem framing and preprocessing through model comparison, threshold tuning, explainability, fairness analysis, and deployment-ready artifact export.

## Executive Summary

Hospitals want to identify high-risk patients early enough to plan discharge support, follow-up care, and intervention workflows. This project frames diabetic readmission as a binary classification problem where missing a true high-risk patient is costlier than raising a false alarm, so model selection is driven by AUC-ROC with explicit operating-threshold optimization for recall.

Final outcome:

- Best model: Random Forest - PCA
- Feature space: 44 PCA components derived from 117 mutual-information-selected features
- Best test AUC-ROC: 0.6446
- Selected operating threshold: 0.4556 using the Constrained strategy
- Test recall at selected threshold: 0.7164
- Test precision at selected threshold: 0.1500
- Test F1 at selected threshold: 0.2481

The final model did not reach the original AUC-ROC target of 0.75, but it substantially exceeded the recall target of 0.50 and aligns with published benchmark performance for this dataset. The resulting pipeline is packaged for reproducible inference and reporting.

## Project Overview

Hospital readmissions are costly for healthcare providers and may indicate gaps in post-discharge care. For diabetic patients, 30-day readmission is an especially important outcome because it affects both patient well-being and hospital performance metrics.

This project uses supervised machine learning to estimate readmission risk from demographic, diagnostic, admission, utilization, and treatment-related signals. The workflow emphasizes reproducibility, honest validation, and interpretable decision support rather than leaderboard-only optimization.

## Problem Statement

Hospitals aim to reduce avoidable 30-day readmissions because they increase healthcare costs and are a proxy for quality of care. This project predicts whether a diabetic patient will be readmitted within 30 days after discharge.

The task is treated as a binary classification problem where:

- Positive class: readmitted within 30 days
- Negative class: not readmitted within 30 days

Success criteria were defined as:

| Criterion | Target | Result |
|---|---:|---:|
| Test AUC-ROC | >= 0.75 | 0.6446 |
| Recall at operating threshold | >= 0.50 | 0.7164 |
| Racial recall gap | <= 0.15 | 0.2189 |

The primary metric is AUC-ROC. Recall is treated as the key operating metric because failing to identify a high-risk patient is costlier than unnecessarily flagging a lower-risk patient.

### Business Goal

Support earlier identification of high-risk diabetic patients so hospitals can improve intervention planning, discharge support, and resource allocation.

### Machine Learning Goal

Develop, compare, and operationalize a classification model that can rank readmission risk and support a recall-prioritized intervention strategy.

## Dataset

- Dataset: Diabetes 130-US Hospitals for Years 1999-2008
- Original source: UCI Machine Learning Repository
- Common mirror used for access: Kaggle
- Size: 101,766 clinical encounters across 130 U.S. hospitals
- Raw schema: 50 columns

The dataset contains demographic variables, admission details, diagnoses, medications, utilization history, and laboratory-related indicators. The raw file uses `?` as the missing-value marker and includes a strongly imbalanced readmission target.

## Project Objectives

- Understand the business and healthcare context of hospital readmissions
- Explore and clean the diabetic patient dataset
- Perform applied exploratory data analysis
- Engineer and select relevant features
- Train and compare multiple machine learning models
- Optimize the decision threshold for a recall-priority healthcare setting
- Interpret model predictions using explainability tools
- Evaluate fairness and bias across sensitive groups
- Export reproducible artifacts for downstream use in reports, slides, or deployment

## Deployment: API & Streamlit UI

This project includes a production-ready deployment layer that wraps the training artifacts in an async FastAPI backend and a Streamlit web interface. See [app/README.md](app/README.md) for full deployment documentation, API reference, and quick-start guides.

**Quick Start:**
```bash
docker-compose up
```
This launches:
- **API** on `http://localhost:8000` (FastAPI with async prediction and LLM-grounded explanation endpoints)
- **UI** on `http://localhost:8501` (Streamlit with 3 tabs: Project Summary, Prediction Tool, Explanation Assistant)

## Repository Structure

```text
diabetic-readmission-prediction/
├── README.md                          # This file; ML project overview
├── app/                               # Deployment layer (API + UI)
│   ├── README.md                      # Comprehensive deployment guide
│   ├── api/                           # FastAPI backend
│   │   ├── main.py                    # App entrypoint
│   │   ├── config.py                  # Settings & environment
│   │   ├── routers/                   # Endpoints: health, predict, explain
│   │   ├── services/                  # Model, adapter, LLM services
│   │   ├── schemas/                   # Pydantic request/response models
│   │   └── core/                      # Logging, error handling
│   ├── ui/                            # Streamlit frontend
│   │   ├── Home.py                    # App entrypoint
│   │   ├── tabs/                      # 3-tab UI: summary, prediction, chat
│   │   └── services/                  # API client
│   ├── requirements.api.txt           # API dependencies
│   └── requirements.ui.txt            # UI dependencies
├── compose.yaml                       # Docker Compose orchestration
├── Dockerfile.api & Dockerfile.ui     # Container definitions
├── requirements.txt                   # Project root dependencies
├── data/
│   ├── raw/                           # Original diabetic_data.csv
│   └── processed/                     # Cleaned, engineered, split datasets
├── models/                            # Trained artifacts (deployment_pipeline.joblib, etc.)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_eda_and_feature_engineering.ipynb
│   ├── 03_modeling_and_evaluation.ipynb
│   └── 04_final_summary_and_results.ipynb
├── reports/
│   ├── figures/                       # EDA, model comparison, SHAP, fairness charts
│   └── tables/                        # CSV comparison & fairness summaries
├── references/                        # Capstone coding templates (read-only)
└── src/                               # Reusable ML modules (preprocessing, training, etc.)
```

## Notebook Guide

### `01_data_understanding.ipynb`

Implements the Data Understanding phase:

- Dataset background and healthcare framing
- Problem framing as binary classification
- Initial loading, profiling, descriptive statistics, and schema inspection
- Data dictionary and missingness review
- Reference mapping tables for administrative codes

### `02_preprocessing_eda_and_feature_engineering.ipynb`

Implements preprocessing, EDA, and feature engineering:

- Missing-value handling for `?` markers and sparse columns
- Identifier and low-value column removal
- Binary target creation for 30-day readmission
- Univariate and bivariate EDA
- ICD-9 grouping and derived utilization features
- Medication and age encoding
- Stratified train/test split with `RANDOM_STATE = 42`
- StandardScaler on continuous features
- Mutual information feature selection from 170 to 117 features
- PCA reduction from 117 selected features to 44 components retaining 95.21% variance
- SMOTE oversampling for training-ready feature tracks
- Export of processed datasets and preprocessing artifacts

### `03_modeling_and_evaluation.ipynb`

Implements the complete modeling and evaluation workflow:

- Loads three modeling tracks: selected, scaled, and PCA
- Trains baseline Logistic Regression, Decision Tree, Random Forest, and XGBoost models
- Uses honest SMOTE-in-CV evaluation to avoid leakage
- Tunes the top ensemble models with `RandomizedSearchCV`
- Compares tuned selected-feature models against PCA-track models
- Selects the final model dynamically by test AUC-ROC
- Evaluates threshold strategies: Default, F2, Youden's J, and Constrained
- Computes SHAP explainability with PCA back-projection to the original clinical feature space
- Runs subgroup fairness analysis across race and gender
- Builds and exports a deployment-ready inference pipeline
- Writes figures, comparison tables, and model artifacts to the repository

### `04_final_summary_and_results.ipynb`

Consolidates the finished pipeline into a presentation-style summary:

- Executive summary and business framing
- Success criteria review
- Final model recommendation and threshold rationale
- Explainability highlights and fairness findings
- Reproducibility checklist and artifact manifest

## Machine Learning Workflow

The project follows a structured ML lifecycle:

1. Problem understanding and framing
2. Data collection and understanding
3. Data preprocessing and feature engineering
4. Model training and comparison
5. Threshold optimization for operational use
6. Explainability and fairness analysis
7. Result communication and artifact export
8. Optional deployment and MLOps extension

## Final Results

### Best Model

- Model: Random Forest - PCA
- Selection rule: highest test AUC-ROC across all baseline, tuned, and PCA-track candidates
- Input representation: 44 PCA components generated from 117 mutual-information-selected features
- Deployment threshold: 0.4556

### Model Comparison Snapshot

| Model | Test AUC | Test Recall | Test Precision | Test F1 |
|---|---:|---:|---:|---:|
| Random Forest - PCA | 0.6446 | 0.5425 | 0.1746 | 0.2642 |
| Random Forest (tuned) | 0.6224 | 0.2391 | 0.2104 | 0.2238 |
| Random Forest | 0.6218 | 0.0396 | 0.2507 | 0.0684 |
| XGBoost (tuned) | 0.6213 | 0.0493 | 0.2430 | 0.0820 |
| XGBoost | 0.6134 | 0.0414 | 0.2103 | 0.0692 |
| XGBoost - PCA | 0.6128 | 0.2122 | 0.2089 | 0.2106 |
| Decision Tree | 0.5874 | 0.2928 | 0.1730 | 0.2175 |
| Logistic Regression | 0.5715 | 0.1369 | 0.1660 | 0.1501 |

### Operating Threshold and Clinical Tradeoff

At the model default threshold of 0.50, the best model reached 0.5425 recall and 0.1746 precision. Because the project prioritizes recall, the threshold was optimized after model selection.

| Strategy | Threshold | Recall | Precision | F1 | F2 | Flagged % |
|---|---:|---:|---:|---:|---:|---:|
| Default (0.50) | 0.5000 | 0.5425 | 0.1746 | 0.2642 | 0.3816 | 35.3 |
| F2 Score (max) | 0.4166 | 0.8142 | 0.1399 | 0.2388 | 0.4146 | 66.0 |
| Youden's J | 0.4986 | 0.5491 | 0.1743 | 0.2646 | 0.3839 | 35.7 |
| Constrained (prec>=15%) | 0.4556 | 0.7164 | 0.1500 | 0.2481 | 0.4082 | 54.2 |

The selected operating point is the Constrained strategy because it maximizes recall while preserving a minimum precision floor of 15 percent. This provides a practical middle ground between Youden's J, which under-prioritizes recall for this use case, and the F2-max strategy, which flags too many patients for follow-up.

### Why the Final Model Is Defensible

- PCA reduced the selected feature space from 117 variables to 44 components while retaining 95.21 percent of variance.
- The PCA Random Forest delivered the strongest test AUC-ROC among all candidates.
- Threshold tuning increased recall from 0.5425 at the default threshold to 0.7164 at the selected operating point.
- The model is packaged in a reusable inference pipeline with preprocessing artifacts preserved.

### Performance Interpretation

The final AUC-ROC of 0.6446 falls short of the original 0.75 target, but the result is consistent with the difficulty of this dataset: strong class imbalance, older encounter data, and limited availability of richer temporal clinical signals. The project still achieves a clinically more useful operating point by optimizing recall, which is the more important downstream behavior for screening high-risk patients.

## Explainability and Fairness

### Explainability Highlights

The project uses SHAP to explain the final model. For PCA-based modeling, SHAP values are back-projected through the PCA loadings so the final explanations remain interpretable in the original feature space.

Top SHAP-ranked drivers include:

- `log_emergency`
- `number_emergency`
- `log_inpatient`
- `number_inpatient`
- `had_inpatient`
- `number_diagnoses`
- `number_outpatient`
- `discharge_disposition_id_home`
- `time_in_hospital`
- `had_emergency`

These features cluster into a few clinically meaningful themes:

- Utilization history: prior emergency and inpatient encounters are the strongest risk signals
- Care complexity: diagnosis burden, hospital time, and utilization intensity drive higher risk
- Discharge pathway: discharge home appears protective relative to more severe disposition patterns

This matters operationally because the strongest risk signals are available in structured EHR data early enough to support proactive intervention planning.

### Fairness Summary

Fairness was evaluated across race and gender subgroups using the optimized threshold.

Race subgroup recall:

| Group | N | Positive Rate | AUC-ROC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Caucasian | 15103 | 0.1143 | 0.6451 | 0.7389 | 0.1492 | 0.2482 |
| AfricanAmerican | 3712 | 0.1158 | 0.6396 | 0.6558 | 0.1573 | 0.2537 |
| Asian | 121 | 0.0992 | 0.6514 | 0.5833 | 0.1750 | 0.2692 |
| Other | 270 | 0.0926 | 0.5535 | 0.5200 | 0.0977 | 0.1646 |
| Unknown | 433 | 0.0762 | 0.6958 | 0.5455 | 0.1250 | 0.2034 |

Gender subgroup recall:

| Group | N | Positive Rate | AUC-ROC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Female | 10805 | 0.1155 | 0.6405 | 0.7228 | 0.1513 | 0.2503 |
| Male | 9213 | 0.1110 | 0.6491 | 0.7087 | 0.1484 | 0.2454 |

Key takeaway:

- Gender performance is relatively stable
- Race subgroup performance shows a wider recall gap, especially in smaller groups
- The fairness target was not met, so subgroup monitoring and broader data improvement remain important next steps

## Saved Artifacts

### Models

Files saved in `models/`:

| File | Description |
|---|---|
| `best_model_random_forest_pca.joblib` | Final selected Random Forest PCA model |
| `best_model_metadata.json` | Final metrics, threshold, and model metadata |
| `deployment_pipeline.joblib` | End-to-end inference pipeline with preprocessing and model |
| `selected_features.json` | Mutual-information-selected feature names |
| `standard_scaler.joblib` | Fitted StandardScaler artifact |
| `pca_transformer.joblib` | Fitted PCA transformer with 44 components |

### Processed Data and Preprocessing Artifacts

The repository includes modeling-ready processed datasets and reusable preprocessing artifacts in `data/processed/`, including:

- Cleaned and preprocessed source tables
- Train and test splits
- Scaled, selected, and PCA-transformed feature sets
- SMOTE-resampled training variants
- Saved scaler, PCA, and selected-feature artifacts

This repository state reflects a completed pipeline rather than a partial checkpoint.

### Reports

Files saved in `reports/figures/`:

- `fig_pca_variance_v1.png`
- `fig_baseline_roc_v1.png`
- `fig_baseline_confusion_v1.png`
- `fig_pca_comparison_v1.png`
- `fig_model_comparison_v1.png`
- `fig_threshold_analysis_v1.png`
- `fig_confusion_threshold_v1.png`
- `fig_shap_beeswarm_v1.png`
- `fig_shap_bar_v1.png`
- `fig_fairness_recall_v1.png`
- `fig_fairness_auc_v1.png`

Files saved in `reports/tables/`:

- `tbl_model_comparison_v1.csv`
- `tbl_threshold_strategies_v1.csv`
- `tbl_fairness_race_v1.csv`
- `tbl_fairness_gender_v1.csv`
- `tbl_shap_importance_v1.csv`

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetic-readmission-prediction.git
cd diabetic-readmission-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the raw dataset in `data/raw/`

Expected raw file:

```text
data/raw/diabetic_data.csv
```

### 5. Run the notebooks

Preferred workflow: open the notebooks in VS Code and run them in order.

Alternative:

```bash
jupyter lab
```

## Reproducibility

This project emphasizes reproducibility through:

- `RANDOM_STATE = 42` for train/test splitting, SMOTE, and model randomness
- Sequential notebooks covering the full pipeline
- Exported processed datasets and preprocessing artifacts
- Saved model and deployment pipeline artifacts in `models/`
- Saved figures and comparison tables in `reports/`
- `requirements.txt` for dependency capture

## Deliverables

Completed deliverables:

- Reproducible notebooks and reusable `src/` modules
- Finished preprocessing and feature-engineering pipeline
- Baseline and tuned model comparison
- Threshold optimization and model recommendation
- SHAP explainability summary
- Fairness audit across race and gender
- Deployment-ready pipeline artifact
- Exported figures and tables for reporting
- Final written report in `reports/final_report.md`

Remaining packaging work:

- Technical presentation slides
- Business-facing presentation slides

## Current Status

Project stage:

- [x] Problem selection
- [x] Data understanding
- [x] EDA
- [x] Preprocessing
- [x] Feature engineering
- [x] Model training
- [x] Hyperparameter tuning
- [x] PCA track comparison
- [x] Threshold optimization
- [x] Explainability (SHAP)
- [x] Fairness audit
- [x] Deployment-ready pipeline
- [x] Final summary notebook
- [x] Final report document
- [ ] Slide decks

## Limitations

The completed pipeline still has important limitations:

- Class imbalance makes pure discrimination difficult even after SMOTE-aware training and threshold tuning
- The dataset is historically bounded and may not represent current hospital processes
- The available features do not include richer temporal signals such as lab trends or medication progression
- Fairness disparity remains across some racial subgroups
- Generalizability outside the source hospitals and population remains uncertain

## Future Improvements

Potential extensions beyond the completed pipeline:

- Increase the hyperparameter search budget with Bayesian optimization
- Add probability calibration for better risk interpretation
- Evaluate additional model families such as LightGBM or CatBoost
- Introduce temporal validation using earlier versus later hospital years
- Explore fairness-aware training or threshold governance rules
- Wrap the deployment pipeline in a Flask or FastAPI scoring service
- Add experiment tracking and monitoring with MLflow or similar tooling

## Author

**Jose Fernando Gonzales**  
Machine Learning Capstone Project

## License

This project is for academic and educational purposes. Please review the original dataset source for dataset-specific licensing and usage restrictions.