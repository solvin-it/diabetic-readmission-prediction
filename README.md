# Predicting 30-Day Hospital Readmission for Diabetic Patients

A machine learning capstone project focused on predicting whether diabetic patients will be readmitted to the hospital within 30 days of discharge. This project applies the end-to-end machine learning lifecycle, including problem framing, data understanding, preprocessing, feature engineering, model training, evaluation, explainability, fairness analysis, and result communication.

## Project Overview

Hospital readmissions are costly for healthcare providers and may indicate gaps in post-discharge care. For diabetic patients, 30-day readmission is a particularly important outcome because it affects both patient well-being and hospital performance metrics.

This project frames the problem as a **supervised classification task**. The objective is to build a model that can identify patients at high risk of readmission within 30 days, using patient encounter, demographic, diagnostic, and treatment-related features.

## Problem Statement

Hospitals aim to reduce avoidable 30-day readmissions because they increase healthcare costs and is an indicator of quality of care. This project uses machine learning to predict whether a diabetic patient will be readmitted within 30 days after discharge.

The project is primarily treated as a **binary classification problem**, where the target is whether a patient is readmitted within 30 days or not. Success is evaluated using technical metrics such as **AUC-ROC, recall, precision, F1-score, and confusion matrix analysis**, with emphasis on recall because failing to identify a high-risk patient may be more costly than a false positive.

### Business Goal
Support early identification of high-risk diabetic patients so hospitals can improve intervention planning, discharge support, and resource allocation.

### Machine Learning Goal
Develop and compare classification models that can accurately predict 30-day readmission risk.

## Dataset

**Dataset:** Diabetic Patients’ Re-admission Prediction  
**Source:** Kaggle  
**Link:** `https://www.kaggle.com/datasets/saurabhtayal/diabetic-patients-readmission-prediction`

The dataset contains approximately 10 years of hospital encounter data for diabetic patients across multiple hospitals. It includes demographic variables, admission details, diagnoses, medications, and laboratory-related indicators.

## Project Objectives

- Understand the business and healthcare context of hospital readmissions
- Explore and clean the diabetic patient dataset
- Perform applied exploratory data analysis
- Engineer and select relevant features
- Train and compare multiple machine learning models
- Interpret model predictions using explainability tools
- Evaluate fairness and bias across sensitive groups
- Present results in both technical and business-friendly formats

## Repository Structure

```text
diabetic-readmission-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        ← original dataset (not versioned)
│   └── processed/                  ← see Data Policy below
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_eda_and_feature_engineering.ipynb
│   └── 03_modeling_and_evaluation.ipynb
│   └── 04_final_summary_and_results.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── feature_engineering.py
├── models/                         ← trained model artifacts & pipeline
├── reports/
└── references/                     ← capstone guides (not versioned)
```

## Notebook Guide

### `01_data_understanding.ipynb`

Implements the **Data Understanding** phase:

* Dataset background (Diabetes 130-US Hospitals, 1999–2008)
* Problem framing as binary classification (readmitted within 30 days vs. not)
* Dataset loading, `df.info()`, descriptive statistics
* Data dictionary (50 columns with types, descriptions, and missingness)
* ID mapping reference tables (admission type, discharge disposition, admission source)

### `02_preprocessing_eda_and_feature_engineering.ipynb`

Implements **Preprocessing, EDA & Feature Engineering**:

* Data cleaning — missing value handling (`?` markers), constant/identifier column removal, binary target creation
* EDA — univariate (target, numerical, categorical), bivariate, and correlation analysis
* 10-step feature engineering pipeline:
  1. ICD-9 diagnosis code grouping (19 clinical chapters)
  2. Utilization features + administrative recoding
  3. Age one-hot encoding (10 decade bands)
  4. Medication ordinal encoding (No / Steady / Adjusted)
  5. Remaining categorical one-hot encoding
  6. Stratified 80/20 train/test split
  7. StandardScaler (11 continuous features, fitted on train only)
  8. Mutual information feature selection (170 → 117 features)
  9. PCA dimensionality reduction (117 → 44 components, 95.21% variance) with component loadings interpretation
  10. SMOTE oversampling (training data only, 3 parallel tracks)
* Feature engineering summary with artifact inventory

### `03_modeling_and_evaluation.ipynb`

Implements **Modeling & Evaluation** (all-in-one):

* Baseline models — Logistic Regression, Decision Tree, Random Forest, XGBoost with SMOTE-in-CV evaluation
* Hyperparameter tuning — RandomizedSearchCV with imblearn Pipeline (SMOTE per fold)
* PCA track comparison — tuned models evaluated on 44-component PCA features
* Best model selection & threshold tuning — dynamic selection across all tracks, clinical threshold optimisation (F2, Youden's J, precision-constrained)
* SHAP explainability — TreeExplainer with PCA-to-original-feature back-projection
* Fairness analysis — performance disparity across race and gender subgroups
* Final model summary — consolidated comparison table and model saving
* Deployment-ready pipeline — sklearn Pipeline with clinical threshold

## Machine Learning Workflow

The project follows a structured ML lifecycle:

1. **Problem Understanding and Framing**
2. **Data Collection and Understanding**
3. **Data Preprocessing and Feature Engineering**
4. **Model Training and Comparison**
5. **Explainability, Bias, and Fairness Analysis**
6. **Communication through reports and presentations**
7. **Optional deployment and MLOps extension**

## Models Trained

The following models were trained and compared across two feature tracks (117 MI-selected features and 44 PCA components):

* Logistic Regression (baseline)
* Decision Tree (baseline)
* Random Forest (baseline + tuned + PCA)
* XGBoost (baseline + tuned + PCA)

**Best model:** Random Forest — PCA (AUC-ROC ≈ 0.645, Recall ≈ 0.54 at optimised threshold)

## Evaluation Metrics

Because the problem involves possible class imbalance, evaluation will not rely only on accuracy.

Primary and secondary metrics include:

* AUC-ROC
* Recall
* Precision
* F1-score
* Confusion Matrix

### Why recall matters

In this healthcare setting, missing a patient who is actually at high risk of readmission may be more harmful than incorrectly flagging a lower-risk patient.

## Explainability and Fairness

This project includes a critical analysis of model behaviour and ethical implications, integrated into Notebook 03:

* **SHAP** — TreeExplainer computes per-feature contributions; for the PCA model, SHAP values are back-projected to the original 117 features via `shap_values @ pca.components_`, restoring clinical interpretability
* **PCA component interpretation** — Loadings heatmap and clinical labels for the top 10 components (documented in Notebook 02)
* **Fairness analysis** — AUC-ROC, recall, precision, and F1 evaluated separately for each race and gender subgroup using the optimised threshold

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetic-readmission-prediction.git
cd diabetic-readmission-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from Kaggle and place the file(s) inside:

```text
data/raw/
```

### 5. Run notebooks

Start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

## Data Policy

Raw data is not versioned. Download the dataset from Kaggle and place it at `data/raw/diabetic_data.csv`.

Only the core processed splits are committed to the repository — these are the minimal artifacts needed to run modeling without re-running preprocessing:

| File | Versioned |
|---|---|
| `X_train.csv`, `X_test.csv` | ✓ |
| `X_train_scaled.csv`, `X_test_scaled.csv` | ✓ |
| `y_train.csv`, `y_test.csv` | ✓ |
| Resampled, selected, PCA variants | ✗ — regenerate by running notebook `02` |
| `diabetic_data_cleaned.csv`, `diabetic_data_preprocessed.csv` | ✗ — regenerate by running notebook `02` |

## Reproducibility

To support reproducibility, this project uses:

* `RANDOM_STATE = 42` for all random seeds
* saved core preprocessing artifacts (see Data Policy above)
* saved trained models under `models/`
* sequentially numbered notebooks
* `requirements.txt` for package versions

## Deliverables

This repository is intended to contain the following capstone deliverables:

* reproducible code
* notebooks
* trained model artifacts
* final report
* technical presentation
* business-facing presentation
* optional deployment demo

## Current Status

Project stage:

* [x] Problem selection
* [x] Data understanding
* [x] EDA
* [x] Preprocessing
* [x] Feature engineering
* [x] Model training
* [x] Hyperparameter tuning
* [x] PCA track comparison
* [x] Threshold optimisation
* [x] Explainability (SHAP)
* [x] Fairness audit
* [x] Deployment-ready pipeline
* [ ] Final report
* [ ] Slide decks

## Limitations

Potential limitations of the project may include:

* class imbalance
* missing or noisy healthcare data
* limited interpretability for some complex models
* possible dataset bias
* reduced generalizability outside the source hospitals or population

These will be explicitly discussed in the report and fairness section.

## Saved Artifacts

Trained model artifacts in `models/`:

| File | Description |
|---|---|
| `best_model_random_forest_pca.joblib` | Best model: Random Forest — PCA (standalone) |
| `best_model_metadata.json` | Model metadata & optimal threshold |
| `deployment_pipeline.joblib` | Complete inference pipeline (FeatureSelector → Scaler → PCA → Model) + threshold |
| `selected_features.json` | MI-selected feature names (117) |
| `standard_scaler.joblib` | Fitted StandardScaler + continuous column list |
| `pca_transformer.joblib` | Fitted PCA transformer (44 components) |

## Future Improvements

Possible future enhancements:

* deploy the best model using Flask or FastAPI
* add experiment tracking with MLflow
* add monitoring and retraining strategy

## Author

**Jose Fernando Gonzales**
Machine Learning Capstone Project

## License

This project is for academic and educational purposes.
Please check the original dataset source for dataset-specific licensing and usage restrictions.