# Predicting 30-Day Hospital Readmission for Diabetic Patients

A machine learning capstone project focused on predicting whether diabetic patients will be readmitted to the hospital within 30 days of discharge. This project applies the end-to-end machine learning lifecycle, including problem framing, data understanding, preprocessing, feature engineering, model training, evaluation, explainability, fairness analysis, and result communication.

## Project Overview

Hospital readmissions are costly for healthcare providers and may indicate gaps in post-discharge care. For diabetic patients, 30-day readmission is a particularly important outcome because it affects both patient well-being and hospital performance metrics.

This project frames the problem as a **supervised classification task**. The objective is to build a model that can identify patients at high risk of readmission within 30 days, using patient encounter, demographic, diagnostic, and treatment-related features.

## Problem Statement

Hospitals aim to reduce avoidable 30-day readmissions because they increase healthcare costs and may lead to financial penalties. This project uses machine learning to predict whether a diabetic patient will be readmitted within 30 days after discharge.

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

> Note: The dataset is not stored directly in this repository if license or size restrictions apply. Please download it from the source link above and place it in the appropriate `data/raw/` directory.

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
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_understanding_and_eda.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_modeling_and_evaluation.ipynb
│   └── 04_explainability_and_fairness.ipynb
│   └── 05_final_summary_and_results.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.pdf
├── slides/
│   ├── technical_presentation.pdf
│   └── business_presentation.pdf
└── app/
````

## Notebook Guide

### `01_data_understanding_and_eda.ipynb`

Covers:

* problem framing
* dataset loading
* dataset overview
* missing values
* duplicate checks
* class distribution
* initial visual analysis
* early insights

### `02_preprocessing_and_feature_engineering.ipynb`

Covers:

* null handling
* outlier treatment
* encoding categorical variables
* scaling
* derived features
* feature selection
* dimensionality reduction if needed

### `03_modeling_and_evaluation.ipynb`

Covers:

* baseline model implementation
* model comparison
* hyperparameter tuning
* evaluation using classification metrics
* final model selection

### `04_explainability_and_fairness.ipynb`

Covers:

* SHAP / LIME analysis
* feature importance interpretation
* fairness checks across selected groups
* limitations and mitigation discussion

## Machine Learning Workflow

The project follows a structured ML lifecycle:

1. **Problem Understanding and Framing**
2. **Data Collection and Understanding**
3. **Data Preprocessing and Feature Engineering**
4. **Model Training and Comparison**
5. **Explainability, Bias, and Fairness Analysis**
6. **Communication through reports and presentations**
7. **Optional deployment and MLOps extension**

## Models to Be Explored

Possible models include:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Support Vector Machine
* Other suitable classifiers

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

This project includes a critical analysis of model behavior and ethical implications.

Planned methods include:

* SHAP
* LIME
* model-based feature importance
* bias checks across available sensitive or demographic groups
* fairness metrics such as demographic parity or equalized odds where applicable

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

## Reproducibility

To support reproducibility, this project aims to include:

* fixed random seeds where applicable
* saved preprocessing artifacts
* saved trained models
* clear notebook ordering
* requirements file for package versions

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
* [ ] Data understanding
* [ ] EDA
* [ ] Preprocessing
* [ ] Feature engineering
* [ ] Model training
* [ ] Explainability
* [ ] Fairness audit
* [ ] Final report
* [ ] Slide decks
* [ ] Deployment

## Limitations

Potential limitations of the project may include:

* class imbalance
* missing or noisy healthcare data
* limited interpretability for some complex models
* possible dataset bias
* reduced generalizability outside the source hospitals or population

These will be explicitly discussed in the report and fairness section.

## Future Improvements

Possible future enhancements:

* deploy the best model using Flask or FastAPI
* add experiment tracking with MLflow
* package preprocessing and inference into a pipeline
* include threshold optimization for hospital intervention scenarios
* add monitoring and retraining strategy

## Author

**Jose Fernando Gonzales**
Machine Learning Capstone Project

## License

This project is for academic and educational purposes.
Please check the original dataset source for dataset-specific licensing and usage restrictions.