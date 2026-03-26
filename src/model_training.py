"""
Reusable model training, evaluation, and deployment functions for
the diabetic 30-day readmission prediction project.

Extracted from notebooks/03_modeling_and_evaluation.ipynb to enable
reuse in the final summary notebook and future experimentation.
"""

import re
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, precision_recall_curve, fbeta_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise column names — replace ``[``, ``]``, ``<`` with ``_``.

    XGBoost forbids these characters in feature names.  Applies in-place and
    returns the same DataFrame for chaining.
    """
    df.columns = [re.sub(r"[\[\]<]", "_", c) for c in df.columns]
    return df


def load_datasets(
    processed_dir: pathlib.Path,
) -> Dict[str, Any]:
    """Load all training/test splits and preprocessing artifacts.

    Returns a dict with keys:
      X_train_sel, X_train_sc, X_train_pca, y_train,
      X_train_sel_raw, X_train_sc_raw, X_train_pca_raw, y_train_raw,
      X_test_sel, X_test_sc, X_test_pca, y_test,
      pca_transformer, scaler_artifact, mi_selected_features.
    """
    data: Dict[str, Any] = {}

    # SMOTE-balanced
    data["X_train_sel"] = pd.read_csv(processed_dir / "X_train_selected_resampled.csv")
    data["X_train_sc"]  = pd.read_csv(processed_dir / "X_train_scaled_selected_resampled.csv")
    data["X_train_pca"] = pd.read_csv(processed_dir / "X_train_pca_resampled.csv")
    data["y_train"]     = pd.read_csv(processed_dir / "y_train_resampled.csv").squeeze()

    # Pre-SMOTE (for honest CV)
    data["X_train_sel_raw"] = pd.read_csv(processed_dir / "X_train_selected.csv")
    data["X_train_sc_raw"]  = pd.read_csv(processed_dir / "X_train_scaled_selected.csv")
    data["X_train_pca_raw"] = pd.read_csv(processed_dir / "X_train_pca.csv")
    data["y_train_raw"]     = pd.read_csv(processed_dir / "y_train.csv").squeeze()

    # Test
    data["X_test_sel"] = pd.read_csv(processed_dir / "X_test_selected.csv")
    data["X_test_sc"]  = pd.read_csv(processed_dir / "X_test_scaled_selected.csv")
    data["X_test_pca"] = pd.read_csv(processed_dir / "X_test_pca.csv")
    data["y_test"]     = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # Preprocessing objects
    data["pca_transformer"]     = joblib.load(processed_dir / "pca_transformer.joblib")
    data["scaler_artifact"]     = joblib.load(processed_dir / "standard_scaler.joblib")
    with open(processed_dir / "selected_features.json") as f:
        data["mi_selected_features"] = json.load(f)

    # Sanitise column names
    for key in [
        "X_train_sel", "X_test_sel", "X_train_sc", "X_test_sc",
        "X_train_pca", "X_test_pca",
        "X_train_sel_raw", "X_train_sc_raw", "X_train_pca_raw",
    ]:
        clean_column_names(data[key])

    data["mi_selected_features"] = [
        re.sub(r"[\[\]<]", "_", c) for c in data["mi_selected_features"]
    ]

    print("✓ Datasets loaded")
    return data


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model: Any,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
) -> Tuple[Dict[str, Any], Any]:
    """Train a single model with honest SMOTE-in-CV evaluation.

    Parameters
    ----------
    X_cv, y_cv : Pre-SMOTE training data (for cross-validation).
    X_fit, y_fit : SMOTE-balanced training data (for final model fit).
    X_test, y_test : Held-out test data.
    cv : Number of CV folds.

    Returns
    -------
    metrics : dict with CV and test-set performance.
    fitted_model : The model fitted on the full SMOTE-balanced data.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    imb_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model),
    ])

    cv_results = cross_validate(
        imb_pipe, X_cv, y_cv, cv=skf,
        scoring=["roc_auc", "recall", "precision", "f1"],
        return_train_score=False, n_jobs=-1,
    )

    model.fit(X_fit, y_fit)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "CV AUC (mean)":    cv_results["test_roc_auc"].mean(),
        "CV AUC (std)":     cv_results["test_roc_auc"].std(),
        "CV Recall (mean)": cv_results["test_recall"].mean(),
        "Test AUC":         roc_auc_score(y_test, y_prob),
        "Test Recall":      recall_score(y_test, y_pred),
        "Test Precision":   precision_score(y_test, y_pred),
        "Test F1":          f1_score(y_test, y_pred),
    }
    return metrics, model


def train_baseline_models(
    data: Dict[str, Any],
    cv: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[Any, pd.DataFrame, pd.DataFrame]]]:
    """Train four baseline models and return results + fitted model registry.

    Returns
    -------
    results_df : DataFrame indexed by model name with CV and test metrics.
    trained_models : dict mapping name → (fitted_model, X_cv, X_test).
    """
    configs = [
        ("Logistic Regression", LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE,
        ), data["X_train_sc_raw"], data["X_train_sc"], data["X_test_sc"]),
        ("Decision Tree", DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE,
        ), data["X_train_sel_raw"], data["X_train_sel"], data["X_test_sel"]),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE,
        ), data["X_train_sel_raw"], data["X_train_sel"], data["X_test_sel"]),
        ("XGBoost", XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ), data["X_train_sel_raw"], data["X_train_sel"], data["X_test_sel"]),
    ]

    results: List[Dict] = []
    trained_models: Dict[str, Tuple] = {}

    for name, model, X_cv, X_fit, X_te in configs:
        print(f"Training {name}...")
        metrics, fitted = evaluate_model(
            name, model,
            X_cv, data["y_train_raw"],
            X_fit, data["y_train"],
            X_te, data["y_test"],
            cv=cv,
        )
        results.append(metrics)
        trained_models[name] = (fitted, X_cv, X_te)
        print(f"  CV AUC={metrics['CV AUC (mean)']:.4f}±{metrics['CV AUC (std)']:.4f}  "
              f"Test AUC={metrics['Test AUC']:.4f}  Recall={metrics['Test Recall']:.4f}")

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n✓ Baseline training complete (SMOTE applied per CV fold)")
    return results_df, trained_models


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_models(
    data: Dict[str, Any],
    n_iter: int = 20,
    cv: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[Any, pd.DataFrame]], Dict[str, Dict]]:
    """Tune Random Forest and XGBoost via RandomizedSearchCV.

    Returns
    -------
    tuned_df : DataFrame with tuned model test metrics.
    tuned_models : dict mapping name → (fitted_model, X_test).
    best_params : dict mapping name → best hyperparameters.
    """
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    X_cv = data["X_train_sel_raw"]
    y_cv = data["y_train_raw"]

    # --- Random Forest ---
    print(f"Tuning Random Forest ({n_iter} iterations, {cv}-fold CV, SMOTE per fold)...")
    rf_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE)),
    ])
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions={
            "model__n_estimators": randint(100, 500),
            "model__max_depth": [6, 8, 10, 12, 15, None],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        },
        n_iter=n_iter, cv=skf,
        scoring="roc_auc", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=1, refit=True,
    )
    rf_search.fit(X_cv, y_cv)
    rf_best_model = rf_search.best_estimator_.named_steps["model"]
    rf_best_params = {
        k.replace("model__", ""): v
        for k, v in rf_search.best_params_.items()
        if k.startswith("model__")
    }
    print(f"  Best RF  CV AUC: {rf_search.best_score_:.4f}")
    print(f"  Best params    : {rf_best_params}")

    # --- XGBoost ---
    print(f"\nTuning XGBoost ({n_iter} iterations, {cv}-fold CV, SMOTE per fold)...")
    xgb_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])
    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions={
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": uniform(0.01, 0.2),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__min_child_weight": randint(1, 10),
            "model__gamma": uniform(0, 0.5),
        },
        n_iter=n_iter, cv=skf,
        scoring="roc_auc", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=1, refit=True,
    )
    xgb_search.fit(X_cv, y_cv)
    xgb_best_model = xgb_search.best_estimator_.named_steps["model"]
    xgb_best_params = {
        k.replace("model__", ""): v
        for k, v in xgb_search.best_params_.items()
        if k.startswith("model__")
    }
    print(f"  Best XGB CV AUC: {xgb_search.best_score_:.4f}")
    print(f"  Best params    : {xgb_best_params}")

    # Test-set evaluation
    tuned_models = {
        "Random Forest (tuned)": (rf_best_model, data["X_test_sel"]),
        "XGBoost (tuned)":       (xgb_best_model, data["X_test_sel"]),
    }
    tuned_results = []
    for name, (model, X_te) in tuned_models.items():
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        tuned_results.append({
            "Model": name,
            "CV AUC": rf_search.best_score_ if "Random" in name else xgb_search.best_score_,
            "Test AUC":       roc_auc_score(data["y_test"], y_prob),
            "Test Recall":    recall_score(data["y_test"], y_pred),
            "Test Precision": precision_score(data["y_test"], y_pred),
            "Test F1":        f1_score(data["y_test"], y_pred),
        })

    tuned_df = pd.DataFrame(tuned_results).set_index("Model")
    best_params = {
        "Random Forest": rf_best_params,
        "XGBoost": xgb_best_params,
    }
    print("\n✓ Tuned model results:")
    print(tuned_df.round(4).to_string())
    return tuned_df, tuned_models, best_params


# ---------------------------------------------------------------------------
# PCA track
# ---------------------------------------------------------------------------

def run_pca_modeling(
    data: Dict[str, Any],
    best_params: Dict[str, Dict],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train tuned RF and XGB on 44-component PCA features.

    Returns
    -------
    pca_df : DataFrame with PCA-track test metrics.
    trained_pca_models : dict mapping name → fitted model.
    """
    print("Training tuned models on PCA track (44 components)...")
    configs = [
        ("Random Forest (PCA)", RandomForestClassifier(
            **best_params["Random Forest"],
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
        ("XGBoost (PCA)", XGBClassifier(
            **best_params["XGBoost"],
            eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]

    results: List[Dict] = []
    trained_pca_models: Dict[str, Any] = {}
    for name, model in configs:
        model.fit(data["X_train_pca"], data["y_train"])
        y_pred = model.predict(data["X_test_pca"])
        y_prob = model.predict_proba(data["X_test_pca"])[:, 1]
        trained_pca_models[name] = model
        results.append({
            "Model": name,
            "Test AUC":       roc_auc_score(data["y_test"], y_prob),
            "Test Recall":    recall_score(data["y_test"], y_pred),
            "Test Precision": precision_score(data["y_test"], y_pred),
            "Test F1":        f1_score(data["y_test"], y_pred),
        })
        print(f"  {name}: AUC={results[-1]['Test AUC']:.4f}  "
              f"Recall={results[-1]['Test Recall']:.4f}")

    pca_df = pd.DataFrame(results).set_index("Model")
    return pca_df, trained_pca_models


# ---------------------------------------------------------------------------
# Model selection and threshold tuning
# ---------------------------------------------------------------------------

def select_best_model(
    results_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    trained_models: Dict[str, Tuple],
    tuned_models: Dict[str, Tuple],
    trained_pca_models: Dict[str, Any],
    X_test_sel: pd.DataFrame,
    X_test_pca: pd.DataFrame,
) -> Dict[str, Any]:
    """Consolidate results and select the best model by Test AUC.

    Returns
    -------
    dict with keys: all_results, best_name, best_model, best_X_te, is_pca_best.
    """
    all_results = pd.concat([
        results_df[["Test AUC", "Test Recall", "Test Precision", "Test F1"]],
        tuned_df[["Test AUC", "Test Recall", "Test Precision", "Test F1"]],
        pca_df.rename(index=lambda x: x.replace("(PCA)", "\u2014 PCA")),
    ]).sort_values("Test AUC", ascending=False)

    best_name = all_results["Test AUC"].idxmax()
    is_pca_best = "PCA" in best_name

    if is_pca_best:
        pca_key = best_name.replace("\u2014 PCA", "(PCA)")
        best_model = trained_pca_models[pca_key]
        best_X_te = X_test_pca
    else:
        if best_name in tuned_models:
            best_model = tuned_models[best_name][0]
        else:
            best_model = trained_models[best_name][0]
        best_X_te = X_test_sel

    print(f"✓ Best model by AUC: {best_name}")
    return {
        "all_results": all_results,
        "best_name": best_name,
        "best_model": best_model,
        "best_X_te": best_X_te,
        "is_pca_best": is_pca_best,
    }


def optimize_threshold(
    y_test: pd.Series,
    y_prob: np.ndarray,
    min_precision: float = 0.15,
) -> Dict[str, Any]:
    """Evaluate three threshold-selection strategies.

    Returns
    -------
    dict with keys: strategy_df, optimal_threshold, thresholds (dict per strategy),
    and intermediate curve data for plotting.
    """
    # F2
    precisions_pr, recalls_pr, thresholds_pr = precision_recall_curve(y_test, y_prob)
    f2_scores = (5 * precisions_pr * recalls_pr) / (4 * precisions_pr + recalls_pr + 1e-9)
    best_f2_idx = f2_scores[:-1].argmax()
    threshold_f2 = float(thresholds_pr[best_f2_idx])

    # Youden's J
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_prob)
    j_scores = tpr_roc - fpr_roc
    best_j_idx = j_scores.argmax()
    threshold_j = float(thresholds_roc[best_j_idx])

    # Constrained
    valid_mask = precisions_pr[:-1] >= min_precision
    if valid_mask.any():
        constrained_recalls = np.where(valid_mask, recalls_pr[:-1], 0)
        best_c_idx = constrained_recalls.argmax()
        threshold_constrained = float(thresholds_pr[best_c_idx])
    else:
        threshold_constrained = 0.50

    strategies = [
        ("Default (0.50)", 0.50),
        ("F2 Score (max)",  threshold_f2),
        ("Youden's J",     threshold_j),
        (f"Constrained (prec\u2265{min_precision:.0%})", threshold_constrained),
    ]

    rows = []
    for strat_name, t in strategies:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "Strategy": strat_name,
            "Threshold": round(t, 4),
            "Recall":    round(recall_score(y_test, y_pred_t), 4),
            "Precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred_t, zero_division=0), 4),
            "F2":        round(fbeta_score(y_test, y_pred_t, beta=2, zero_division=0), 4),
            "Flagged %": round(float(y_pred_t.mean()) * 100, 1),
        })

    strategy_df = pd.DataFrame(rows).set_index("Strategy")
    optimal_threshold = threshold_constrained

    print(f"✓ Recommended threshold: {optimal_threshold:.4f} (Constrained: max recall, prec≥{min_precision:.0%})")
    return {
        "strategy_df": strategy_df,
        "optimal_threshold": optimal_threshold,
        "thresholds": {
            "f2": threshold_f2,
            "youdens_j": threshold_j,
            "constrained": threshold_constrained,
        },
        # Curve data for plotting
        "pr_curve": (precisions_pr, recalls_pr, thresholds_pr, f2_scores),
        "roc_curve": (fpr_roc, tpr_roc, thresholds_roc, j_scores, best_j_idx),
    }


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap_values(
    best_model: Any,
    best_X_te: pd.DataFrame,
    is_pca_best: bool,
    pca_transformer: Any,
    mi_selected_features: List[str],
    X_test_sc: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute SHAP values and back-project from PCA if needed.

    Returns
    -------
    dict with keys: shap_original, feature_names, X_display, mean_abs_shap.
    """
    print(f"Computing SHAP values...")
    explainer = shap.TreeExplainer(best_model)
    shap_values_raw = explainer.shap_values(best_X_te)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    if is_pca_best:
        shap_original = shap_values @ pca_transformer.components_
        feature_names = mi_selected_features
        X_display = pd.DataFrame(
            X_test_sc.values,
            columns=mi_selected_features,
        )
        print(f"✓ SHAP back-projected to original features: {shap_original.shape}")
    else:
        shap_original = shap_values
        feature_names = list(best_X_te.columns)
        X_display = best_X_te

    mean_abs_shap = pd.Series(
        data=abs(shap_original).mean(axis=0),
        index=feature_names,
        name="Mean |SHAP|",
    ).sort_values(ascending=False)

    print(f"✓ SHAP values computed: {shap_original.shape}")
    return {
        "shap_original": shap_original,
        "feature_names": feature_names,
        "X_display": X_display,
        "mean_abs_shap": mean_abs_shap,
    }


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------

def compute_fairness_slices(
    group_prefix: str,
    X_test_df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    min_group_size: int = 50,
) -> pd.DataFrame:
    """Evaluate model performance for each one-hot demographic subgroup.

    Parameters
    ----------
    group_prefix : Column prefix to filter (e.g. ``"race_"``, ``"gender_"``).
    X_test_df : DataFrame with one-hot demographic columns (non-PCA test set).
    y_true : True labels.
    y_prob : Predicted probabilities for the positive class.
    y_pred : Binary predictions.
    min_group_size : Minimum samples required for a subgroup to be included.

    Returns
    -------
    DataFrame indexed by subgroup name with AUC, Recall, Precision, F1.
    """
    cols = [c for c in X_test_df.columns if c.startswith(group_prefix)]
    rows = []
    for col in cols:
        mask = X_test_df[col] == 1
        if mask.sum() < min_group_size:
            continue
        group_name = col.replace(group_prefix, "")
        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_prob[mask]
        rows.append({
            "Group": group_name,
            "N": int(mask.sum()),
            "Positive Rate": round(float(yt.mean()), 4),
            "AUC-ROC":   round(roc_auc_score(yt, yprob), 4) if yt.nunique() > 1 else float("nan"),
            "Recall":    round(recall_score(yt, yp, zero_division=0), 4),
            "Precision": round(precision_score(yt, yp, zero_division=0), 4),
            "F1":        round(f1_score(yt, yp, zero_division=0), 4),
        })
    return pd.DataFrame(rows).set_index("Group")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_baseline_roc(
    trained_models: Dict[str, Tuple],
    y_test: pd.Series,
    results_df: pd.DataFrame,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """ROC curves + AUC bar chart for baseline models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["steelblue", "darkorange", "forestgreen", "firebrick"]

    for (name, (model, _, X_te)), color in zip(trained_models.items(), colors):
        y_prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} (AUC = {roc_auc_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves \u2014 Baseline Models")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    models_sorted = results_df["Test AUC"].sort_values()
    axes[1].barh(models_sorted.index, models_sorted.values,
                 color=["steelblue"] * len(models_sorted))
    axes[1].set_xlabel("Test AUC-ROC")
    axes[1].set_title("Test AUC-ROC by Model")
    axes[1].axvline(x=0.75, color="red", linestyle="--", alpha=0.5, label="Target: 0.75")
    for i, v in enumerate(models_sorted.values):
        axes[1].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    axes[1].legend()
    axes[1].set_xlim([0.5, 1.0])

    plt.tight_layout()
    plt.suptitle("Baseline Model Performance", y=1.01, fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_baseline_confusion_matrices(
    trained_models: Dict[str, Tuple],
    y_test: pd.Series,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Confusion matrices for all baseline models."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (name, (model, _, X_te)) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_te)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax,
            cmap="Blues", colorbar=False, display_labels=["No Readmit", "Readmit"],
        )
        ax.set_title(name, fontsize=10)
    plt.suptitle("Confusion Matrices \u2014 Baseline Models (Test Set)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pca_comparison(
    tuned_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Side-by-side bar chart: PCA vs selected-feature track."""
    compare_rows = []
    for base, pca in [("Random Forest (tuned)", "Random Forest (PCA)"),
                      ("XGBoost (tuned)", "XGBoost (PCA)")]:
        for metric in ["Test AUC", "Test Recall", "Test F1"]:
            compare_rows.append({
                "Model": base.replace(" (tuned)", ""),
                "Track": "Selected (117)", "Metric": metric,
                "Score": tuned_df.loc[base, metric],
            })
            compare_rows.append({
                "Model": base.replace(" (tuned)", ""),
                "Track": "PCA (44)", "Metric": metric,
                "Score": pca_df.loc[pca, metric],
            })
    compare_df = pd.DataFrame(compare_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(["Test AUC", "Test Recall", "Test F1"]):
        subset = compare_df[compare_df["Metric"] == metric]
        ax = axes[i]
        for j, (model_name, grp) in enumerate(subset.groupby("Model")):
            x = [j - 0.15, j + 0.15]
            bars = ax.bar(
                x, grp.set_index("Track").loc[["Selected (117)", "PCA (44)"], "Score"],
                width=0.28,
                label=["Selected (117)", "PCA (44)"] if (i == 0 and j == 0) else ["", ""],
                color=["steelblue", "coral"],
            )
        ax.set_title(metric)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Random Forest", "XGBoost"])
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend()

    plt.suptitle("PCA (44) vs Selected-Feature (117) Track Comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pca_variance(
    pca_transformer,
    target_variance: float = 0.95,
    save_path: Optional[pathlib.Path] = None,
) -> Tuple[plt.Figure, int, float]:
    """Plot per-component and cumulative variance explained by PCA.

    Left subplot: per-component bar chart, highlighting components retained
    (steelblue) vs discarded (lightgray), with a red dashed vertical cutoff.
    Right subplot: cumulative variance curve with a red dashed horizontal
    target line and a green dashed vertical line at the crossover component.

    Returns
    -------
    fig : matplotlib Figure
    n_at_target : int — number of components reaching ``target_variance``
    variance_at_target : float — actual cumulative variance at that component
    """
    variance_ratio = pca_transformer.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    n_components_total = len(variance_ratio)
    n_at_target = int(np.argmax(cumulative_variance >= target_variance) + 1)
    variance_at_target = float(cumulative_variance[n_at_target - 1])

    bar_colors = [
        "steelblue" if i < n_at_target else "lightgray"
        for i in range(n_components_total)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: per-component variance ──────────────────────────────
    axes[0].bar(range(1, n_components_total + 1), variance_ratio,
                color=bar_colors, edgecolor="none", width=1.0)
    axes[0].axvline(x=n_at_target + 0.5, color="red", linestyle="--", linewidth=1.5,
                    label=f"Cutoff: {n_at_target} components")
    axes[0].set(
        xlabel="Principal Component",
        ylabel="Individual Variance Explained",
        title="Variance Explained by Each Principal Component",
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # ── Right: cumulative variance ─────────────────────────────────
    x_vals = range(1, n_components_total + 1)
    axes[1].plot(x_vals, cumulative_variance, color="steelblue", linewidth=2,
                 label="Cumulative variance")
    axes[1].axhline(y=target_variance, color="red", linestyle="--", linewidth=1.5,
                    label=f"{target_variance:.0%} target")
    axes[1].axvline(x=n_at_target, color="green", linestyle="--", linewidth=1.5,
                    label=f"n={n_at_target} components")
    axes[1].plot(n_at_target, variance_at_target, "r*", markersize=14, zorder=5,
                 label=f"{variance_at_target:.2%} variance retained")
    axes[1].annotate(
        f"  {n_at_target} components\n  {variance_at_target:.2%} variance",
        xy=(n_at_target, variance_at_target),
        xytext=(n_at_target + n_components_total * 0.05, variance_at_target - 0.07),
        fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
    )
    axes[1].set(
        xlabel="Number of Components",
        ylabel="Cumulative Variance Explained",
        title="Cumulative Variance Explained vs. Number of Components",
        ylim=[0, 1.05],
    )
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.3)

    pct_reduction = (1 - n_at_target / n_components_total) * 100
    plt.suptitle(
        f"PCA Dimensionality Reduction — {n_at_target} of {n_components_total} components "
        f"retained ({pct_reduction:.1f}% reduction, {variance_at_target:.2%} variance)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, n_at_target, variance_at_target


def plot_threshold_analysis(
    best_name: str,
    threshold_result: Dict[str, Any],
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Precision-Recall-F2 curves and ROC with threshold operating points."""
    precisions_pr, recalls_pr, thresholds_pr, f2_scores = threshold_result["pr_curve"]
    fpr_roc, tpr_roc, thresholds_roc, j_scores, best_j_idx = threshold_result["roc_curve"]
    thresholds = threshold_result["thresholds"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(thresholds_pr, precisions_pr[:-1], "b-", label="Precision", alpha=0.8)
    axes[0].plot(thresholds_pr, recalls_pr[:-1], "r-", label="Recall", alpha=0.8)
    axes[0].plot(thresholds_pr, f2_scores[:-1], "g-", label="F2 Score", alpha=0.8)
    for label, t, color in [
        ("F2",          thresholds["f2"],           "purple"),
        ("Youden's J",  thresholds["youdens_j"],    "gray"),
        ("Constrained", thresholds["constrained"],  "darkorange"),
    ]:
        axes[0].axvline(x=t, linestyle="--", color=color, alpha=0.7,
                        label=f"{label} = {t:.3f}")
    axes[0].set_xlabel("Classification Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision, Recall, and F2 vs Threshold")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ROC curve with both Youden's J (reference) and Constrained (selected)
    axes[1].plot(fpr_roc, tpr_roc, "b-", lw=2,
                 label=f"ROC (AUC={auc(fpr_roc, tpr_roc):.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].plot(fpr_roc[best_j_idx], tpr_roc[best_j_idx], "o", color="gray",
                 markersize=8, alpha=0.6, label=f"Youden's J ({thresholds['youdens_j']:.3f})")
    constrained_roc_idx = int(np.argmin(np.abs(thresholds_roc - thresholds["constrained"])))
    axes[1].plot(fpr_roc[constrained_roc_idx], tpr_roc[constrained_roc_idx],
                 "r*", markersize=14, zorder=5,
                 label=f"Constrained ({thresholds['constrained']:.3f}) ← selected")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve with Threshold Operating Points")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Threshold Selection \u2014 {best_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix_comparison(
    best_name: str,
    y_test: pd.Series,
    y_prob: np.ndarray,
    optimal_threshold: float,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Side-by-side confusion matrices: default vs tuned threshold."""
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned = (y_prob >= optimal_threshold).astype(int)

    fig, ax_cm = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_default, ax=ax_cm[0],
        cmap="Blues", colorbar=False, display_labels=["No Readmit", "Readmit"],
    )
    ax_cm[0].set_title(f"{best_name}\nDefault threshold (0.50)")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_tuned, ax=ax_cm[1],
        cmap="Blues", colorbar=False, display_labels=["No Readmit", "Readmit"],
    )
    ax_cm[1].set_title(f"{best_name}\nOptimised threshold ({optimal_threshold:.3f})")

    plt.tight_layout()
    plt.suptitle("Impact of Threshold Tuning on Predictions",
                 fontsize=12, fontweight="bold", y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_shap_summary(
    shap_original: np.ndarray,
    X_display: pd.DataFrame,
    feature_names: List[str],
    best_name: str,
    save_dir: Optional[pathlib.Path] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Beeswarm and bar SHAP summary plots.

    Returns (fig_beeswarm, fig_bar).
    """
    fig_bee = plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_original, X_display,
        feature_names=feature_names, max_display=20,
        plot_type="dot", show=False,
    )
    plt.title(f"SHAP Beeswarm \u2014 Top 20 Features\n({best_name})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig_bee.savefig(save_dir / "fig_shap_beeswarm_v1.png", dpi=300, bbox_inches="tight")

    fig_bar = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_original, X_display,
        feature_names=feature_names, max_display=20,
        plot_type="bar", show=False,
    )
    plt.title(f"Mean |SHAP| \u2014 Top 20 Features\n({best_name})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig_bar.savefig(save_dir / "fig_shap_bar_v1.png", dpi=300, bbox_inches="tight")

    return fig_bee, fig_bar


def plot_fairness(
    race_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    best_name: str,
    y_test: pd.Series,
    y_pred_final: np.ndarray,
    y_prob_final: np.ndarray,
    optimal_threshold: float,
    save_dir: Optional[pathlib.Path] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Recall and AUC disparity bar charts across demographic groups.

    Returns (fig_recall, fig_auc).
    """
    overall_recall = recall_score(y_test, y_pred_final)
    overall_auc = roc_auc_score(y_test, y_prob_final)

    fig_recall, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, df, title in [
        (axes[0], race_df, "Recall by Race"),
        (axes[1], gender_df, "Recall by Gender"),
    ]:
        df["Recall"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Recall")
        ax.set_ylim([0, 1])
        ax.axhline(overall_recall, color="red", linestyle="--",
                    label=f"Overall recall = {overall_recall:.3f}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.suptitle(
        f"Fairness Analysis \u2014 Recall Disparity across Demographic Groups\n"
        f"(Best model: {best_name}, threshold = {optimal_threshold:.2f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        fig_recall.savefig(save_dir / "fig_fairness_recall_v1.png", dpi=300, bbox_inches="tight")

    fig_auc, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, df, title in [
        (axes[0], race_df, "AUC-ROC by Race"),
        (axes[1], gender_df, "AUC-ROC by Gender"),
    ]:
        df["AUC-ROC"].dropna().plot(kind="bar", ax=ax, color="coral", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim([0.5, 1.0])
        ax.axhline(overall_auc, color="blue", linestyle="--",
                    label=f"Overall AUC = {overall_auc:.3f}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.suptitle("Fairness Analysis \u2014 AUC-ROC Disparity",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig_auc.savefig(save_dir / "fig_fairness_auc_v1.png", dpi=300, bbox_inches="tight")

    return fig_recall, fig_auc


def plot_final_model_comparison(
    all_results: pd.DataFrame,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing all model tracks on 4 metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(all_results))
    width = 0.2
    metrics_to_plot = ["Test AUC", "Test Recall", "Test Precision", "Test F1"]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors_bar = ["steelblue", "firebrick", "forestgreen", "darkorange"]

    for offset, metric, c in zip(offsets, metrics_to_plot, colors_bar):
        ax.bar([xi + offset for xi in x], all_results[metric],
               width=width, label=metric, color=c, alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(all_results.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Final Model Comparison \u2014 All Tracks",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Deployment pipeline
# ---------------------------------------------------------------------------

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select a fixed set of columns by name."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class ColumnScaler(BaseEstimator, TransformerMixin):
    """Apply a pre-fitted StandardScaler to specific continuous columns."""

    def __init__(self, scaler: StandardScaler, continuous_cols: List[str]):
        self.scaler = scaler
        self.continuous_cols = continuous_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out[self.continuous_cols] = self.scaler.transform(X[self.continuous_cols])
        return X_out


class DataFrameToArray(BaseEstimator, TransformerMixin):
    """Convert DataFrame to numpy array, stripping column names."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values if hasattr(X, "values") else np.asarray(X)


def build_deployment_pipeline(
    best_model: Any,
    is_pca_best: bool,
    selected_features: List[str],
    scaler_artifact: Dict[str, Any],
    pca_transformer: Any,
) -> Pipeline:
    """Assemble a deployment-ready sklearn Pipeline.

    Adapts to the best model type (PCA or direct features).
    """
    if is_pca_best:
        continuous_cols = scaler_artifact["continuous_cols"]
        scaler = scaler_artifact["scaler"]
        pipeline_cont_cols = [c for c in continuous_cols if c in selected_features]
        cont_idx = [continuous_cols.index(c) for c in pipeline_cont_cols]

        deploy_scaler = StandardScaler()
        deploy_scaler.mean_ = scaler.mean_[cont_idx]
        deploy_scaler.scale_ = scaler.scale_[cont_idx]
        deploy_scaler.var_ = scaler.var_[cont_idx]
        deploy_scaler.n_features_in_ = len(pipeline_cont_cols)

        pipeline = Pipeline([
            ("feature_selector", FeatureSelector(selected_features)),
            ("scaler", ColumnScaler(deploy_scaler, pipeline_cont_cols)),
            ("to_array", DataFrameToArray()),
            ("pca", pca_transformer),
            ("model", best_model),
        ])
        print("✓ Deployment pipeline: FeatureSelector → Scaler → Array → PCA → Model")
    else:
        pipeline = Pipeline([
            ("feature_selector", FeatureSelector(selected_features)),
            ("model", best_model),
        ])
        print("✓ Deployment pipeline: FeatureSelector → Model")

    return pipeline


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_artifacts(
    models_dir: pathlib.Path,
    best_name: str,
    best_model: Any,
    optimal_threshold: float,
    all_results: pd.DataFrame,
    y_test: pd.Series,
    y_pred_final: np.ndarray,
    is_pca_best: bool,
    selected_features: List[str],
    scaler_artifact: Dict[str, Any],
    pca_transformer: Any,
    deployment_pipeline: Pipeline,
) -> None:
    """Save all model artifacts to the models directory."""
    models_dir.mkdir(exist_ok=True)

    # Model
    name_base = (best_name.lower().replace(" ", "_")
                 .replace("\u2014", "").replace("__", "_").strip("_"))
    model_path = models_dir / f"best_model_{name_base}.joblib"
    joblib.dump(best_model, model_path)
    print(f"✓ Best model saved: {model_path}")

    # Metadata
    meta = {
        "model": best_name,
        "optimal_threshold": float(optimal_threshold),
        "test_auc": float(all_results.loc[best_name, "Test AUC"]),
        "test_recall_at_threshold": float(recall_score(y_test, y_pred_final)),
        "test_precision_at_threshold": float(precision_score(y_test, y_pred_final)),
        "test_f1_at_threshold": float(f1_score(y_test, y_pred_final)),
        "feature_track": "PCA (44 components)" if is_pca_best else "MI-selected (117 features)",
        "training_data": "SMOTE-balanced",
    }
    (models_dir / "best_model_metadata.json").write_text(json.dumps(meta, indent=2))
    print("✓ Metadata saved: best_model_metadata.json")

    # Feature list
    with open(models_dir / "selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"✓ Feature list saved ({len(selected_features)} features)")

    # Scaler
    joblib.dump(scaler_artifact, models_dir / "standard_scaler.joblib")
    print("✓ StandardScaler saved")

    # PCA
    if is_pca_best:
        joblib.dump(pca_transformer, models_dir / "pca_transformer.joblib")
        print("✓ PCA transformer saved")

    # Pipeline
    joblib.dump({
        "pipeline": deployment_pipeline,
        "optimal_threshold": float(optimal_threshold),
        "features_expected": selected_features,
        "model_name": best_name,
        "is_pca": is_pca_best,
    }, models_dir / "deployment_pipeline.joblib")
    print("✓ Deployment pipeline saved")


def export_tables(
    tables_dir: pathlib.Path,
    all_results: pd.DataFrame,
    strategy_df: pd.DataFrame,
    race_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    mean_abs_shap: pd.Series,
) -> None:
    """Export curated summary tables as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_results.round(4).to_csv(tables_dir / "tbl_model_comparison_v1.csv")
    strategy_df.to_csv(tables_dir / "tbl_threshold_strategies_v1.csv")
    race_df.to_csv(tables_dir / "tbl_fairness_race_v1.csv")
    gender_df.to_csv(tables_dir / "tbl_fairness_gender_v1.csv")
    mean_abs_shap.head(20).round(5).to_frame().to_csv(
        tables_dir / "tbl_shap_importance_v1.csv"
    )
    print(f"✓ {5} tables exported to {tables_dir}")
