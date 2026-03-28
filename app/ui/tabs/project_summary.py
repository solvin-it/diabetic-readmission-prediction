import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui.services.api_client import get_model_info

# Assume reports/ is accessible relative to the app directory
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Sample patient presets for demo purposes
SAMPLE_PRESETS = {
    "High-Risk Elderly": {
        "age_band": "80-90",
        "gender": "Female",
        "race": "Caucasian",
        "admission_source_id": "7",  # Referral
        "readmitted": "Yes",
        "diabetic_medication": "Yes",
        "insulin": "Up",
        "max_glu_serum": ">200",
        "A1Cresult": ">8",
        "time_in_hospital": 8,
        "num_medications": 45,
        "number_procedures": 2,
        "number_diagnoses": 8,
        "number_outpatient": 0,
        "number_inpatient": 3,
        "number_emergency": 2,
        "specialty": "InternalMedicine",
        "description": "Elderly patient with complex medical history, multiple comorbidities, high medication burden, recent emergency and inpatient visits."
    },
    "Moderate-Risk": {
        "age_band": "50-60",
        "gender": "Male",
        "race": "AfricanAmerican",
        "admission_source_id": "1",  # Physician Referral
        "readmitted": "No",
        "diabetic_medication": "Yes",
        "insulin": "Steady",
        "max_glu_serum": "Normal",
        "A1Cresult": ">7",
        "time_in_hospital": 5,
        "num_medications": 20,
        "number_procedures": 1,
        "number_diagnoses": 5,
        "number_outpatient": 2,
        "number_inpatient": 1,
        "number_emergency": 1,
        "specialty": "Cardiology",
        "description": "Middle-aged patient with moderate comorbidities, controlled insulin regimen, recent inpatient stay."
    },
    "Low-Risk": {
        "age_band": "30-40",
        "gender": "Female",
        "race": "Caucasian",
        "admission_source_id": "1",  # Physician Referral
        "readmitted": "No",
        "diabetic_medication": "No",
        "insulin": "No",
        "max_glu_serum": "Normal",
        "A1Cresult": "<7",
        "time_in_hospital": 2,
        "num_medications": 5,
        "number_procedures": 0,
        "number_diagnoses": 2,
        "number_outpatient": 3,
        "number_inpatient": 0,
        "number_emergency": 0,
        "specialty": "GeneralPractice",
        "description": "Younger patient with minimal complications, no insulin requirement, outpatient-focused care."
    }
}


def _load_csv_table(file_path: Path) -> pd.DataFrame | None:
    """Safely load CSV table."""
    try:
        if file_path.exists():
            return pd.read_csv(file_path)
        return None
    except Exception as e:
        st.warning(f"Could not load {file_path.name}: {e}")
        return None


def _display_figure(fig_path: Path, caption: str) -> None:
    """Display a figure with caption."""
    try:
        if fig_path.exists():
            st.image(str(fig_path), caption=caption, width="stretch")
        else:
            st.info(f"Figure not found: {fig_path.name}")
    except Exception as e:
        st.warning(f"Could not display figure {fig_path.name}: {e}")


def render() -> None:
    st.subheader("Project Summary")
    st.caption("Technical summary of the diabetic readmission prediction capstone.")

    # ==================== SECTION 1: Model Performance Metrics ====================
    try:
        info = get_model_info()
        st.markdown(f"**Model:** `{info['model_name']}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{info['test_auc']:.4f}")
        c2.metric("Recall @ Threshold", f"{info['test_recall_at_threshold']:.4f}")
        c3.metric("Operating Threshold", f"{info['optimal_threshold']:.4f}")
    except Exception as exc:
        st.error(f"Could not load model info from API: {exc}")

    st.divider()

    # ==================== SECTION 2: Model Comparison ====================
    st.markdown("### Model Comparison")
    st.write(
        "Performance across 8 candidate configurations — baseline, tuned, and PCA feature tracks — "
        "evaluated on test AUC-ROC. Random Forest and XGBoost dominated; the PCA track (44 components "
        "from 117 MI-selected features) retained 95.2% of variance while reducing dimensionality by 62%, "
        "matching tuned full-feature models on AUC."
    )
    tbl_comparison = _load_csv_table(TABLES_DIR / "tbl_model_comparison_v1.csv")
    if tbl_comparison is not None:
        st.dataframe(tbl_comparison, width="stretch")
    else:
        st.info("Model comparison table not available.")

    st.divider()

    # ==================== SECTION 3: Feature Importance (SHAP) ====================
    st.markdown("### Top Feature Importance (SHAP)")
    st.write(
        "Mean absolute SHAP values measure each feature's average contribution to predictions across all "
        "test patients. For this PCA-based model, SHAP values are **back-projected** onto the original "
        "117 clinical features via the PCA loadings matrix, restoring interpretability. "
        "Prior emergency and inpatient utilization dominate — both are available in EHR data at "
        "*admission time*, enabling early intervention before discharge planning begins."
    )

    tbl_shap = _load_csv_table(TABLES_DIR / "tbl_shap_importance_v1.csv")
    if tbl_shap is not None:
        if "Unnamed: 0" in tbl_shap.columns:
            tbl_shap = tbl_shap.rename(columns={"Unnamed: 0": "Feature"})
        st.dataframe(tbl_shap, width="stretch")
    else:
        st.info("SHAP importance table not available.")

    st.divider()

    # ==================== SECTION 4: Fairness Analysis ====================
    st.markdown("### Fairness Analysis")
    st.write(
        "Model performance stratified by demographic subgroups at the Constrained operating threshold. "
        "Gender groups show broadly consistent performance. Racial subgroups exhibit moderate recall "
        "disparity — likely driven by lower sample sizes for underrepresented groups — and warrant "
        "ongoing monitoring in deployment alongside consideration of fairness constraints in threshold selection."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.write("**By Gender:**")
        tbl_fairness_gender = _load_csv_table(TABLES_DIR / "tbl_fairness_gender_v1.csv")
        if tbl_fairness_gender is not None:
            st.dataframe(tbl_fairness_gender, width="stretch")
        else:
            st.info("Gender fairness table not available.")

    with col2:
        st.write("**By Race/Ethnicity:**")
        tbl_fairness_race = _load_csv_table(TABLES_DIR / "tbl_fairness_race_v1.csv")
        if tbl_fairness_race is not None:
            st.dataframe(tbl_fairness_race, width="stretch")
        else:
            st.info("Race/ethnicity fairness table not available.")

    st.divider()

    # ==================== SECTION 5: Key Visualizations ====================
    st.markdown("### Key Visualizations")

    # --- Model Comparison & ROC ---
    st.markdown("#### Model Selection")
    st.caption(
        "Random Forest and XGBoost outperformed tree-based and linear baselines across all tracks. "
        "The final model (Random Forest + PCA) achieves AUC 0.6446 — consistent with published benchmarks "
        "of 0.63–0.68 for this class-imbalanced dataset (~11% positive rate, 1999–2008)."
    )
    col1, col2 = st.columns(2)
    with col1:
        _display_figure(
            FIGURES_DIR / "fig_model_comparison_v1.png",
            "AUC-ROC across 8 candidate configurations (baseline, tuned, PCA tracks)"
        )
    with col2:
        _display_figure(
            FIGURES_DIR / "fig_baseline_roc_v1.png",
            "ROC curve for the best model — Random Forest with PCA (AUC 0.6446)"
        )

    st.divider()

    # --- Threshold Strategies ---
    st.markdown("#### Threshold Selection & Operational Tradeoffs")
    st.caption(
        "At the default 0.50 threshold, the model prioritizes specificity, missing most readmissions. "
        "Three strategies were compared: **F2** (highest recall ~81%, but flags 66% of patients), "
        "**Youden's J** (~55% recall, equal sensitivity/specificity weight), and **Constrained** "
        "(precision ≥ 15% floor, ~72% recall, ~54% flagged). The **Constrained** strategy was adopted "
        "as the operational threshold given the cost asymmetry: missing a high-risk readmission is "
        "costlier than a false alarm."
    )
    col1, col2 = st.columns(2)
    with col1:
        _display_figure(
            FIGURES_DIR / "fig_baseline_confusion_v1.png",
            "Confusion matrix at the Constrained threshold (~0.456) — ~72% recall on the test set"
        )
    with col2:
        _display_figure(
            FIGURES_DIR / "fig_confusion_threshold_v1.png",
            "Effect of threshold strategy on confusion matrix (F2, Youden's J, Constrained)"
        )

    st.divider()

    # --- SHAP ---
    st.markdown("#### Interpretability (SHAP)")
    st.caption(
        "SHAP values are back-projected from 44 PCA components onto the original 117 clinical features. "
        "The beeswarm plot shows each test patient as a dot (red = high feature value, blue = low); "
        "rightward displacement increases predicted readmission risk."
    )
    col1, col2 = st.columns(2)
    with col1:
        _display_figure(
            FIGURES_DIR / "fig_shap_bar_v1.png",
            "Global SHAP importance — mean |SHAP| per feature across all test patients"
        )
    with col2:
        _display_figure(
            FIGURES_DIR / "fig_shap_beeswarm_v1.png",
            "SHAP beeswarm — feature value vs. prediction impact for each test patient"
        )

    st.divider()

    # --- Fairness ---
    st.markdown("#### Subgroup Fairness")
    st.caption(
        "AUC and recall computed per demographic subgroup at the Constrained threshold. "
        "Gender parity is strong; racial subgroups show moderate recall disparity, "
        "consistent with underrepresentation in the 1999–2008 training data."
    )
    col1, col2 = st.columns(2)
    with col1:
        _display_figure(
            FIGURES_DIR / "fig_fairness_auc_v1.png",
            "AUC-ROC parity across race and gender subgroups"
        )
    with col2:
        _display_figure(
            FIGURES_DIR / "fig_fairness_recall_v1.png",
            "Recall parity across race and gender subgroups at the Constrained threshold"
        )

    st.divider()

    # ==================== SECTION 6: Sample Patients (Quick Demo) ====================
    st.markdown("### Sample Patients for Quick Demo")
    st.write("Click a preset below to pre-fill the prediction form on the **Prediction Tool** tab:")

    for preset_name, preset_data in SAMPLE_PRESETS.items():
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(f"📋 {preset_name}", key=f"preset_{preset_name}"):
                st.session_state.active_preset = preset_name
                st.session_state.preset_data = preset_data
                st.success(f"✅ Loaded preset: **{preset_name}**. Switch to **Prediction Tool** tab to apply.")

        with col2:
            st.caption(preset_data.get("description", ""))

    st.divider()

    # ==================== SECTION 7: Disclaimers & Context ====================
    st.markdown("### Important Notes")
    st.write(
        "- This UI is a **capstone demo** and **educational tool** for demonstrating binary classification techniques."
    )
    st.write(
        "- Prediction outputs are **decision support only**, not medical directives. Always consult healthcare providers."
    )
    st.write(
        "- Model prioritizes **recall** over precision to minimize missing high-risk patients. False positives (low-risk predicted as high-risk) are acceptable trade-offs."
    )
    st.write(
        "- **No actual patient data** is stored or transmitted. All predictions are ephemeral."
    )
