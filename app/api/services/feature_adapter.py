from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from app.api.schemas.request import PredictRequest


@dataclass
class FeatureAdapter:
    """Transform business-level request input into selected model features."""

    expected_features: list[str]

    _INSULIN_MAP = {
        "No": 0.0,
        "Steady": 1.0,
        "Down": 2.0,
        "Up": 2.0,
    }

    @staticmethod
    def _age_feature(age_band: str) -> str:
        # Training columns were sanitized from age_[70-80) to age__70-80)
        return f"age__{age_band}"

    @staticmethod
    def _set_if_exists(row: dict[str, float], key: str, value: float = 1.0) -> None:
        if key in row:
            row[key] = value

    def transform(self, payload: PredictRequest) -> pd.DataFrame:
        row = {k: 0.0 for k in self.expected_features}

        # Selected feature space keeps a reduced set of one-hot categories.
        a1c_bucket = payload.A1Cresult if payload.A1Cresult in {">7", "Norm", "none"} else ">7"
        max_glu_bucket = payload.max_glu_serum if payload.max_glu_serum in {">300", "Norm", "none"} else ">300"

        # Raw continuous/count features
        raw_values = {
            "time_in_hospital": payload.time_in_hospital,
            "num_lab_procedures": payload.num_lab_procedures,
            "num_procedures": payload.num_procedures,
            "num_medications": payload.num_medications,
            "number_diagnoses": payload.number_diagnoses,
            "number_outpatient": payload.number_outpatient,
            "number_emergency": payload.number_emergency,
            "number_inpatient": payload.number_inpatient,
            "change": float(payload.change),
            "diabetesMed": float(payload.diabetesMed),
            "insulin": self._INSULIN_MAP[payload.insulin],
        }
        for k, v in raw_values.items():
            if k in row:
                row[k] = float(v)

        # Derived engineered features
        derived_values = {
            "had_outpatient": float(payload.number_outpatient > 0),
            "had_emergency": float(payload.number_emergency > 0),
            "had_inpatient": float(payload.number_inpatient > 0),
            "had_procedures": float(payload.num_procedures > 0),
            "log_outpatient": float(np.log1p(payload.number_outpatient)),
            "log_emergency": float(np.log1p(payload.number_emergency)),
            "log_inpatient": float(np.log1p(payload.number_inpatient)),
            "specialty_known": float(payload.medical_specialty != "Unknown"),
            "insulin_adjusted": float(payload.insulin in {"Down", "Up"}),
            "glucose_tested": float(payload.max_glu_serum != "none"),
            "A1C_tested": float(payload.A1Cresult != "none"),
        }
        for k, v in derived_values.items():
            if k in row:
                row[k] = v

        # One-hot categorical features where selected columns include them
        categorical_keys = {
            f"gender_{payload.gender}",
            f"race_{payload.race}",
            f"admission_type_id_{payload.admission_type_group}",
            f"admission_source_id_{payload.admission_source_group}",
            f"discharge_disposition_id_{payload.discharge_disposition_group}",
            f"A1Cresult_{a1c_bucket}",
            f"max_glu_serum_{max_glu_bucket}",
            f"medical_specialty_{payload.medical_specialty}",
            f"diag_1_chapter_{payload.diag_1_chapter}",
            f"diag_2_chapter_{payload.diag_2_chapter}",
            f"diag_3_chapter_{payload.diag_3_chapter}",
            self._age_feature(payload.age_band),
        }

        for key in categorical_keys:
            self._set_if_exists(row, key, 1.0)

        # Only one insulin one-hot selected in training set
        if payload.insulin in {"Down", "Up"}:
            self._set_if_exists(row, "insulin_adjusted", 1.0)

        # Preserve exact expected ordering
        df = pd.DataFrame([[row[f] for f in self.expected_features]], columns=self.expected_features)
        self._validate_output(df)
        return df

    def _validate_output(self, df: pd.DataFrame) -> None:
        if list(df.columns) != self.expected_features:
            raise ValueError("Adapter output column order does not match expected feature order.")

        if df.shape != (1, len(self.expected_features)):
            raise ValueError("Adapter output has invalid shape for single-record inference.")

        if df.isna().any().any():
            raise ValueError("Adapter output contains NaN values.")

        values = df.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Adapter output contains non-finite numeric values.")

    @staticmethod
    def top_drivers_from_input(payload: PredictRequest) -> list[dict[str, str | int | float]]:
        """Identify the most relevant risk drivers from patient inputs.

        Rules are ordered by SHAP importance (from tbl_shap_importance_v1.csv).
        Each rule fires only when the input value is clinically notable.
        Both risk-increasing and risk-decreasing (protective) factors are included.
        """
        # (shap_rank, feature, value, hint) — shap_rank used to sort, lower = more important
        candidates: list[tuple[float, str, str | int | float, str]] = []

        # --- Emergency utilization (SHAP #1-2: log_emergency 0.033, number_emergency 0.032) ---
        if payload.number_emergency > 0:
            candidates.append((
                1, "number_emergency", payload.number_emergency,
                "Prior emergency visits are the strongest predictor of readmission risk."
            ))
        else:
            candidates.append((
                2, "number_emergency", 0,
                "No prior emergency visits — this is protective (88.8% of the training population had zero)."
            ))

        # --- Inpatient utilization (SHAP #3-4: log_inpatient 0.030, number_inpatient 0.027) ---
        if payload.number_inpatient > 0:
            candidates.append((
                3, "number_inpatient", payload.number_inpatient,
                "Prior inpatient admissions are a strong predictor of readmission risk."
            ))
        else:
            candidates.append((
                4, "number_inpatient", 0,
                "No prior inpatient admissions — this is protective (66.5% of training data had zero)."
            ))

        # --- Diagnosis burden (SHAP #6: 0.013) ---
        if payload.number_diagnoses >= 7:
            candidates.append((
                6, "number_diagnoses", payload.number_diagnoses,
                "High diagnosis burden (>=7) reflects complex care needs and increases risk."
            ))
        elif payload.number_diagnoses <= 3:
            candidates.append((
                6.5, "number_diagnoses", payload.number_diagnoses,
                "Low diagnosis count suggests simpler clinical picture, associated with lower risk."
            ))

        # --- Outpatient utilization (SHAP #7: 0.011) ---
        if payload.number_outpatient > 0:
            candidates.append((
                7, "number_outpatient", payload.number_outpatient,
                "Prior outpatient visits indicate ongoing care utilization, moderately associated with risk."
            ))

        # --- Discharge disposition (SHAP #8-9: home 0.009, facility 0.008) ---
        if payload.discharge_disposition_group == "home":
            candidates.append((
                8, "discharge_disposition_group", "home",
                "Home discharge is a protective factor associated with lower readmission risk."
            ))
        elif payload.discharge_disposition_group == "facility":
            candidates.append((
                9, "discharge_disposition_group", "facility",
                "Discharge to facility (not home) is associated with higher readmission risk."
            ))

        # --- Time in hospital (SHAP #9: 0.009) ---
        if payload.time_in_hospital >= 5:
            candidates.append((
                10, "time_in_hospital", payload.time_in_hospital,
                "Longer hospital stays (>=5 days) are associated with more complex cases."
            ))
        elif payload.time_in_hospital <= 2:
            candidates.append((
                10.5, "time_in_hospital", payload.time_in_hospital,
                "Short hospital stay suggests lower acuity."
            ))

        # --- Insulin management (SHAP #12: 0.007) ---
        if payload.insulin in {"Down", "Up"}:
            candidates.append((
                12, "insulin", payload.insulin,
                "Active insulin adjustment (dose changed) indicates less stable glycemic control."
            ))
        elif payload.insulin == "Steady":
            candidates.append((
                12.5, "insulin", "Steady",
                "Stable insulin dosing suggests well-managed glycemic control."
            ))

        # --- Lab procedures (SHAP #14: 0.007) ---
        if payload.num_lab_procedures >= 40:
            candidates.append((
                14, "num_lab_procedures", payload.num_lab_procedures,
                "High lab procedure count may reflect diagnostic workup complexity."
            ))

        # --- Age (SHAP #15: 0.007 for 80-90) ---
        if payload.age_band in {"80-90)", "90-100)"}:
            candidates.append((
                15, "age_band", payload.age_band,
                "Advanced age (80+) is associated with higher readmission risk."
            ))

        # --- Primary diagnosis chapter (SHAP #16: 0.007 for endocrine) ---
        if payload.diag_1_chapter == "endocrine":
            candidates.append((
                16, "diag_1_chapter", "endocrine",
                "Endocrine primary diagnosis directly relates to diabetes management complexity."
            ))
        elif payload.diag_1_chapter == "circulatory":
            candidates.append((
                16.5, "diag_1_chapter", "circulatory",
                "Circulatory primary diagnosis is common in diabetic patients and a known comorbidity."
            ))

        # --- Glucose / A1C testing (SHAP contributing via derived features) ---
        if payload.A1Cresult == "Norm":
            candidates.append((
                18, "A1Cresult", "Norm",
                "Normal A1C result indicates well-controlled blood sugar."
            ))
        elif payload.A1Cresult in {">7", ">8"}:
            candidates.append((
                18, "A1Cresult", payload.A1Cresult,
                "Elevated A1C indicates suboptimal glycemic control over recent months."
            ))

        if payload.max_glu_serum in {">200", ">300"}:
            candidates.append((
                19, "max_glu_serum", payload.max_glu_serum,
                "Elevated serum glucose indicates poor glycemic control during this admission."
            ))

        # --- Medication burden (SHAP via num_medications: 0.012) ---
        if payload.num_medications >= 20:
            candidates.append((
                11, "num_medications", payload.num_medications,
                "High medication count (>=20) reflects polypharmacy and complex management."
            ))
        elif payload.num_medications <= 3:
            candidates.append((
                11.5, "num_medications", payload.num_medications,
                "Low medication count suggests simpler treatment regimen."
            ))

        # --- Medication change flag (SHAP via change) ---
        if payload.change:
            candidates.append((
                20, "change", True,
                "Medication change during admission may indicate treatment instability."
            ))

        # --- Diabetes medication (SHAP via diabetesMed) ---
        if not payload.diabetesMed:
            candidates.append((
                21, "diabetesMed", False,
                "No diabetes medication prescribed — may indicate diet-controlled or early-stage diabetes."
            ))

        # Sort by SHAP importance rank and return top 5
        candidates.sort(key=lambda c: c[0])
        return [
            {"feature": feat, "value": val, "contribution_hint": hint}
            for _, feat, val, hint in candidates[:5]
        ]
