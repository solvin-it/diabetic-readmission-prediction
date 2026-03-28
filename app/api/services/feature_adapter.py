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
        drivers: list[dict[str, str | int | float]] = []

        if payload.number_emergency > 0:
            drivers.append({
                "feature": "number_emergency",
                "value": payload.number_emergency,
                "contribution_hint": "Prior emergency utilization tends to increase risk.",
            })
        if payload.number_inpatient > 0:
            drivers.append({
                "feature": "number_inpatient",
                "value": payload.number_inpatient,
                "contribution_hint": "Prior inpatient utilization tends to increase risk.",
            })
        if payload.number_diagnoses >= 8:
            drivers.append({
                "feature": "number_diagnoses",
                "value": payload.number_diagnoses,
                "contribution_hint": "Higher diagnosis burden often reflects more complex care.",
            })
        if payload.discharge_disposition_group == "home":
            drivers.append({
                "feature": "discharge_disposition_group",
                "value": payload.discharge_disposition_group,
                "contribution_hint": "Home discharge is frequently associated with lower relative risk.",
            })

        if not drivers:
            drivers.append({
                "feature": "utilization_pattern",
                "value": "low",
                "contribution_hint": "No dominant high-risk utilization pattern was detected from the submitted inputs.",
            })

        return drivers[:5]
