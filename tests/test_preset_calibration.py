import unittest

from fastapi.testclient import TestClient

from app.api.main import app
from app.ui.preset_validator import PresetManifestValidator
from app.ui.tabs.project_summary import SAMPLE_PRESETS


def _admission_source_map(source_id: str) -> str:
    mapping = {
        "1": "referral",
        "2": "transfer",
        "3": "referral",
        "4": "transfer",
        "5": "transfer",
        "6": "transfer",
        "7": "emergency_room",
        "10": "transfer",
        "22": "transfer",
        "25": "transfer",
        "26": "other",
    }
    return mapping.get(str(source_id), "referral")


def _to_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"yes", "true", "1", "y"}:
            return True
        if norm in {"no", "false", "0", "n"}:
            return False
    return default


def _transform_preset_to_api_request(preset: dict) -> dict:
    return {
        "age_band": preset.get("age_band", "70-80)"),
        "gender": preset.get("gender", "Female"),
        "race": preset.get("race", "Caucasian"),
        "admission_type_group": str(preset.get("admission_type_group", "1")),
        "admission_source_group": _admission_source_map(preset.get("admission_source_id", "1")),
        "discharge_disposition_group": preset.get("discharge_disposition_group", "home"),
        "time_in_hospital": preset.get("time_in_hospital", 4),
        "num_lab_procedures": preset.get("num_lab_procedures", 40),
        "num_procedures": preset.get("number_procedures", 0),
        "num_medications": preset.get("num_medications", 12),
        "number_diagnoses": preset.get("number_diagnoses", 7),
        "number_outpatient": preset.get("number_outpatient", 0),
        "number_emergency": preset.get("number_emergency", 0),
        "number_inpatient": preset.get("number_inpatient", 0),
        "A1Cresult": preset.get("A1Cresult", "none"),
        "max_glu_serum": preset.get("max_glu_serum", "none"),
        "insulin": preset.get("insulin", "Steady"),
        "change": _to_bool(preset.get("change", False)),
        "diabetesMed": _to_bool(preset.get("diabetic_medication", True)),
        "medical_specialty": preset.get("specialty", "Unknown"),
        "diag_1_chapter": preset.get("diag_1_chapter", "circulatory"),
        "diag_2_chapter": preset.get("diag_2_chapter", "circulatory"),
        "diag_3_chapter": preset.get("diag_3_chapter", "other"),
    }


class TestPresetCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.validator = PresetManifestValidator()

    def test_presets_match_manifest_targets(self) -> None:
        preset_map = {
            "High-Risk Elderly": "high_risk",
            "Moderate-Risk": "moderate_risk",
            "Low-Risk": "low_risk",
        }

        for preset_name, manifest_key in preset_map.items():
            with self.subTest(preset=preset_name):
                payload = _transform_preset_to_api_request(SAMPLE_PRESETS[preset_name])
                response = self.client.post("/v1/predict", json=payload)
                self.assertEqual(response.status_code, 200)

                data = response.json()
                validation = self.validator.validate_preset(
                    manifest_key,
                    data["readmission_probability"],
                    data["risk_band"],
                )

                self.assertTrue(validation.passes_all, msg=validation.message)

    def test_presets_return_multiple_top_drivers(self) -> None:
        for preset_name, preset in SAMPLE_PRESETS.items():
            with self.subTest(preset=preset_name):
                payload = _transform_preset_to_api_request(preset)
                response = self.client.post("/v1/predict", json=payload)
                self.assertEqual(response.status_code, 200)

                data = response.json()
                drivers = data.get("top_drivers", [])
                self.assertGreaterEqual(
                    len(drivers),
                    3,
                    msg=f"Expected >=3 top drivers for {preset_name}, got {len(drivers)}",
                )

                unique_features = {driver["feature"] for driver in drivers}
                self.assertGreaterEqual(
                    len(unique_features),
                    3,
                    msg=f"Expected diverse driver features for {preset_name}",
                )


if __name__ == "__main__":
    unittest.main()
