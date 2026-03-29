#!/usr/bin/env python
"""Calibration loop: test presets against manifest targets and iterate.

Usage:
    python scripts/calibrate_presets.py [start_server]

If start_server is 'true', this script will start the FastAPI server before testing.
Otherwise, assumes server is already running on http://localhost:8000.
"""

import json
import sys
import subprocess
import time
from pathlib import Path
import asyncio
import httpx

# Import preset validator and presets
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ui.preset_validator import PresetManifestValidator
from app.ui.tabs.project_summary import SAMPLE_PRESETS


async def test_presets_against_manifest(api_base_url: str = "http://localhost:8000") -> dict:
    """Test all presets and return validation results.

    Args:
        api_base_url: Base URL for the API (default localhost:8000)

    Returns:
        Dict mapping preset_name -> (probability, band, validation_result)
    """
    validator = PresetManifestValidator()
    results = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for preset_name, preset_data in SAMPLE_PRESETS.items():
            try:
                # Transform preset data to API schema
                payload = _transform_preset_to_api_request(preset_data)

                # Call API with preset data
                response = await client.post(
                    f"{api_base_url}/v1/predict",
                    json=payload
                )
                response.raise_for_status()
                prediction = response.json()

                prob = prediction["readmission_probability"]
                band = prediction["risk_band"]
                label = prediction["prediction_label"]

                # Map preset name to manifest key
                manifest_key = {
                    "High-Risk Elderly": "high_risk",
                    "Moderate-Risk": "moderate_risk",
                    "Low-Risk": "low_risk",
                }.get(preset_name)

                if manifest_key:
                    validation = validator.validate_preset(manifest_key, prob, band)
                    results[preset_name] = {
                        "probability": prob,
                        "band": band,
                        "label": label,
                        "validation": validation,
                    }
                    print(f"\n✓ {preset_name}")
                    print(f"  Probability: {prob:.6f} | Band: {band} | Label: {label}")
                    print(f"  Status: {validation.message}")
                else:
                    print(f"\n✗ {preset_name} - not in manifest")

            except Exception as e:
                print(f"\n✗ {preset_name} - API error: {e}")
                results[preset_name] = {"error": str(e)}

    return results


def _transform_preset_to_api_request(preset: dict) -> dict:
    """Transform SAMPLE_PRESETS data to API request schema.

    Maps preset field names to API field names and includes only expected fields.

    Args:
        preset: Dict from SAMPLE_PRESETS

    Returns:
        Dict matching PredictRequest schema
    """
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


def _admission_source_map(source_id: str) -> str:
    """Map admission_source_id from preset to API request value."""
    mapping = {
        "1": "referral",
        "2": "transfer",
        "7": "referral",
    }
    return mapping.get(str(source_id), "referral")


def _to_bool(value: object, default: bool = True) -> bool:
    """Coerce typical preset values to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"yes", "true", "1", "y"}:
            return True
        if norm in {"no", "false", "0", "n"}:
            return False
    return default


def print_calibration_summary(results: dict) -> bool:
    """Print summary and return True if all presets pass.

    Args:
        results: Dict from test_presets_against_manifest

    Returns:
        True if all presets pass validation, False otherwise
    """
    print("\n" + "=" * 80)
    print("PRESET CALIBRATION SUMMARY")
    print("=" * 80)

    all_pass = True
    for preset_name, result in results.items():
        if "error" in result:
            print(f"✗ {preset_name}: {result['error']}")
            all_pass = False
        else:
            validator_result = result["validation"]
            status = "✓ PASS" if validator_result.passes_all else "✗ FAIL"
            print(f"{status} | {preset_name}")
            print(f"        Prob: {result['probability']:.4f} | Band: {result['band']}")
            print(f"        {validator_result.message}")
            all_pass = all_pass and validator_result.passes_all

    print("=" * 80)
    if all_pass:
        print("✓ All presets pass calibration targets!")
    else:
        print("✗ Some presets need tuning. Review messages above.")
    print("=" * 80 + "\n")

    return all_pass


async def main():
    """Main calibration driver."""
    start_server = len(sys.argv) > 1 and sys.argv[1].lower() == "true"

    if start_server:
        print("Starting FastAPI server...")
        proc = subprocess.Popen(
            ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)  # Wait for server to start
        print("Server started.\n")
    else:
        proc = None

    try:
        # Run preset calibration
        print("Testing presets against manifest targets...\n")
        results = await test_presets_against_manifest()

        # Print summary
        all_pass = print_calibration_summary(results)

        # Return exit code based on pass/fail
        sys.exit(0 if all_pass else 1)

    finally:
        if proc:
            print("Stopping server...")
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
