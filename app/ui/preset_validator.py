"""Preset validator: ensures presets conform to manifest intent and probability targets."""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class PresetValidationResult:
    """Result of validating a preset against manifest targets."""
    preset_name: str
    intended_band: str
    target_range: tuple[float, float]
    actual_probability: float
    actual_band: str
    passes_target_range: bool
    passes_band: bool
    passes_all: bool
    tolerance: float
    message: str


class PresetManifestValidator:
    """Load and validate presets against manifest targets."""

    def __init__(self, manifest_path: Optional[Path] = None):
        """Initialize validator with manifest path.

        Args:
            manifest_path: Path to presets_manifest.json. Defaults to
                          app/ui/presets_manifest.json relative to project root.
        """
        if manifest_path is None:
            # Default to app/ui/presets_manifest.json
            project_root = Path(__file__).parent.parent.parent
            manifest_path = project_root / "app" / "ui" / "presets_manifest.json"

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.calibration_threshold = self.manifest["calibration_threshold"]
        self.policy_floor = self.manifest["policy_floor"]

    def validate_preset(
        self,
        preset_name: str,
        predicted_probability: float,
        predicted_band: str,
    ) -> PresetValidationResult:
        """Validate a single preset prediction against manifest targets.

        Args:
            preset_name: Name of preset (e.g., "high_risk", "moderate_risk", "low_risk")
            predicted_probability: Model's predicted probability for this preset
            predicted_band: Model's assigned risk band for this preset

        Returns:
            PresetValidationResult with pass/fail details and message
        """
        if preset_name not in self.manifest["presets"]:
            return PresetValidationResult(
                preset_name=preset_name,
                intended_band="unknown",
                target_range=(0.0, 1.0),
                actual_probability=predicted_probability,
                actual_band=predicted_band,
                passes_target_range=False,
                passes_band=False,
                passes_all=False,
                tolerance=0.03,
                message=f"Preset '{preset_name}' not found in manifest",
            )

        preset_spec = self.manifest["presets"][preset_name]
        intended_band = preset_spec["intended_band"]
        target_range = tuple(preset_spec["target_probability_range"])
        tolerance = preset_spec["tolerance"]

        # Check band agreement
        band_match = predicted_band == intended_band
        band_message = (
            f"✓ Band matches intent"
            if band_match
            else f"✗ Band mismatch: expected {intended_band}, got {predicted_band}"
        )

        # Check probability range
        lower, upper = target_range
        lower_with_tolerance = lower - tolerance
        upper_with_tolerance = upper + tolerance

        in_strict_range = lower <= predicted_probability <= upper
        in_tolerance_range = lower_with_tolerance <= predicted_probability <= upper_with_tolerance

        range_message = (
            f"✓ Probability {predicted_probability:.6f} in target range [{lower}, {upper}]"
            if in_strict_range
            else f"⚠ Probability {predicted_probability:.6f} outside strict range [{lower}, {upper}] "
                 f"(within tolerance ±{tolerance})" if in_tolerance_range
            else f"✗ Probability {predicted_probability:.6f} outside tolerance range "
                 f"[{lower_with_tolerance:.4f}, {upper_with_tolerance:.4f}]"
        )

        passes_all = band_match and in_tolerance_range

        full_message = f"{preset_name}: {band_message}. {range_message}"

        return PresetValidationResult(
            preset_name=preset_name,
            intended_band=intended_band,
            target_range=target_range,
            actual_probability=predicted_probability,
            actual_band=predicted_band,
            passes_target_range=in_tolerance_range,
            passes_band=band_match,
            passes_all=passes_all,
            tolerance=tolerance,
            message=full_message,
        )

    def validate_all_presets(
        self,
        preset_results: dict[str, tuple[float, str]],
    ) -> dict[str, PresetValidationResult]:
        """Validate all presets in one call.

        Args:
            preset_results: Dict mapping preset_name -> (probability, band)

        Returns:
            Dict mapping preset_name -> PresetValidationResult
        """
        results = {}
        for preset_name, (prob, band) in preset_results.items():
            results[preset_name] = self.validate_preset(preset_name, prob, band)
        return results

    def print_validation_report(
        self,
        validation_results: dict[str, PresetValidationResult],
    ) -> None:
        """Pretty-print validation results.

        Args:
            validation_results: Dict from validate_all_presets or dict of individual results
        """
        print("\n" + "=" * 80)
        print("PRESET VALIDATION REPORT")
        print("=" * 80)
        print(f"Calibration Threshold: {self.calibration_threshold}")
        print(f"Policy Floor: {self.policy_floor}")
        print("=" * 80)

        all_pass = True
        for preset_name in ["high_risk", "moderate_risk", "low_risk"]:
            if preset_name not in validation_results:
                continue

            result = validation_results[preset_name]
            status = "✓ PASS" if result.passes_all else "✗ FAIL"
            print(f"\n{status} | {result.message}")
            all_pass = all_pass and result.passes_all

        print("\n" + "=" * 80)
        overall = "✓ All presets pass" if all_pass else "✗ Some presets fail"
        print(f"SUMMARY: {overall}")
        print("=" * 80 + "\n")

        return all_pass


if __name__ == "__main__":
    # Example usage
    validator = PresetManifestValidator()
    test_results = {
        "high_risk": (0.58, "high"),
        "moderate_risk": (0.35, "moderate"),
        "low_risk": (0.25, "low"),
    }
    results = validator.validate_all_presets(test_results)
    validator.print_validation_report(results)
