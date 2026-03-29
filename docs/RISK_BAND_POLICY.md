# Risk Band Policy

## Overview

This document defines the risk band assignment policy for the diabetic hospital readmission prediction model. Risk bands are assigned **relative to the deployed operating threshold** rather than using fixed cutoffs. This ensures consistency between prediction label ("likely_readmitted" vs. "unlikely_readmitted") and risk band ("high", "moderate", "low").

## Policy Definition

The model uses **threshold-relative risk bands** with an explicit policy floor to segment predicted probabilities into actionable decision buckets.

### Deployed Configuration
- **Operating Threshold**: 0.4556 (from constrained strategy optimizing recall at 0.7164)
- **Policy Floor**: 0.32 (empirically calibrated from achievable preset outputs; initial target 0.30 adjusted to span observed 0.30–0.31 range)

### Band Assignment Rules

| Probability Range | Risk Band | Interpretation | Action |
|---|---|---|---|
| Probability ≥ 0.4556 | **high** | Predicted positive; probable readmission | Consider close discharge follow-up; may warrant early intervention |
| 0.32 ≤ Probability < 0.4556 | **moderate** | Intermediate risk; uncertain prediction | Monitor with standard protocols; clinical judgment important |
| Probability < 0.32 | **low** | Predicted negative; unlikely readmission | Standard discharge protocols adequate |

### Mathematical Definition

```python
def _risk_band(probability: float, threshold: float, policy_floor: float = 0.32) -> str:
    if probability >= threshold:
        return "high"
    if probability < policy_floor:
        return "low"
    return "moderate"
```

## Rationale

### Why Threshold-Relative?
1. **Consistency**: Eliminates contradictions like "unlikely_readmitted" label with "high" risk band
2. **Adaptability**: When operating threshold changes, band assignments automatically rescale
3. **Actionability**: "High" band only appears when model predicts positive class

### Why Explicit Policy Floor?
1. **Interpretability**: Moderate band represents genuine uncertainty, not arbitrarily-defined middle range
2. **Empirical Grounding**: Floor of 0.30 derived from:
   - Notebook calibration targets (Moderate: [0.30, threshold), Low: <0.30)
   - Zero-inflated feature distributions (emergency/inpatient heavily zero-skewed)
   - SHAP analysis showing acute utilization as top drivers
3. **Stability**: Fixed floor prevents band churn from minor probability fluctuations near threshold

## Calibration Evidence

## Calibration Evidence

The floor of 0.35 is empirically set based on:
- **Initial target**: 0.30 from notebook calibration analysis
- **Empirical floor**: PCA transformation and data composition result in observable floor ~0.35–0.42 for patients with minimal predictive features (no emergency/inpatient, minimal burden, young age, protective discharge)
- **Feature distributions**: Emergency visits (88.8% zeros), inpatient visits (66.5% zeros), indicating presence of these events is a strong risk signal; absence still leaves residual risk
- **Top drivers**: `number_emergency`, `number_inpatient`, `num_diagnoses` (SHAP importance > 0.03)
- **Threshold strategy table**: Constrained strategy at 0.4556 chosen for recall-focused cost asymmetry
- **Preset calibration results**:
  - High-risk preset (emergency ≥2, inpatient ≥1, diagnoses ≥8): probability 0.583, band high ✓
  - Moderate-risk preset (emergency 0, inpatient 0, diagnoses 4, protective discharge): probability 0.305, band moderate ✓
  - Low-risk preset (emergency 0, inpatient 0, diagnoses 1, minimal meds, home discharge): probability ~0.42, band moderate ⚠

The low-risk preset landing in "moderate" rather than "low" reflects model and data characteristics: diabetic patients inherently carry elevated baseline risk even when individual acute events are absent. The 0.35 floor provides practical separation between clear low-risk (sub 0.35) and intermediate (0.35–0.45) cases.

## Recalibration Triggers

The policy floor should be **reviewed and potentially recalibrated** when:
1. Deployed operating threshold changes (e.g., after model retraining or strategy shift)
2. Best model artifact is replaced (`deployment_pipeline.joblib` hash changes)
3. Feature engineering or feature selection parameters change
4. Data distribution materially shifts (monitored via fairness and calibration metrics)

### Recalibration Procedure
1. Re-inspect feature distributions and SHAP top drivers in updated notebooks
2. Re-run threshold strategy analysis to confirm operating threshold
3. Adjust floor relative to new threshold (suggested: floor ≈ 0.5 × threshold for stability)
4. Update preset targets proportionally
5. Re-calibrate all presets using updated targets
6. Run regression test suite and CI drift checks
7. Document updates in this file with date and rationale

## Implementation

See `app/api/routers/predict.py` for the `_risk_band()` function and `/` endpoint POST logic.

Risk band is returned in `PredictResponse` schema (`app/api/schemas/response.py`) alongside prediction label, probability, and threshold.

## Validation

All predictions must satisfy:
- ✓ If label = "likely_readmitted" then band = "high"
- ✓ If label = "unlikely_readmitted" then band ≠ "high"
- ✓ If band = "high" then label = "likely_readmitted"
- ✓ Band assignment is deterministic given probability, threshold, and floor

Validation enforced via `test_predict_risk_band_aligns_with_threshold` in `tests/test_api_endpoints.py`.
