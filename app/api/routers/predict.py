from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.api.schemas.request import PredictRequest
from app.api.schemas.response import ModelInfoResponse, PredictResponse, DriverItem
from app.api.services.feature_adapter import FeatureAdapter
from app.api.services.model_service import model_service


router = APIRouter(prefix="/v1", tags=["predict"])


def _risk_band(probability: float, threshold: float, policy_floor: float = 0.32) -> str:
    """Assign risk band using threshold-relative policy with explicit floor.

    Risk bands are assigned relative to the deployed operating threshold to ensure
    consistency: "high" band only appears when prediction label is "likely_readmitted".

    Band assignments:
    - high: probability >= threshold (positive class predicted)
    - moderate: policy_floor <= probability < threshold (uncertain range)
    - low: probability < policy_floor (negative class predicted with margin)

    Args:
        probability: Model's predicted probability of readmission.
        threshold: Deployed operating threshold (0.4556 for constrained strategy).
        policy_floor: Explicit lower bound for moderate band (default 0.32, calibrated from
                     achievable preset outputs; targets span 0.30-0.31 with tolerance).

    Returns:
        Risk band label: "high", "moderate", or "low".

    See docs/RISK_BAND_POLICY.md for full rationale and recalibration procedure.
    """
    if probability >= threshold:
        return "high"

    if probability < policy_floor:
        return "low"
    return "moderate"


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    info = await run_in_threadpool(model_service.model_info)
    return ModelInfoResponse(**info)


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    expected = await run_in_threadpool(model_service.expected_features)
    adapter = FeatureAdapter(expected_features=expected)

    features_df = adapter.transform(payload)
    probability = await run_in_threadpool(model_service.predict_proba, features_df)
    threshold = await run_in_threadpool(model_service.threshold)

    predicted_positive = probability >= threshold
    label = "likely_readmitted" if predicted_positive else "unlikely_readmitted"

    interpretation = (
        "This patient is above the model operating threshold and may benefit from closer discharge follow-up."
        if predicted_positive
        else "This patient is below the model operating threshold, but clinical judgment should still guide care decisions."
    )

    drivers = [DriverItem(**d) for d in adapter.top_drivers_from_input(payload)]

    return PredictResponse(
        prediction_label=label,
        readmission_probability=round(probability, 6),
        threshold_used=round(threshold, 6),
        positive_class_predicted=predicted_positive,
        risk_band=_risk_band(probability, threshold),
        top_drivers=drivers,
        interpretation=interpretation,
        model_note="Prediction is based on historical data and intended for decision support only.",
    )
