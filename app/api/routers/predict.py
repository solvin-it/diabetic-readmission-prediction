from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.api.schemas.request import PredictRequest
from app.api.schemas.response import ModelInfoResponse, PredictResponse, DriverItem
from app.api.services.feature_adapter import FeatureAdapter
from app.api.services.model_service import model_service


router = APIRouter(prefix="/v1", tags=["predict"])


def _risk_band(probability: float) -> str:
    if probability < 0.10:
        return "low"
    if probability <= 0.20:
        return "moderate"
    return "high"


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
        risk_band=_risk_band(probability),
        top_drivers=drivers,
        interpretation=interpretation,
        model_note="Prediction is based on historical data and intended for decision support only.",
    )
