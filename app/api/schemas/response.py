from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    loaded_artifacts: list[str]
    details: dict | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_track: str
    optimal_threshold: float
    test_auc: float
    test_recall_at_threshold: float
    test_precision_at_threshold: float
    test_f1_at_threshold: float
    training_data: str


class DriverItem(BaseModel):
    feature: str
    value: float | int | str
    contribution_hint: str


class PredictResponse(BaseModel):
    prediction_label: str
    readmission_probability: float
    threshold_used: float
    positive_class_predicted: bool
    risk_band: str
    top_drivers: list[DriverItem]
    interpretation: str
    model_note: str


class DummyResponse(BaseModel):
    generated_value: float
    generated_at: str


class ExplainResponse(BaseModel):
    concise_explanation: str
    cautionary_note: str
    evidence_points: list[str]
    source_refs: list[str]
