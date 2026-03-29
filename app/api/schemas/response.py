from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Liveness status.", examples=["ok"])
    service: str = Field(description="Service identifier.", examples=["api"])
    version: str = Field(description="API version.", examples=["1.0.0"])


class ReadyResponse(BaseModel):
    status: str = Field(description="Readiness status.", examples=["ready"])
    loaded_artifacts: list[str] = Field(description="Loaded model/preprocessing artifacts.")
    details: dict | None = Field(default=None, description="Optional readiness diagnostics.")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(description="Deployed model name.")
    feature_track: str = Field(description="Feature pipeline variant used by the model.")
    optimal_threshold: float = Field(description="Operating threshold used to assign positive class.")
    test_auc: float = Field(description="AUC-ROC on held-out test set.")
    test_recall_at_threshold: float = Field(description="Recall measured at operating threshold.")
    test_precision_at_threshold: float = Field(description="Precision measured at operating threshold.")
    test_f1_at_threshold: float = Field(description="F1 score measured at operating threshold.")
    training_data: str = Field(description="Training data balancing/preparation note.")


class DriverItem(BaseModel):
    feature: str = Field(description="Feature name surfaced as a top driver hint.")
    value: float | int | str = Field(description="Input value associated with the driver.")
    contribution_hint: str = Field(description="Human-readable directional hint for this driver.")


class PredictResponse(BaseModel):
    prediction_label: str = Field(description="Classification label derived from threshold comparison.")
    readmission_probability: float = Field(description="Predicted probability of readmission within 30 days.")
    threshold_used: float = Field(description="Threshold applied for positive class decision.")
    positive_class_predicted: bool = Field(description="True when probability >= threshold.")
    risk_band: str = Field(description="Band assignment under threshold-relative policy: low/moderate/high.")
    top_drivers: list[DriverItem] = Field(description="Top explanatory drivers derived from input features.")
    interpretation: str = Field(description="Natural-language interpretation for decision support.")
    model_note: str = Field(description="Usage caveat/disclaimer for model output.")


class ExplainResponse(BaseModel):
    concise_explanation: str = Field(description="Primary explanation answer.")
    cautionary_note: str = Field(description="Safety-oriented reminder and usage limitation.")
    evidence_points: list[str] = Field(description="Evidence bullets supporting the explanation.")
    source_refs: list[str] = Field(description="Source references used for grounding.")
