from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.api.config import settings


class ModelService:
    """Thread-safe lazy loader for deployment artifacts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded = False
        self._bundle: dict[str, Any] | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def model_dir(self) -> Path:
        return Path(settings.model_dir)

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            bundle_path = self.model_dir / "deployment_pipeline.joblib"
            metadata_path = self.model_dir / "best_model_metadata.json"

            self._bundle = joblib.load(bundle_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self._loaded = True

    def readiness(self) -> tuple[bool, list[str], dict[str, str] | None]:
        artifacts = [
            "deployment_pipeline.joblib",
            "best_model_metadata.json",
            "selected_features.json",
        ]
        missing = [a for a in artifacts if not (self.model_dir / a).exists()]
        if missing:
            return False, [], {"missing": ", ".join(missing)}

        try:
            self._load()
        except Exception as exc:  # pragma: no cover
            return False, [], {"load_error": str(exc)}

        return True, artifacts, None

    def model_info(self) -> dict[str, Any]:
        self._load()
        assert self._metadata is not None
        return {
            "model_name": self._metadata["model"],
            "feature_track": self._metadata["feature_track"],
            "optimal_threshold": float(self._metadata["optimal_threshold"]),
            "test_auc": float(self._metadata["test_auc"]),
            "test_recall_at_threshold": float(self._metadata["test_recall_at_threshold"]),
            "test_precision_at_threshold": float(self._metadata["test_precision_at_threshold"]),
            "test_f1_at_threshold": float(self._metadata["test_f1_at_threshold"]),
            "training_data": self._metadata.get("training_data", "unknown"),
        }

    def predict_proba(self, features_df: pd.DataFrame) -> float:
        self._load()
        assert self._bundle is not None
        pipeline = self._bundle["pipeline"]

        # Suppress non-actionable sklearn warnings from mixed feature-name metadata
        # in persisted artifacts (some steps were fitted with names, others without).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"X has feature names, but .* was fitted without feature names",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"X does not have valid feature names, but .* was fitted with feature names",
                category=UserWarning,
            )
            proba = pipeline.predict_proba(features_df)[:, 1]
        return float(proba[0])

    def threshold(self) -> float:
        self._load()
        assert self._bundle is not None
        return float(self._bundle["optimal_threshold"])

    def expected_features(self) -> list[str]:
        self._load()
        assert self._bundle is not None
        return list(self._bundle["features_expected"])


model_service = ModelService()
