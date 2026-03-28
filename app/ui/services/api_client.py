from __future__ import annotations

import os
from typing import Any

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_model_info() -> dict[str, Any]:
    r = requests.get(f"{API_BASE_URL}/v1/model-info", timeout=20)
    r.raise_for_status()
    return r.json()


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{API_BASE_URL}/v1/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def explain(question: str, session_id: str, prediction_context: dict | None) -> dict[str, Any]:
    r = requests.post(
        f"{API_BASE_URL}/v1/explain",
        json={
            "question": question,
            "session_id": session_id,
            "prediction_context": prediction_context,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()
