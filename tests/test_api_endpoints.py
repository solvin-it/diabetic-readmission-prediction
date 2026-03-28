import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.config import settings
from app.api.main import app
from app.api.services.explanation_service import explanation_service


class TestApiEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _valid_payload(self) -> dict:
        return {
            "age_band": "10-20)",
            "gender": "Female",
            "race": "Caucasian",
            "admission_type_group": "1",
            "admission_source_group": "emergency_room",
            "discharge_disposition_group": "home",
            "time_in_hospital": 4,
            "num_lab_procedures": 40,
            "num_procedures": 0,
            "num_medications": 12,
            "number_diagnoses": 7,
            "number_outpatient": 1,
            "number_emergency": 0,
            "number_inpatient": 0,
            "A1Cresult": ">8",
            "max_glu_serum": ">200",
            "insulin": "Steady",
            "change": False,
            "diabetesMed": True,
            "medical_specialty": "Unknown",
            "diag_1_chapter": "circulatory",
            "diag_2_chapter": "other",
            "diag_3_chapter": "other",
        }

    def test_health_routes(self) -> None:
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.get("/health/ready").status_code, 200)

    def test_predict_success(self) -> None:
        response = self.client.post("/v1/predict", json=self._valid_payload())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("readmission_probability", data)
        self.assertIn("risk_band", data)
        self.assertIn("threshold_used", data)

    def test_predict_validation(self) -> None:
        payload = self._valid_payload()
        payload["age_band"] = "999-1000)"
        response = self.client.post("/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_explain_fallback_mode(self) -> None:
        with patch.object(settings, "openai_api_key", None), patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            response = self.client.post(
                "/v1/explain",
                json={"question": "Summarize the model selection.", "session_id": "s1", "prediction_context": None},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("fallback mode", data["concise_explanation"].lower())

    def test_explain_guardrail_blocks_medical_directive(self) -> None:
        with patch.object(settings, "openai_api_key", None), patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            response = self.client.post(
                "/v1/explain",
                json={
                    "question": "What dosage should I prescribe for this patient?",
                    "session_id": "s2",
                    "prediction_context": {"risk_band": "high"},
                },
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cannot provide diagnosis", data["concise_explanation"].lower())

    def test_explain_timeout_returns_504(self) -> None:
        with patch.object(explanation_service, "explain", side_effect=TimeoutError("slow response")):
            response = self.client.post(
                "/v1/explain",
                json={"question": "Explain the result.", "session_id": "s3", "prediction_context": None},
            )
        self.assertEqual(response.status_code, 504)

    def test_explain_session_isolation(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.turns: dict[str, int] = {}

            def invoke(self, _payload, config):
                thread_id = config["configurable"]["thread_id"]
                turn = self.turns.get(thread_id, 0) + 1
                self.turns[thread_id] = turn
                return {"messages": [SimpleNamespace(content=f"thread={thread_id};turn={turn}")]}

        fake_agent = FakeAgent()
        with patch.object(settings, "openai_api_key", "test-key"), patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch.object(explanation_service, "_ensure_agent", return_value=None):
                with patch.object(explanation_service, "_agent", fake_agent):
                    r1 = self.client.post(
                        "/v1/explain",
                        json={"question": "Explain this prediction.", "session_id": "session-a", "prediction_context": None},
                    )
                    r2 = self.client.post(
                        "/v1/explain",
                        json={"question": "And now summarize.", "session_id": "session-a", "prediction_context": None},
                    )
                    r3 = self.client.post(
                        "/v1/explain",
                        json={"question": "Start fresh.", "session_id": "session-b", "prediction_context": None},
                    )

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r3.status_code, 200)
        self.assertIn("thread=session-a;turn=1", r1.json()["concise_explanation"])
        self.assertIn("thread=session-a;turn=2", r2.json()["concise_explanation"])
        self.assertIn("thread=session-b;turn=1", r3.json()["concise_explanation"])

    def test_explain_reuses_prediction_context_only_once_per_session(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def invoke(self, payload, config=None):
                prompt = payload["messages"][-1]["content"]
                self.prompts.append(prompt)
                return {"messages": [SimpleNamespace(content="ok")]}

        fake_agent = FakeAgent()
        prediction_context = {
            "prediction_label": "readmitted",
            "readmission_probability": 0.62,
            "risk_band": "high",
        }

        with patch.object(settings, "openai_api_key", "test-key"), patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch.object(explanation_service, "_ensure_agent", return_value=None):
                with patch.object(explanation_service, "_agent", fake_agent):
                    explanation_service._session_prediction_contexts.clear()
                    first = self.client.post(
                        "/v1/explain",
                        json={
                            "question": "Explain the result.",
                            "session_id": "context-session",
                            "prediction_context": prediction_context,
                        },
                    )
                    second = self.client.post(
                        "/v1/explain",
                        json={
                            "question": "What matters most?",
                            "session_id": "context-session",
                            "prediction_context": prediction_context,
                        },
                    )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(fake_agent.prompts), 2)
        self.assertIn("Prediction context:", fake_agent.prompts[0])
        self.assertNotIn("Prediction context:", fake_agent.prompts[1])


if __name__ == "__main__":
    unittest.main()
