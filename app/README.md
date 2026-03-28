# Diabetic Readmission Prediction — Deployment Guide

This directory contains the deployment layer for the diabetic readmission prediction capstone: a FastAPI backend with async inference and an LLM-grounded explanation chatbot, plus a Streamlit web interface.

**System Architecture:**

```
Client (Browser)
    ↓
Streamlit UI (port 8501)
    ├→ GET /health/live, GET /health/ready        [Health checks]
    ├→ GET /v1/model-info                          [Model metadata]
    ├→ POST /v1/predict                            [Risk predictions]
    └→ POST /v1/explain                            [LLM-grounded Q&A]
    ↓
FastAPI (port 8000)
    ├→ ModelService (thread-safe lazy-load)
    ├→ FeatureAdapter (pre-pipeline transformer)
    ├→ ExplanationService (LangChain CAG + session memory)
    └→ scikit-learn Pipeline (StandardScaler → PCA → RandomForest)
    ↓
Model Artifacts
    ├→ models/deployment_pipeline.joblib
    ├→ models/best_model_metadata.json
    └→ models/selected_features.json
```

---

## Quick Start

### Option 1: Docker Compose (Recommended)

**Prerequisites:**
- Docker & Docker Compose installed
- `.env` file with `OPENAI_API_KEY` (see `.env.template` example below)

**Launch:**
```bash
cd /path/to/diabetic-readmission-prediction
docker compose up
```

**Access:**
- **API**: `http://localhost:8000`
- **UI**: `http://localhost:8501`
- **API Docs**: `http://localhost:8000/docs` (interactive Swagger)

**Stop:**
```bash
docker compose down
```

### Option 2: Local Development (Python)

**Prerequisites:**
- Python 3.12+
- Virtual environment set up and activated
- `.venv` directory with dependencies installed

**Install dependencies:**
```bash
# API
pip install -r app/requirements.api.txt

# UI
pip install -r app/requirements.ui.txt

# Or both
pip install -r requirements.txt
```

**Start API:**
```bash
cd app/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Start UI (in a separate terminal):**
```bash
cd app/ui
export API_BASE_URL=http://localhost:8000
streamlit run Home.py
```

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Health Checks

#### `GET /health/live`
Readiness check for load balancers / Kubernetes.

**Response (200):**
```json
{
  "status": "ok",
  "timestamp": "2026-03-29T12:34:56.789Z"
}
```

#### `GET /health/ready`
Detailed readiness check including model artifact validation.

**Response (200):**
```json
{
  "status": "ready",
  "model_available": true,
  "artifact_path": "models/deployment_pipeline.joblib",
  "feature_count": 117,
  "expected_threshold": 0.4556
}
```

**Response (503) if model artifacts unavailable:**
```json
{
  "status": "not_ready",
  "model_available": false,
  "error": "Model artifacts not found or could not be loaded."
}
```

---

### Model Information

#### `GET /v1/model-info`
Fetch model metadata and performance metrics.

**Response (200):**
```json
{
  "model_name": "Random Forest — PCA",
  "test_auc": 0.6446,
  "test_recall_at_threshold": 0.7164,
  "test_precision_at_threshold": 0.15,
  "optimal_threshold": 0.4556,
  "feature_count": 117,
  "pca_components": 44,
  "pca_variance_retained": 0.9521
}
```

---

### Prediction Endpoint

#### `POST /v1/predict`
Estimate 30-day readmission risk for a patient.

**Request:**
```json
{
  "age_band": "70-80)",
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
  "A1Cresult": "none",
  "max_glu_serum": "none",
  "insulin": "Steady",
  "change": false,
  "diabetesMed": true,
  "medical_specialty": "Unknown",
  "diag_1_chapter": "circulatory",
  "diag_2_chapter": "circulatory",
  "diag_3_chapter": "other"
}
```

**Valid Enumerations:**
- `age_band`: `"0-10)"`, `"10-20)"`, ..., `"90-100)"`
- `gender`: `"Female"`, `"Male"`
- `race`: `"AfricanAmerican"`, `"Asian"`, `"Caucasian"`, `"Other"`, `"Unknown"`
- `admission_type_group`: `"1"`, `"2"`, `"3"`, `"4"`, `"Unknown"`
- `admission_source_group`: `"emergency_room"`, `"referral"`, `"transfer"`, `"other"`
- `discharge_disposition_group`: `"facility"`, `"home"`, `"inpatient"`, `"other"`
- `A1Cresult`: `">7"`, `">8"`, `"Norm"`, `"none"`
- `max_glu_serum`: `">200"`, `">300"`, `"Norm"`, `"none"`
- `insulin`: `"No"`, `"Steady"`, `"Down"`, `"Up"`
- `medical_specialty`: (20 values) `"Cardiology"`, `"Emergency/Trauma"`, `"Endocrinology"`, `"Family/GeneralPractice"`, etc.
- `diag_*_chapter`: (13 ICD-9 chapters) `"circulatory"`, `"congenital"`, `"digestive"`, `"endocrine"`, etc.

**Response (200):**
```json
{
  "readmission_probability": 0.62,
  "prediction_label": "Readmit",
  "risk_band": "moderate",
  "threshold_used": 0.4556,
  "top_drivers": {
    "log_emergency": 0.045,
    "number_emergency": 0.038,
    "log_inpatient": 0.032
  },
  "interpretation": "Moderate readmission risk. High utilization history (emergency and inpatient visits) is the primary risk signal. Consider outpatient follow-up and care coordination.",
  "disclaimer": "For decision support only. Not a clinical diagnosis or treatment recommendation."
}
```

**Response (422) for invalid input:**
```json
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body", "age_band"],
      "msg": "Input should be 'clinical' or 'research' [type=enum, input_value='invalid', input_type=str]",
      "input": "invalid"
    }
  ]
}
```

---

### Explanation Endpoint

#### `POST /v1/explain`
Ask context-aware questions about the model and predictions using an LLM.

**Request:**
```json
{
  "question": "Why did the model flag this patient as high-risk?",
  "session_id": "user-123-session",
  "prediction_context": {
    "readmission_probability": 0.62,
    "risk_band": "moderate",
    "top_drivers": {
      "log_emergency": 0.045,
      "number_emergency": 0.038
    }
  }
}
```

**Response (200):**
```json
{
  "concise_explanation": "The model flagged this patient as higher risk primarily because of prior utilization history, especially emergency and inpatient encounters. Those features were among the strongest signals in the final model and indicate a pattern associated with higher short-term readmission risk.",
  "cautionary_note": "This output is for decision support and education only. It is not a substitute for clinical judgment.",
  "evidence_points": [
    "Answer generated with repository-grounded context.",
    "Conversation memory persisted per session_id."
  ],
  "source_refs": [
    "reports/final_report.md",
    "README.md"
  ]
}
```

**Response (200) with fallback (no OpenAI key):**
```json
{
  "concise_explanation": "AI explanation is running in fallback mode because no API key is configured. Predicted probability is 62.00% (readmitted).",
  "cautionary_note": "This is a decision-support educational output, not a clinical directive.",
  "evidence_points": [
    "Grounded by repository metadata and final report context."
  ],
  "source_refs": [
    "reports/final_report.md",
    "README.md"
  ]
}
```

**Response (504) for timeout:**
```json
{
  "detail": "Explanation request timed out."
}
```

---

## UI Tabs Overview

### Tab 1: Project Summary
Displays the complete model analysis including:
- **Performance metrics**: AUC, recall, precision, threshold
- **Model comparison table**: 8 candidate models ranked by test AUC
- **Feature importance (SHAP)**: Top 9 drivers with mean absolute impact
- **Fairness analysis**: AUC and recall parity across gender and race groups
- **Visualizations**: ROC curves, confusion matrices, SHAP beeswarm, threshold analysis, fairness charts
- **Sample patient presets**: Quick-demo templates (high-risk elderly, moderate-risk, low-risk)

### Tab 2: Prediction Tool
Interactive form to estimate readmission risk:
- **Input sections**: Demographics, admission details, diagnoses, utilization history, medications
- **Grouped form layout**: Intuitive two-column design
- **Sample presets**: Click to pre-fill the form with realistic patient profiles
- **Results display**: Probability, label, risk band, interpretation, top 3 drivers
- **Reset button**: Clear the form and active presets

### Tab 3: Explanation Assistant
Conversational interface for model and prediction questions:
- **Question input**: Free-text field for user queries
- **Session persistence**: Each user gets a separate conversation thread ID
- **Prediction context**: Automatically references the last prediction from Tab 2 if available
- **Response display**: AI-generated explanation with sources and disclaimer
- **Fallback mode**: Safe response if OpenAI API key is not configured

---

## GenAI Safeguards & Fallback Behavior

### System Prompt Grounding
The explanation chatbot uses a system prompt grounded in:
1. `reports/final_report.md` — Detailed capstone findings and methodology
2. `README.md` — Project overview and fairness summary

This ensures responses stay within the scope of the trained model and documented analysis.

### Guardrails
- **Boundary detection**: System prompt explicitly discourages medical directives or prescriptions
- **Scope limitation**: Prevents the model from claims about causality or clinical guidance beyond decision support
- **Fallback mode**: If `OPENAI_API_KEY` is not set, the service returns a deterministic, safe response based on model facts

### Timeout Policy
- Explanation requests default to a 60-second timeout
- On timeout, API returns HTTP 504 with a helpful message

---

## Environment Variables

Create a `.env` file in the project root (or in `compose.yaml` `env_file`):

```env
# API & UI Configuration
LOG_LEVEL=INFO
API_BASE_URL=http://localhost:8000

# OpenAI API (for explanation chatbot; optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-nano

# Model Paths (defaults: models/deployment_pipeline.joblib, etc.)
MODEL_PATH=models/deployment_pipeline.joblib

# Docker/Production
PORT=8000
WORKERS=4
RELOAD=false
```

### Example `.env.template`
```env
# Copy this to .env and fill in your values

# Logging
LOG_LEVEL=INFO

# OpenAI Integration (required for explanation chatbot; see fallback behavior above)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.4-nano

# API Configuration
PORT=8000
WORKERS=4
RELOAD=false

# UI Configuration
API_BASE_URL=http://localhost:8000
STREAMLIT_SERVER_PORT=8501

# Model Paths
MODEL_PATH=models/deployment_pipeline.joblib
```

---

## Troubleshooting

### API won't start: "No module named 'app'"

**Cause**: Missing `app/__init__.py` or incorrect working directory.

**Solution**:
```bash
touch app/__init__.py
cd /path/to/diabetic-readmission-prediction
python -m uvicorn app.api.main:app --reload
```

---

### API returns 503 "Model artifacts not found"

**Cause**: `models/deployment_pipeline.joblib` or dependencies are missing.

**Solution**:
```bash
# Verify files exist
ls -la models/
# Should list: deployment_pipeline.joblib, best_model_metadata.json, selected_features.json

# If missing, ensure you've run the full Jupyter workflow.
# See ../notebooks/03_modeling_and_evaluation.ipynb
```

---

### Prediction endpoint returns 422 "Invalid enum value"

**Cause**: One or more request fields have a value not in the allowed enumeration.

**Solution**: Check the request against the valid enumerations listed in the API Reference (e.g., age_band must be one of `"0-10)"`, `"10-20)"`, etc.).

**Example:**
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age_band": "invalid_value", ...}' 
# Returns 422 with field-level error details
```

---

### Explanation endpoint returns fallback response

**Cause**: `OPENAI_API_KEY` is not set in `.env` or environment.

**Solution**:
1. Obtain an OpenAI API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Set in `.env`:
   ```env
   OPENAI_API_KEY=sk-...
   ```
3. Restart the API:
   ```bash
  docker compose restart api
   # or
   Ctrl+C and restart uvicorn
   ```

---

### Streamlit can't connect to API

**Cause**: `API_BASE_URL` is incorrect or API is not running.

**Solution**:
1. Verify API is running: `curl http://localhost:8000/health/live`
2. Check `API_BASE_URL` in `app/ui/services/api_client.py` or `.env`
3. If using Docker Compose, ensure both services are running: `docker compose ps`

---

### Port 8000 or 8501 already in use

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000
# Kill it or use a different port
docker compose down  # Clean up containers
# Then restart
docker compose up
```

---

## Local Development Workflow

### Running tests
```bash
cd /path/to/diabetic-readmission-prediction
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests cover:
- Feature adapter parity (shape, column order, insulin encoding)
- API endpoint validation (health checks, predict success/failure)
- Schema enforcement (invalid enum values rejected with 422)

### Viewing API documentation
```
curl http://localhost:8000/docs
```
This opens an interactive Swagger interface where you can test all endpoints.

### Adding a new prediction field
1. Update `app/api/schemas/request.py` (add field to `PredictRequest`)
2. Update `app/api/services/feature_adapter.py` (add transformation logic)
3. Update `app/ui/tabs/prediction_tool.py` (add form input)
4. Add tests in `tests/test_feature_adapter.py`

---

## Production Deployment: GCP Cloud Run

### Prerequisites
- GCP project with Cloud Run enabled
- gcloud CLI installed and authenticated
- Docker image pushed to Artifact Registry

### High-Level Steps

**1. Build and push image to Artifact Registry:**
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/diabetic-api:latest app/
gcloud builds submit --tag gcr.io/$PROJECT_ID/diabetic-ui:latest app/
```

**2. Create secrets in Secret Manager:**
```bash
echo -n "sk-..." | gcloud secrets create OPENAI_API_KEY --data-file=-
```

**3. Deploy API to Cloud Run:**
```bash
gcloud run deploy diabetic-api \
  --image gcr.io/$PROJECT_ID/diabetic-api:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars LOG_LEVEL=INFO \
  --secret OPENAI_API_KEY=OPENAI_API_KEY:latest \
  --allow-unauthenticated
```

**4. Deploy UI to Cloud Run:**
```bash
gcloud run deploy diabetic-ui \
  --image gcr.io/$PROJECT_ID/diabetic-ui:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars API_BASE_URL=https://diabetic-api-xxx.a.run.app
```

### Monitoring
- View logs: `gcloud run logs read diabetic-api --limit 50`
- View metrics: Cloud Console → Cloud Run → Service

---

## Contributing & Support

For issues or questions:
1. Check Troubleshooting section above
2. Review API test cases in `tests/`
3. Consult model analysis in `../reports/final_report.md` and `../notebooks/03_modeling_and_evaluation.ipynb`

---

## License

See [../LICENSE](../LICENSE)

---

## Disclaimer

**This is an educational and decision-support tool.** Predictions are NOT clinical diagnoses or medical recommendations. Always consult with healthcare providers for actual patient care decisions. The model was trained on historical data from 1999–2008 and may not reflect current clinical practice or population characteristics.
