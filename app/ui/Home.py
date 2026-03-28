import sys
from pathlib import Path

# Add project root to sys.path so absolute imports work under Streamlit
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from app.ui.tabs.project_summary import render as render_summary
from app.ui.tabs.prediction_tool import render as render_prediction
from app.ui.tabs.explanation_assistant import render as render_assistant


st.set_page_config(
    page_title="Diabetic Readmission Demo",
    page_icon="🏥",
    layout="wide",
)

st.title("30-Day Readmission Risk Demo")

summary_tab, prediction_tab, assistant_tab = st.tabs(
    ["Project Summary", "Prediction Tool", "Explanation Assistant"]
)

with summary_tab:
    render_summary()

with prediction_tab:
    render_prediction()

with assistant_tab:
    render_assistant()
