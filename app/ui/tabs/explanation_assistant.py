import time
import uuid
from collections.abc import Iterator
from typing import Any

import streamlit as st

from app.ui.services.api_client import explain

_WELCOME = (
    "👋 Hi! I'm the **Explanation Assistant**, grounded in this project's "
    "capstone documentation and model outputs.\n\n"
    "Ask me about:\n"
    "- **Model behavior** — feature importance, PCA components, Random Forest\n"
    "- **Threshold tradeoffs** — precision vs. recall at different operating points\n"
    "- **Fairness analysis** — performance across race and gender subgroups\n"
    "- **Your latest prediction** — explain individual risk scores"
)


def _stream_response(text: str) -> Iterator[str]:
    """Stream text word-by-word while preserving all whitespace for markdown."""
    for i, line in enumerate(text.split("\n")):
        if i > 0:
            yield "\n"
        for j, word in enumerate(line.split(" ")):
            if j > 0:
                yield " "
            yield word
            if word:
                time.sleep(0.015)


def _render_message_extras(message: dict[str, Any]) -> None:
    if message.get("cautionary_note"):
        st.caption(message["cautionary_note"])
    if message.get("source_refs"):
        with st.expander("📚 Sources"):
            for ref in message["source_refs"]:
                st.markdown(f"- {ref}")


def _finalize_pending_message(message_index: int) -> None:
    pending_message = st.session_state.chat_messages[message_index]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                result = explain(
                    question=pending_message["question"],
                    session_id=st.session_state.chat_session_id,
                    prediction_context=st.session_state.last_prediction,
                )

            answer = result["concise_explanation"]
            st.write_stream(_stream_response(answer))

            finalized_message = {
                "role": "assistant",
                "content": answer,
                "cautionary_note": result.get("cautionary_note"),
                "source_refs": result.get("source_refs", []),
            }
            _render_message_extras(finalized_message)
            st.session_state.chat_messages[message_index] = finalized_message
        except Exception as exc:
            error_message = {
                "role": "assistant",
                "content": f"⚠️ Request failed: {exc}",
                "cautionary_note": None,
                "source_refs": [],
            }
            st.error(error_message["content"])
            st.session_state.chat_messages[message_index] = error_message


def render() -> None:
    # -- Session state --
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    # -- Compact toolbar --
    cols = st.columns([6, 1, 1])
    with cols[0]:
        if st.session_state.last_prediction:
            risk = str(st.session_state.last_prediction.get("risk_band", "")).title()
            prob = st.session_state.last_prediction.get("readmission_probability")
            if isinstance(prob, (float, int)):
                st.caption(f"🔗 Prediction context — {risk} risk ({prob:.2%})")
    with cols[1]:
        if st.button("🗑️ Clear", width="stretch"):
            st.session_state.chat_messages = []
            st.rerun()
    with cols[2]:
        if st.button("🔄 New", width="stretch"):
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.rerun()

    st.divider()

    # -- Welcome message (only when history is empty; not stored) --
    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.markdown(_WELCOME)

    # -- Display chat messages from history on app rerun --
    for index, msg in enumerate(st.session_state.chat_messages):
        if msg.get("pending"):
            _finalize_pending_message(index)
            continue

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_message_extras(msg)

    # -- Accept user input --
    if prompt := st.chat_input("Ask about model behavior, metrics, or your latest prediction"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": "",
                "pending": True,
                "question": prompt,
            }
        )
        st.rerun()
