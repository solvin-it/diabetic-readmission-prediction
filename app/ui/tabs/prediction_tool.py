import streamlit as st

from app.ui.services.api_client import predict


def _default_payload() -> dict:
    return {
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
        "change": False,
        "diabetesMed": True,
        "medical_specialty": "Unknown",
        "diag_1_chapter": "circulatory",
        "diag_2_chapter": "circulatory",
        "diag_3_chapter": "other",
    }


def _admission_source_map(source_id: str) -> str:
    """Map admission_source_id from preset to form value."""
    mapping = {
        "1": "referral",
        "2": "transfer",
        "3": "referral",
        "4": "transfer",
        "5": "transfer",
        "6": "transfer",
        "7": "emergency_room",
        "10": "transfer",
        "22": "transfer",
        "25": "transfer",
        "26": "other",
    }
    return mapping.get(source_id, "referral")


def _to_bool(value: object, default: bool = True) -> bool:
    """Coerce typical preset values to boolean for toggle fields."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"yes", "true", "1", "y"}:
            return True
        if norm in {"no", "false", "0", "n"}:
            return False
    return default


def _normalize_age_band(age_band: str) -> str:
    """Normalize preset age band values to model form categories."""
    value = age_band.strip()
    if value.endswith(")"):
        return value
    return f"{value})"


def _normalize_a1c(value: str) -> str:
    """Normalize alternate A1C labels from presets."""
    norm = value.strip().lower()
    if norm in {"<7", "norm", "normal"}:
        return "Norm"
    if norm in {">7", ">8", "none"}:
        return value if value in {">7", ">8", "none"} else value.capitalize()
    return "none"


def _normalize_max_glu(value: str) -> str:
    """Normalize alternate glucose labels from presets."""
    norm = value.strip().lower()
    if norm in {"norm", "normal"}:
        return "Norm"
    if norm in {">200", ">300", "none"}:
        return value if value in {">200", ">300", "none"} else value.capitalize()
    return "none"


def _normalize_specialty(value: str) -> str:
    """Normalize specialty aliases used in presets."""
    if value == "GeneralPractice":
        return "Family/GeneralPractice"
    return value


def _get_specialty_index(specialty: str) -> int:
    """Get selectbox index for medical specialty."""
    specialties = [
        "Cardiology", "Emergency/Trauma", "Endocrinology", "Family/GeneralPractice",
        "Gastroenterology", "Hematology/Oncology", "InternalMedicine", "Nephrology", "Neurology",
        "ObstetricsandGynecology", "Orthopedics", "Orthopedics-Reconstructive", "Other", "Podiatry",
        "Psychology", "Pulmonology", "Radiologist", "Surgery-General", "Surgery-Neuro", "Unknown",
    ]
    try:
        return specialties.index(specialty)
    except ValueError:
        return 19  # Default to Unknown


def _get_age_band_index(age_band: str) -> int:
    """Get selectbox index for age band."""
    bands = ["0-10)", "10-20)", "20-30)", "30-40)", "40-50)", "50-60)", "60-70)", "70-80)", "80-90)", "90-100)"]
    try:
        return bands.index(age_band)
    except ValueError:
        return 7  # Default to 70-80)


def _get_race_index(race: str) -> int:
    """Get selectbox index for race."""
    races = ["AfricanAmerican", "Asian", "Caucasian", "Other", "Unknown"]
    try:
        return races.index(race)
    except ValueError:
        return 2  # Default to Caucasian


def _get_a1c_index(a1c: str) -> int:
    """Get selectbox index for A1C result."""
    options = [">7", ">8", "Norm", "none"]
    try:
        return options.index(a1c)
    except ValueError:
        return 3  # Default to none


def _get_glu_index(glu: str) -> int:
    """Get selectbox index for max glucose serum."""
    options = [">200", ">300", "Norm", "none"]
    try:
        return options.index(glu)
    except ValueError:
        return 3  # Default to none


def _get_insulin_index(insulin: str) -> int:
    """Get selectbox index for insulin."""
    options = ["No", "Steady", "Down", "Up"]
    try:
        return options.index(insulin)
    except ValueError:
        return 1  # Default to Steady


def _get_diag_chapter_index(chapter: str, position: int) -> int:
    """Get selectbox index for diagnosis chapter."""
    if position == 1:
        options = ["circulatory", "congenital", "digestive", "endocrine", "external", "genitourinary",
                   "musculoskeletal", "neoplasm", "nervous_sensory", "other", "pregnancy", "skin", "symptoms"]
    elif position == 2:
        options = ["blood", "circulatory", "congenital", "endocrine", "external", "genitourinary",
                   "infectious", "neoplasm", "other", "pregnancy", "symptoms"]
    else:  # position == 3
        options = ["circulatory", "congenital", "endocrine", "mental", "musculoskeletal", "nervous_sensory",
                   "other", "respiratory", "skin", "supplementary"]
    try:
        return options.index(chapter)
    except ValueError:
        return 0


def render() -> None:
    st.subheader("Prediction Tool")
    st.caption("Estimate 30-day readmission risk from patient encounter attributes.")
    st.warning("Educational capstone demo. Decision support only, not a replacement for clinical judgment.")

    # Initialize payload from preset if available
    if "preset_data" in st.session_state and st.session_state.active_preset:
        preset = st.session_state.preset_data
        payload = _default_payload()
        # Map preset fields to form fields
        payload.update({
            "age_band": _normalize_age_band(str(preset.get("age_band", payload["age_band"]))),
            "gender": preset.get("gender", payload["gender"]),
            "race": preset.get("race", payload["race"]),
            "admission_type_group": str(preset.get("admission_type_group", payload["admission_type_group"])),
            "admission_source_group": _admission_source_map(preset.get("admission_source_id", "1")),
            "discharge_disposition_group": str(preset.get("discharge_disposition_group", payload["discharge_disposition_group"])),
            "time_in_hospital": preset.get("time_in_hospital", payload["time_in_hospital"]),
            "num_lab_procedures": preset.get("num_lab_procedures", payload["num_lab_procedures"]),
            "num_medications": preset.get("num_medications", payload["num_medications"]),
            "num_procedures": preset.get("number_procedures", payload["num_procedures"]),
            "number_diagnoses": preset.get("number_diagnoses", payload["number_diagnoses"]),
            "number_outpatient": preset.get("number_outpatient", payload["number_outpatient"]),
            "number_inpatient": preset.get("number_inpatient", payload["number_inpatient"]),
            "number_emergency": preset.get("number_emergency", payload["number_emergency"]),
            "insulin": preset.get("insulin", payload["insulin"]),
            "max_glu_serum": _normalize_max_glu(str(preset.get("max_glu_serum", payload["max_glu_serum"]))),
            "A1Cresult": _normalize_a1c(str(preset.get("A1Cresult", payload["A1Cresult"]))),
            "change": _to_bool(preset.get("change", payload["change"]), default=payload["change"]),
            "diabetesMed": _to_bool(preset.get("diabetic_medication", payload["diabetesMed"]), default=payload["diabetesMed"]),
            "medical_specialty": _normalize_specialty(str(preset.get("specialty", payload["medical_specialty"]))),
            "diag_1_chapter": str(preset.get("diag_1_chapter", payload["diag_1_chapter"])),
            "diag_2_chapter": str(preset.get("diag_2_chapter", payload["diag_2_chapter"])),
            "diag_3_chapter": str(preset.get("diag_3_chapter", payload["diag_3_chapter"])),
        })
    else:
        payload = _default_payload()

    # Header info for active preset
    if "active_preset" in st.session_state and st.session_state.active_preset:
        st.info(f"📋 **Preset Active:** {st.session_state.active_preset}")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)

        with c1:
            payload["age_band"] = st.selectbox("Age band", ["0-10)", "10-20)", "20-30)", "30-40)", "40-50)", "50-60)", "60-70)", "70-80)", "80-90)", "90-100)"], index=_get_age_band_index(payload["age_band"]))
            payload["gender"] = st.selectbox("Gender", ["Female", "Male"], index=0 if payload["gender"] == "Female" else 1)
            payload["race"] = st.selectbox("Race", ["AfricanAmerican", "Asian", "Caucasian", "Other", "Unknown"], index=_get_race_index(payload["race"]))
            admission_type_options = ["1", "2", "3", "4", "Unknown"]
            payload["admission_type_group"] = st.selectbox(
                "Admission type",
                admission_type_options,
                index=admission_type_options.index(payload["admission_type_group"]) if payload["admission_type_group"] in admission_type_options else 0,
            )
            admission_source_options = ["emergency_room", "referral", "transfer", "other"]
            payload["admission_source_group"] = st.selectbox(
                "Admission source",
                admission_source_options,
                index=admission_source_options.index(payload["admission_source_group"]) if payload["admission_source_group"] in admission_source_options else 0,
            )
            discharge_options = ["facility", "home", "inpatient", "other"]
            payload["discharge_disposition_group"] = st.selectbox(
                "Discharge disposition",
                discharge_options,
                index=discharge_options.index(payload["discharge_disposition_group"]) if payload["discharge_disposition_group"] in discharge_options else 1,
            )
            payload["A1Cresult"] = st.selectbox("A1C result", [">7", ">8", "Norm", "none"], index=_get_a1c_index(payload["A1Cresult"]))
            payload["max_glu_serum"] = st.selectbox("Max glucose serum", [">200", ">300", "Norm", "none"], index=_get_glu_index(payload["max_glu_serum"]))
            payload["insulin"] = st.selectbox("Insulin", ["No", "Steady", "Down", "Up"], index=_get_insulin_index(payload["insulin"]))
            payload["medical_specialty"] = st.selectbox(
                "Medical specialty",
                [
                    "Cardiology", "Emergency/Trauma", "Endocrinology", "Family/GeneralPractice",
                    "Gastroenterology", "Hematology/Oncology", "InternalMedicine", "Nephrology", "Neurology",
                    "ObstetricsandGynecology", "Orthopedics", "Orthopedics-Reconstructive", "Other", "Podiatry",
                    "Psychology", "Pulmonology", "Radiologist", "Surgery-General", "Surgery-Neuro", "Unknown",
                ],
                index=_get_specialty_index(payload["medical_specialty"]),
            )

        with c2:
            payload["time_in_hospital"] = st.number_input("Time in hospital", 1, 30, payload["time_in_hospital"])
            payload["num_lab_procedures"] = st.number_input("Lab procedures", 0, 200, payload.get("num_lab_procedures", 40))
            payload["num_procedures"] = st.number_input("Procedures", 0, 50, payload["num_procedures"])
            payload["num_medications"] = st.number_input("Medications", 0, 100, payload["num_medications"])
            payload["number_diagnoses"] = st.number_input("Number of diagnoses", 1, 20, payload["number_diagnoses"])
            payload["number_outpatient"] = st.number_input("Outpatient visits", 0, 100, payload["number_outpatient"])
            payload["number_emergency"] = st.number_input("Emergency visits", 0, 100, payload["number_emergency"])
            payload["number_inpatient"] = st.number_input("Inpatient visits", 0, 100, payload["number_inpatient"])
            payload["change"] = st.toggle("Medication change", value=payload["change"])
            payload["diabetesMed"] = st.toggle("Diabetes medication", value=payload["diabetesMed"])
            payload["diag_1_chapter"] = st.selectbox(
                "Primary diagnosis chapter",
                ["circulatory", "congenital", "digestive", "endocrine", "external", "genitourinary",
                 "musculoskeletal", "neoplasm", "nervous_sensory", "other", "pregnancy", "skin", "symptoms"],
                index=_get_diag_chapter_index(payload["diag_1_chapter"], 1),
            )
            payload["diag_2_chapter"] = st.selectbox(
                "Secondary diagnosis chapter",
                ["blood", "circulatory", "congenital", "endocrine", "external", "genitourinary",
                 "infectious", "neoplasm", "other", "pregnancy", "symptoms"],
                index=_get_diag_chapter_index(payload["diag_2_chapter"], 2),
            )
            payload["diag_3_chapter"] = st.selectbox(
                "Tertiary diagnosis chapter",
                ["circulatory", "congenital", "endocrine", "mental", "musculoskeletal", "nervous_sensory",
                 "other", "respiratory", "skin", "supplementary"],
                index=_get_diag_chapter_index(payload["diag_3_chapter"], 3),
            )

        col_predict, col_reset, col_spacer = st.columns([1, 1, 2])
        with col_predict:
            submit = st.form_submit_button("🚀 Predict")
        with col_reset:
            reset = st.form_submit_button("🔄 Reset Form")

    if reset:
        st.session_state.active_preset = None
        st.session_state.preset_data = None
        st.rerun()

    if submit:
        try:
            result = predict(payload)
            st.session_state.last_prediction = result
            st.success("Prediction completed")

            label = result["prediction_label"].replace("_", " ").title()
            band = result["risk_band"].title()

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Probability", f"{result['readmission_probability']:.2%}")
            c2.metric("Prediction", label)
            c3.metric("Risk Band", band)

            st.info(result["interpretation"])

            st.markdown("### Top Drivers")
            import pandas as pd
            drivers_df = pd.DataFrame(result["top_drivers"])
            drivers_df.columns = [c.replace("_", " ").title() for c in drivers_df.columns]
            st.dataframe(drivers_df, width="stretch", hide_index=True)

            st.divider()
            st.caption(
                "💬 Want to understand this prediction? Switch to the **Explanation Assistant** tab "
                "and ask a question — your prediction result is automatically shared with the chatbot."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
