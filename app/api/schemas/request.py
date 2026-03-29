from pydantic import BaseModel, Field
from typing import Literal


AgeBand = Literal["0-10)", "10-20)", "20-30)", "30-40)", "40-50)", "50-60)", "60-70)", "70-80)", "80-90)", "90-100)"]
Gender = Literal["Female", "Male"]
Race = Literal["AfricanAmerican", "Asian", "Caucasian", "Other", "Unknown"]
AdmissionType = Literal["1", "2", "3", "4", "Unknown"]
AdmissionSource = Literal["emergency_room", "referral", "transfer", "other"]
DischargeDisposition = Literal["facility", "home", "inpatient", "other"]
A1CResult = Literal[">7", ">8", "Norm", "none"]
MaxGluSerum = Literal[">200", ">300", "Norm", "none"]
Insulin = Literal["No", "Steady", "Down", "Up"]

Diag1Chapter = Literal[
    "circulatory", "congenital", "digestive", "endocrine", "external", "genitourinary",
    "musculoskeletal", "neoplasm", "nervous_sensory", "other", "pregnancy", "skin", "symptoms",
]
Diag2Chapter = Literal[
    "blood", "circulatory", "congenital", "endocrine", "external", "genitourinary",
    "infectious", "neoplasm", "other", "pregnancy", "symptoms",
]
Diag3Chapter = Literal[
    "circulatory", "congenital", "endocrine", "mental", "musculoskeletal", "nervous_sensory",
    "other", "respiratory", "skin", "supplementary",
]

MedicalSpecialty = Literal[
    "Cardiology", "Emergency/Trauma", "Endocrinology", "Family/GeneralPractice",
    "Gastroenterology", "Hematology/Oncology", "InternalMedicine", "Nephrology", "Neurology",
    "ObstetricsandGynecology", "Orthopedics", "Orthopedics-Reconstructive", "Other", "Podiatry",
    "Psychology", "Pulmonology", "Radiologist", "Surgery-General", "Surgery-Neuro", "Unknown",
]


class PredictRequest(BaseModel):
    age_band: AgeBand = Field(description="Age band at encounter time.", examples=["70-80)"])
    gender: Gender = Field(description="Administrative gender value.", examples=["Female"])
    race: Race = Field(description="Race category from source dataset.", examples=["Caucasian"])
    admission_type_group: AdmissionType = Field(
        description="Grouped admission type code (1=Emergency, 2=Urgent, 3=Elective, 4=Newborn).",
        examples=["1"],
    )
    admission_source_group: AdmissionSource = Field(
        description="Grouped admission source category used in modeling.",
        examples=["emergency_room"],
    )
    discharge_disposition_group: DischargeDisposition = Field(
        description="Grouped discharge disposition category.",
        examples=["home"],
    )

    time_in_hospital: int = Field(ge=1, le=30, description="Length of stay in days.", examples=[4])
    num_lab_procedures: int = Field(ge=0, le=200, description="Number of lab procedures during encounter.", examples=[40])
    num_procedures: int = Field(ge=0, le=50, description="Number of non-lab procedures during encounter.", examples=[1])
    num_medications: int = Field(ge=0, le=100, description="Count of medications administered.", examples=[12])
    number_diagnoses: int = Field(ge=1, le=20, description="Number of diagnoses coded for the encounter.", examples=[8])
    number_outpatient: int = Field(ge=0, le=100, description="Prior outpatient visits in lookback period.", examples=[0])
    number_emergency: int = Field(ge=0, le=100, description="Prior emergency visits in lookback period.", examples=[0])
    number_inpatient: int = Field(ge=0, le=100, description="Prior inpatient visits in lookback period.", examples=[0])

    A1Cresult: A1CResult = Field(description="A1C test bucket.", examples=[">8"])
    max_glu_serum: MaxGluSerum = Field(description="Maximum serum glucose bucket.", examples=[">200"])
    insulin: Insulin = Field(description="Insulin adjustment category.", examples=["Steady"])
    change: bool = Field(description="Whether diabetic medication was changed during encounter.", examples=[False])
    diabetesMed: bool = Field(description="Whether diabetic medication is prescribed.", examples=[True])

    medical_specialty: MedicalSpecialty = Field(default="Unknown", description="Medical specialty group.", examples=["InternalMedicine"])
    diag_1_chapter: Diag1Chapter = Field(default="other", description="Primary diagnosis ICD chapter group.", examples=["circulatory"])
    diag_2_chapter: Diag2Chapter = Field(default="other", description="Secondary diagnosis ICD chapter group.", examples=["other"])
    diag_3_chapter: Diag3Chapter = Field(default="other", description="Tertiary diagnosis ICD chapter group.", examples=["other"])

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_band": "70-80)",
                "gender": "Female",
                "race": "Caucasian",
                "admission_type_group": "1",
                "admission_source_group": "emergency_room",
                "discharge_disposition_group": "home",
                "time_in_hospital": 4,
                "num_lab_procedures": 40,
                "num_procedures": 1,
                "num_medications": 12,
                "number_diagnoses": 8,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 0,
                "A1Cresult": ">8",
                "max_glu_serum": ">200",
                "insulin": "Steady",
                "change": False,
                "diabetesMed": True,
                "medical_specialty": "InternalMedicine",
                "diag_1_chapter": "circulatory",
                "diag_2_chapter": "other",
                "diag_3_chapter": "other",
            }
        }
    }


class ExplainRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, description="User question about model output or behavior.")
    session_id: str = Field(default="default", min_length=1, max_length=128, description="Conversation/session identifier for context continuity.")
    prediction_context: dict | None = Field(
        default=None,
        description="Optional prediction payload to ground the explanation.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Why is this patient in the moderate risk band?",
                "session_id": "demo-session-1",
                "prediction_context": {
                    "prediction_label": "unlikely_readmitted",
                    "readmission_probability": 0.43,
                    "risk_band": "moderate",
                },
            }
        }
    }
