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
    age_band: AgeBand
    gender: Gender
    race: Race
    admission_type_group: AdmissionType
    admission_source_group: AdmissionSource
    discharge_disposition_group: DischargeDisposition

    time_in_hospital: int = Field(ge=1, le=30)
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=50)
    num_medications: int = Field(ge=0, le=100)
    number_diagnoses: int = Field(ge=1, le=20)
    number_outpatient: int = Field(ge=0, le=100)
    number_emergency: int = Field(ge=0, le=100)
    number_inpatient: int = Field(ge=0, le=100)

    A1Cresult: A1CResult
    max_glu_serum: MaxGluSerum
    insulin: Insulin
    change: bool
    diabetesMed: bool

    medical_specialty: MedicalSpecialty = "Unknown"
    diag_1_chapter: Diag1Chapter = "other"
    diag_2_chapter: Diag2Chapter = "other"
    diag_3_chapter: Diag3Chapter = "other"


class ExplainRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    prediction_context: dict | None = None


class DummyRequest(BaseModel):
    seed: int | None = None
