import unittest

from app.api.schemas.request import PredictRequest
from app.api.services.feature_adapter import FeatureAdapter
from app.api.services.model_service import model_service


class TestFeatureAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = FeatureAdapter(expected_features=model_service.expected_features())

    def test_output_shape_and_order(self) -> None:
        payload = PredictRequest(
            age_band="10-20)",
            gender="Female",
            race="Caucasian",
            admission_type_group="1",
            admission_source_group="emergency_room",
            discharge_disposition_group="home",
            time_in_hospital=3,
            num_lab_procedures=35,
            num_procedures=1,
            num_medications=10,
            number_diagnoses=6,
            number_outpatient=0,
            number_emergency=1,
            number_inpatient=0,
            A1Cresult=">8",
            max_glu_serum=">200",
            insulin="Down",
            change=True,
            diabetesMed=True,
            medical_specialty="Unknown",
            diag_1_chapter="circulatory",
            diag_2_chapter="other",
            diag_3_chapter="other",
        )

        df = self.adapter.transform(payload)

        self.assertEqual(df.shape, (1, 117))
        self.assertEqual(list(df.columns), model_service.expected_features())

    def test_engineered_flags_present(self) -> None:
        payload = PredictRequest(
            age_band="70-80)",
            gender="Male",
            race="AfricanAmerican",
            admission_type_group="2",
            admission_source_group="transfer",
            discharge_disposition_group="facility",
            time_in_hospital=8,
            num_lab_procedures=60,
            num_procedures=0,
            num_medications=20,
            number_diagnoses=10,
            number_outpatient=2,
            number_emergency=3,
            number_inpatient=1,
            A1Cresult=">7",
            max_glu_serum=">300",
            insulin="Up",
            change=True,
            diabetesMed=True,
            medical_specialty="InternalMedicine",
            diag_1_chapter="endocrine",
            diag_2_chapter="circulatory",
            diag_3_chapter="respiratory",
        )

        df = self.adapter.transform(payload)

        if "had_emergency" in df.columns:
            self.assertEqual(float(df.iloc[0]["had_emergency"]), 1.0)
        if "had_inpatient" in df.columns:
            self.assertEqual(float(df.iloc[0]["had_inpatient"]), 1.0)
        if "log_emergency" in df.columns:
            self.assertGreater(float(df.iloc[0]["log_emergency"]), 0.0)

    def test_insulin_and_bucket_mappings(self) -> None:
        payload = PredictRequest(
            age_band="20-30)",
            gender="Female",
            race="Other",
            admission_type_group="3",
            admission_source_group="referral",
            discharge_disposition_group="other",
            time_in_hospital=5,
            num_lab_procedures=30,
            num_procedures=2,
            num_medications=15,
            number_diagnoses=8,
            number_outpatient=1,
            number_emergency=0,
            number_inpatient=0,
            A1Cresult=">8",
            max_glu_serum=">200",
            insulin="Up",
            change=True,
            diabetesMed=True,
            medical_specialty="Unknown",
            diag_1_chapter="other",
            diag_2_chapter="other",
            diag_3_chapter="other",
        )

        df = self.adapter.transform(payload)

        if "insulin" in df.columns:
            self.assertEqual(float(df.iloc[0]["insulin"]), 2.0)

        if "A1Cresult_>7" in df.columns:
            self.assertEqual(float(df.iloc[0]["A1Cresult_>7"]), 1.0)

        if "max_glu_serum_>300" in df.columns:
            self.assertEqual(float(df.iloc[0]["max_glu_serum_>300"]), 1.0)


if __name__ == "__main__":
    unittest.main()
