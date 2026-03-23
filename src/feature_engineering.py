import numpy as np
import pandas as pd

def add_utilization_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer binary flags and log1p transforms for the zero-inflated utilization columns.

    All three prior-utilization variables (number_outpatient, number_emergency,
    number_inpatient) are heavily zero-inflated with long right tails. This function
    addresses both issues simultaneously:

    Binary flags ('had_outpatient', 'had_emergency', 'had_inpatient', 'had_procedures')
        Capture any-vs-none variation cleanly. For highly zero-inflated columns the
        binary flag often carries as much discriminative signal as the raw count and is
        robust to extreme values.

    Log1p transforms ('log_outpatient', 'log_emergency', 'log_inpatient')
        Compress the right tail for linear and distance-based models. log1p(x) = log(1+x)
        is safe for zero values (log1p(0) = 0) while substantially reducing skewness.

    The original count columns are retained so that tree-based models (Random Forest,
    XGBoost) can partition on raw values without any information loss.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after leakage filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with utilization flags and log1p transforms added as new columns.
    """
    df_out = dataframe.copy()

    util_cols  = ['number_outpatient', 'number_emergency', 'number_inpatient']
    flag_names = ['had_outpatient',    'had_emergency',    'had_inpatient']
    log_names  = ['log_outpatient',    'log_emergency',    'log_inpatient']

    for raw, flag, log in zip(util_cols, flag_names, log_names):
        df_out[flag] = (df_out[raw] > 0).astype(int)
        df_out[log]  = np.log1p(df_out[raw])

    df_out['had_procedures'] = (df_out['num_procedures'] > 0).astype(int)

    print("✓ Binary utilization flags added:", flag_names + ['had_procedures'])
    print("✓ Log1p transforms added:", log_names)
    return df_out

def recode_admission_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate rare/unknown admission_type_id codes into a single 'Unknown' category.

    The EDA identified that codes 5, 6, and 8 represent:
      - 5: Not Available
      - 6: NULL
      - 8: Not Mapped
    
    These ~10% of records represent administratively undefined admission types.
    Consolidating them into a single 'Unknown' category prevents the model from
    learning spurious patterns on rare administrative codes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after gender cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame with admission codes 5,6,8 consolidated to 'Unknown'.
    """
    df_out = dataframe.copy()
    unknown_admission_codes = {5, 6, 8}
    
    # Map unknown codes to the string 'Unknown'
    df_out['admission_type_id'] = df_out['admission_type_id'].apply(
        lambda x: 'Unknown' if x in unknown_admission_codes else x
    )
    
    n_unknown = (dataframe['admission_type_id'].isin(unknown_admission_codes)).sum()
    print(f"✓ Consolidated {n_unknown:,} records ({100*n_unknown/len(dataframe):.2f}%) with admission codes {{5,6,8}} → 'Unknown'")
    print(f"  Admission type value counts:")
    print(df_out['admission_type_id'].value_counts().to_string())
    return df_out

def group_discharge_disposition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse discharge_disposition_id into four clinically coherent string groups.

    The 25+ distinct numeric codes are replaced with interpretable labels that
    reflect the patient's post-discharge care trajectory:

        'home'      Discharged to home (with or without home-health support).
                    Raw codes: 1, 6, 8
                    Baseline group; lowest mean readmission rate (~9.3%).

        'facility'  Transferred to a post-acute or long-term care facility
                    (SNF, ICF, rehab unit, long-term hospital, federal facility,
                    psychiatric hospital, CAH, etc.).
                    Raw codes: 2, 3, 4, 5, 10, 15, 22, 23, 24, 25, 26, 27, 28, 29, 30
                    Elevated readmission risk; includes the highest-risk SNF/rehab codes.

        'inpatient' Still in care or expected to return for inpatient services.
                    Raw codes: 9, 12

        'other'     AMA (left against medical advice), hospice discharge, outpatient
                    referrals, unknown, unmapped, and any remaining codes not assigned
                    to the three groups above.

    Note: discharge_disposition_id == 11 (Expired) is excluded upstream by
    filter_leakage_records(), so it will not appear here.

    The original numeric column is replaced in-place with the string labels.
    One-hot encoding is deferred to the feature-engineering step.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after utilization feature engineering.

    Returns
    -------
    pd.DataFrame
        DataFrame with discharge_disposition_id replaced by string group labels.
    """
    df_out = dataframe.copy()

    discharge_map = {
        1: 'home', 6: 'home', 8: 'home',
        2: 'facility', 3: 'facility',  4: 'facility',  5: 'facility',
        10: 'facility', 15: 'facility', 22: 'facility', 23: 'facility',
        24: 'facility', 25: 'facility', 26: 'facility', 27: 'facility',
        28: 'facility', 29: 'facility', 30: 'facility',
        9: 'inpatient', 12: 'inpatient',
    }

    df_out['discharge_disposition_id'] = (
        df_out['discharge_disposition_id']
        .map(discharge_map)
        .fillna('other')
    )

    print("✓ discharge_disposition_id grouped into clinical categories:")
    print(df_out['discharge_disposition_id'].value_counts().to_string())
    return df_out

def group_admission_source(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse admission_source_id into four clinically interpretable string groups.

    The coded numeric values are mapped to interpretable labels that capture the
    clinical meaning of how a patient arrived at the hospital:

        'emergency_room'  Arrived directly via the emergency department.
                          Raw codes: 7
                          Highest-volume group; likely higher acuity on arrival.

        'referral'        Admitted via a physician or clinic referral, representing
                          a planned or semi-planned admission.
                          Raw codes: 1, 2

        'transfer'        Transferred from another healthcare facility (hospital, SNF,
                          HMO, ambulatory surgery center, hospice, or other care setting).
                          Raw codes: 3, 4, 5, 6, 10, 18, 22, 25, 26
                          Bivariate analysis showed HMO referral (code 3, 15.51%) and
                          transfer codes carried elevated readmission rates, motivating
                          a dedicated group rather than folding into 'other'.

        'other'           Court/Law Enforcement, not-available, null, unknown,
                          and any remaining codes not assigned above.

    The original numeric column is replaced with the string labels.
    One-hot encoding is deferred to the feature-engineering step.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after discharge disposition grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with admission_source_id replaced by string group labels.
    """
    df_out = dataframe.copy()

    admission_map = {
        7:  'emergency_room',
        1:  'referral', 2: 'referral',
        3:  'transfer', 4: 'transfer', 5:  'transfer', 6: 'transfer',
        10: 'transfer', 18: 'transfer', 22: 'transfer', 25: 'transfer', 26: 'transfer',
    }

    df_out['admission_source_id'] = (
        df_out['admission_source_id']
        .map(admission_map)
        .fillna('other')
    )

    print("✓ admission_source_id grouped into clinical categories:")
    print(df_out['admission_source_id'].value_counts().to_string())
    return df_out

def encode_clinical_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicator flags from categorical clinical variables and recode
    change/diabetesMed to 0/1 integers.

    New columns created:

        specialty_known   1 if medical_specialty is not 'Unknown', else 0.
                          Separates the informative specialty signal from pure
                          missingness; allows models to use both dimensions of
                          this feature without conflating them.

        insulin_adjusted  1 if insulin was actively changed during admission
                          ('Down' or 'Up'), else 0. Captures whether the
                          treating physician adjusted dosage — a signal of
                          suboptimal glycaemic control on entry.

        glucose_tested    1 if max_glu_serum is not 'none', else 0.
                          Whether an inpatient glucose test was ordered is a
                          clinical decision that may independently distinguish
                          high-acuity from routine admissions.

        A1C_tested        1 if A1Cresult is not 'none', else 0.
                          Analogous reasoning to glucose_tested.

    Columns recoded in-place:

        change      'Ch' → 1 (medication change occurred), 'No' → 0.
        diabetesMed 'Yes' → 1 (diabetes medication prescribed), 'No' → 0.

    The original insulin, max_glu_serum, and A1Cresult columns are retained
    with their level categories for use in feature engineering.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after admission source grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with the four indicator columns added and change/diabetesMed recoded.
    """
    df_out = dataframe.copy()

    df_out['specialty_known']  = (df_out['medical_specialty'] != 'Unknown').astype(int)
    df_out['insulin_adjusted'] = df_out['insulin'].isin(['Down', 'Up']).astype(int)
    df_out['glucose_tested']   = (df_out['max_glu_serum'] != 'none').astype(int)
    df_out['A1C_tested']       = (df_out['A1Cresult'] != 'none').astype(int)

    df_out['change']      = df_out['change'].map({'Ch': 1, 'No': 0})
    df_out['diabetesMed'] = df_out['diabetesMed'].map({'Yes': 1, 'No': 0})

    print("✓ Binary indicator flags added: specialty_known, insulin_adjusted, glucose_tested, A1C_tested")
    print("✓ Recoded: change (Ch→1 / No→0), diabetesMed (Yes→1 / No→0)")
    return df_out

def map_icd9_to_chapter(code) -> str:
    """
    Map an ICD-9 code string to one of 16 standard clinical chapter labels.

    Handles numeric codes, V-codes (supplementary), and E-codes (external causes).
    Unrecognized or missing values are mapped to 'other'.
    """
    if pd.isna(code):
        return 'other'
    code = str(code).strip()
    if code.upper().startswith('V'):
        return 'supplementary'
    if code.upper().startswith('E'):
        return 'external'
    try:
        num = float(code)
    except ValueError:
        return 'other'
    if   1   <= num <= 139: return 'infectious'
    elif 140 <= num <= 239: return 'neoplasm'
    elif 240 <= num <= 279: return 'endocrine'
    elif 280 <= num <= 289: return 'blood'
    elif 290 <= num <= 319: return 'mental'
    elif 320 <= num <= 389: return 'nervous_sensory'
    elif 390 <= num <= 459: return 'circulatory'
    elif 460 <= num <= 519: return 'respiratory'
    elif 520 <= num <= 579: return 'digestive'
    elif 580 <= num <= 629: return 'genitourinary'
    elif 630 <= num <= 679: return 'pregnancy'
    elif 680 <= num <= 709: return 'skin'
    elif 710 <= num <= 739: return 'musculoskeletal'
    elif 740 <= num <= 759: return 'congenital'
    elif 760 <= num <= 799: return 'symptoms'
    elif 800 <= num <= 999: return 'injury'
    return 'other'