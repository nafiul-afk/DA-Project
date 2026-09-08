import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('diabetic_data.csv')
print("RAW SHAPE:", df.shape)

# --- Missingness check (dataset uses '?' for missing) ---
miss = (df == '?').sum().sort_values(ascending=False)
miss_pct = (miss / len(df) * 100).round(1)
print("\nMISSINGNESS (top 10):")
print(miss_pct.head(10))

# --- Step 1: Drop encounters discharged to hospice/expired (avoid label leakage) ---
# discharge_disposition_id: 11,13,14,19,20,21 = expired/hospice related
exclude_codes = [11, 13, 14, 19, 20, 21]
before = len(df)
df = df[~df['discharge_disposition_id'].isin(exclude_codes)].copy()
print(f"\nDropped {before - len(df)} hospice/expired encounters -> {len(df)} remain")

# --- Step 2: Drop sparse / low-value columns ---
drop_cols = ['weight', 'payer_code', 'medical_specialty',
             'encounter_id',
             'examide', 'citoglipton']  # near-zero variance
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# --- Step 3: Keep first encounter per patient (avoid patient-level leakage across rows) ---
before = len(df)
df = df.sort_values('patient_nbr').drop_duplicates(subset='patient_nbr', keep='first')
print(f"Kept first encounter per patient: {before} -> {len(df)} rows")

# --- Step 4: Collapse ICD-9 diagnosis codes into 9 clinical groups ---
def map_icd9(code):
    if pd.isna(code) or code == '?':
        return 'Other'
    code = str(code)
    if code.startswith('250'):
        return 'Diabetes'
    try:
        val = float(code)
    except ValueError:
        return 'Other'  # V or E codes
    if 390 <= val <= 459 or val == 785:
        return 'Circulatory'
    if 460 <= val <= 519 or val == 786:
        return 'Respiratory'
    if 520 <= val <= 579 or val == 787:
        return 'Digestive'
    if 800 <= val <= 999:
        return 'Injury'
    if 710 <= val <= 739:
        return 'Musculoskeletal'
    if 580 <= val <= 629 or val == 788:
        return 'Genitourinary'
    if 140 <= val <= 239:
        return 'Neoplasms'
    return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col + '_group'] = df[col].apply(map_icd9)
df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

# --- Step 5: Impute remaining '?' as 'Unknown' ---
df = df.replace('?', 'Unknown')

# --- Step 6: Target variable ---
# Binary target: 1 if readmitted <30 days, else 0
df['readmit_30'] = (df['readmitted'] == '<30').astype(int)
print("\nTarget distribution:")
print(df['readmit_30'].value_counts(normalize=True).round(4))

# --- Step 7: Age band -> ordinal encoding ---
age_map = {f'[{i}-{i+10})': i // 10 for i in range(0, 100, 10)}
df['age_ordinal'] = df['age'].map(age_map)

# --- Step 8: Simplify medication columns (23 drug columns -> keep as categorical Up/Down/Steady/No) ---
med_cols = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
            'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
            'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
            'insulin','glyburide-metformin','glipizide-metformin',
            'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']
med_cols = [c for c in med_cols if c in df.columns]

# count of medications actually changed (Up/Down) - a useful engineered feature
df['num_med_changed'] = (df[med_cols].isin(['Up', 'Down'])).sum(axis=1)

df.to_csv('cleaned_data.csv', index=False)
print("\nCLEANED SHAPE:", df.shape)
print("Saved cleaned_data.csv")
