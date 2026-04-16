import pandas as pd
import numpy as np
import ast
from datetime import datetime

# ============================================================================
# LOAD FILES
# ============================================================================
print("Loading data...")
symptoms = pd.read_csv("P1000/symptoms.csv")
careplans = pd.read_csv("P1000/careplans.csv")
conditions = pd.read_csv("P1000/conditions.csv")
patients = pd.read_csv("P1000/patients.csv")

print(f"Loaded: {len(symptoms)} symptoms, {len(careplans)} careplans, {len(conditions)} conditions, {len(patients)} patients")

# ============================================================================
# CLEAN IDs AND BASIC VALIDATION
# ============================================================================
print("\nCleaning IDs...")
symptoms["PATIENT"] = symptoms["PATIENT"].astype(str).str.strip()
careplans["PATIENT"] = careplans["PATIENT"].astype(str).str.strip()
conditions["PATIENT"] = conditions["PATIENT"].astype(str).str.strip()
patients["Id"] = patients["Id"].astype(str).str.strip()

valid_ids = set(patients["Id"])
symptoms = symptoms[symptoms["PATIENT"].isin(valid_ids)].copy()
careplans = careplans[careplans["PATIENT"].isin(valid_ids)].copy()
conditions = conditions[conditions["PATIENT"].isin(valid_ids)].copy()

print(f"After ID validation: {len(symptoms)} symptoms, {len(careplans)} careplans, {len(conditions)} conditions")

# ============================================================================
# CLEAN TEXT FIELDS
# ============================================================================
print("\nCleaning text fields...")
symptoms["PATHOLOGY"] = symptoms["PATHOLOGY"].astype(str).str.strip().str.lower()
symptoms["SYMPTOMS"] = symptoms["SYMPTOMS"].astype(str).str.strip()
careplans["DESCRIPTION"] = careplans["DESCRIPTION"].astype(str).str.strip().str.lower()
careplans["REASONDESCRIPTION"] = careplans["REASONDESCRIPTION"].astype(str).str.strip().str.lower()

# Remove 'nan' strings
symptoms["PATHOLOGY"] = symptoms["PATHOLOGY"].replace("nan", "")
careplans["REASONDESCRIPTION"] = careplans["REASONDESCRIPTION"].replace("nan", "")

# ============================================================================
# PARSE SYMPTOM LIST
# ============================================================================
def parse_symptom_list(x):
    """Parse SYMPTOMS column which may be a list, delimited string, or single value"""
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    # Try to parse as literal list (Python syntax)
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip() != ""]
    except:
        pass

    # Try common delimiters
    if ";" in x:
        parts = x.split(";")
    elif "," in x:
        parts = x.split(",")
    else:
        parts = [x]

    return [p.strip().lower() for p in parts if p.strip() != ""]

print("\nParsing symptom lists...")
symptoms["SYMPTOM_LIST"] = symptoms["SYMPTOMS"].apply(parse_symptom_list)

# ============================================================================
# KEY FIX: Use REASONDESCRIPTION to link careplans to pathologies
# ============================================================================
print("\nCreating pathology-symptom base table...")
# Group symptoms by (PATIENT, PATHOLOGY) to preserve the relationship
pathology_symptom = symptoms.groupby(["PATIENT", "PATHOLOGY"], as_index=False).agg({
    "GENDER": "first",
    "RACE": "first",
    "ETHNICITY": "first",
    "AGE_BEGIN": "first",
    "AGE_END": "first",
    "NUM_SYMPTOMS": "max",
    "SYMPTOM_LIST": lambda x: sorted(list(set([item for sublist in x for item in sublist])))
})

print(f"Created {len(pathology_symptom)} (patient, pathology) combinations")

# ============================================================================
# LINK CAREPLANS TO PATHOLOGIES VIA REASONDESCRIPTION
# ============================================================================
print("\nLinking careplans to pathologies...")

def get_careplans_for_pathology(patient_id, pathology):
    """
    Find all careplans for a patient that match a specific pathology.
    
    Strategy:
    1. Exact match: REASONDESCRIPTION == PATHOLOGY (primary signal)
    2. Fallback: Use any careplan if no match (ensures data completeness)
    """
    patient_careplans = careplans[careplans["PATIENT"] == patient_id]
    
    if len(patient_careplans) == 0:
        return []
    
    # Try exact match on REASONDESCRIPTION
    matching = patient_careplans[
        (patient_careplans["REASONDESCRIPTION"] != "") & 
        (patient_careplans["REASONDESCRIPTION"] == pathology)
    ]
    
    if len(matching) > 0:
        return sorted(matching["DESCRIPTION"].unique().tolist())
    
    # If no exact match AND pathology is clearly a real diagnosis (not routine/review)
    # then don't assign random careplans (avoids false positives)
    # Only use fallback for common review/check-up pathologies
    routine_keywords = ["medication review", "preventive", "check", "screening"]
    is_routine = any(kw in pathology for kw in routine_keywords)
    
    if is_routine:
        # For routine care, use all patient's careplans
        return sorted(patient_careplans["DESCRIPTION"].unique().tolist())
    else:
        # For specific diagnoses, only link if REASONDESCRIPTION matched
        return []

# Apply the linking function
pathology_symptom["CAREPLAN_LIST"] = pathology_symptom.apply(
    lambda row: get_careplans_for_pathology(row["PATIENT"], row["PATHOLOGY"]),
    axis=1
)

pathology_symptom["NUM_CAREPLANS"] = pathology_symptom["CAREPLAN_LIST"].apply(len)

print(f"Records with careplans: {(pathology_symptom['NUM_CAREPLANS'] > 0).sum()} / {len(pathology_symptom)}")

# ============================================================================
# FILTER: Only keep records with symptoms
# ============================================================================
print("\nFiltering for records with symptoms...")
# Keep only records with symptoms AND at least one careplan
pathology_symptom_filtered = pathology_symptom[
    (pathology_symptom["SYMPTOM_LIST"].apply(len) > 0) &
    (pathology_symptom["NUM_CAREPLANS"] > 0)
].copy()

print(f"After filtering (symptoms only): {len(pathology_symptom_filtered)} records")

# ============================================================================
# CREATE BINARY FEATURES FOR SYMPTOMS
# ============================================================================
print("\nCreating binary symptom features...")
all_symptoms = sorted(
    set(symptom for sublist in pathology_symptom_filtered["SYMPTOM_LIST"] 
        for symptom in sublist)
)

for symptom in all_symptoms:
    pathology_symptom_filtered[f"SYMPTOM__{symptom}"] = (
        pathology_symptom_filtered["SYMPTOM_LIST"].apply(
            lambda x: int(symptom in x)
        )
    )

pathology_symptom_filtered["NUM_SYMPTOMS_COMPUTED"] = (
    pathology_symptom_filtered["SYMPTOM_LIST"].apply(len)
)

print(f"Total unique symptoms: {len(all_symptoms)}")

# ============================================================================
# CREATE BINARY FEATURES FOR CAREPLANS
# ============================================================================
print("\nCreating binary careplan features...")
all_careplans = sorted(
    set(cp for sublist in pathology_symptom_filtered["CAREPLAN_LIST"] 
        for cp in sublist)
)

for cp in all_careplans:
    pathology_symptom_filtered[f"CAREPLAN__{cp}"] = (
        pathology_symptom_filtered["CAREPLAN_LIST"].apply(
            lambda x: int(cp in x)
        )
    )

print(f"Total unique careplans: {len(all_careplans)}")

# ============================================================================
# CLEAN UP & HANDLE MISSING VALUES
# ============================================================================
print("\nHandling missing values...")
final_data = pathology_symptom_filtered.copy()

# Fill numeric columns
numeric_fill_cols = ["AGE_BEGIN", "AGE_END", "NUM_SYMPTOMS", "NUM_SYMPTOMS_COMPUTED", "NUM_CAREPLANS"]
for col in numeric_fill_cols:
    if col in final_data.columns:
        final_data[col] = pd.to_numeric(final_data[col], errors="coerce").fillna(0)

final_data["PATHOLOGY"] = final_data["PATHOLOGY"].fillna("unknown")
final_data["SYMPTOM_LIST"] = final_data["SYMPTOM_LIST"].apply(
    lambda x: x if isinstance(x, list) else []
)
final_data["CAREPLAN_LIST"] = final_data["CAREPLAN_LIST"].apply(
    lambda x: x if isinstance(x, list) else []
)

# ============================================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\nEncoding categorical variables...")
final_encoded = pd.get_dummies(
    final_data,
    columns=["GENDER", "RACE", "ETHNICITY"],
    dummy_na=True
)

# ============================================================================
# NORMALIZE NUMERICAL FEATURES
# ============================================================================
print("\nNormalizing numerical features...")
scale_cols = [
    c for c in ["AGE_BEGIN", "AGE_END", "NUM_SYMPTOMS", "NUM_SYMPTOMS_COMPUTED", "NUM_CAREPLANS"]
    if c in final_encoded.columns
]

for col in scale_cols:
    col_min = final_encoded[col].min()
    col_max = final_encoded[col].max()
    if col_max > col_min:
        final_encoded[col] = (final_encoded[col] - col_min) / (col_max - col_min)
    else:
        final_encoded[col] = 0

# ============================================================================
# SANITY CHECKS
# ============================================================================
print("\n" + "="*80)
print("FINAL CHECKS")
print("="*80)
print(f"Final shape: {final_encoded.shape}")
print(f"Unique patients: {final_data['PATIENT'].nunique()}")
print(f"Unique (patient, pathology) combinations: {len(final_data)}")
print(f"Missing values in final dataset: {final_encoded.isna().sum().sum()}")
print(f"Duplicate patient-pathology rows: {final_data.duplicated(subset=['PATIENT', 'PATHOLOGY']).sum()}")

# Check class balance for careplans
print(f"\nCareplan distribution:")
print(f"  Records with 0 careplans: {(final_data['NUM_CAREPLANS'] == 0).sum()}")
print(f"  Records with 1+ careplans: {(final_data['NUM_CAREPLANS'] > 0).sum()}")
print(f"  Mean careplans per record: {final_data['NUM_CAREPLANS'].mean():.2f}")

print(f"\nSymptom distribution:")
print(f"  Mean symptoms per record: {final_data['NUM_SYMPTOMS_COMPUTED'].mean():.2f}")
print(f"  Max symptoms in one record: {final_data['NUM_SYMPTOMS_COMPUTED'].max()}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

final_data.to_csv("../meta_dataset_readable.csv", index=False)
final_encoded.to_csv("../meta_dataset_ml_ready.csv", index=False)

print("✓ ../meta_dataset_readable.csv")
print("✓ ../meta_dataset_ml_ready.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

print(f"\nInput:")
print(f"  - {len(symptoms)} symptom records from {symptoms['PATIENT'].nunique()} patients")
print(f"  - {len(careplans)} careplan records")
print(f"  - {symptoms['PATHOLOGY'].nunique()} unique pathologies")

print(f"\nOutput:")
print(f"  - {len(final_data)} (patient × pathology) records")
print(f"  - {final_data['PATIENT'].nunique()} unique patients")
print(f"  - {final_data['PATHOLOGY'].nunique()} unique pathologies")
print(f"  - {len(all_symptoms)} unique symptoms (binary features: SYMPTOM__*)")
print(f"  - {len(all_careplans)} unique careplans (binary features: CAREPLAN__*)")

print(f"\nPredictive task:")
print(f"  INPUT: symptoms + pathology + demographics")
print(f"  OUTPUT: which careplans are needed?")
print(f"  TARGET: binary classification for each careplan (or multi-label)")

print(f"\nTop 10 most common careplans:")
careplan_counts = final_encoded[[c for c in final_encoded.columns if c.startswith("CAREPLAN__")]].sum().sort_values(ascending=False)
for cp, count in careplan_counts.head(10).items():
    cp_name = cp.replace("CAREPLAN__", "")
    print(f"  {cp_name}: {count} records ({100*count/len(final_encoded):.1f}%)")

print(f"\nTop 10 most common pathologies:")
pathology_counts = final_data['PATHOLOGY'].value_counts().head(10)
for path, count in pathology_counts.items():
    print(f"  {path}: {count} records ({100*count/len(final_data):.1f}%)")

print("\nPreview of readable dataset (first 5 rows):")
print(final_data[["PATIENT", "PATHOLOGY", "NUM_SYMPTOMS_COMPUTED", "SYMPTOM_LIST", "NUM_CAREPLANS", "CAREPLAN_LIST"]].head())

print("\n✓ Preprocessing complete!")