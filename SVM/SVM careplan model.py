import warnings
warnings.filterwarnings("ignore")
##Importing libss
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# SETTINGS
CSV_PATH        = "meta_dataset_ml_ready.csv"
FEATURE_COLS_PATH  = "feature_cols.pkl"
CAREPLAN_COLS_PATH = "careplan_cols.pkl"

MIN_PATHOLOGY_COUNT = 2
TEST_SIZE           = 0.20
RANDOM_STATE        = 42
CAREPLAN_THRESHOLD  = 0.30   # a careplan is recommended if ≥30 % of matching
                              # training samples had it


# STEP 6.0: LOAD DATA & FEATURE DEFINITIONS
print("=" * 60)
print("STEP 6 — Load Data")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")

feature_cols  = joblib.load(FEATURE_COLS_PATH)
careplan_cols = joblib.load(CAREPLAN_COLS_PATH)

# Keep only feature/careplan cols that actually exist in the CSV
feature_cols  = [c for c in feature_cols  if c in df.columns]
careplan_cols = [c for c in careplan_cols if c in df.columns]
print(f"Feature columns used : {len(feature_cols)}")
print(f"Careplan columns used: {len(careplan_cols)}")

# Drop rare pathologies so every class has enough samples to stratify
path_counts      = df["PATHOLOGY"].value_counts()
valid_pathologies = path_counts[path_counts >= MIN_PATHOLOGY_COUNT].index
df = df[df["PATHOLOGY"].isin(valid_pathologies)].copy()
print(f"Rows after rare-class filter: {len(df)}")
print(f"Unique pathologies           : {df['PATHOLOGY'].nunique()}")


# STEP 6.1: TRAIN / TEST SPLIT
print("\n" + "=" * 60)
print("STEP 6.1 — Train / Test Split (80 / 20, stratified)")
print("=" * 60)

X = df[feature_cols].astype(np.float32).values
y = df["PATHOLOGY"].astype(str).values
y_care = df[careplan_cols].astype(np.float32).values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test, y_care_train, y_care_test = train_test_split(
    X, y_encoded, y_care,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

print(f"Train samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}")

# Scale features — SVMs are sensitive to feature magnitude
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# STEP 6.2: TRAIN BASELINE SVM
print("\n" + "=" * 60)
print("STEP 6.2 — Train Baseline SVM (RBF kernel)")
print("=" * 60)

baseline_svm = SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=RANDOM_STATE)
baseline_svm.fit(X_train, y_train)

y_pred_base = baseline_svm.predict(X_test)
base_acc = accuracy_score(y_test, y_pred_base)
base_f1  = f1_score(y_test, y_pred_base, average="weighted", zero_division=0)
print(f"Baseline Accuracy : {base_acc:.4f}")
print(f"Baseline F1 (wtd) : {base_f1:.4f}")


# STEP 6.3: HYPERPARAMETER TUNING
print("\n" + "=" * 60)
print("STEP 6.3 — Hyperparameter Tuning (GridSearchCV)")
print("=" * 60)

param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C"     : [0.1, 1.0, 10.0],
    "gamma" : ["scale", "auto"],   # only used by rbf/poly but ignored for linear
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    SVC(probability=True, random_state=RANDOM_STATE),
    param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
    refit=True,
)

grid_search.fit(X_train, y_train)

print(f"\nBest params : {grid_search.best_params_}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")

best_svm = grid_search.best_estimator_

# ============================================================
# EVALUATE TUNED MODEL ON HELD-OUT TEST SET
# ============================================================
print("\n" + "=" * 60)
print("STEP 6.3 — Tuned SVM — Test-Set Evaluation")
print("=" * 60)

y_pred  = best_svm.predict(X_test)
y_proba = best_svm.predict_proba(X_test)

tuned_acc = accuracy_score(y_test, y_pred)
tuned_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# Top-3 accuracy
top3_correct = 0
for i, true_label in enumerate(y_test):
    top3 = np.argsort(y_proba[i])[-3:]
    if true_label in top3:
        top3_correct += 1
top3_acc = top3_correct / len(y_test)

print(f"Accuracy (top-1) : {tuned_acc:.4f}")
print(f"F1 (weighted)    : {tuned_f1:.4f}")
print(f"Top-3 Accuracy   : {top3_acc:.4f}")

print("\nClassification Report:")
# Only include labels that actually appear in the test set
present_labels = sorted(set(y_test))
present_names  = label_encoder.inverse_transform(present_labels)
print(classification_report(
    y_test, y_pred,
    labels=present_labels,
    target_names=present_names,
    zero_division=0,
))


# STEP 7: CAREPLAN RECOMMENDATION LOGIC (TWO-STAGE)
print("=" * 60)
print("STEP 7 — Build Pathology → Careplan Lookup Table")
print("=" * 60)

"""
Stage 1: SVM predicts pathology from symptoms.
Stage 2: Predicted pathology is used to look up the most common
         careplans seen for that condition in the training data.

We build a per-pathology careplan frequency table from X_train so
the recommendation is grounded only in training-set evidence
(no data leakage from the test set).
"""

# Decode training labels back to string pathology names
y_train_names = label_encoder.inverse_transform(y_train)

# Build lookup: pathology → average careplan presence in training data
careplan_lookup = {}
for path in label_encoder.classes_:
    mask = (y_train_names == path)
    if mask.sum() == 0:
        careplan_lookup[path] = []
        continue
    # Mean presence of each careplan for this pathology
    mean_presence = y_care_train[mask].mean(axis=0)
    # Keep careplans that appear in at least CAREPLAN_THRESHOLD of cases
    recommended = [
        careplan_cols[i].replace("CAREPLAN__", "")
        for i, freq in enumerate(mean_presence)
        if freq >= CAREPLAN_THRESHOLD
    ]
    careplan_lookup[path] = recommended

print("Careplan lookup table built.")
print(f"Pathologies with ≥1 recommended careplan: "
      f"{sum(1 for v in careplan_lookup.values() if len(v) > 0)}"
      f" / {len(careplan_lookup)}\n")

# Quick sanity peek
for path, cps in list(careplan_lookup.items())[:4]:
    print(f"  {path}")
    for cp in cps:
        print(f"    → {cp}")
    if not cps:
        print("    (none above threshold)")


# STEP 7: VALIDATE CAREPLAN RECOMMENDATIONS ON TEST SET
print("\n" + "=" * 60)
print("STEP 7 — Careplan Recommendation Validation (test set)")
print("=" * 60)

"""
Metric: sample-wise F1 between the recommended careplan set
(derived from predicted pathology) and the actual careplan set.
"""

care_f1_scores = []
for i in range(len(X_test)):
    pred_path_name = label_encoder.inverse_transform([y_pred[i]])[0]
    rec_careplans  = set(careplan_lookup.get(pred_path_name, []))

    # Ground-truth careplan set
    true_care_flags = y_care_test[i]
    true_careplans  = set(
        careplan_cols[j].replace("CAREPLAN__", "")
        for j, flag in enumerate(true_care_flags) if flag == 1
    )

    # Sample F1
    tp = len(rec_careplans & true_careplans)
    fp = len(rec_careplans - true_careplans)
    fn = len(true_careplans - rec_careplans)

    if tp == 0 and fp == 0 and fn == 0:
        f1_i = 1.0
    elif tp == 0:
        f1_i = 0.0
    else:
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        f1_i = 2 * prec * rec / (prec + rec)

    care_f1_scores.append(f1_i)

careplan_rec_f1 = float(np.mean(care_f1_scores))
print(f"Careplan Recommendation Sample-wise F1: {careplan_rec_f1:.4f}")

# STEP 8 — RECOMMENDATION FUNCTION FOR NEW PATIENTS
print("\n" + "=" * 60)
print("STEP 8 — generate_recommendation() for new patients")
print("=" * 60)

def generate_recommendation(
    symptom_list,
    age=None,
    num_symptoms=None,
    top_k_pathologies=3,
):
    """
    Given a list of symptom strings, return the top-k predicted
    pathologies and their recommended careplans.

    Parameters
    ----------
    symptom_list : list[str]
        e.g. ["SYMPTOM__wheezing", "SYMPTOM__cough"]
        Must match the exact column names in feature_cols.
    age : float, optional
        Normalised age in [0, 1].  Pass None to leave at 0.
    num_symptoms : float, optional
        Normalised symptom count in [0, 1].  Pass None to leave at 0.
    top_k_pathologies : int
        How many top pathologies to return.

    Returns
    -------
    list of dicts with keys: rank, pathology, confidence, careplans
    """
    # Build feature vector
    input_dict = {col: 0.0 for col in feature_cols}
    for s in symptom_list:
        if s in input_dict:
            input_dict[s] = 1.0
        else:
            print(f"  [warning] '{s}' not found in feature_cols — skipped")

    if age is not None and "AGE_BEGIN" in input_dict:
        input_dict["AGE_BEGIN"] = float(age)
    if num_symptoms is not None and "NUM_SYMPTOMS" in input_dict:
        input_dict["NUM_SYMPTOMS"] = float(num_symptoms)

    x_raw = np.array([input_dict[c] for c in feature_cols],
                     dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x_raw)

    # Stage 1 — predict pathology probabilities
    proba = best_svm.predict_proba(x_scaled)[0]
    top_k_idx = np.argsort(proba)[-top_k_pathologies:][::-1]

    results = []
    for rank, idx in enumerate(top_k_idx, start=1):
        path_name  = label_encoder.inverse_transform([idx])[0]
        confidence = float(proba[idx])
        # Stage 2 — look up careplans for this pathology
        careplans  = careplan_lookup.get(path_name, [])

        results.append({
            "rank"      : rank,
            "pathology" : path_name,
            "confidence": round(confidence, 4),
            "careplans" : careplans,
        })

    return results


# DEMO — three example patients

# Helper: find all feature columns whose name contains a keyword
def find_symptom_cols(keyword):
    """Return all feature_col names that contain `keyword` (case-insensitive)."""
    kw = keyword.lower()
    return [c for c in feature_cols if kw in c.lower()]

# NOTE: Symptom columns in this dataset are named like:
#   SYMPTOM__<symptom text>:<severity_integer>
# Use find_symptom_cols() to turn plain text keywords into real column names.

demo_cases = [
    {
        "label"   : "Patient A — asthma-like symptoms",
        "symptoms": (find_symptom_cols("wheez") +
                     find_symptom_cols("cough") +
                     find_symptom_cols("shortness of breath")),
        "age"         : 0.25,
        "num_symptoms": 0.3,
    },
    {
        "label"   : "Patient B — joint / knee pain",
        "symptoms": (find_symptom_cols("knee pain") +
                     find_symptom_cols("joint pain") +
                     find_symptom_cols("range of motion")),
        "age"         : 0.55,
        "num_symptoms": 0.3,
    },
    {
        "label"   : "Patient C — cardiac symptoms",
        "symptoms": (find_symptom_cols("dyspnea") +
                     find_symptom_cols("fatigue") +
                     find_symptom_cols("ankle")),
        "age"         : 0.70,
        "num_symptoms": 0.3,
    },
]

for case in demo_cases:
    print(f"\n{'─'*55}")
    print(f"  {case['label']}")
    print(f"{'─'*55}")
    recommendations = generate_recommendation(
        symptom_list  = case["symptoms"],
        age           = case.get("age"),
        num_symptoms  = case.get("num_symptoms"),
        top_k_pathologies=3,
    )
    for r in recommendations:
        print(f"\n  Rank {r['rank']} — {r['pathology']}")
        print(f"          Confidence : {r['confidence']:.2%}")
        if r["careplans"]:
            print(f"          Careplans  :")
            for cp in r["careplans"]:
                print(f"            • {cp}")
        else:
            print("          Careplans  : (none above threshold for this condition)")


# SAVE ARTEFACTS
print("\n" + "=" * 60)
print("Saving model artefacts...")
print("=" * 60)

joblib.dump(best_svm,        "svm_model.pkl")
joblib.dump(scaler,          "svm_scaler.pkl")
joblib.dump(label_encoder,   "svm_label_encoder.pkl")
joblib.dump(careplan_lookup, "careplan_lookup.pkl")

print("Saved:")
print("  svm_model.pkl         — tuned SVC")
print("  svm_scaler.pkl        — StandardScaler fitted on X_train")
print("  svm_label_encoder.pkl — LabelEncoder for pathology names")
print("  careplan_lookup.pkl   — pathology → careplan mapping")


# FINAL SUMMARY
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Best SVM params    : {grid_search.best_params_}")
print(f"  Pathology Accuracy : {tuned_acc:.4f}")
print(f"  Pathology F1 (wtd) : {tuned_f1:.4f}")
print(f"  Top-3 Accuracy     : {top3_acc:.4f}")
print(f"  Careplan Rec F1    : {careplan_rec_f1:.4f}")
print("\nDone.")