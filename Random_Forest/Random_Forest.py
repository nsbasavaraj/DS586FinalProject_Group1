import pandas as pd
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

meta_encoded   = pd.read_csv("../Data/meta_dataset_ml_ready.csv", index_col="PATIENT")
meta_readable  = pd.read_csv("../Data/meta_dataset_readable.csv", index_col="PATIENT")

print("Loaded:", meta_encoded.shape)

# ============================================================
# 2. FEATURES & LABELS
# ============================================================

# Identify careplan columns (targets for other tasks)
careplan_cols = [c for c in meta_encoded.columns if c.startswith("CAREPLAN__")]

# Use all numeric features EXCEPT careplans
X = meta_encoded.drop(columns=careplan_cols)

# Pathology label
y_pathology = meta_readable["PATHOLOGY"].astype(str).str.strip().str.lower()
le = LabelEncoder()
y = le.fit_transform(y_pathology)

# ============================================================
# 3. FILTER RARE PATHOLOGIES
# ============================================================
counts = Counter(y)
valid_labels = {label for label, cnt in counts.items() if cnt >= 10}
mask = [label in valid_labels for label in y]

X = X[mask]
y = y[mask]

print("Remaining samples:", len(y))
print("Remaining pathologies:", len(set(y)))

# ============================================================
# 4. KEEP ONLY NUMERIC COLUMNS
# ============================================================
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

# Remove zero-variance columns
X = X.loc[:, X.var() > 0]

print("Final feature count:", X.shape[1])

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train:", len(X_train), "Test:", len(X_test))

# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ============================================================
# 7. EVALUATION
# ============================================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

present = sorted(set(y_test))
names = le.inverse_transform(present)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=present, target_names=names))

# ============================================================
# 8. SAVE MODEL
# ============================================================
joblib.dump(rf, "../Random_Forest/rf_model.pkl")
joblib.dump(le, "../Random_Forest/label_encoder.pkl")

print("\nSaved model + encoder")
print("=" * 60)
print("DONE")
print("=" * 60)

