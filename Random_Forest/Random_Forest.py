import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from collections import defaultdict
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 1. LOAD DATA

print("=" * 60)
print("LOADING DATA")
print("=" * 60)
 
meta_encoded   = pd.read_csv("../meta_dataset_ml_ready.csv", index_col="PATIENT")
meta_readable  = pd.read_csv("../meta_dataset_readable.csv", index_col="PATIENT")
 
print(f"Dataset shape: {meta_encoded.shape}")

# 2. FEATURES AND LABELS
symptom_cols = [c for c in meta_encoded.columns if c.startswith("SYMPTOM__")]
X = meta_encoded[symptom_cols]
 
y_pathology = meta_readable["PATHOLOGY"].astype(str).str.strip().str.lower()
le = LabelEncoder()
y = le.fit_transform(y_pathology)
 
print(f"Features (symptoms): {X.shape[1]}")
print(f"Unique pathologies : {len(le.classes_)}")

# Remove pathologies with fewer than 2 patients
from collections import Counter
counts = Counter(y)
valid_indices = [i for i, label in enumerate(y) if counts[label] >= 2]

X = X.iloc[valid_indices]
y = y[valid_indices]

print(f"Patients after filtering rare pathologies: {len(y)}")
print(f"Remaining pathologies: {len(set(y))}")
 
# 3. TRAIN / TEST SPLIT  80 / 20
print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT  80 / 20")
print("=" * 60)
 
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.20, random_state=42, stratify=y
)
 
print(f"Train size: {len(X_train)} patients")
print(f"Test  size: {len(X_test)}  patients")

# 4. TRAIN RANDOM FOREST
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)
 
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
 
y_pred = rf.predict(X_test)
 
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
present_classes = sorted(set(y_test))
present_names = le.inverse_transform(present_classes)
print(classification_report(y_test, y_pred, labels=present_classes, target_names=present_names)) 
# 5. FEATURE IMPORTANCE PLOT
importances = pd.Series(rf.feature_importances_, index=symptom_cols)
top15 = importances.sort_values(ascending=False).head(15)
 
fig, ax = plt.subplots(figsize=(10, 6))
top15.plot(kind="bar", ax=ax)
ax.set_title("Top 15 Most Important Symptoms")
ax.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../feature_importance.png", dpi=150)
plt.close()
print("\nSaved: ../feature_importance.png")
 
# 6. SAVE MODEL
joblib.dump(rf, "../rf_model.pkl")
joblib.dump(le, "../label_encoder.pkl")
print("Saved: ../rf_model.pkl")
print("Saved: ../label_encoder.pkl")
 
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
 