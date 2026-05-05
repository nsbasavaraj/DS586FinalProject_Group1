import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


from model import DualDataset, DualHeadMLP

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "../Code/meta_dataset_ml_ready_large.csv"
MODEL_PATH = "dual_model.pt"

MIN_PATHOLOGY_COUNT = 5
RANDOM_STATE = 42

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 5.575e-4
WEIGHT_DECAY = 7.633e-05

VAL_TEST_SIZE = 0.30   # 70% train, 30% temp
TEST_SIZE_FROM_TEMP = 0.50  # 15% validation, 15% test

PATIENCE = 10

# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Original shape: {df.shape}")

if "PATHOLOGY" not in df.columns:
    raise ValueError("PATHOLOGY column not found in dataset.")

# =========================================================
# FILTER RARE PATHOLOGIES
# =========================================================
path_counts = df["PATHOLOGY"].value_counts()
valid_pathologies = path_counts[path_counts >= MIN_PATHOLOGY_COUNT].index
df = df[df["PATHOLOGY"].isin(valid_pathologies)].copy()

print(f"Rows after pathology filtering: {len(df)}")
print(f"Pathology classes after filtering: {df['PATHOLOGY'].nunique()}")

if len(df) == 0:
    raise ValueError("No rows left after pathology filtering.")

# =========================================================
# DEFINE FEATURES / TARGETS
# =========================================================
symptom_cols = [c for c in df.columns if c.startswith("SYMPTOM__")]
careplan_cols = [c for c in df.columns if c.startswith("CAREPLAN__")]
demo_cols = [
    c for c in df.columns
    if c.startswith("GENDER_") or c.startswith("RACE_") or c.startswith("ETHNICITY_")
]
numeric_cols = [
    c for c in ["AGE_BEGIN", "AGE_END", "NUM_SYMPTOMS", "NUM_SYMPTOMS_COMPUTED"]
    if c in df.columns
]

feature_cols = symptom_cols + demo_cols + numeric_cols

print(f"Feature columns: {len(feature_cols)}")
print(f"Careplan target columns: {len(careplan_cols)}")

if len(feature_cols) == 0:
    raise ValueError("No feature columns found.")
if len(careplan_cols) == 0:
    raise ValueError("No careplan target columns found.")

# =========================================================
# PREPARE ARRAYS
# =========================================================
X = df[feature_cols].astype(np.float32).values

label_encoder = LabelEncoder()
y_path = label_encoder.fit_transform(df["PATHOLOGY"].astype(str))

y_care = df[careplan_cols].astype(np.float32).values

# Save metadata for prediction / test
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(careplan_cols, "careplan_cols.pkl")
joblib.dump(label_encoder, "pathology_label_encoder.pkl")

print("Saved:")
print(" - feature_cols.pkl")
print(" - careplan_cols.pkl")
print(" - pathology_label_encoder.pkl")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
# First split: train + temp
X_train, X_temp, y_path_train, y_path_temp, y_care_train, y_care_temp = train_test_split(
    X,
    y_path,
    y_care,
    test_size=VAL_TEST_SIZE,  # 70% train, 30% temp
    random_state=RANDOM_STATE,
    stratify=y_path,
)

# Second split: validation + test
X_val, X_test, y_path_val, y_path_test, y_care_val, y_care_test = train_test_split(
    X_temp,
    y_path_temp,
    y_care_temp,
    test_size=TEST_SIZE_FROM_TEMP,  # 15% val, 15% test
    random_state=RANDOM_STATE,
    stratify=y_path_temp,
)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")

# Save test data so test.py evaluates the same test split
np.save("X_test.npy", X_test)
np.save("y_path_test.npy", y_path_test)
np.save("y_care_test.npy", y_care_test)

# =========================================================
# DATASET & DATALOADER
# =========================================================
train_dataset = DualDataset(X_train, y_path_train, y_care_train)
val_dataset = DualDataset(X_val, y_path_val, y_care_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# MODEL
# =========================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DualHeadMLP(
    input_dim=X.shape[1],
    num_pathologies=len(label_encoder.classes_),
    num_careplans=len(careplan_cols),
).to(device)

# =========================================================
# LOSSES & OPTIMIZER
# =========================================================

pathology_loss_fn = nn.CrossEntropyLoss()
careplan_loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Optional scheduler (simple cosine)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS
)

# =========================================================
# TRAINING WITH VALIDATION
# =========================================================
best_val_f1 = 0.0
patience_counter = 0

print("\nTraining...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb_path, yb_care in train_loader:
        xb, yb_path, yb_care = xb.to(device), yb_path.to(device), yb_care.to(device)

        optimizer.zero_grad()

        path_logits, care_logits = model(xb)

        loss = (
            pathology_loss_fn(path_logits, yb_path)
            + careplan_loss_fn(care_logits, yb_care)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_true, val_pred = [], []
    val_loss = 0

    with torch.no_grad():
        for xb, yb_path, yb_care in val_loader:
            xb, yb_path, yb_care = xb.to(device), yb_path.to(device), yb_care.to(device)

            path_logits, care_logits = model(xb)

            loss = (
                pathology_loss_fn(path_logits, yb_path)
                + careplan_loss_fn(care_logits, yb_care)
            )

            val_loss += loss.item()

            preds = torch.argmax(path_logits, dim=1)
            val_true.extend(yb_path.cpu().numpy())
            val_pred.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_true, val_pred, average="weighted")

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✔ Saved best model (Val F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

print("\nDone training.")