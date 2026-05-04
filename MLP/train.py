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

from model import DualDataset, DualHeadMLP

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "../Code/meta_dataset_ml_ready.csv"
MODEL_PATH = "dual_model.pt"

MIN_PATHOLOGY_COUNT = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

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
X_train, X_test, y_path_train, y_path_test, y_care_train, y_care_test = train_test_split(
    X,
    y_path,
    y_care,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_path,
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# =========================================================
# DATASET & DATALOADER
# =========================================================
train_dataset = DualDataset(X_train, y_path_train, y_care_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================================================
# MODEL
# =========================================================
device = torch.device("cpu")
print(f"Using device: {device}")

model = DualHeadMLP(
    input_dim=X.shape[1],
    num_pathologies=len(label_encoder.classes_),
    num_careplans=len(careplan_cols),
).to(device)

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
# TRAINING
# =========================================================
print("\nStarting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_total_loss = 0.0
    epoch_path_loss = 0.0
    epoch_care_loss = 0.0

    for xb, yb_path, yb_care in train_loader:
        xb = xb.to(device)
        yb_path = yb_path.to(device)
        yb_care = yb_care.to(device)

        optimizer.zero_grad()

        path_logits, care_logits = model(xb)

        loss_path = pathology_loss_fn(path_logits, yb_path)
        loss_care = careplan_loss_fn(care_logits, yb_care)
        loss = loss_path + loss_care

        loss.backward()
        optimizer.step()

        epoch_total_loss += loss.item()
        epoch_path_loss += loss_path.item()
        epoch_care_loss += loss_care.item()

    if scheduler is not None:
        scheduler.step()

    avg_total_loss = epoch_total_loss / len(train_loader)
    avg_path_loss = epoch_path_loss / len(train_loader)
    avg_care_loss = epoch_care_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Total Loss: {avg_total_loss:.4f} | "
        f"Path Loss: {avg_path_loss:.4f} | "
        f"Care Loss: {avg_care_loss:.4f}"
    )

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model was not saved: {MODEL_PATH}")

print("Done training.")
