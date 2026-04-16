import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "meta_dataset_ml_ready.csv"
MODEL_PATH = "dual_model.pt"

MIN_PATHOLOGY_COUNT = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3

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

# Save metadata for prediction script
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
# DATASET
# =========================================================
class DualDataset(Dataset):
    def __init__(self, X_data, y_path_data, y_care_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y_path = torch.tensor(y_path_data, dtype=torch.long)
        self.y_care = torch.tensor(y_care_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_path[idx], self.y_care[idx]


train_dataset = DualDataset(X_train, y_path_train, y_care_train)
test_dataset = DualDataset(X_test, y_path_test, y_care_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# MODEL
# =========================================================
class DualHeadMLP(nn.Module):
    def __init__(self, input_dim, num_pathologies, num_careplans):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.pathology_head = nn.Linear(64, num_pathologies)
        self.careplan_head = nn.Linear(64, num_careplans)

    def forward(self, x):
        shared_features = self.shared(x)
        pathology_logits = self.pathology_head(shared_features)
        careplan_logits = self.careplan_head(shared_features)
        return pathology_logits, careplan_logits


device = torch.device("cpu")
print(f"Using device: {device}")

model = DualHeadMLP(
    input_dim=X.shape[1],
    num_pathologies=len(label_encoder.classes_),
    num_careplans=len(careplan_cols),
).to(device)

pathology_loss_fn = nn.CrossEntropyLoss()
careplan_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

# =========================================================
# EVALUATION
# =========================================================
model.eval()

all_path_true = []
all_path_pred = []
all_path_top3 = []

all_care_true = []
all_care_pred = []

with torch.no_grad():
    for xb, yb_path, yb_care in test_loader:
        xb = xb.to(device)

        path_logits, care_logits = model(xb)

        # Pathology predictions
        path_probs = torch.softmax(path_logits, dim=1).cpu().numpy()
        path_pred = np.argmax(path_probs, axis=1)
        top3_idx = np.argsort(path_probs, axis=1)[:, -3:][:, ::-1]

        # Careplan predictions
        care_probs = torch.sigmoid(care_logits).cpu().numpy()
        care_pred = (care_probs >= 0.5).astype(int)

        all_path_true.extend(yb_path.numpy())
        all_path_pred.extend(path_pred.tolist())
        all_path_top3.extend(top3_idx.tolist())

        all_care_true.append(yb_care.numpy())
        all_care_pred.append(care_pred)

all_care_true = np.vstack(all_care_true)
all_care_pred = np.vstack(all_care_pred)

# Pathology metrics
path_acc = accuracy_score(all_path_true, all_path_pred)
path_f1 = f1_score(all_path_true, all_path_pred, average="weighted")

top3_correct = 0
for true_label, top3 in zip(all_path_true, all_path_top3):
    if true_label in top3:
        top3_correct += 1
top3_acc = top3_correct / len(all_path_true)

# Careplan metrics
care_exact_match_acc = np.mean(np.all(all_care_true == all_care_pred, axis=1))
care_element_acc = np.mean(all_care_true == all_care_pred)

care_f1_scores = []
for i in range(len(all_care_true)):
    true_i = all_care_true[i]
    pred_i = all_care_pred[i]

    tp = np.sum((true_i == 1) & (pred_i == 1))
    fp = np.sum((true_i == 0) & (pred_i == 1))
    fn = np.sum((true_i == 1) & (pred_i == 0))

    if tp == 0 and fp == 0 and fn == 0:
        f1_i = 1.0
    elif tp == 0:
        f1_i = 0.0
    else:
        precision_i = tp / (tp + fp)
        recall_i = tp / (tp + fn)
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i)

    care_f1_scores.append(f1_i)

care_sample_f1 = float(np.mean(care_f1_scores))

# =========================================================
# PRINT RESULTS
# =========================================================
print("\n===== TEST RESULTS =====")
print(f"Pathology Accuracy: {path_acc:.4f}")
print(f"Pathology Weighted F1: {path_f1:.4f}")
print(f"Pathology Top-3 Accuracy: {top3_acc:.4f}")
print(f"Careplan Exact Match Accuracy: {care_exact_match_acc:.4f}")
print(f"Careplan Element-wise Accuracy: {care_element_acc:.4f}")
print(f"Careplan Sample-wise F1: {care_sample_f1:.4f}")

print("\nDone.")