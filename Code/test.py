import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Code.MLP.model import DualDataset, DualHeadMLP

CSV_PATH = "meta_dataset_ml_ready.csv"
MODEL_PATH = "dual_model.pt"

MIN_PATHOLOGY_COUNT = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32

# =========================================================
# LOAD METADATA
# =========================================================
feature_cols = joblib.load("feature_cols.pkl")
careplan_cols = joblib.load("careplan_cols.pkl")
label_encoder = joblib.load("pathology_label_encoder.pkl")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)

path_counts = df["PATHOLOGY"].value_counts()
valid_pathologies = path_counts[path_counts >= MIN_PATHOLOGY_COUNT].index
df = df[df["PATHOLOGY"].isin(valid_pathologies)].copy()

X = df[feature_cols].astype(np.float32).values
y_care = df[careplan_cols].astype(np.float32).values
y_path = label_encoder.transform(df["PATHOLOGY"].astype(str))

X_train, X_test, y_path_train, y_path_test, y_care_train, y_care_test = train_test_split(
    X,
    y_path,
    y_care,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_path,
)

test_dataset = DualDataset(X_test, y_path_test, y_care_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# LOAD MODEL
# =========================================================
device = torch.device("cpu")
model = DualHeadMLP(
    input_dim=X.shape[1],
    num_pathologies=len(label_encoder.classes_),
    num_careplans=len(careplan_cols),
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================================================
# EVALUATION
# =========================================================
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

print("\n===== TEST RESULTS =====")
print(f"Pathology Accuracy: {path_acc:.4f}")
print(f"Pathology Weighted F1: {path_f1:.4f}")
print(f"Pathology Top-3 Accuracy: {top3_acc:.4f}")
print(f"Careplan Exact Match Accuracy: {care_exact_match_acc:.4f}")
print(f"Careplan Element-wise Accuracy: {care_element_acc:.4f}")
print(f"Careplan Sample-wise F1: {care_sample_f1:.4f}")

print("\nDone.")
