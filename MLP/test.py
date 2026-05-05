import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import DualDataset, DualHeadMLP

MODEL_PATH = "dual_model.pt"
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
X_test = np.load("X_test.npy")
y_path_test = np.load("y_path_test.npy")
y_care_test = np.load("y_care_test.npy")

test_dataset = DualDataset(X_test, y_path_test, y_care_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# LOAD MODEL
# =========================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DualHeadMLP(
    input_dim=X_test.shape[1],
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

print(classification_report(
    all_path_true,
    all_path_pred,
    target_names=label_encoder.classes_
))

#per-class fl score plot
report = classification_report(
    all_path_true,
    all_path_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)


df_report = pd.DataFrame(report).T

df_report = df_report.iloc[:-3]  # remove avg rows
df_report = df_report[df_report["f1-score"] > 0]
df_report = df_report.sort_values("f1-score")

plt.figure(figsize=(10, 6))
plt.barh(
    [l.replace(" (disorder)", "") for l in df_report.index],
    df_report["f1-score"]
)

plt.xlabel("F1 Score")
plt.title("Per-Disease F1 Score", fontsize=18)
plt.xlim(0, 1)

plt.tight_layout()
plt.savefig("per_class_f1.png", dpi=300)
plt.show()


# confusion matrix plot
cm = confusion_matrix(all_path_true, all_path_pred)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=[l.replace(" (disorder)", "") for l in label_encoder.classes_],
    yticklabels=[l.replace(" (disorder)", "") for l in label_encoder.classes_],
)


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.xlabel("Predicted", fontsize=20)
plt.ylabel("Actual", fontsize=20)
plt.title("Pathology Confusion Matrix", fontsize=25)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)



plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


#top-3 vs. top-1 performance plot
metrics = {
    "Top-1 Accuracy": 0.8517,
    "Top-3 Accuracy": 0.9994
}

# care-plan performance breakdown
metrics = {
    "Exact Match": 0.9873,
    "Element-wise": 0.9988,
    "Sample F1": 0.9958
}

plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values())

plt.ylabel("Score")
plt.title("Care Plan Prediction Performance")
plt.ylim(0.95, 1.0)

plt.tight_layout()
plt.savefig("careplan_performance.png", dpi=300)
plt.show()

print("\nDone.")


model_name = "DualHeadMLP"  # change this for each folder/model

results = {
    "model": model_name,
    "pathology_accuracy": path_acc,
    "pathology_weighted_f1": path_f1,
    "pathology_top3_accuracy": top3_acc,
    "careplan_exact_match": care_exact_match_acc,
    "careplan_element_accuracy": care_element_acc,
    "careplan_sample_f1": care_sample_f1,
}

results_df = pd.DataFrame([results])
results_df.to_csv("model_results.csv", index=False)

print("\nSaved results to model_results.csv")