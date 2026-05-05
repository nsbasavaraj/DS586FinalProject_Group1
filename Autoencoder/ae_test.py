import joblib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from ae_model import DualDataset, AutoencoderDualHeadClassifier

MODEL_PATH = "ae_dual_model.pt"
BATCH_SIZE = 64
LATENT_DIM = 128
DROPOUT = 0.3


def test_model():
    X_test = np.load("ae_X_test.npy")
    y_path_test = np.load("ae_y_path_test.npy")
    y_care_test = np.load("ae_y_care_test.npy")

    careplan_cols = joblib.load("ae_careplan_cols.pkl")
    label_encoder = joblib.load("ae_pathology_label_encoder.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoencoderDualHeadClassifier(
        input_dim=X_test.shape[1],
        num_pathologies=len(label_encoder.classes_),
        num_careplans=len(careplan_cols),
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_loader = DataLoader(
        DualDataset(X_test, y_path_test, y_care_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_path_true = []
    all_path_pred = []
    all_path_top3 = []

    all_care_true = []
    all_care_pred = []

    with torch.no_grad():
        for xb, yb_path, yb_care in test_loader:
            xb = xb.to(device)

            _, path_logits, care_logits = model(xb)

            # Pathology predictions
            path_probs = torch.softmax(path_logits, dim=1).cpu().numpy()
            path_pred = np.argmax(path_probs, axis=1)
            top3_idx = np.argsort(path_probs, axis=1)[:, -3:][:, ::-1]

            # Care plan predictions
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

    top3_correct = sum(
        true_label in top3
        for true_label, top3 in zip(all_path_true, all_path_top3)
    )
    top3_acc = top3_correct / len(all_path_true)

    # Care plan metrics
    care_exact_match_acc = np.mean(np.all(all_care_true == all_care_pred, axis=1))
    care_element_acc = np.mean(all_care_true == all_care_pred)

    care_sample_f1 = f1_score(
        all_care_true,
        all_care_pred,
        average="samples",
        zero_division=0,
    )

    print("\n===== AUTOENCODER + CLASSIFIER TEST RESULTS =====")
    print(f"Pathology Accuracy: {path_acc:.4f}")
    print(f"Pathology Weighted F1: {path_f1:.4f}")
    print(f"Pathology Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Careplan Exact Match Accuracy: {care_exact_match_acc:.4f}")
    print(f"Careplan Element-wise Accuracy: {care_element_acc:.4f}")
    print(f"Careplan Sample-wise F1: {care_sample_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_path_true,
            all_path_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(all_path_true, all_path_pred)

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(
        cm,
        cmap="Blues",
        cbar=True,
        xticklabels=[l.replace(" (disorder)", "") for l in label_encoder.classes_],
        yticklabels=[l.replace(" (disorder)", "") for l in label_encoder.classes_],
    )

    # Bigger colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Autoencoder Pathology Confusion Matrix", fontsize=14)

    plt.tight_layout()
    plt.savefig("ae_confusion_matrix.png", dpi=300)
    plt.show()


    # =========================================================
    # PER-DISEASE F1 SCORES
    # =========================================================
    report = classification_report(
        all_path_true,
        all_path_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    df_report = pd.DataFrame(report).T

    # Remove avg rows
    df_report = df_report.iloc[:-3]

    # Remove zero-F1 diseases
    df_report = df_report[df_report["f1-score"] > 0]

    # Sort
    df_report = df_report.sort_values("f1-score")

    # Clean labels
    labels = [l.replace(" (disorder)", "") for l in df_report.index]

    plt.figure(figsize=(10, 6))

    plt.barh(labels, df_report["f1-score"])

    plt.xlabel("F1 Score", fontsize=12)
    plt.title("Autoencoder Per-Disease F1 Scores", fontsize=14)

    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig("ae_per_class_f1.png", dpi=300)
    plt.show()

    model_name = "Autoencoder"  # change this for each folder/model

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



if __name__ == "__main__":
    test_model()