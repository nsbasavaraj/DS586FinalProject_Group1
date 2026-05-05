import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from ae_model import DualDataset, AutoencoderDualHeadClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "../Code/meta_dataset_ml_ready_large.csv"
MODEL_PATH = "ae_dual_model.pt"

MIN_PATHOLOGY_COUNT = 5
RANDOM_STATE = 42

BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10

LATENT_DIM = 128
DROPOUT = 0.3

VAL_TEST_SIZE = 0.30
TEST_SIZE_FROM_TEMP = 0.50

RECON_LOSS_WEIGHT = 0.2


def train_model():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Original shape: {df.shape}")

    if "PATHOLOGY" not in df.columns:
        raise ValueError("PATHOLOGY column not found.")

    # Filter rare classes
    path_counts = df["PATHOLOGY"].value_counts()
    valid_pathologies = path_counts[path_counts >= MIN_PATHOLOGY_COUNT].index
    df = df[df["PATHOLOGY"].isin(valid_pathologies)].copy()

    print(f"Rows after filtering: {len(df)}")
    print(f"Pathology classes: {df['PATHOLOGY'].nunique()}")

    # Features / targets
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

    X = df[feature_cols].astype(np.float32).values

    label_encoder = LabelEncoder()
    y_path = label_encoder.fit_transform(df["PATHOLOGY"].astype(str))

    y_care = df[careplan_cols].astype(np.float32).values

    # Save metadata
    joblib.dump(feature_cols, "ae_feature_cols.pkl")
    joblib.dump(careplan_cols, "ae_careplan_cols.pkl")
    joblib.dump(label_encoder, "ae_pathology_label_encoder.pkl")

    # Split train / val / test
    X_train, X_temp, y_path_train, y_path_temp, y_care_train, y_care_temp = train_test_split(
        X,
        y_path,
        y_care,
        test_size=VAL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_path,
    )

    X_val, X_test, y_path_val, y_path_test, y_care_val, y_care_test = train_test_split(
        X_temp,
        y_path_temp,
        y_care_temp,
        test_size=TEST_SIZE_FROM_TEMP,
        random_state=RANDOM_STATE,
        stratify=y_path_temp,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    # Save test set
    np.save("ae_X_test.npy", X_test)
    np.save("ae_y_path_test.npy", y_path_test)
    np.save("ae_y_care_test.npy", y_care_test)

    train_loader = DataLoader(
        DualDataset(X_train, y_path_train, y_care_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        DualDataset(X_val, y_path_val, y_care_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoencoderDualHeadClassifier(
        input_dim=X.shape[1],
        num_pathologies=len(label_encoder.classes_),
        num_careplans=len(careplan_cols),
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(device)

    reconstruction_loss_fn = nn.MSELoss()
    pathology_loss_fn = nn.CrossEntropyLoss()
    careplan_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = 0.0
    patience_counter = 0

    train_loss_history = []
    val_f1_history = []

    print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_total_loss = 0.0

        for xb, yb_path, yb_care in train_loader:
            xb = xb.to(device)
            yb_path = yb_path.to(device)
            yb_care = yb_care.to(device)

            optimizer.zero_grad()

            reconstructed, path_logits, care_logits = model(xb)

            loss_recon = reconstruction_loss_fn(reconstructed, xb)
            loss_path = pathology_loss_fn(path_logits, yb_path)
            loss_care = careplan_loss_fn(care_logits, yb_care)

            loss = loss_path + loss_care + (RECON_LOSS_WEIGHT * loss_recon)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

        avg_train_loss = train_total_loss / len(train_loader)

        # Validation
        model.eval()
        val_true = []
        val_pred = []

        with torch.no_grad():
            for xb, yb_path, yb_care in val_loader:
                xb = xb.to(device)
                yb_path = yb_path.to(device)

                _, path_logits, _ = model(xb)

                preds = torch.argmax(path_logits, dim=1)

                val_true.extend(yb_path.cpu().numpy())
                val_pred.extend(preds.cpu().numpy())

        val_f1 = f1_score(val_true, val_pred, average="weighted")

        train_loss_history.append(avg_train_loss)
        val_f1_history.append(val_f1)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Weighted F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    np.save("ae_train_loss_history.npy", np.array(train_loss_history))
    np.save("ae_val_f1_history.npy", np.array(val_f1_history))

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    print(f"Saved model to {MODEL_PATH}")
    print("Done training.")


if __name__ == "__main__":
    train_model()