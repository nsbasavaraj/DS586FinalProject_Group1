import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from model import DualHeadTransformer


CSV_PATH = "../Code/meta_dataset_ml_ready.csv"
MODEL_PATH = "dual_transformer_model.pt"

MIN_PATHOLOGY_COUNT = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3


class DualDataset(Dataset):
    def __init__(self, X_data, y_path_data, y_care_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y_path = torch.tensor(y_path_data, dtype=torch.long)
        self.y_care = torch.tensor(y_care_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_path[idx], self.y_care[idx]


def train_model():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Original shape: {df.shape}")

    path_counts = df["PATHOLOGY"].value_counts()
    valid_pathologies = path_counts[path_counts >= MIN_PATHOLOGY_COUNT].index
    df = df[df["PATHOLOGY"].isin(valid_pathologies)].copy()

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

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    label_encoder = LabelEncoder()
    y_path = label_encoder.fit_transform(df["PATHOLOGY"].astype(str))

    y_care = df[careplan_cols].astype(np.float32).values

    joblib.dump(feature_cols, "feature_cols.pkl")
    joblib.dump(careplan_cols, "careplan_cols.pkl")
    joblib.dump(label_encoder, "pathology_label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")

    X_train, X_test, y_path_train, y_path_test, y_care_train, y_care_test = train_test_split(
        X,
        y_path,
        y_care,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_path,
    )

    np.save("X_test.npy", X_test)
    np.save("y_path_test.npy", y_path_test)
    np.save("y_care_test.npy", y_care_test)

    train_dataset = DualDataset(X_train, y_path_train, y_care_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DualHeadTransformer(
        input_dim=X.shape[1],
        num_pathologies=len(label_encoder.classes_),
        num_careplans=len(careplan_cols),
    ).to(device)

    pathology_loss_fn = torch.nn.CrossEntropyLoss()
    careplan_loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

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

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()