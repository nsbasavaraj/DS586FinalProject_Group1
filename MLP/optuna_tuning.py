import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from train import (
    X_train, y_path_train, y_care_train,
    X_val, y_path_val, y_care_val,
    label_encoder, careplan_cols
)
from model import DualHeadMLP, DualDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout1 = trial.suggest_float("dropout1", 0.2, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.4)

    train_loader = DataLoader(
        DualDataset(X_train, y_path_train, y_care_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        DualDataset(X_val, y_path_val, y_care_val),
        batch_size=batch_size,
    )

    model = DualHeadMLP(
        input_dim=X_train.shape[1],
        num_pathologies=len(label_encoder.classes_),
        num_careplans=len(careplan_cols),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    path_loss = nn.CrossEntropyLoss()
    care_loss = nn.BCEWithLogitsLoss()

    # Train only few epochs for speed
    for epoch in range(10):
        model.train()
        for xb, yb_path, yb_care in train_loader:
            xb, yb_path, yb_care = xb.to(device), yb_path.to(device), yb_care.to(device)

            optimizer.zero_grad()
            p, c = model(xb)

            loss = path_loss(p, yb_path) + care_loss(c, yb_care)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_true, val_pred = [], []

    with torch.no_grad():
        for xb, yb_path, _ in val_loader:
            xb = xb.to(device)
            logits, _ = model(xb)

            preds = torch.argmax(logits, dim=1)
            val_true.extend(yb_path.numpy())
            val_pred.extend(preds.cpu().numpy())

    val_f1 = f1_score(val_true, val_pred, average="weighted")

    return val_f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\nBest parameters:")
    print(study.best_params)

    print("\nBest F1:")
    print(study.best_value)

    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)

    print("\nSaved results to optuna_results.csv")