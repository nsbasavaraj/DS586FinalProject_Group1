import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from model import DualHeadTransformer


MODEL_PATH = "dual_transformer_model.pt"
BATCH_SIZE = 32


class TestDataset(Dataset):
    def __init__(self, X_data, y_path_data, y_care_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y_path = torch.tensor(y_path_data, dtype=torch.long)
        self.y_care = torch.tensor(y_care_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_path[idx], self.y_care[idx]


def test_model():
    X_test = np.load("X_test.npy")
    y_path_test = np.load("y_path_test.npy")
    y_care_test = np.load("y_care_test.npy")

    careplan_cols = joblib.load("careplan_cols.pkl")
    label_encoder = joblib.load("pathology_label_encoder.pkl")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DualHeadTransformer(
        input_dim=X_test.shape[1],
        num_pathologies=len(label_encoder.classes_),
        num_careplans=len(careplan_cols),
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_dataset = TestDataset(X_test, y_path_test, y_care_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_path_true = []
    all_path_pred = []
    all_path_top3 = []

    all_care_true = []
    all_care_pred = []

    with torch.no_grad():
        for xb, yb_path, yb_care in test_loader:
            xb = xb.to(device)

            path_logits, care_logits = model(xb)

            path_probs = torch.softmax(path_logits, dim=1).cpu().numpy()
            path_pred = np.argmax(path_probs, axis=1)
            top3_idx = np.argsort(path_probs, axis=1)[:, -3:][:, ::-1]

            care_probs = torch.sigmoid(care_logits).cpu().numpy()
            care_pred = (care_probs >= 0.5).astype(int)

            all_path_true.extend(yb_path.numpy())
            all_path_pred.extend(path_pred.tolist())
            all_path_top3.extend(top3_idx.tolist())

            all_care_true.append(yb_care.numpy())
            all_care_pred.append(care_pred)

    all_care_true = np.vstack(all_care_true)
    all_care_pred = np.vstack(all_care_pred)

    path_acc = accuracy_score(all_path_true, all_path_pred)
    path_f1 = f1_score(all_path_true, all_path_pred, average="weighted")

    top3_correct = sum(
        true_label in top3
        for true_label, top3 in zip(all_path_true, all_path_top3)
    )
    top3_acc = top3_correct / len(all_path_true)

    care_exact_match_acc = np.mean(np.all(all_care_true == all_care_pred, axis=1))
    care_element_acc = np.mean(all_care_true == all_care_pred)

    care_sample_f1 = f1_score(
        all_care_true,
        all_care_pred,
        average="samples",
        zero_division=0,
    )

    print("\n===== TEST RESULTS =====")
    print(f"Pathology Accuracy: {path_acc:.4f}")
    print(f"Pathology Weighted F1: {path_f1:.4f}")
    print(f"Pathology Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Careplan Exact Match Accuracy: {care_exact_match_acc:.4f}")
    print(f"Careplan Element-wise Accuracy: {care_element_acc:.4f}")
    print(f"Careplan Sample-wise F1: {care_sample_f1:.4f}")


if __name__ == "__main__":
    test_model()