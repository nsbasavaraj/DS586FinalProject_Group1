import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DualDataset(Dataset):
    def __init__(self, X, y_path, y_care):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_path = torch.tensor(y_path, dtype=torch.long)
        self.y_care = torch.tensor(y_care, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_path[idx], self.y_care[idx]


class DualHeadMLP(nn.Module):
    def __init__(self, input_dim, num_pathologies, num_careplans, dropout1=0.3, dropout2=0.2):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pathology_head = nn.Linear(128, num_pathologies)
        self.careplan_head = nn.Linear(128, num_careplans)

    def forward(self, x):
        shared = self.shared(x)
        path_logits = self.pathology_head(shared)
        care_logits = self.careplan_head(shared)
        return path_logits, care_logits
