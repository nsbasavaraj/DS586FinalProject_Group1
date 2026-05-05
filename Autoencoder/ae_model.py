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


class AutoencoderDualHeadClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_pathologies,
        num_careplans,
        latent_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        # Encoder compresses input features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        # Decoder reconstructs original input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, input_dim),
            nn.Sigmoid(),
        )

        # Prediction heads use compressed representation
        self.pathology_head = nn.Linear(latent_dim, num_pathologies)
        self.careplan_head = nn.Linear(latent_dim, num_careplans)

    def forward(self, x):
        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        path_logits = self.pathology_head(latent)
        care_logits = self.careplan_head(latent)

        return reconstructed, path_logits, care_logits