import torch
import torch.nn as nn


class DualHeadTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_pathologies,
        num_careplans,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super().__init__()

        self.feature_embedding = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.pathology_head = nn.Linear(d_model, num_pathologies)
        self.careplan_head = nn.Linear(d_model, num_careplans)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = self.transformer(x)

        shared_features = x.mean(dim=1)

        pathology_logits = self.pathology_head(shared_features)
        careplan_logits = self.careplan_head(shared_features)

        return pathology_logits, careplan_logits