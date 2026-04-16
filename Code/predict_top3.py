import numpy as np
import torch
import torch.nn as nn
import joblib

feature_cols = joblib.load("feature_cols.pkl")
careplan_cols = joblib.load("careplan_cols.pkl")
label_encoder = joblib.load("pathology_label_encoder.pkl")

class DualHeadMLP(nn.Module):
    def __init__(self, input_dim, num_pathologies, num_careplans):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.pathology_head = nn.Linear(64, num_pathologies)
        self.careplan_head = nn.Linear(64, num_careplans)

    def forward(self, x):
        h = self.shared(x)
        return self.pathology_head(h), self.careplan_head(h)

device = torch.device("cpu")

model = DualHeadMLP(
    input_dim=len(feature_cols),
    num_pathologies=len(label_encoder.classes_),
    num_careplans=len(careplan_cols)
)
model.load_state_dict(torch.load("dual_model.pt", map_location=device))
model.eval()

# blank input
input_dict = {col: 0 for col in feature_cols}

# turn on symptoms that are present
chosen_symptoms = [
    "SYMPTOM__wheezing",
    "SYMPTOM__cough",
    "SYMPTOM__shortness of breath"
]

for s in chosen_symptoms:
    if s in input_dict:
        input_dict[s] = 1

# optional demographics
if "AGE_BEGIN" in input_dict:
    input_dict["AGE_BEGIN"] = 0.4
if "AGE_END" in input_dict:
    input_dict["AGE_END"] = 0.4
if "NUM_SYMPTOMS" in input_dict:
    input_dict["NUM_SYMPTOMS"] = 0.2

x = np.array([input_dict[col] for col in feature_cols], dtype=np.float32)
x = torch.tensor(x).unsqueeze(0)

with torch.no_grad():
    path_logits, care_logits = model(x)

    path_probs = torch.softmax(path_logits, dim=1).numpy()[0]
    care_probs = torch.sigmoid(care_logits).numpy()[0]

top3_path = np.argsort(path_probs)[-3:][::-1]
top3_care = np.argsort(care_probs)[-3:][::-1]

print("\nTop 3 predicted pathologies:")
for i in top3_path:
    print(label_encoder.inverse_transform([i])[0], float(path_probs[i]))

print("\nTop 3 predicted careplans:")
for i in top3_care:
    print(careplan_cols[i].replace("CAREPLAN__", ""), float(care_probs[i]))