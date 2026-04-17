import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("meta_dataset_ml_ready.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# ----------------------------
# 2. Define target and features
# ----------------------------
target_col = "pathology"

# Columns to remove if they exist
drop_cols = ["pathology", "patient", "symptom_list"]
existing_drop_cols = [col for col in drop_cols if col in df.columns]

if target_col not in df.columns:
    raise ValueError("Column 'pathology' not found in dataset.")

# Remove rows with missing target
df = df.dropna(subset=[target_col])

# Remove rare classes with fewer than 2 samples
class_counts = df[target_col].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df[target_col].isin(valid_classes)].copy()

print(f"Remaining samples: {len(df)}")
print(f"Remaining classes: {df[target_col].nunique()}")

# Build X and y
X = df.drop(columns=existing_drop_cols)
y = df[target_col]

# ----------------------------
# 3. Keep numeric features only
# ----------------------------
X = X.select_dtypes(include=[np.number]).copy()

# Fill missing values if any
X = X.fillna(0)

print("Feature matrix shape:", X.shape)

# ----------------------------
# 4. Encode labels
# ----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)
print("Number of pathology classes:", num_classes)

# ----------------------------
# 5. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ----------------------------
# 6. Scale features
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 7. Convert to tensors
# ----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ----------------------------
# 8. Define neural network
# ----------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = SimpleNN(input_dim=input_dim, num_classes=num_classes)

# ----------------------------
# 9. Loss and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 10. Train model
# ----------------------------
num_epochs = 50

for epoch in range(num_epochs):
    model.train()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        _, predicted_train = torch.max(outputs, 1)
        train_acc = (predicted_train == y_train_tensor).float().mean().item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")

# ----------------------------
# 11. Evaluate model
# ----------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted_test = torch.max(test_outputs, 1)
    test_acc = (predicted_test == y_test_tensor).float().mean().item()

print(f"Test Accuracy: {test_acc:.4f}")

# ----------------------------
# 12. Predict one example
# ----------------------------
def predict_pathology(sample_row):
    """
    sample_row should be a pandas DataFrame with one row
    and the same feature columns as X.
    """
    sample_row = sample_row.select_dtypes(include=[np.number]).fillna(0)
    sample_scaled = scaler.transform(sample_row)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(sample_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return label_encoder.inverse_transform([pred_idx])[0]

# Example:
sample_prediction = predict_pathology(X.iloc[[0]])
print("Predicted pathology:", sample_prediction)
print("Actual pathology:", y.iloc[0])