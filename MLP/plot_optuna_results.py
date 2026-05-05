import pandas as pd
import matplotlib.pyplot as plt

# Load Optuna results
df = pd.read_csv("optuna_results.csv")

# Keep only completed trials
df = df[df["state"] == "COMPLETE"].copy()

# Sort by validation F1
df = df.sort_values("value", ascending=False)

# Create readable trial labels
df["trial_label"] = "Trial " + df["number"].astype(str)

# Plot top 10 trials
top_df = df.head(10)

plt.figure(figsize=(10, 6))
plt.bar(top_df["trial_label"], top_df["value"])

plt.xlabel("Optuna Trial")
plt.ylabel("Validation Weighted F1")
plt.title("Top 10 Optuna Hyperparameter Tuning Results")
plt.ylim(df["value"].min() - 0.01, df["value"].max() + 0.01)
plt.xticks(rotation=45, ha="right")
for i, v in enumerate(df["value"]):
    plt.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("optuna_top_trials.png", dpi=300)
plt.show()