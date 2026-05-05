import pandas as pd
import matplotlib.pyplot as plt

result_files = [
    "MLP/model_results.csv",
    "Autoencoder/model_results.csv",
]

dfs = []

for file in result_files:
    df = pd.read_csv(file)
    dfs.append(df)

results = pd.concat(dfs, ignore_index=True)

print("\n===== MODEL COMPARISON =====")
print(results)

results.to_csv("all_model_comparison.csv", index=False)

# Plot pathology performance
plt.figure(figsize=(6, 5))
colors = ["darkorange", "seagreen"]
plt.bar(results["model"], results["pathology_weighted_f1"], color=colors)
plt.ylabel("Pathology Weighted F1")
plt.title("Model Comparison: Disease Prediction", fontsize=15)
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("model_comparison_pathology_f1.png", dpi=300)
plt.show()

# Plot care plan performance
plt.figure(figsize=(6, 5))
colors = ["darkorange", "seagreen"]
plt.bar(results["model"], results["careplan_sample_f1"], color=colors)
plt.ylabel("Care Plan Sample-wise F1")
plt.title("Model Comparison: Care Plan Prediction", fontsize=15)
plt.ylim(0.95, 1.0)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("model_comparison_careplan_f1.png", dpi=300)
plt.show()