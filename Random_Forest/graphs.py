import matplotlib.pyplot as plt
import numpy as np
import os

# ============================
# INSERT YOUR METRICS HERE
# ============================
baseline_acc = 0.800
baseline_macro_f1 = 0.720
baseline_weighted_f1 = 0.800

tuned_acc = 0.7988137603795967
tuned_macro_f1 = 0.76
tuned_weighted_f1 = 0.80

# ============================
# PREPARE DATA
# ============================
metrics = ["Accuracy", "Macro F1", "Weighted F1"]

baseline_vals = [baseline_acc, baseline_macro_f1, baseline_weighted_f1]
tuned_vals = [tuned_acc, tuned_macro_f1, tuned_weighted_f1]

x = np.arange(len(metrics))
width = 0.35

# ============================
# CREATE MULTI-BAR PLOT
# ============================
plt.figure(figsize=(8,5))

plt.bar(x - width/2, baseline_vals, width, label="Baseline RF", color="skyblue")
plt.bar(x + width/2, tuned_vals, width, label="Tuned RF", color="orange")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Random Forest Comparison: Accuracy, Macro F1, Weighted F1")
plt.ylim(0, 1)
plt.legend()

# ============================
# SAVE PNG TO Random_Forest
# ============================
save_path = os.path.join(os.getcwd(), "rf_metric_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved multi-bar comparison plot to: {save_path}")

