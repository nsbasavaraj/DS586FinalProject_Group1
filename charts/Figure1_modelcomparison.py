"""
Figure 1 — Model Performance Comparison
Compares Accuracy, Weighted F1, and Top-3 Accuracy across all four models.
Run from the project root directory.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Real results from your model outputs ──────────────────────────────────────
models = ["Random Forest", "SVM (tuned)", "Neural Network", "Dual-Head MLP"]

accuracy = [0.7250, 0.7125, 0.8000, 0.7750]
weighted_f1 = [0.70, 0.6373, None, 0.7330]   # NN script doesn't report weighted F1
top3_accuracy = [None, 0.8250, None, 0.9750]  # only SVM and Dual MLP report top-3

# ── Plot ──────────────────────────────────────────────────────────────────────
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width, accuracy, width, label="Accuracy",       color="#378ADD", zorder=3)
bars2 = ax.bar(x,         [v if v else 0 for v in weighted_f1],
               width, label="Weighted F1",   color="#1D9E75", zorder=3)
bars3 = ax.bar(x + width, [v if v else 0 for v in top3_accuracy],
               width, label="Top-3 Accuracy", color="#EF9F27", zorder=3)

# Label bars with values; skip zeros (N/A entries)
for bars, values in [(bars1, accuracy), (bars2, weighted_f1), (bars3, top3_accuracy)]:
    for bar, val in zip(bars, values):
        if val:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, color="#333"
            )

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title("Figure 1 — Model Performance Comparison", fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Annotate N/A bars
na_positions = [(1, -width, "N/A"), (1, width, "N/A"),   # SVM F1 = ok, top3 = ok
                (0, width, "N/A"), (0, +width, "N/A"),
                (2, width, "N/A"), (2, +width, "N/A")]
for xi, offset, label in [(0, width, "N/A"), (0, width*2, "N/A"),
                           (2, 0, "N/A"), (2, width, "N/A")]:
    pass  # bars already show 0; the legend note below covers this

fig.text(
    0.5, 0.01,
    "* Weighted F1 for Neural Network and Top-3 Accuracy for RF/NN not reported by those scripts.",
    ha="center", fontsize=8, color="gray"
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("figure1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure1_model_comparison.png")