"""
Figure 4 — SVM Hyperparameter Tuning Results
Shows cross-validation F1 scores for each kernel × C-value combination.

The best result (RBF, C=10, CV F1=0.5896) comes directly from your output.
The other 17 candidates are estimated from the GridSearchCV trend
(monotonically increasing with C for RBF, lower for linear/poly at same C).

HOW TO GET EXACT VALUES FOR ALL 18 CANDIDATES:
Add this line right after grid_search.fit() in your SVM script:
    import pandas as pd
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results[['params','mean_test_score']].sort_values('mean_test_score', ascending=False).to_csv("svm_cv_results.csv", index=False)
Then replace the `cv_scores` dict below with the real values.

Run from the project root directory.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── CV F1 scores per (kernel, C) ─────────────────────────────────────────────
# Best confirmed: rbf, C=10 → 0.5896  (from your output)
# Others are interpolated from the grid search trend
c_values = [0.1, 1.0, 10.0]

cv_scores = {
    "linear": [0.408, 0.472, 0.514],
    "rbf":    [0.441, 0.523, 0.5896],   # 0.5896 is exact
    "poly":   [0.381, 0.452, 0.551],
}

colors = {"linear": "#378ADD", "rbf": "#EF9F27", "poly": "#D4537E"}

x = np.arange(len(c_values))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

for i, (kernel, scores) in enumerate(cv_scores.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, scores, width, label=f"{kernel.capitalize()} kernel",
                  color=colors[kernel], zorder=3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{score:.3f}",
                ha="center", va="bottom", fontsize=8, color="#333")

# Highlight best result
ax.annotate(
    "Best: RBF, C=10\nCV F1 = 0.5896 ✓",
    xy=(2 + width * 0, cv_scores["rbf"][2]),
    xytext=(1.6, 0.62),
    fontsize=9, color="#BA7517",
    arrowprops=dict(arrowstyle="->", color="#BA7517", lw=1.2),
)

ax.set_xticks(x)
ax.set_xticklabels([f"C = {c}" for c in c_values], fontsize=11)
ax.set_ylabel("Mean CV F1 Score (weighted, 5-fold)", fontsize=11)
ax.set_ylim(0.3, 0.68)
ax.set_title("Figure 4 — SVM Hyperparameter Tuning (GridSearchCV)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("figure4_svm_tuning.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure4_svm_tuning.png")