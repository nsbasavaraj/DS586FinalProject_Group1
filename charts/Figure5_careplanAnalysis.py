"""
Figure 5 — Careplan Distribution & SVM Two-Stage Pipeline Performance
Left panel: careplan frequency across the dataset (from preprocessing output).
Right panel: SVM pipeline metrics showing pathology acc → top-3 acc → careplan F1.
Uses exact values from preprocessing.py and SVM careplan model.py outputs.
Run from the project root directory.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Careplan distribution (from preprocessing.py output) ─────────────────────
careplan_labels = [
    "Hyperlipidemia mgmt.",
    "Musculoskeletal care",
    "Asthma self-mgmt.",
    "Skin condition care",
    "COPD mgmt. plan",
    "Dementia mgmt.",
    "Cancer care plan",
    "Heart failure plan",
    "Inpatient care plan",
    "Major surgery care",
]
careplan_counts = [101, 92, 66, 33, 28, 27, 26, 25, 24, 9]

# ── SVM two-stage pipeline metrics (from SVM careplan model.py output) ────────
pipeline_labels = ["Pathology\nAccuracy", "Top-3\nAccuracy", "Careplan\nRec. F1"]
pipeline_scores = [0.7125, 0.8250, 0.8500]
pipeline_colors = ["#378ADD", "#EF9F27", "#1D9E75"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                gridspec_kw={"width_ratios": [1.6, 1]})
fig.suptitle("Figure 5 — Careplan Analysis & SVM Pipeline Performance",
             fontsize=13, fontweight="bold", y=1.01)

# ── Left: careplan distribution ───────────────────────────────────────────────
y_pos = np.arange(len(careplan_labels))
bars = ax1.barh(y_pos, careplan_counts, color="#5DCAA5", height=0.65, zorder=3)

for bar, val in zip(bars, careplan_counts):
    pct = val / 398 * 100
    ax1.text(bar.get_width() + 0.8,
             bar.get_y() + bar.get_height() / 2,
             f"{val}  ({pct:.1f}%)", va="center", fontsize=9)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(careplan_labels, fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel("Number of Records", fontsize=11)
ax1.set_title("Careplan Frequency (398 Records, 10 Careplans)", fontsize=11, pad=10)
ax1.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)
ax1.set_xlim(0, 135)

# ── Right: SVM pipeline performance ──────────────────────────────────────────
bars2 = ax2.bar(pipeline_labels, pipeline_scores, color=pipeline_colors,
                width=0.45, zorder=3)

for bar, score in zip(bars2, pipeline_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.008,
             f"{score:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Arrow annotation showing improvement
ax2.annotate("", xy=(2, 0.85), xytext=(0, 0.7125),
             arrowprops=dict(arrowstyle="-", color="#aaa", lw=1, linestyle="dashed"))

ax2.set_ylim(0, 1.0)
ax2.set_ylabel("Score", fontsize=11)
ax2.set_title("SVM Two-Stage Pipeline\n(Pathology → Careplan)", fontsize=11, pad=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

# Note
ax2.text(0.5, 0.08,
         "Careplan F1 (85%) exceeds\npathology accuracy (71.3%)\nbecause similar diseases\nshare careplans.",
         transform=ax2.transAxes, ha="center", fontsize=8.5,
         color="#555", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", alpha=0.8))

plt.tight_layout()
plt.savefig("figure5_careplan_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure5_careplan_analysis.png")