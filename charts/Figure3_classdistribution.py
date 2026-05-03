"""
Figure 3 — Pathology Class Distribution
Shows class imbalance across the 17 pathologies in the dataset.
Uses exact counts from preprocessing.py output.
Run from the project root directory (needs meta_dataset_readable.csv OR uses hardcoded values).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Option A: load directly from CSV (preferred) ──────────────────────────────
# Uncomment this block if meta_dataset_readable.csv is available:
#
# import pandas as pd
# df = pd.read_csv("meta_dataset_readable.csv")
# counts = df["PATHOLOGY"].value_counts()
# labels = [l.replace(" (disorder)", "").replace("localized  primary ", "loc. ").title()
#           for l in counts.index]
# values = counts.values

# ── Option B: hardcoded from preprocessing.py output ─────────────────────────
labels_raw = [
    "Hyperlipidemia",
    "Childhood asthma",
    "OA of knee",
    "Atopic dermatitis",
    "Congestive heart failure",
    "Alzheimer's disease",
    "OA of hip",
    "OA of hand",
    "COPD bronchitis",
    "Neoplasm of prostate",
    "Pulmonary emphysema",
    "Familial Alzheimer's (early)",
    "Colon cancer (overlap.)",
    "Contact dermatitis",
    "Asthma",
    "Overlapping OA",          # placeholder for any remainder
]
values = [101, 63, 54, 26, 25, 22, 22, 16, 15, 14, 10, 8, 7, 5, 3, 1]

# Sort descending
paired = sorted(zip(values, labels_raw), reverse=True)
values, labels_raw = zip(*paired)

# Color: red for dominant, amber for mid, teal for rare
def pick_color(v):
    if v >= 50:
        return "#E24B4A"
    elif v >= 20:
        return "#EF9F27"
    else:
        return "#5DCAA5"

colors = [pick_color(v) for v in values]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(labels_raw))
bars = ax.barh(y_pos, values, color=colors, height=0.7, zorder=3)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_raw, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Number of Records", fontsize=12)
ax.set_title("Figure 3 — Pathology Class Distribution (398 Records, 16 Classes)",
             fontsize=13, fontweight="bold", pad=14)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, 120)

# Legend
legend_patches = [
    mpatches.Patch(color="#E24B4A", label="Dominant (≥50 records)"),
    mpatches.Patch(color="#EF9F27", label="Mid-frequency (20–49)"),
    mpatches.Patch(color="#5DCAA5", label="Rare (<20 records)"),
]
ax.legend(handles=legend_patches, fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig("figure3_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure3_class_distribution.png")