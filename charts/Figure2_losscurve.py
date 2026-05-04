"""
Figure 2 — Dual-Head MLP Training Loss Curve
Plots total loss, pathology head loss, and careplan head loss over 25 epochs.
Uses the exact values printed by train_dual_model.py.
Run from the project root directory.
"""

import matplotlib.pyplot as plt

# ── Exact values from train_dual_model.py output ─────────────────────────────
epochs = list(range(1, 26))

total_loss = [
    3.4359, 3.2667, 2.8225, 2.5896, 2.3422,
    2.0128, 1.7058, 1.4315, 1.2344, 1.0876,
    0.9628, 0.8467, 0.7637, 0.7010, 0.5995,
    0.5562, 0.5127, 0.4555, 0.4445, 0.4123,
    0.3794, 0.3601, 0.3368, 0.3466, 0.3280,
]

path_loss = [
    2.7534, 2.6333, 2.3560, 2.2426, 2.0056,
    1.7041, 1.4251, 1.1854, 1.0283, 0.9145,
    0.8183, 0.7251, 0.6585, 0.6068, 0.5172,
    0.4827, 0.4498, 0.3990, 0.3935, 0.3645,
    0.3350, 0.3189, 0.2990, 0.3095, 0.2935,
]

care_loss = [
    0.6825, 0.6334, 0.4665, 0.3470, 0.3366,
    0.3087, 0.2807, 0.2462, 0.2060, 0.1731,
    0.1445, 0.1216, 0.1052, 0.0942, 0.0824,
    0.0735, 0.0629, 0.0565, 0.0510, 0.0479,
    0.0444, 0.0412, 0.0378, 0.0370, 0.0345,
]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, total_loss, color="#378ADD", linewidth=2.5, label="Total loss", zorder=3)
ax.plot(epochs, path_loss,  color="#D4537E", linewidth=1.8, linestyle="--",
        label="Pathology head loss", zorder=3)
ax.plot(epochs, care_loss,  color="#1D9E75", linewidth=1.8,
        label="Careplan head loss", zorder=3)

# Annotate final values
for loss_list, color, label in [
    (total_loss, "#378ADD", f"  {total_loss[-1]:.4f}"),
    (path_loss,  "#D4537E", f"  {path_loss[-1]:.4f}"),
    (care_loss,  "#1D9E75", f"  {care_loss[-1]:.4f}"),
]:
    ax.annotate(label, xy=(25, loss_list[-1]), fontsize=9, color=color, va="center")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Figure 2 — Dual-Head MLP Training Loss Curve (25 Epochs)",
             fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(1, 25)
ax.set_ylim(0, 3.7)

plt.tight_layout()
plt.savefig("figure2_loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure2_loss_curve.png")