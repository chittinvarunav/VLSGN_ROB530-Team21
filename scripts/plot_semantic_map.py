import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ground_truth = [
    {"label": "red cylinder",   "x": -2.00, "y":  2.00, "color": "red"},
    {"label": "blue box",       "x":  2.00, "y":  2.00, "color": "blue"},
    {"label": "yellow box",     "x":  2.50, "y": -2.50, "color": "gold"},
    {"label": "white cylinder", "x":  2.50, "y":  0.50, "color": "lightgray"},
    {"label": "green cylinder", "x": -2.00, "y": -2.00, "color": "green"},
]

perceived = [
    {"label": "red cylinder",   "x": -1.61, "y":  2.05, "color": "red"},
    {"label": "blue box",       "x":  1.36, "y":  0.84, "color": "blue"},
    {"label": "yellow box",     "x":  2.87, "y": -2.93, "color": "gold"},
    {"label": "white cylinder", "x":  2.29, "y":  0.02, "color": "lightgray"},
    {"label": "green cylinder", "x": -1.79, "y": -2.04, "color": "green"},
]

# Object name label offsets in points (anchored to ground truth dot)
LABEL_OFFSETS = {
    "red cylinder":   (-70, 18),
    "blue box":       ( 15, 18),
    "yellow box":     ( 15,-32),
    "white cylinder": ( 15, 18),
    "green cylinder": (-80,-30),
}

# Error label offsets in data coordinates (placed at midpoint of dashed line)
ERR_OFFSETS = {
    "red cylinder":   ( 0.10,  0.12),
    "blue box":       (-0.55,  0.12),   # shifted left to avoid white cylinder
    "yellow box":     ( 0.10,  0.12),
    "white cylinder": ( 0.10, -0.22),   # below line to avoid blue box label
    "green cylinder": ( 0.10,  0.10),
}

fig, ax = plt.subplots(figsize=(9, 9))

# Room boundary
ax.add_patch(mpatches.FancyBboxPatch(
    (-4.5, -4.5), 9.0, 9.0,
    boxstyle="square,pad=0", linewidth=2.5,
    edgecolor="#555555", facecolor="#F8F8F8", zorder=0
))

for gt, per in zip(ground_truth, perceived):
    err = np.sqrt((gt["x"] - per["x"])**2 + (gt["y"] - per["y"])**2)

    # Dashed error line between GT and perceived
    ax.plot([gt["x"], per["x"]], [gt["y"], per["y"]],
            color="gray", linewidth=1.2, linestyle="--", zorder=3, alpha=0.7)

    # Error value at midpoint
    mid_x = (gt["x"] + per["x"]) / 2
    mid_y = (gt["y"] + per["y"]) / 2
    ex, ey = ERR_OFFSETS[gt["label"]]
    ax.text(mid_x + ex, mid_y + ey, f"Δ{err:.2f}m",
            fontsize=8.5, color="#333333", ha="center", va="bottom",
            bbox=dict(facecolor="lightyellow", edgecolor="#CCCCCC",
                      boxstyle="round,pad=0.25", alpha=0.95))

    # Ground truth: large filled circle
    ax.scatter(gt["x"], gt["y"], s=600, c=gt["color"],
               edgecolors="black", linewidths=2.0, zorder=5, marker="o")

    # Perceived: star
    ax.scatter(per["x"], per["y"], s=350, c=per["color"],
               edgecolors="black", linewidths=1.5, zorder=6, marker="*")

    # Object name label anchored to ground truth
    ox, oy = LABEL_OFFSETS[gt["label"]]
    ax.annotate(
        gt["label"],
        (gt["x"], gt["y"]),
        textcoords="offset points",
        xytext=(ox, oy),
        fontsize=10, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="-", color="gray",
                        lw=0.8, connectionstyle="arc3,rad=0.0"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.95),
    )

# Robot start
ax.scatter(0, 0, s=250, marker="^", color="#FF6F00", zorder=7)
ax.annotate("Robot\nstart", (0, 0),
            textcoords="offset points", xytext=(12, 8),
            fontsize=8, color="#FF6F00", fontweight="bold")

# Legend
legend_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black",
                   label="● Ground Truth (Gazebo)"),
    mpatches.Patch(facecolor="white", edgecolor="black",
                   label="★ Perceived (semantic map)"),
    mpatches.Patch(facecolor="lightyellow", edgecolor="#CCCCCC",
                   label="Δ Localization error"),
    plt.Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#FF6F00", markersize=10, label="Robot start"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.95)

ax.set_xlabel("X (meters)", fontsize=12)
ax.set_ylabel("Y (meters)", fontsize=12)
ax.set_title("Semantic Map: Ground Truth vs Perceived Positions",
             fontsize=14, fontweight="bold", pad=15)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig("semantic_map_comparison.png", dpi=150, bbox_inches="tight")
print("Saved!")
