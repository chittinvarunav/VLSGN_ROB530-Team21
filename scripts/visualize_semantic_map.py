"""
Visualize a semantic map as a 2D plot (standalone, no ROS needed).
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

# Map noisy labels to clean names
LABEL_REMAP = {
    "blue cube":       "blue box",
    "red pillar":      "red cylinder",
    "pillar":          "white cylinder",
    "colored obstacle":"green cylinder",
}

LABEL_COLORS = {
    "red cylinder":   "red",
    "blue box":       "blue",
    "green cylinder": "green",
    "yellow box":     "gold",
    "white cylinder": "lightgray",
}

def visualize_map(
    semantic_map_path: str,
    output_path: str = "semantic_map_viz.png",
    show_labels: bool = True,
    show_walls: bool = True,
    min_observations: int = 50,
):
    with open(semantic_map_path) as f:
        data = json.load(f)

    all_objects = data["objects"]

    # Filter by min observations
    filtered = [o for o in all_objects if o.get("observations", 1) >= min_observations]

    # Remap labels to clean names
    for o in filtered:
        o["label"] = LABEL_REMAP.get(o["label"], o["label"])

    # Deduplicate: keep highest-obs entry per (label, nearby location)
    merged = []
    used = [False] * len(filtered)
    for i, o in enumerate(filtered):
        if used[i]:
            continue
        best = o
        used[i] = True
        for j, o2 in enumerate(filtered):
            if used[j] or o2["label"] != o["label"]:
                continue
            dx = o["position"][0] - o2["position"][0]
            dy = o["position"][1] - o2["position"][1]
            if (dx**2 + dy**2) ** 0.5 < 1.0:
                used[j] = True
                if o2["observations"] > best["observations"]:
                    best = o2
        merged.append(best)

    print(f"Final objects ({len(merged)}):")
    for o in merged:
        print(f"  {o['label']} at ({o['position'][0]:.2f}, {o['position'][1]:.2f}) obs={o['observations']}")

    fig, ax = plt.subplots(figsize=(8, 8))

    xs = [o["position"][0] for o in merged]
    ys = [o["position"][1] for o in merged]
    margin = 2.0
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    if show_walls:
        ax.plot([x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                color="#888888", linewidth=2.5)

    for o in merged:
        x, y = o["position"][0], o["position"][1]
        label = o["label"]
        color = LABEL_COLORS.get(label, "purple")
        ax.scatter(x, y, s=400, c=color, edgecolors="black",
                   linewidths=1.5, zorder=5, alpha=0.95)
        if show_labels:
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 18),
                fontsize=11,
                fontweight="bold",
                ha="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9),
            )

    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(f"Semantic Map ({len(merged)} objects)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("semantic_map")
    parser.add_argument("--output", default="semantic_map_viz.png")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument("--min-obs", type=int, default=50)
    args = parser.parse_args()
    visualize_map(args.semantic_map, args.output,
                  show_labels=not args.no_labels,
                  show_walls=not args.no_walls,
                  min_observations=args.min_obs)
