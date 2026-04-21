"""
Analyze experiment results and generate plots/tables for the paper.

Generates:
  - Success rate bar charts by command category
  - Localization error distribution
  - Failure mode breakdown
  - LaTeX tables for the IEEE paper
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

# Optional: matplotlib for plots
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


def load_results(json_path: str) -> tuple[list, dict]:
    """Load results from experiment JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data["results"], data["metrics"]


def plot_success_rates(metrics: dict, output_path: str):
    """Bar chart of success rates by category."""
    if not HAS_MATPLOTLIB:
        return

    categories = list(metrics["by_category"].keys())
    rates = [metrics["by_category"][c]["success_rate"] * 100 for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(categories, rates, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Command Category", fontsize=12)
    ax.set_title("Navigation Success Rate by Command Category", fontsize=14)
    ax.set_ylim(0, 110)
    ax.axhline(y=metrics["overall_success_rate"] * 100, color="red",
               linestyle="--", label=f"Overall: {metrics['overall_success_rate']:.1%}")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_localization_errors(results: list, output_path: str):
    """Histogram of localization errors."""
    if not HAS_MATPLOTLIB:
        return

    errors = [r["localization_error"] for r in results if r["localization_error"] is not None]
    if not errors:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=20, color="#2196F3", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(errors), color="red", linestyle="--",
               label=f"Mean: {np.mean(errors):.2f}m")
    ax.axvline(np.median(errors), color="green", linestyle="--",
               label=f"Median: {np.median(errors):.2f}m")
    ax.set_xlabel("Localization Error (m)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Localization Errors", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def failure_analysis(results: list) -> dict:
    """Categorize failures."""
    failures = [r for r in results if not r["found"]]

    categories = {
        "object_not_in_map": 0,
        "attribute_mismatch": 0,
        "spatial_query_fail": 0,
        "other": 0,
    }

    for f in failures:
        if f["target_attribute"]:
            categories["attribute_mismatch"] += 1
        elif f["spatial_relation"]:
            categories["spatial_query_fail"] += 1
        else:
            categories["object_not_in_map"] += 1

    return {
        "total_failures": len(failures),
        "breakdown": categories,
        "failed_commands": [f["command"] for f in failures],
    }


def generate_latex_table(metrics: dict) -> str:
    """Generate LaTeX table for IEEE paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Navigation Success Rates by Command Category}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Category} & \textbf{Total} & \textbf{Success} & \textbf{Rate} \\",
        r"\hline",
    ]

    for cat, data in metrics["by_category"].items():
        name = cat.replace("_", " ").title()
        lines.append(
            f"{name} & {data['total']} & {data['found']} & "
            f"{data['success_rate']:.1%} \\\\"
        )

    lines.append(r"\hline")
    lines.append(
        f"\\textbf{{Overall}} & {metrics['total_commands']} & "
        f"{metrics['total_found']} & "
        f"\\textbf{{{metrics['overall_success_rate']:.1%}}} \\\\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="data/experiments", help="Output directory")
    args = parser.parse_args()

    results, metrics = load_results(args.results_json)
    os.makedirs(args.output_dir, exist_ok=True)

    # Print summary
    print("=" * 60)
    print("EXPERIMENT ANALYSIS")
    print("=" * 60)
    print(f"Total commands: {metrics['total_commands']}")
    print(f"Overall success rate: {metrics['overall_success_rate']:.1%}")
    print()

    for cat, data in metrics["by_category"].items():
        print(f"  {cat}: {data['success_rate']:.1%} ({data['found']}/{data['total']})")

    # Failure analysis
    failures = failure_analysis(results)
    print(f"\nFailure analysis ({failures['total_failures']} failures):")
    for mode, count in failures["breakdown"].items():
        print(f"  {mode}: {count}")

    # Generate plots
    if HAS_MATPLOTLIB:
        plot_success_rates(
            metrics, os.path.join(args.output_dir, "success_rates.png")
        )
        plot_localization_errors(
            results, os.path.join(args.output_dir, "localization_errors.png")
        )

    # Generate LaTeX table
    latex = generate_latex_table(metrics)
    latex_path = os.path.join(args.output_dir, "results_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")
    print("\n" + latex)


if __name__ == "__main__":
    main()
