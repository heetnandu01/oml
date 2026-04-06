import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def save_results_table(results, output_csv_path):
    """Save experiment summary as CSV."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    fieldnames = [
        "swarm_size",
        "inertia_weight",
        "iterations",
        "best_cost",
        "best_accuracy",
        "convergence_iteration",
        "best_C",
        "best_gamma",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r[key] for key in fieldnames})


def print_results_table(results):
    """Print a clean text table for all parameter combinations."""
    headers = [
        "Swarm",
        "Inertia",
        "Best Cost",
        "Best Acc",
        "Conv Iter",
        "Best C",
        "Best Gamma",
    ]
    row_format = "{:<8} {:<8} {:<10} {:<10} {:<10} {:<12} {:<12}"

    print("\nComparison Table")
    print("-" * 80)
    print(row_format.format(*headers))
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x["swarm_size"], x["inertia_weight"])):
        print(
            row_format.format(
                r["swarm_size"],
                r["inertia_weight"],
                f"{r['best_cost']:.4f}",
                f"{r['best_accuracy']:.4f}",
                r["convergence_iteration"],
                f"{r['best_C']:.4f}",
                f"{r['best_gamma']:.6f}",
            )
        )


def get_best_result(results):
    """Select best config by lowest cost; tie-break by earlier convergence."""
    return sorted(
        results,
        key=lambda x: (x["best_cost"], x["convergence_iteration"]),
    )[0]


def plot_convergence_by_swarm(results, output_dir):
    """For each swarm size, plot convergence curves across inertia weights."""
    os.makedirs(output_dir, exist_ok=True)

    swarm_sizes = sorted({r["swarm_size"] for r in results})

    for swarm_size in swarm_sizes:
        subset = [r for r in results if r["swarm_size"] == swarm_size]
        subset = sorted(subset, key=lambda x: x["inertia_weight"])

        plt.figure(figsize=(9, 5))
        for r in subset:
            plt.plot(
                r["cost_history"],
                label=f"w={r['inertia_weight']}",
                linewidth=2,
            )

        plt.title(f"Convergence Curves (Swarm Size = {swarm_size})")
        plt.xlabel("Iteration")
        plt.ylabel("Cost (1 - Accuracy)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"convergence_swarm_{swarm_size}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()


def plot_convergence_all(results, output_dir):
    """Plot all parameter combinations on one graph."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sorted_results = sorted(results, key=lambda x: (x["swarm_size"], x["inertia_weight"]))

    for r in sorted_results:
        label = f"N={r['swarm_size']}, w={r['inertia_weight']}"
        plt.plot(r["cost_history"], label=label, linewidth=1.6)

    plt.title("PSO Convergence for All Configurations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (1 - Accuracy)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "convergence_all_configs.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_heatmap(results, output_dir):
    """Plot heatmap for final accuracy across swarm size and inertia weight."""
    os.makedirs(output_dir, exist_ok=True)

    swarm_sizes = sorted({r["swarm_size"] for r in results})
    inertia_weights = sorted({r["inertia_weight"] for r in results})

    matrix = np.zeros((len(swarm_sizes), len(inertia_weights)))

    for i, swarm_size in enumerate(swarm_sizes):
        for j, inertia_weight in enumerate(inertia_weights):
            match = [
                r
                for r in results
                if r["swarm_size"] == swarm_size and r["inertia_weight"] == inertia_weight
            ][0]
            matrix[i, j] = match["best_accuracy"]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Final Best Accuracy")

    plt.xticks(range(len(inertia_weights)), inertia_weights)
    plt.yticks(range(len(swarm_sizes)), swarm_sizes)
    plt.xlabel("Inertia Weight")
    plt.ylabel("Swarm Size")
    plt.title("Final Accuracy Heatmap")

    for i in range(len(swarm_sizes)):
        for j in range(len(inertia_weights)):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
            )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
