import os

import numpy as np

from oml.src.objective import load_iris_data, pso_objective_factory
from oml.src.experiments import run_grid_experiments
from oml.src.visualize import (
    get_best_result,
    plot_accuracy_heatmap,
    plot_convergence_all,
    plot_convergence_by_swarm,
    print_results_table,
    save_results_table,
)


def main():
    # Experiment settings requested in the project statement.
    swarm_sizes = [10, 30, 50]
    inertia_weights = [0.4, 0.7, 0.9]
    iterations = 100

    # Search space for SVC hyperparameters in log10 scale.
    # position[0] -> log10(C) in [-2, 2]
    # position[1] -> log10(gamma) in [-3, 1]
    lower_bounds = np.array([-2.0, -3.0])
    upper_bounds = np.array([2.0, 1.0])
    bounds = (lower_bounds, upper_bounds)

    output_dir = "outputs"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    X_train, X_val, y_train, y_val = load_iris_data()
    objective_func = pso_objective_factory(X_train, X_val, y_train, y_val)

    results = run_grid_experiments(
        objective_func=objective_func,
        swarm_sizes=swarm_sizes,
        inertia_weights=inertia_weights,
        iterations=iterations,
        bounds=bounds,
        dimensions=2,
    )

    print_results_table(results)

    csv_path = os.path.join(output_dir, "results_summary.csv")
    save_results_table(results, csv_path)

    plot_convergence_by_swarm(results, plots_dir)
    plot_convergence_all(results, plots_dir)
    plot_accuracy_heatmap(results, plots_dir)

    best = get_best_result(results)
    print("\nBest Performing Configuration")
    print("-" * 40)
    print(f"Swarm Size       : {best['swarm_size']}")
    print(f"Inertia Weight   : {best['inertia_weight']}")
    print(f"Best Cost        : {best['best_cost']:.4f}")
    print(f"Best Accuracy    : {best['best_accuracy']:.4f}")
    print(f"Convergence Iter : {best['convergence_iteration']}")
    print(f"Best C           : {best['best_C']:.4f}")
    print(f"Best Gamma       : {best['best_gamma']:.6f}")

    print("\nArtifacts generated:")
    print(f"- Table CSV: {csv_path}")
    print(f"- Plots dir: {plots_dir}")


if __name__ == "__main__":
    main()
