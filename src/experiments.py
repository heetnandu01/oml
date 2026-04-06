import numpy as np
import pyswarms as ps

from oml.src.objective import decode_position


def estimate_convergence_iteration(cost_history):
    """
    Estimate convergence speed by finding first iteration reaching
    within 5% of the total possible improvement.
    """
    history = np.array(cost_history)
    start = history[0]
    end = history[-1]

    if np.isclose(start, end):
        return 0

    target = end + 0.05 * (start - end)
    hit_indices = np.where(history <= target)[0]
    if len(hit_indices) == 0:
        return len(history) - 1
    return int(hit_indices[0])


def run_single_experiment(
    objective_func,
    swarm_size,
    inertia_weight,
    iterations,
    bounds,
    dimensions=2,
):
    """Run one PSO experiment and return structured results."""
    options = {
        "c1": 1.5,
        "c2": 1.5,
        "w": inertia_weight,
    }

    optimizer = ps.single.GlobalBestPSO(
        n_particles=swarm_size,
        dimensions=dimensions,
        options=options,
        bounds=bounds,
    )

    best_cost, best_position = optimizer.optimize(
        objective_func,
        iters=iterations,
        verbose=False,
    )

    cost_history = np.array(optimizer.cost_history)
    best_accuracy = 1.0 - float(best_cost)
    best_c, best_gamma = decode_position(best_position)

    result = {
        "swarm_size": swarm_size,
        "inertia_weight": inertia_weight,
        "iterations": iterations,
        "best_cost": float(best_cost),
        "best_accuracy": best_accuracy,
        "best_C": float(best_c),
        "best_gamma": float(best_gamma),
        "convergence_iteration": estimate_convergence_iteration(cost_history),
        "cost_history": cost_history,
    }
    return result


def run_grid_experiments(
    objective_func,
    swarm_sizes,
    inertia_weights,
    iterations,
    bounds,
    dimensions=2,
):
    """Run all swarm size and inertia weight combinations."""
    results = []

    for swarm_size in swarm_sizes:
        for inertia_weight in inertia_weights:
            result = run_single_experiment(
                objective_func=objective_func,
                swarm_size=swarm_size,
                inertia_weight=inertia_weight,
                iterations=iterations,
                bounds=bounds,
                dimensions=dimensions,
            )
            results.append(result)
            print(
                "Finished: swarm_size={0}, w={1}, best_cost={2:.4f}, "
                "best_accuracy={3:.4f}, convergence_iter={4}".format(
                    result["swarm_size"],
                    result["inertia_weight"],
                    result["best_cost"],
                    result["best_accuracy"],
                    result["convergence_iteration"],
                )
            )

    return results
