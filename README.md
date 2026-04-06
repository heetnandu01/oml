# Particle Swarm Optimization Parameter Analysis

This project studies the impact of:
- Swarm size (`10`, `30`, `50`)
- Inertia weight (`0.4`, `0.7`, `0.9`)

on PSO optimization performance using the Iris dataset.

## Objective

Particle positions represent SVC hyperparameters in log scale:
- `position[0] = log10(C)`
- `position[1] = log10(gamma)`

The objective minimized by PSO is:
- `cost = 1 - validation_accuracy`

So lower cost means better model performance.

## Project Structure

- `requirements.txt` : dependencies
- `src/objective.py` : data loading and objective function
- `src/experiments.py` : PSO setup and experiment loop
- `src/visualize.py` : table and plots
- `src/main.py` : full experiment pipeline
- `outputs/` : generated results

## Setup & Run

### Option 1: Using Virtual Environment (Recommended)

**Activate virtual environment:**
```bash
.\.venv\Scripts\Activate.ps1
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the project:**
```bash
python src/main.py
```

**Deactivate virtual environment:**
```bash
deactivate
```

---

### Option 2: Run Directly Without Virtual Environment

**Run the project:**
```bash
C:/Users/heetn/AppData/Local/Programs/Python/Python312/python.exe src/main.py
```

**Note:** No venv activation or dependency installation needed with this method.

## Outputs

After running, you get:
- `outputs/results_summary.csv` : comparison table for all parameter combinations
- `outputs/plots/convergence_swarm_10.png`
- `outputs/plots/convergence_swarm_30.png`
- `outputs/plots/convergence_swarm_50.png`
- `outputs/plots/convergence_all_configs.png`
- `outputs/plots/accuracy_heatmap.png`

The console also prints:
- Comparison table
- Best performing configuration

## Notes

- Iterations are fixed at `100` for each experiment.
- Convergence speed is estimated using the first iteration where cost reaches within 5% of total improvement.
- This implementation is intentionally simple and readable for an undergraduate project report/demo.
