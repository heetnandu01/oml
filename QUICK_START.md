# OML Project - Quick Start Guide

## 🚀 Option 1: With Virtual Environment (Recommended)

### Step 1: Activate Virtual Environment
```bash
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Project
```bash
python src/main.py
```

### Step 4: Deactivate When Done
```bash
deactivate
```

---

## ⚡ Option 2: Without Virtual Environment (Direct)

### Run Project Directly
```bash
C:/Users/heetn/AppData/Local/Programs/Python/Python312/python.exe src/main.py
```

---

## 📊 What the Project Does

- Runs **9 PSO experiments** (3 swarm sizes × 3 inertia weights)
- Generates **comparison table** in console
- Saves **CSV results** to `outputs/results_summary.csv`
- Creates **5 visualization plots** in `outputs/plots/`
- Prints **best configuration** summary

---

## 📁 Project Structure

```
oml-project/
├── src/
│   ├── main.py          # Run this!
│   ├── objective.py     # Iris data + PSO objective
│   ├── experiments.py   # PSO grid experiments
│   └── visualize.py     # Plots & tables
├── outputs/             # Generated results
│   ├── results_summary.csv
│   └── plots/
├── requirements.txt     # Dependencies
└── README.md           # Full documentation
```

---

## ✅ Troubleshooting

**Issue:** `activate : The term 'activate' is not recognized`
- **Solution:** Use full path: `.\.venv\Scripts\Activate.ps1`

**Issue:** Permission denied when activating
- **Solution:** Run this once:
  ```bash
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

**Issue:** ModuleNotFoundError after activating venv
- **Solution:** Ensure dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```
