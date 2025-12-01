# Milestone III: NFL Game Outcome Prediction 

## Overview

This project presents a data-driven machine learning pipeline to predict NFL regular-season game outcomes (home vs away win) using team-level offensive and defensive efficiency metrics.  
Using publicly available play-by-play data (1999–2025), we engineer features such as expected points added (EPA), yardage per play, pressure rate, turnovers, and explosive plays for both home and away teams.  

Multiple classification models are trained and evaluated, including logistic regression, random forest, and XGBoost. Hyperparameters are optimized using Bayesian search, and model performance is assessed through held-out testing, cross-validation, and out-of-time extrapolation.

The final system produces well-calibrated win probabilities, enabling probabilistic forecasting for real-world usage such as game prediction, simulation, or betting strategy research.

---

## Repository Structure

Milestone3Final/
- Data/ — Raw and generated datasets
- outputs/ — Figures, tables, predictions, and saved models
- defensive_NFL_Stats.py — Builds defensive game-level logs
- offensive_NFL_Stats.py — Builds offensive aggregates
- repro_m2.ipynb — Main notebook (training, evaluation, prediction)
- README.md — Project documentation


---

## Installation & Dependencies

### Python Version
Python 3.9+

### Required Libraries
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `nfl_data_py`
- `matplotlib`
- `seaborn`

### Recommended Setup

python3 -m venv .venv
source .venv/bin/activate # for macOS/Linux
.venv\Scripts\activate # for Windows


---

## Usage

### 1) Build Defensive Game Logs (1999–2025)

python3 defensive_NFL_Stats.py

This generates:

Data/team_defense_game_logs_1999_2025.csv

### 2) Build Offensive Game Logs (1999-2025)

python3 offensive_NFL_Stats.py

This generates:

Data/offensive_team_logs_from_nfl_data_py_1999_2025.csv


---

### 2) Train Models & Run Experiments

Open:

repro_m2.ipynb


This notebook handles:
- Data cleaning and merging (OFF + DEF)
- Train/test split
- Feature scaling
- Hyperparameter tuning
- Model training
- Cross-validation
- Out-of-time testing
- Prediction output (Week 13, 2025)
- confusion matrices, ROC, PR curves

---

## 4) Results Summary

### Dataset
- 7,158 regular-season games (1999–2025)

### Baseline
- Always predicting home team (56.5% accuracy)

### Test Performance (30% Split)

| Model                     | Accuracy | Precision | Recall | F1   | AUROC |
|---------------------------|----------|-----------|--------|------|-------|
| Random Forest (tuned)     | 0.870    | 0.871     | 0.903  | 0.886 | 0.945 |
| Logistic Regression       | 0.857    | 0.854     | 0.901  | 0.877 | 0.938 |
| XGBoost (tuned)           | 0.876    | 0.875     | 0.909  | 0.892 | 0.952 |

### Cross-Validation (pre-2019)
- Random Forest: 0.878  
- Logistic Regression: 0.866

### Out-of-Time Test (2019+)
- Voting ensemble: **91.99% betting win rate**

---

## Included Features

### Offensive
- Total EPA
- Rushing and passing EPA
- QB EPA

### Defensive
- EPA allowed
- Yards per play
- Pressure rate
- Sack rate
- Turnovers forced
- Explosive plays allowed

### Context
- Weather
- Betting lines
- Home favorite flags

---

## Limitations & Future Work

- Binary classification only (no margin prediction)
- Static training (no automation for weekly refresh)
- No player-level injury modeling
- No calibration curves (planned)
- Could extend with:
  - Injury reports
  - Travel/rest days
  - Weather severity
  - Market efficiency modeling

---

## Author

**Johnathan Gutierrez-Diaz**  
GitHub: https://github.com/JohnathanGD  

---







