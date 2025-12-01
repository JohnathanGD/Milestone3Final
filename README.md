# Milestone III: NFL Game Outcome Prediction 

## Overview

This project presents a data-driven machine learning pipeline to predict NFL regular-season game outcomes (home vs away win) using team-level offensive and defensive efficiency metrics.  
Using publicly available play-by-play data (1999–2025), we engineer features such as expected points added (EPA), yardage per play, pressure rate, turnovers, and explosive plays for both home and away teams.  

Multiple classification models are trained and evaluated, including logistic regression, random forest, and XGBoost. Hyperparameters are optimized using Bayesian search, and model performance is assessed through held-out testing, cross-validation, and out-of-time extrapolation.

The final system produces well-calibrated win probabilities, enabling probabilistic forecasting for real-world usage such as game prediction, simulation, or betting strategy research.

---

## Repository Structure

Milestone3Final/
│
├── Data/ # Raw and generated datasets
│
├── outputs/ # Figures, tables, predictions, and saved models
│
├── defensive_NFL_Stats.py # Script to compute defensive game-level logs
├── offensive_NFL_Stats.py # Builds/loads offensive team aggregates
├── repro_m2.ipynb # Main notebook (preprocessing, modeling, results)
├── README.md # Project documentation
└── 


