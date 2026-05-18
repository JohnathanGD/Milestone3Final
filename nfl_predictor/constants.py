from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

OFFENSE_PATH = DATA_DIR / "offensive_team_logs_from_nfl_data_py_1999_2025.csv"
DEFENSE_PATH = DATA_DIR / "team_defense_game_logs_1999_2025.csv"

SEED = 42

TEAM_FULL = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "STL": "St. Louis Rams",
    "SD": "San Diego Chargers",
    "OAK": "Oakland Raiders",
}

TEAM_ABBR_NORMALIZE = {
    "OAK": "LV",
    "STL": "LAR",
    "SD": "LAC",
}

# Hyperparameters from repro_m2.ipynb tuning runs
TUNED_RF = {
    "n_estimators": 300,
    "max_depth": 14,
    "criterion": "entropy",
}
TUNED_LR_C = 0.001
TUNED_XGB = {
    "learning_rate": 0.013413910048382898,
    "max_depth": 2,
    "n_estimators": 1500,
}

DROP_LEAKAGE = [
    "home_def_points_allowed",
    "home_def_points_scored_by_team",
    "away_def_points_allowed",
    "away_def_points_scored_by_team",
    "total_home_epa",
    "total_home_rush_epa",
    "total_home_pass_epa",
    "home_qb_epa",
    "total_away_epa",
    "total_away_rush_epa",
    "total_away_pass_epa",
    "away_qb_epa",
    "home_wp",
    "away_wp",
    "spread_line",
    "total_line",
    "over_under_line",
    "spread_favorite",
    "home_favorite",
    "has_betting_line",
]
