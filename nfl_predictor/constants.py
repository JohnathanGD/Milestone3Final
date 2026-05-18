from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MILESTONE_PDF_PATH = ROOT / "milestone3.pdf"
DATA_DIR = ROOT / "Data"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

OFFENSE_PATH = DATA_DIR / "offensive_team_logs_from_nfl_data_py_1999_2025.csv"
DEFENSE_PATH = DATA_DIR / "team_defense_game_logs_1999_2025.csv"

SEED = 42
EXTRAPOLATION_CUTOFF_SEASON = 2019

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

OFF_HOME_STAT_COLS = [
    "total_home_epa",
    "total_home_rush_epa",
    "total_home_pass_epa",
    "home_qb_epa",
]
OFF_AWAY_STAT_COLS = [
    "total_away_epa",
    "total_away_rush_epa",
    "total_away_pass_epa",
    "away_qb_epa",
]

# Do not aggregate these into season-to-date defensive priors
DEF_AGG_EXCLUDE = {
    "season",
    "week",
    "season_type",
    "defteam",
    "offteam",
    "home_team",
    "away_team",
    "is_home",
    "result",
    "points_allowed",
    "points_scored_by_team",
    "home_score_final",
    "away_score_final",
    "game_id",
}

# Home minus away differentials (lagged columns)
DIFF_COLUMNS = [
    ("home_def_epa_per_play_allowed", "away_def_epa_per_play_allowed", "def_epa_per_play_diff"),
    ("home_def_success_rate_allowed", "away_def_success_rate_allowed", "def_success_rate_diff"),
    ("home_def_pressure_rate", "away_def_pressure_rate", "def_pressure_rate_diff"),
    ("home_def_yards_per_play_allowed", "away_def_yards_per_play_allowed", "def_yards_per_play_diff"),
    ("total_home_epa", "total_away_epa", "off_epa_diff"),
    ("total_home_pass_epa", "total_away_pass_epa", "off_pass_epa_diff"),
]

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

# Post-game leakage (same-game offense removed in features.py before lag merge)
DROP_LEAKAGE = [
    "home_wp",
    "away_wp",
    "home_wp_post",
    "away_wp_post",
    "spread_line",
    "total_line",
    "over_under_line",
    "spread_favorite",
    "has_betting_line",
    "score_home",
    "score_away",
    "game_result",
]
