"""Load offensive / defensive logs and build lagged pre-game feature tables."""

from __future__ import annotations

import pandas as pd

from .constants import DEFENSE_PATH, OFFENSE_PATH, TEAM_ABBR_NORMALIZE
from .features import build_lagged_game_table


def data_status() -> dict[str, bool]:
    return {
        "offense": OFFENSE_PATH.is_file(),
        "defense": DEFENSE_PATH.is_file(),
    }


def _normalize_abbr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ABBR_NORMALIZE)
    return df


def load_offense() -> pd.DataFrame:
    if not OFFENSE_PATH.is_file():
        raise FileNotFoundError(
            f"Offensive logs not found at {OFFENSE_PATH}. "
            "Run: python offensive_NFL_Stats.py"
        )
    off = pd.read_csv(OFFENSE_PATH)
    return _normalize_abbr(off, ["home_abbr", "away_abbr"])


def load_defense() -> pd.DataFrame:
    if not DEFENSE_PATH.is_file():
        raise FileNotFoundError(
            f"Defensive logs not found at {DEFENSE_PATH}. "
            "Run: python defensive_NFL_Stats.py"
        )
    defense = pd.read_csv(DEFENSE_PATH)
    return _normalize_abbr(
        defense, ["defteam", "offteam", "home_team", "away_team"]
    )


def load_merged_game_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns (X, meta, y) with season-to-date features through the prior week.
  Training and live week predictions use the same logic.
    """
    off = load_offense()
    defense = load_defense()
    return build_lagged_game_table(off, defense)
