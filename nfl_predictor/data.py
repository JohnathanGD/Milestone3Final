"""Load and merge offensive / defensive game logs."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .constants import (
    OFFENSE_PATH,
    DEFENSE_PATH,
    TEAM_ABBR_NORMALIZE,
)


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
    Returns (feature_matrix X, full processed df with meta, target y).
    Mirrors repro_m2.ipynb preprocessing.
    """
    off = load_offense()
    defense = load_defense()

    score_meta = (
        off[["game_id", "score_home", "score_away"]]
        .drop_duplicates("game_id")
        .copy()
    )

    defense = defense.drop(
        columns=[
            "points_allowed",
            "points_scored_by_team",
            "home_score_final",
            "away_score_final",
            "result",
        ],
        errors="ignore",
    )
    defense = defense.merge(score_meta, on="game_id", how="left")

    def get_points_allowed(row):
        if row["defteam"] == row["home_team"]:
            return row["score_away"]
        if row["defteam"] == row["away_team"]:
            return row["score_home"]
        return np.nan

    def get_points_scored(row):
        if row["defteam"] == row["home_team"]:
            return row["score_home"]
        if row["defteam"] == row["away_team"]:
            return row["score_away"]
        return np.nan

    defense["points_allowed"] = defense.apply(get_points_allowed, axis=1)
    defense["points_scored_by_team"] = defense.apply(get_points_scored, axis=1)
    defense["result"] = defense["points_scored_by_team"] - defense["points_allowed"]
    defense = defense.drop(columns=["score_home", "score_away"], errors="ignore")

    df = off.copy()
    df["result"] = np.where(
        df["score_home"] > df["score_away"],
        1,
        np.where(df["score_home"] < df["score_away"], 0, np.nan),
    )
    df = df.dropna(subset=["result"]).copy()
    df["result"] = df["result"].astype(int)

    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df["schedule_playoff"] = df["schedule_playoff"].astype(int)

    df["home_favorite"] = np.where(
        df["spread_line"] < 0,
        1,
        np.where(df["spread_line"] > 0, 0, np.nan),
    )
    df["home_favorite"] = df["home_favorite"].fillna(0).astype(int)

    home_def = defense[defense["is_home"] == 1]
    away_def = defense[defense["is_home"] == 0]

    from .features import prep_def_side

    home_def_prepped = prep_def_side(home_def, "home_def", "home_abbr")
    away_def_prepped = prep_def_side(away_def, "away_def", "away_abbr")

    df = df.merge(home_def_prepped, on=["game_id", "home_abbr"], how="left")
    df = df.merge(away_def_prepped, on=["game_id", "away_abbr"], how="left")

    if "weather" in df.columns:
        df = df.drop(columns=["weather"])

    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    num_cols_all = df.select_dtypes(include=[np.number]).columns
    df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].mean())

    meta_cols = [
        "schedule_season",
        "schedule_week",
        "team_home",
        "team_away",
        "home_abbr",
        "away_abbr",
        "score_home",
        "score_away",
        "game_result",
    ]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()

    drop_candidates = [
        "game_id",
        "schedule_date",
        "team_home",
        "team_away",
        "home_abbr",
        "away_abbr",
        "score_home",
        "score_away",
        "home_wp_post",
        "away_wp_post",
        "game_result",
    ]
    df_proc = df.drop(columns=[c for c in drop_candidates if c in df.columns])

    from .constants import DROP_LEAKAGE

    X = df_proc.drop(columns=["result"])
    y = df_proc["result"].copy()
    X = X.drop(columns=[c for c in DROP_LEAKAGE if c in X.columns])

    return X, meta, y
