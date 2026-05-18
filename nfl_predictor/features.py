"""Feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import TEAM_FULL, TEAM_ABBR_NORMALIZE


def prep_def_side(df_def: pd.DataFrame, side_prefix: str, team_col_name: str) -> pd.DataFrame:
    df_def = df_def.copy()
    drop_cols = [
        "season",
        "week",
        "season_type",
        "offteam",
        "home_team",
        "away_team",
        "is_home",
        "result",
        "home_score_final",
        "away_score_final",
        "score_home",
        "score_away",
    ]
    for c in drop_cols:
        if c in df_def.columns:
            df_def = df_def.drop(columns=c)

    df_def = df_def.rename(columns={"defteam": team_col_name})
    key_cols = ["game_id", team_col_name]
    rename_map = {
        c: f"{side_prefix}_{c}" for c in df_def.columns if c not in key_cols
    }
    return df_def.rename(columns=rename_map)


def build_week_features(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    season: int,
    week: int,
    feature_columns: list[str],
    train_means: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Season-to-date defensive averages for games in (season, week).
    Returns (X_features, meta with team names and optional actual scores).
    """
    wk = off[(off["schedule_season"] == season) & (off["schedule_week"] == week)].copy()
    if wk.empty:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame()

    for col in ["home_abbr", "away_abbr"]:
        if col in wk.columns:
            wk[col] = wk[col].replace(TEAM_ABBR_NORMALIZE)

    for side in ["home", "away"]:
        name_col, abbr_col = f"team_{side}", f"{side}_abbr"
        if name_col in wk.columns and abbr_col in wk.columns:
            wk[name_col] = wk[name_col].fillna(wk[abbr_col].map(TEAM_FULL))

    past_def = defense[
        (defense["season"] == season) & (defense["week"] < week)
    ].copy()

    def_id_cols = {
        "season",
        "week",
        "season_type",
        "defteam",
        "offteam",
        "home_team",
        "away_team",
        "is_home",
        "result",
        "home_score_final",
        "away_score_final",
    }
    def_num_cols = [
        c
        for c in past_def.columns
        if c not in def_id_cols and np.issubdtype(past_def[c].dtype, np.number)
    ]

    if past_def.empty or not def_num_cols:
        def_team_agg = pd.DataFrame(columns=["defteam"] + def_num_cols)
    else:
        def_team_agg = (
            past_def.groupby("defteam")[def_num_cols].mean().reset_index()
        )

    home_def_agg = def_team_agg.rename(
        columns={"defteam": "home_abbr", **{c: f"home_def_{c}" for c in def_num_cols}}
    )
    away_def_agg = def_team_agg.rename(
        columns={"defteam": "away_abbr", **{c: f"away_def_{c}" for c in def_num_cols}}
    )

    wk = wk.merge(home_def_agg, on="home_abbr", how="left")
    wk = wk.merge(away_def_agg, on="away_abbr", how="left")
    wk = wk.loc[:, ~wk.columns.duplicated()]

    X_wk = wk.reindex(columns=feature_columns)
    for col in X_wk.columns:
        if np.issubdtype(X_wk[col].dtype, np.number):
            if col in train_means.index:
                X_wk[col] = X_wk[col].fillna(train_means[col])
            else:
                X_wk[col] = X_wk[col].fillna(0)

    meta_cols = [
        "schedule_season",
        "schedule_week",
        "team_home",
        "team_away",
        "score_home",
        "score_away",
        "game_result",
        "spread_line",
        "total_line",
    ]
    meta = wk[[c for c in meta_cols if c in wk.columns]].copy()
    return X_wk, meta
