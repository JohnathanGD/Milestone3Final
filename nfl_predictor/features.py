"""Feature engineering: season-to-date (lagged) offensive and defensive stats."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    DEF_AGG_EXCLUDE,
    DIFF_COLUMNS,
    DROP_LEAKAGE,
    OFF_AWAY_STAT_COLS,
    OFF_HOME_STAT_COLS,
    TEAM_ABBR_NORMALIZE,
    TEAM_FULL,
)


def _normalize_game_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["home_abbr", "away_abbr"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ABBR_NORMALIZE)
    for side in ["home", "away"]:
        name_col, abbr_col = f"team_{side}", f"{side}_abbr"
        if name_col in df.columns and abbr_col in df.columns:
            df[name_col] = df[name_col].fillna(df[abbr_col].map(TEAM_FULL))
    return df


def defense_numeric_cols(defense: pd.DataFrame) -> list[str]:
    exclude = DEF_AGG_EXCLUDE
    return [
        c
        for c in defense.columns
        if c not in exclude and np.issubdtype(defense[c].dtype, np.number)
    ]


def _prior_expanding_means(
    df: pd.DataFrame,
    group_cols: list[str],
    sort_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Per-group expanding mean of prior games only (current game excluded)."""
    if df.empty or not value_cols:
        return df.copy()

    out = df.sort_values(group_cols + [sort_col]).copy()
    grouped = out.groupby(group_cols, sort=False)[value_cols]
    prior = grouped.transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
    prior.columns = [f"prior_{c}" for c in value_cols]
    return pd.concat([out, prior], axis=1)


def build_offense_team_games(off: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team per game with that team's offensive box score."""
    home_cols = ["schedule_season", "schedule_week", "home_abbr"] + [
        c for c in OFF_HOME_STAT_COLS if c in off.columns
    ]
    away_cols = ["schedule_season", "schedule_week", "away_abbr"] + [
        c for c in OFF_AWAY_STAT_COLS if c in off.columns
    ]

    home = off[home_cols].copy()
    home = home.rename(
        columns={
            "home_abbr": "team_abbr",
            **{
                "total_home_epa": "off_epa",
                "total_home_rush_epa": "off_rush_epa",
                "total_home_pass_epa": "off_pass_epa",
                "home_qb_epa": "off_qb_epa",
            },
        }
    )

    away = off[away_cols].copy()
    away = away.rename(
        columns={
            "away_abbr": "team_abbr",
            **{
                "total_away_epa": "off_epa",
                "total_away_rush_epa": "off_rush_epa",
                "total_away_pass_epa": "off_pass_epa",
                "away_qb_epa": "off_qb_epa",
            },
        }
    )

    return pd.concat([home, away], ignore_index=True)


def build_lagged_game_table(
    off: pd.DataFrame,
    defense: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    One row per game with features known before kickoff:
    season-to-date offensive/defensive averages through the prior week.
    """
    off = _normalize_game_teams(off)
    defense = defense.copy()
    for col in ["defteam", "home_team", "away_team"]:
        if col in defense.columns:
            defense[col] = defense[col].replace(TEAM_ABBR_NORMALIZE)

    same_game_off = OFF_HOME_STAT_COLS + OFF_AWAY_STAT_COLS
    leak_cols = same_game_off + ["home_wp", "away_wp", "home_wp_post", "away_wp_post"]
    games = off.drop(columns=[c for c in leak_cols if c in off.columns], errors="ignore").copy()
    games["result"] = np.where(
        games["score_home"] > games["score_away"],
        1,
        np.where(games["score_home"] < games["score_away"], 0, np.nan),
    )
    games = games.dropna(subset=["result"]).copy()
    games["result"] = games["result"].astype(int)
    games["schedule_playoff"] = games["schedule_playoff"].astype(int)
    games["temp"] = pd.to_numeric(games.get("temp"), errors="coerce")

    if "spread_line" in games.columns:
        games["home_favorite"] = np.where(
            games["spread_line"] < 0,
            1,
            np.where(games["spread_line"] > 0, 0, np.nan),
        )
        games["home_favorite"] = games["home_favorite"].fillna(0).astype(int)

    def_num = defense_numeric_cols(defense)
    def_long = _prior_expanding_means(
        defense,
        ["season", "defteam"],
        "week",
        def_num,
    )
    def_prior_cols = [f"prior_{c}" for c in def_num]
    def_keys = ["season", "week", "defteam"] + def_prior_cols
    def_lookup = def_long[def_keys].rename(
        columns={
            "season": "schedule_season",
            "week": "schedule_week",
            "defteam": "team_abbr",
        }
    )

    home_def = def_lookup.rename(
        columns={"team_abbr": "home_abbr", **{c: f"home_def_{c[6:]}" for c in def_prior_cols}}
    )
    away_def = def_lookup.rename(
        columns={"team_abbr": "away_abbr", **{c: f"away_def_{c[6:]}" for c in def_prior_cols}}
    )

    off_long = build_offense_team_games(off)
    off_num = [c for c in off_long.columns if c.startswith("off_")]
    off_prior = _prior_expanding_means(
        off_long,
        ["schedule_season", "team_abbr"],
        "schedule_week",
        off_num,
    )
    off_prior_cols = [f"prior_{c}" for c in off_num]
    off_lookup = off_prior[
        ["schedule_season", "schedule_week", "team_abbr"] + off_prior_cols
    ]

    home_off = off_lookup.rename(
        columns={
            "team_abbr": "home_abbr",
            **{
                "prior_off_epa": "total_home_epa",
                "prior_off_rush_epa": "total_home_rush_epa",
                "prior_off_pass_epa": "total_home_pass_epa",
                "prior_off_qb_epa": "home_qb_epa",
            },
        }
    )
    away_off = off_lookup.rename(
        columns={
            "team_abbr": "away_abbr",
            **{
                "prior_off_epa": "total_away_epa",
                "prior_off_rush_epa": "total_away_rush_epa",
                "prior_off_pass_epa": "total_away_pass_epa",
                "prior_off_qb_epa": "away_qb_epa",
            },
        }
    )

    merge_keys = ["schedule_season", "schedule_week"]
    games = games.merge(
        home_def, on=merge_keys + ["home_abbr"], how="left"
    )
    games = games.merge(
        away_def, on=merge_keys + ["away_abbr"], how="left"
    )
    games = games.merge(
        home_off, on=merge_keys + ["home_abbr"], how="left"
    )
    games = games.merge(
        away_off, on=merge_keys + ["away_abbr"], how="left"
    )
    games = games.loc[:, ~games.columns.duplicated()]

    games = add_differential_features(games)

    label_meta = [
        "team_home",
        "team_away",
        "home_abbr",
        "away_abbr",
        "score_home",
        "score_away",
        "game_result",
    ]
    meta = games[
        ["schedule_season", "schedule_week"]
        + [c for c in label_meta if c in games.columns]
    ].copy()
    non_feature = set(label_meta) | {
        "game_id",
        "schedule_date",
        "result",
        "weather",
    }

    feature_cols = [
        c
        for c in games.columns
        if c not in non_feature
        and c not in DROP_LEAKAGE
        and not c.endswith("_post")
    ]
    feature_cols = [c for c in feature_cols if c in games.columns]

    X = games[feature_cols].copy()
    y = games["result"].copy()
    return X, meta, y


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Home minus away for key rate stats (pre-game comparable)."""
    out = df.copy()
    for home_col, away_col, diff_name in DIFF_COLUMNS:
        if home_col in out.columns and away_col in out.columns:
            out[diff_name] = out[home_col] - out[away_col]
    return out


def build_week_features(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    season: int,
    week: int,
    feature_columns: list[str],
    train_means: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pre-game features for all games in (season, week).
    Uses the same lagged logic as training (stats through week-1).
    """
    off = _normalize_game_teams(off)
    wk = off[(off["schedule_season"] == season) & (off["schedule_week"] == week)].copy()
    if wk.empty:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame()

    full_X, full_meta, _ = build_lagged_game_table(off, defense)
    mask = (full_meta["schedule_season"] == season) & (
        full_meta["schedule_week"] == week
    )
    if not mask.any():
        return pd.DataFrame(columns=feature_columns), pd.DataFrame()

    X_wk = full_X.loc[mask].copy()
    meta = full_meta.loc[mask].copy()

    X_wk = X_wk.reindex(columns=feature_columns)
    for col in X_wk.columns:
        if np.issubdtype(X_wk[col].dtype, np.number):
            fill = train_means[col] if col in train_means.index else 0
            X_wk[col] = X_wk[col].fillna(fill)

    return X_wk, meta
