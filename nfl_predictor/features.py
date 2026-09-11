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


def _team_season_means(
    long_df: pd.DataFrame,
    season_col: str,
    team_col: str,
    value_cols: list[str],
    season: int,
) -> pd.DataFrame:
    """Full-season team averages for one season (used as next-season week-1 priors)."""
    if not value_cols or long_df.empty:
        return pd.DataFrame(columns=[team_col] + value_cols)
    season_df = long_df[long_df[season_col] == season]
    if season_df.empty:
        return pd.DataFrame(columns=[team_col] + value_cols)
    return season_df.groupby(team_col, as_index=False)[value_cols].mean()


def _fill_priors_from_prev_season(
    lookup: pd.DataFrame,
    *,
    season_col: str,
    team_col: str,
    prior_cols: list[str],
    history: pd.DataFrame,
    hist_season_col: str,
    hist_team_col: str,
    hist_value_cols: list[str],
) -> pd.DataFrame:
    """
    Where within-season priors are missing (typically week 1), fill with
    that team's previous-season averages. Later weeks keep within-season priors.
    """
    if lookup.empty or not prior_cols:
        return lookup

    out = lookup.copy()
    col_map = {
        f"prior_{raw}": raw
        for raw in hist_value_cols
        if f"prior_{raw}" in prior_cols
    }
    if not col_map:
        return out

    for season in sorted(out[season_col].dropna().unique()):
        season = int(season)
        means = _team_season_means(
            history, hist_season_col, hist_team_col, list(col_map.values()), season - 1
        )
        if means.empty:
            continue
        means = means.set_index(hist_team_col)
        mask = out[season_col] == season
        for prior_col, raw_col in col_map.items():
            missing = mask & out[prior_col].isna()
            if not missing.any() or raw_col not in means.columns:
                continue
            out.loc[missing, prior_col] = out.loc[missing, team_col].map(means[raw_col])
    return out


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
    Week 1 (no within-season history) uses each team's previous-season averages.
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
    def_long = _fill_priors_from_prev_season(
        def_long,
        season_col="season",
        team_col="defteam",
        prior_cols=def_prior_cols,
        history=defense,
        hist_season_col="season",
        hist_team_col="defteam",
        hist_value_cols=def_num,
    )
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
    # Priors only from played games (EPA present)
    off_scored = off_long.dropna(subset=off_num, how="all")
    off_prior = _prior_expanding_means(
        off_scored,
        ["schedule_season", "team_abbr"],
        "schedule_week",
        off_num,
    )
    off_prior_cols = [f"prior_{c}" for c in off_num]
    off_prior = _fill_priors_from_prev_season(
        off_prior,
        season_col="schedule_season",
        team_col="team_abbr",
        prior_cols=off_prior_cols,
        history=off_scored,
        hist_season_col="schedule_season",
        hist_team_col="team_abbr",
        hist_value_cols=off_num,
    )
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


def _as_of_team_means(
    long_df: pd.DataFrame,
    season_col: str,
    week_col: str,
    team_col: str,
    season: int,
    before_week: int,
    value_cols: list[str],
) -> pd.DataFrame:
    """Season-to-date means per team using only games strictly before `before_week`."""
    if not value_cols:
        return pd.DataFrame(columns=[team_col])
    prior = long_df[
        (long_df[season_col] == season) & (long_df[week_col] < before_week)
    ].copy()
    if prior.empty:
        return pd.DataFrame(columns=[team_col] + value_cols)
    means = prior.groupby(team_col, as_index=False)[value_cols].mean()
    return means


def _as_of_team_means_with_prev_season(
    long_df: pd.DataFrame,
    season_col: str,
    week_col: str,
    team_col: str,
    season: int,
    before_week: int,
    value_cols: list[str],
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """
    Within-season means before `before_week`, falling back to previous-season
    averages for teams (or columns) still missing — e.g. week 1 cold start.
    """
    within = _as_of_team_means(
        long_df, season_col, week_col, team_col, season, before_week, value_cols
    )
    prev = _team_season_means(long_df, season_col, team_col, value_cols, season - 1)

    if teams is None:
        team_set: set[str] = set()
        if not within.empty:
            team_set.update(within[team_col].tolist())
        if not prev.empty:
            team_set.update(prev[team_col].tolist())
        teams = sorted(team_set)

    if not teams:
        return pd.DataFrame(columns=[team_col] + value_cols)

    base = pd.DataFrame({team_col: teams})
    if not prev.empty:
        base = base.merge(prev, on=team_col, how="left")
    else:
        for c in value_cols:
            base[c] = np.nan

    if not within.empty:
        # Within-season priors override previous-season carryover where present.
        base = base.merge(within, on=team_col, how="left", suffixes=("_prev", ""))
        for c in value_cols:
            prev_c = f"{c}_prev"
            if prev_c in base.columns:
                base[c] = base[c].combine_first(base[prev_c])
                base = base.drop(columns=[prev_c])
    return base


def build_week_features(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    season: int,
    week: int,
    feature_columns: list[str],
    train_means: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pre-game features for all scheduled games in (season, week), including unplayed.

    Uses season-to-date offense/defense through the prior week. When those are
    missing (week 1), fills from each team's previous-season averages.
    """
    off = _normalize_game_teams(off)
    defense = defense.copy()
    for col in ["defteam", "home_team", "away_team"]:
        if col in defense.columns:
            defense[col] = defense[col].replace(TEAM_ABBR_NORMALIZE)

    wk = off[(off["schedule_season"] == season) & (off["schedule_week"] == week)].copy()
    if wk.empty:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame()

    # Drop same-game offense / WP columns so prior-season merges can populate them.
    same_game_off = OFF_HOME_STAT_COLS + OFF_AWAY_STAT_COLS
    leak_cols = same_game_off + ["home_wp", "away_wp", "home_wp_post", "away_wp_post"]
    games = wk.drop(columns=[c for c in leak_cols if c in wk.columns], errors="ignore").copy()
    if "schedule_playoff" in games.columns:
        games["schedule_playoff"] = games["schedule_playoff"].astype(int)
    if "temp" in games.columns:
        games["temp"] = pd.to_numeric(games["temp"], errors="coerce")
    if "spread_line" in games.columns:
        games["home_favorite"] = np.where(
            games["spread_line"] < 0,
            1,
            np.where(games["spread_line"] > 0, 0, np.nan),
        )
        games["home_favorite"] = games["home_favorite"].fillna(0).astype(int)

    teams = sorted(
        set(games["home_abbr"].dropna().tolist())
        | set(games["away_abbr"].dropna().tolist())
    )

    def_num = defense_numeric_cols(defense)
    def_means = _as_of_team_means_with_prev_season(
        defense, "season", "week", "defteam", season, week, def_num, teams=teams
    )
    if not def_means.empty:
        home_def = def_means.rename(
            columns={"defteam": "home_abbr", **{c: f"home_def_{c}" for c in def_num}}
        )
        away_def = def_means.rename(
            columns={"defteam": "away_abbr", **{c: f"away_def_{c}" for c in def_num}}
        )
        games = games.merge(home_def, on="home_abbr", how="left")
        games = games.merge(away_def, on="away_abbr", how="left")

    off_long = build_offense_team_games(off)
    off_num = [c for c in off_long.columns if c.startswith("off_")]
    scored = off_long.dropna(subset=[c for c in off_num if c in off_long.columns], how="all")
    off_means = _as_of_team_means_with_prev_season(
        scored,
        "schedule_season",
        "schedule_week",
        "team_abbr",
        season,
        week,
        off_num,
        teams=teams,
    )
    if not off_means.empty:
        home_off = off_means.rename(
            columns={
                "team_abbr": "home_abbr",
                "off_epa": "total_home_epa",
                "off_rush_epa": "total_home_rush_epa",
                "off_pass_epa": "total_home_pass_epa",
                "off_qb_epa": "home_qb_epa",
            }
        )
        away_off = off_means.rename(
            columns={
                "team_abbr": "away_abbr",
                "off_epa": "total_away_epa",
                "off_rush_epa": "total_away_rush_epa",
                "off_pass_epa": "total_away_pass_epa",
                "off_qb_epa": "away_qb_epa",
            }
        )
        games = games.merge(home_off, on="home_abbr", how="left")
        games = games.merge(away_off, on="away_abbr", how="left")

    games = add_differential_features(games)
    games = games.loc[:, ~games.columns.duplicated()]

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

    X_wk = games.reindex(columns=feature_columns)
    for col in X_wk.columns:
        if np.issubdtype(X_wk[col].dtype, np.number) or X_wk[col].dtype == object:
            try:
                X_wk[col] = pd.to_numeric(X_wk[col], errors="coerce")
            except (TypeError, ValueError):
                pass
        fill = train_means[col] if col in train_means.index else 0
        X_wk[col] = X_wk[col].fillna(fill)

    return X_wk, meta
