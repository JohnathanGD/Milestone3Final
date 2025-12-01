#!/usr/bin/env python3

import pandas as pd
import nfl_data_py as nfl

def safe_div(n, d):
    return n / d if d not in (0, None) else 0.0

def load_pbp(start_season=1999, end_season=2025):
    years = list(range(start_season, end_season + 1))
    pbp = nfl.import_pbp_data(years, downcast=True, cache=False)
    return pbp

def build_defense_logs(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    numeric_cols = [
        "yards_gained",
        "passing_yards",
        "rushing_yards",
        "epa",
        "success",
        "sack",
        "interception",
        "fumble_lost",
        "qb_hit",
    ]
    for col in numeric_cols:
        if col in pbp.columns:
            pbp[col] = pbp[col].fillna(0)

    if "passing_yards" not in pbp.columns:
        if "pass" in pbp.columns:
            pbp["passing_yards"] = pbp["yards_gained"].where(pbp["pass"] == 1, 0)
        else:
            pbp["passing_yards"] = 0

    if "rushing_yards" not in pbp.columns:
        if "rush" in pbp.columns:
            pbp["rushing_yards"] = pbp["yards_gained"].where(pbp["rush"] == 1, 0)
        else:
            pbp["rushing_yards"] = 0

    pbp["explosive_pass"] = (
        (pbp.get("pass", 0) == 1) & (pbp["yards_gained"] >= 20)
    )
    pbp["explosive_run"] = (
        (pbp.get("rush", 0) == 1) & (pbp["yards_gained"] >= 10)
    )

    if "qb_hit" in pbp.columns:
        pbp["pressure"] = (pbp["sack"] > 0) | (pbp["qb_hit"] > 0)
    else:
        pbp["qb_hit"] = 0
        pbp["pressure"] = pbp["sack"] > 0

    group_cols = ["game_id", "defteam"]

    agg = pbp.groupby(group_cols).agg(
        plays_faced=("play_id", "count"),
        total_yards_allowed=("yards_gained", "sum"),
        pass_yards_allowed=("passing_yards", "sum"),
        rush_yards_allowed=("rushing_yards", "sum"),
        epa_allowed=("epa", "sum"),
        success_rate_allowed=("success", "mean"),
        sacks=("sack", "sum"),
        interceptions=("interception", "sum"),
        forced_fumbles=("fumble_lost", "sum"),
        fumbles_recovered=("fumble_lost", "sum"),
        explosive_passes_allowed=("explosive_pass", "sum"),
        explosive_runs_allowed=("explosive_run", "sum"),
        qb_hits=("qb_hit", "sum"),
        pressures=("pressure", "sum"),
        dropbacks=("pass", "sum") if "pass" in pbp.columns else ("play_id", "count"),
    ).reset_index()

    agg["yards_per_play_allowed"] = agg.apply(
        lambda r: safe_div(r["total_yards_allowed"], r["plays_faced"]), axis=1
    )
    agg["epa_per_play_allowed"] = agg.apply(
        lambda r: safe_div(r["epa_allowed"], r["plays_faced"]), axis=1
    )
    agg["explosive_plays_allowed"] = (
        agg["explosive_passes_allowed"] + agg["explosive_runs_allowed"]
    )
    agg["pressure_rate"] = agg.apply(
        lambda r: safe_div(r["pressures"], r["dropbacks"]), axis=1
    )

    game_meta_cols = ["game_id", "season", "week", "season_type",
                      "home_team", "away_team"]

    score_cols = []
    for c in ["total_home_score", "total_away_score", "home_score", "away_score"]:
        if c in pbp.columns:
            score_cols.append(c)

    meta = pbp[game_meta_cols + score_cols].drop_duplicates("game_id")

    if "total_home_score" in meta.columns and "total_away_score" in meta.columns:
        meta = meta.rename(
            columns={
                "total_home_score": "home_score_final",
                "total_away_score": "away_score_final",
            }
        )
    elif "home_score" in meta.columns and "away_score" in meta.columns:
        meta = meta.rename(
            columns={
                "home_score": "home_score_final",
                "away_score": "away_score_final",
            }
        )
    else:
        meta["home_score_final"] = 0
        meta["away_score_final"] = 0

    defense = agg.merge(meta, on="game_id", how="left")

    def get_offteam(row):
        if row["defteam"] == row["home_team"]:
            return row["away_team"]
        if row["defteam"] == row["away_team"]:
            return row["home_team"]
        return None

    defense["offteam"] = defense.apply(get_offteam, axis=1)

    def get_points_allowed(row):
        if row["defteam"] == row["home_team"]:
            return row["away_score_final"]
        if row["defteam"] == row["away_team"]:
            return row["home_score_final"]
        return 0

    def get_points_scored(row):
        if row["defteam"] == row["home_team"]:
            return row["home_score_final"]
        if row["defteam"] == row["away_team"]:
            return row["away_score_final"]
        return 0

    defense["points_allowed"] = defense.apply(get_points_allowed, axis=1)
    defense["points_scored_by_team"] = defense.apply(get_points_scored, axis=1)
    defense["result"] = defense["points_scored_by_team"] - defense["points_allowed"]
    defense["is_home"] = (defense["defteam"] == defense["home_team"]).astype(int)

    defense = defense[defense["season_type"] == "REG"].copy()

    cols_order = [
        "game_id",
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
        "plays_faced",
        "total_yards_allowed",
        "pass_yards_allowed",
        "rush_yards_allowed",
        "yards_per_play_allowed",
        "epa_allowed",
        "epa_per_play_allowed",
        "success_rate_allowed",
        "sacks",
        "interceptions",
        "forced_fumbles",
        "fumbles_recovered",
        "explosive_passes_allowed",
        "explosive_runs_allowed",
        "explosive_plays_allowed",
        "qb_hits",
        "pressures",
        "dropbacks",
        "pressure_rate",
    ]

    cols_order = [c for c in cols_order if c in defense.columns]
    defense = defense[cols_order]

    return defense

def main():
    pbp = load_pbp(1999, 2025)
    defense = build_defense_logs(pbp)
    out_path = "Data/team_defense_game_logs_1999_2025.csv"
    defense.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
