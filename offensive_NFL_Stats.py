#!/usr/bin/env python3

import pandas as pd
import nfl_data_py as nfl

OUTPUT_PATH = "Data/offensive_team_logs_from_nfl_data_py_1999_2025.csv"

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
    "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",
    "LV":  "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF":  "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "STL": "St. Louis Rams",
    "SD":  "San Diego Chargers",
    "OAK": "Oakland Raiders",
}

def main():
    # Seasons to pull
    years = list(range(1999, 2026))
    sched = nfl.import_schedules(years)
    pbp = nfl.import_pbp_data(years)

    game_type_col = "game_type" if "game_type" in sched.columns else "season_type"

    off_epa = (
        pbp.groupby(["game_id", "posteam"])["epa"]
           .sum()
           .reset_index()
           .rename(columns={"epa": "total_epa"})
    )

    # Rushing EPA 
    rush_epa = (
        pbp[pbp["rush"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .sum()
        .reset_index()
        .rename(columns={"epa": "rush_epa"})
    )

    # Passing EPA 
    pass_epa = (
        pbp[pbp["pass"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .sum()
        .reset_index()
        .rename(columns={"epa": "pass_epa"})
    )

    # QB EPA 
    qb_epa = (
        pbp.groupby(["game_id", "posteam"])["qb_epa"]
           .sum()
           .reset_index()
           .rename(columns={"qb_epa": "qb_epa_total"})
    )

    off_stats = (
        off_epa
        .merge(rush_epa, on=["game_id", "posteam"], how="left")
        .merge(pass_epa, on=["game_id", "posteam"], how="left")
        .merge(qb_epa,  on=["game_id", "posteam"], how="left")
    )


    # Sorting play by play data so first/last are meaningful
    pbp_sorted = pbp.sort_values(["game_id", "play_id"])

    # Pre-game WP
    pre_wp = (
        pbp_sorted.groupby("game_id")[["home_wp", "away_wp"]]
        .first()
        .reset_index()
    )

    # Post-game WP
    post_wp = (
        pbp_sorted.groupby("game_id")[["home_wp_post", "away_wp_post"]]
        .last()
        .reset_index()
    )

    wp_stats = pre_wp.merge(post_wp, on="game_id", how="left")


    df = pd.DataFrame()

    # Basic identifiers
    df["schedule_season"] = sched["season"]
    df["schedule_week"] = sched["week"]
    df["schedule_date"] = sched["gameday"]

    # Playoff flag 
    df["schedule_playoff"] = sched[game_type_col] != "REG"

    # Game id
    df["game_id"] = sched["game_id"]

    # Team abbreviations from nfl_data_py
    df["home_abbr"] = sched["home_team"]
    df["away_abbr"] = sched["away_team"]

    # Full team names (similar to spreadspoke_scores)
    df["team_home"] = df["home_abbr"].map(TEAM_FULL)
    df["team_away"] = df["away_abbr"].map(TEAM_FULL)

    # Final scores
    df["score_home"] = sched["home_score"]
    df["score_away"] = sched["away_score"]

    # Weather: temp + weather string, if available
    if "temp" in sched.columns:
        df["temp"] = sched["temp"]
    else:
        df["temp"] = None

    if "weather" in sched.columns:
        df["weather"] = sched["weather"]
    else:
        df["weather"] = None

    # Home offensive stats
    home_off = off_stats.rename(
        columns={
            "posteam": "home_abbr",
            "total_epa": "total_home_epa",
            "rush_epa": "total_home_rush_epa",
            "pass_epa": "total_home_pass_epa",
            "qb_epa_total": "home_qb_epa",
        }
    )

    df = df.merge(
        home_off,
        on=["game_id", "home_abbr"],
        how="left",
    )

    # Away offensive stats
    away_off = off_stats.rename(
        columns={
            "posteam": "away_abbr",
            "total_epa": "total_away_epa",
            "rush_epa": "total_away_rush_epa",
            "pass_epa": "total_away_pass_epa",
            "qb_epa_total": "away_qb_epa",
        }
    )

    df = df.merge(
        away_off,
        on=["game_id", "away_abbr"],
        how="left",
    )

    df = df.merge(wp_stats, on="game_id", how="left")
    # Columns added: home_wp, away_wp, home_wp_post, away_wp_post

    bet_cols = ["game_id", "spread_line", "total_line"]

    # Get one row per game with its betting lines
    betting = (
        pbp[bet_cols]
        .dropna(subset=["spread_line", "total_line"], how="all")
        .drop_duplicates(subset=["game_id"])                      
    )

    # Merge onto main df
    df = df.merge(betting, on="game_id", how="left")

    df["spread_favorite"] = df["spread_line"]      
    df["over_under_line"] = df["total_line"]       

    df["has_betting_line"] = df["spread_favorite"].notna().astype(int)

    def outcome(row):
        if pd.isna(row["score_home"]) or pd.isna(row["score_away"]):
            return "unknown"
        if row["score_home"] > row["score_away"]:
            return "home_win"
        elif row["score_home"] < row["score_away"]:
            return "away_win"
        else:
            return "tie"

    df["game_result"] = df.apply(outcome, axis=1)

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

