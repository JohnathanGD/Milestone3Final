"""Grade saved week predictions against final scores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .constants import OUTPUTS_DIR
from .data import load_offense

_PRED_NAME = re.compile(r"predictions_(\d{4})_wk(\d+)\.csv$")


@dataclass
class WeekPickResults:
    season: int
    week: int
    games: pd.DataFrame
    n_graded: int = 0
    n_correct: int = 0
    n_wrong: int = 0
    n_pending: int = 0

    @property
    def accuracy(self) -> float | None:
        if self.n_graded == 0:
            return None
        return self.n_correct / self.n_graded


@dataclass
class PickHistory:
    weeks: list[WeekPickResults] = field(default_factory=list)

    @property
    def n_graded(self) -> int:
        return sum(w.n_graded for w in self.weeks)

    @property
    def n_correct(self) -> int:
        return sum(w.n_correct for w in self.weeks)

    @property
    def n_wrong(self) -> int:
        return sum(w.n_wrong for w in self.weeks)

    @property
    def accuracy(self) -> float | None:
        if self.n_graded == 0:
            return None
        return self.n_correct / self.n_graded


def _prediction_files() -> list[tuple[int, int, Path]]:
    found: list[tuple[int, int, Path]] = []
    for path in sorted(OUTPUTS_DIR.glob("predictions_*_wk*.csv")):
        m = _PRED_NAME.match(path.name)
        if not m:
            continue
        found.append((int(m.group(1)), int(m.group(2)), path))
    return found


def _actual_winner(row: pd.Series) -> str | None:
    if pd.isna(row.get("score_home")) or pd.isna(row.get("score_away")):
        return None
    if row["score_home"] > row["score_away"]:
        return str(row["team_home"])
    if row["score_home"] < row["score_away"]:
        return str(row["team_away"])
    return None  # tie — not graded


def grade_prediction_frame(preds: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Attach actual scores/winners and correct flag to a predictions table."""
    out = preds.copy()
    score_cols = [
        c
        for c in [
            "schedule_season",
            "schedule_week",
            "home_abbr",
            "away_abbr",
            "score_home",
            "score_away",
            "team_home",
            "team_away",
        ]
        if c in scores.columns
    ]
    keyed = scores[score_cols].drop_duplicates(
        subset=["schedule_season", "schedule_week", "home_abbr", "away_abbr"]
    )

    drop_scores = [
        c for c in ["score_home", "score_away", "actual_winner", "correct"] if c in out.columns
    ]
    out = out.drop(columns=drop_scores, errors="ignore")

    merge_keys = ["schedule_season", "schedule_week", "home_abbr", "away_abbr"]
    if not all(k in out.columns for k in merge_keys):
        merge_keys = ["schedule_season", "schedule_week", "team_home", "team_away"]

    available_keys = [k for k in merge_keys if k in out.columns and k in keyed.columns]
    if len(available_keys) < 3:
        out["actual_winner"] = None
        out["correct"] = pd.NA
        return out

    out = out.merge(keyed, on=available_keys, how="left", suffixes=("", "_score"))
    if "team_home" not in out.columns and "team_home_score" in out.columns:
        out["team_home"] = out["team_home_score"]
    if "team_away" not in out.columns and "team_away_score" in out.columns:
        out["team_away"] = out["team_away_score"]

    out["actual_winner"] = out.apply(_actual_winner, axis=1)
    out["correct"] = pd.NA
    has_actual = out["actual_winner"].notna() & out["predicted_winner"].notna()
    out.loc[has_actual, "correct"] = (
        out.loc[has_actual, "predicted_winner"] == out.loc[has_actual, "actual_winner"]
    )
    return out


def load_pick_history(offense: pd.DataFrame | None = None) -> PickHistory:
    """Load all prediction CSVs and grade against offense game logs."""
    scores = offense if offense is not None else load_offense()
    weeks: list[WeekPickResults] = []

    for season, week, path in _prediction_files():
        preds = pd.read_csv(path)
        if "schedule_season" not in preds.columns:
            preds["schedule_season"] = season
        if "schedule_week" not in preds.columns:
            preds["schedule_week"] = week

        graded = grade_prediction_frame(preds, scores)
        correct_mask = graded["correct"] == True  # noqa: E712
        wrong_mask = graded["correct"] == False  # noqa: E712
        pending_mask = graded["correct"].isna()

        weeks.append(
            WeekPickResults(
                season=season,
                week=week,
                games=graded.sort_values(["team_home"]).reset_index(drop=True),
                n_graded=int(correct_mask.sum() + wrong_mask.sum()),
                n_correct=int(correct_mask.sum()),
                n_wrong=int(wrong_mask.sum()),
                n_pending=int(pending_mask.sum()),
            )
        )

    weeks.sort(key=lambda w: (w.season, w.week))
    return PickHistory(weeks=weeks)
