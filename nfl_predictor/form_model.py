"""Home baseline + offense/defense form edge predictor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .constants import (
    EXTRAPOLATION_CUTOFF_SEASON,
    FORM_PARAMS_PATH,
    TEAM_ABBR_NORMALIZE,
)
from .features import (
    _as_of_team_means,
    _normalize_game_teams,
    build_offense_team_games,
)

# Higher = better after sign flip for allowed stats.
OFFENSE_FORM_COLS = ["off_epa", "off_pass_epa", "off_qb_epa"]
# User-selected defensive deciding factors (game-log columns).
DEFENSE_FORM_RAW = [
    "total_yards_allowed",
    "success_rate_allowed",
    "forced_fumbles",
    "fumbles_recovered",
    "explosive_runs_allowed",
    "qb_hits",
    "pressure_rate",
    "pressures",
]
# Signed components used in the strength score (after z-scoring).
# Defense signs: allowed/explosive negative; takeaways/pressure positive.
STRENGTH_COMPONENTS = [
    ("off_epa", 1.0),
    ("off_pass_epa", 1.0),
    ("off_qb_epa", 1.0),
    ("total_yards_allowed", -1.0),
    ("success_rate_allowed", -1.0),
    ("forced_fumbles", 1.0),
    ("fumbles_recovered", 1.0),
    ("explosive_runs_allowed", -1.0),
    ("qb_hits", 1.0),
    ("pressure_rate", 1.0),
    ("pressures", 1.0),
]
OFFENSE_COMPONENT_NAMES = {"off_epa", "off_pass_epa", "off_qb_epa"}

# Blend: once the current season has games, weight them vs a multi-year history window.
HIST_LOOKBACK_YEARS = 5
IN_SEASON_WEIGHT = 0.60
HIST_WEIGHT = 0.40


@dataclass
class FormParams:
    p0: float = 0.56
    beta: float = 1.0
    # Offense slightly ahead of defense (55% / 45%).
    offense_weight: float = 0.55
    # Expected home margin ≈ spread_intercept + spread_scale * edge
    spread_intercept: float = 2.5
    spread_scale: float = 4.0
    feature_means: dict[str, float] | None = None
    feature_stds: dict[str, float] | None = None
    train_metrics: dict[str, float] | None = None
    holdout_metrics: dict[str, float] | None = None

    def save(self, path: Path | None = None) -> Path:
        path = path or FORM_PARAMS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> FormParams:
        path = path or FORM_PARAMS_PATH
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _logit(p: float | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _prepare_defense(defense: pd.DataFrame) -> pd.DataFrame:
    d = defense.copy()
    for col in ["defteam", "home_team", "away_team", "offteam"]:
        if col in d.columns:
            d[col] = d[col].replace(TEAM_ABBR_NORMALIZE)
    for col in DEFENSE_FORM_RAW:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def _prepare_offense_long(off: pd.DataFrame) -> pd.DataFrame:
    off = _normalize_game_teams(off)
    long = build_offense_team_games(off)
    for col in OFFENSE_FORM_COLS:
        if col in long.columns:
            long[col] = pd.to_numeric(long[col], errors="coerce")
    # Keep only played games for offense priors.
    return long.dropna(subset=[c for c in OFFENSE_FORM_COLS if c in long.columns], how="all")


def _game_counts(
    long_df: pd.DataFrame,
    season_col: str,
    week_col: str,
    team_col: str,
    season: int,
    before_week: int,
) -> pd.Series:
    prior = long_df[
        (long_df[season_col] == season) & (long_df[week_col] < before_week)
    ]
    if prior.empty:
        return pd.Series(dtype=int)
    return prior.groupby(team_col).size()


def _team_history_means(
    long_df: pd.DataFrame,
    season_col: str,
    team_col: str,
    value_cols: list[str],
    season: int,
    lookback_years: int = HIST_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """Team averages over the previous `lookback_years` completed seasons."""
    if not value_cols or long_df.empty:
        return pd.DataFrame(columns=[team_col] + value_cols)
    lo = season - lookback_years
    hi = season - 1
    hist = long_df[(long_df[season_col] >= lo) & (long_df[season_col] <= hi)]
    if hist.empty:
        return pd.DataFrame(columns=[team_col] + value_cols)
    return hist.groupby(team_col, as_index=False)[value_cols].mean()


def _blended_team_means(
    long_df: pd.DataFrame,
    season_col: str,
    week_col: str,
    team_col: str,
    season: int,
    before_week: int,
    value_cols: list[str],
    teams: list[str],
) -> pd.DataFrame:
    """
    Within-season means before `before_week`, blended with past-5-year history.

    - Week 1 / no games yet (n=0): 100% past 5 seasons
    - Once current-season games exist: 60% in-season + 40% past 5 seasons
    """
    value_cols = [c for c in value_cols if c in long_df.columns]
    within = _as_of_team_means(
        long_df, season_col, week_col, team_col, season, before_week, value_cols
    )
    hist = _team_history_means(long_df, season_col, team_col, value_cols, season)
    counts = _game_counts(long_df, season_col, week_col, team_col, season, before_week)

    within_i = (
        within.set_index(team_col) if not within.empty else pd.DataFrame(columns=value_cols)
    )
    hist_i = hist.set_index(team_col) if not hist.empty else pd.DataFrame(columns=value_cols)

    rows: list[dict] = []
    for team in teams:
        n = float(counts.get(team, 0) or 0)
        w_in = 0.0 if n <= 0 else IN_SEASON_WEIGHT
        w_hist = HIST_WEIGHT if n > 0 else 1.0
        # If somehow both missing one side, combine_first logic below still applies.

        row: dict = {team_col: team}
        for c in value_cols:
            wv = (
                within_i.at[team, c]
                if team in within_i.index and c in within_i.columns
                else np.nan
            )
            hv = (
                hist_i.at[team, c]
                if team in hist_i.index and c in hist_i.columns
                else np.nan
            )
            if np.isnan(wv) and np.isnan(hv):
                row[c] = np.nan
            elif np.isnan(wv):
                row[c] = float(hv)
            elif np.isnan(hv):
                row[c] = float(wv)
            else:
                row[c] = float(w_in * wv + w_hist * hv)
        rows.append(row)
    return pd.DataFrame(rows)


def team_priors(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    season: int,
    week: int,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Pre-game offense/defense form priors for teams as of (season, week)."""
    off_long = _prepare_offense_long(off)
    defense = _prepare_defense(defense)

    if teams is None:
        hist_lo = season - HIST_LOOKBACK_YEARS
        teams = sorted(
            set(off_long.loc[off_long["schedule_season"] == season, "team_abbr"].dropna())
            | set(defense.loc[defense["season"] == season, "defteam"].dropna())
            | set(
                off_long.loc[
                    (off_long["schedule_season"] >= hist_lo)
                    & (off_long["schedule_season"] < season),
                    "team_abbr",
                ].dropna()
            )
            | set(
                defense.loc[
                    (defense["season"] >= hist_lo) & (defense["season"] < season),
                    "defteam",
                ].dropna()
            )
        )

    off_cols = [c for c in OFFENSE_FORM_COLS if c in off_long.columns]
    def_cols = [c for c in DEFENSE_FORM_RAW if c in defense.columns]

    off_priors = _blended_team_means(
        off_long,
        "schedule_season",
        "schedule_week",
        "team_abbr",
        season,
        week,
        off_cols,
        teams,
    )
    def_priors = _blended_team_means(
        defense,
        "season",
        "week",
        "defteam",
        season,
        week,
        def_cols,
        teams,
    ).rename(columns={"defteam": "team_abbr"})

    return off_priors.merge(def_priors, on="team_abbr", how="outer")


def _component_matrix(priors: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for name, _sign in STRENGTH_COMPONENTS:
        if name in priors.columns:
            cols[name] = pd.to_numeric(priors[name], errors="coerce")
        else:
            cols[name] = np.nan
    return pd.DataFrame(cols, index=priors.index)


def fit_feature_scaler(priors_frames: list[pd.DataFrame]) -> tuple[dict[str, float], dict[str, float]]:
    if not priors_frames:
        names = [n for n, _ in STRENGTH_COMPONENTS]
        return {n: 0.0 for n in names}, {n: 1.0 for n in names}
    mat = pd.concat([_component_matrix(p) for p in priors_frames], ignore_index=True)
    means = mat.mean(numeric_only=True).to_dict()
    stds = mat.std(numeric_only=True).replace(0, 1.0).fillna(1.0).to_dict()
    for n, _ in STRENGTH_COMPONENTS:
        means.setdefault(n, 0.0)
        stds.setdefault(n, 1.0)
        if stds[n] == 0 or np.isnan(stds[n]):
            stds[n] = 1.0
    return means, stds


def team_strength(
    priors: pd.DataFrame,
    params: FormParams,
) -> pd.Series:
    """One strength score per team row in priors (aligned index)."""
    means = params.feature_means or {}
    stds = params.feature_stds or {}
    off_w = float(params.offense_weight)
    def_w = 1.0 - off_w

    off_names = OFFENSE_COMPONENT_NAMES
    scores = np.zeros(len(priors), dtype=float)
    off_parts: list[np.ndarray] = []
    def_parts: list[np.ndarray] = []

    for name, sign in STRENGTH_COMPONENTS:
        raw = pd.to_numeric(priors.get(name), errors="coerce").to_numpy(dtype=float)
        mu = means.get(name, np.nanmean(raw) if np.isfinite(raw).any() else 0.0)
        sd = stds.get(name, 1.0) or 1.0
        z = (raw - mu) / sd
        z = np.nan_to_num(z, nan=0.0)
        contrib = sign * z
        if name in off_names:
            off_parts.append(contrib)
        else:
            def_parts.append(contrib)

    if off_parts:
        scores += off_w * np.mean(np.vstack(off_parts), axis=0)
    if def_parts:
        # Defense block is averaged over the selected indicators, then scaled by def_w.
        scores += def_w * np.mean(np.vstack(def_parts), axis=0)
    return pd.Series(scores, index=priors.index)


def edge_to_home_prob(edge: np.ndarray | float, params: FormParams) -> np.ndarray:
    logit = _logit(params.p0) + params.beta * np.asarray(edge, dtype=float)
    return np.clip(_sigmoid(logit), 0.05, 0.95)


def edge_to_margin(edge: np.ndarray | float, params: FormParams) -> np.ndarray:
    """Expected home − away point margin from form edge."""
    return params.spread_intercept + params.spread_scale * np.asarray(edge, dtype=float)


def margin_to_home_spread(margin: np.ndarray | float) -> np.ndarray:
    """Home betting line (negative = home favored), rounded to half-points."""
    spread = -np.asarray(margin, dtype=float)
    return np.round(spread * 2) / 2


def format_home_spread(spread: float, home: str, away: str) -> str:
    """Human label like 'Buffalo Bills -3.5' or 'Detroit Lions +3.5'."""
    if abs(spread) < 0.25:
        return "PICK"
    if spread < 0:
        return f"{home} {spread:.1f}"
    return f"{away} {-spread:.1f}"


def predict_week_form(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    season: int,
    week: int,
    params: FormParams | None = None,
) -> pd.DataFrame:
    params = params or FormParams.load()
    off = _normalize_game_teams(off)
    wk = off[(off["schedule_season"] == season) & (off["schedule_week"] == week)].copy()
    if wk.empty:
        return pd.DataFrame()

    teams = sorted(
        set(wk["home_abbr"].dropna().tolist()) | set(wk["away_abbr"].dropna().tolist())
    )
    priors = team_priors(off, defense, season, week, teams=teams)
    priors = priors.set_index("team_abbr")
    strength = team_strength(priors.reset_index(), params)
    strength.index = priors.index

    out = wk[
        [
            c
            for c in [
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
            if c in wk.columns
        ]
    ].copy()

    out["home_strength"] = out["home_abbr"].map(strength)
    out["away_strength"] = out["away_abbr"].map(strength)
    out["home_strength"] = out["home_strength"].fillna(0.0)
    out["away_strength"] = out["away_strength"].fillna(0.0)
    out["edge"] = out["home_strength"] - out["away_strength"]
    out["pred_home_win_prob"] = edge_to_home_prob(out["edge"].to_numpy(), params)
    out["pred_away_win_prob"] = 1.0 - out["pred_home_win_prob"]
    out["pred_margin"] = edge_to_margin(out["edge"].to_numpy(), params)
    out["predicted_spread"] = margin_to_home_spread(out["pred_margin"].to_numpy())
    out["spread_label"] = [
        format_home_spread(float(s), h, a)
        for s, h, a in zip(out["predicted_spread"], out["team_home"], out["team_away"])
    ]
    out["predicted_winner"] = np.where(
        out["pred_home_win_prob"] >= 0.5, out["team_home"], out["team_away"]
    )
    out["confidence"] = np.where(
        out["pred_home_win_prob"] >= 0.5,
        out["pred_home_win_prob"],
        1 - out["pred_home_win_prob"],
    )

    if "score_home" in out.columns and "score_away" in out.columns:
        has_scores = out["score_home"].notna() & out["score_away"].notna()
        out.loc[has_scores, "actual_winner"] = np.where(
            out.loc[has_scores, "score_home"] > out.loc[has_scores, "score_away"],
            out.loc[has_scores, "team_home"],
            out.loc[has_scores, "team_away"],
        )
        out.loc[has_scores, "correct"] = (
            out.loc[has_scores, "predicted_winner"] == out.loc[has_scores, "actual_winner"]
        )

    return out.sort_values(["team_home"]).reset_index(drop=True)


def _schedule_games(off: pd.DataFrame) -> pd.DataFrame:
    off = _normalize_game_teams(off)
    games = off.copy()
    games["result"] = np.where(
        games["score_home"] > games["score_away"],
        1,
        np.where(games["score_home"] < games["score_away"], 0, np.nan),
    )
    games = games.dropna(subset=["result"]).copy()
    games["result"] = games["result"].astype(int)
    if "schedule_playoff" in games.columns:
        games = games[games["schedule_playoff"].astype(bool) == False]  # noqa: E712
    return games


def build_historical_edges(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    params: FormParams,
) -> pd.DataFrame:
    """Pre-game edges + labels for all scored regular-season games."""
    games = _schedule_games(off)
    rows: list[dict] = []
    for (season, week), slate in games.groupby(["schedule_season", "schedule_week"], sort=True):
        season_i, week_i = int(season), int(week)
        teams = sorted(
            set(slate["home_abbr"].dropna()) | set(slate["away_abbr"].dropna())
        )
        priors = team_priors(off, defense, season_i, week_i, teams=teams)
        if priors.empty:
            continue
        strength = team_strength(priors, params)
        strength.index = priors["team_abbr"].values
        for _, g in slate.iterrows():
            hs = float(strength.get(g["home_abbr"], 0.0) or 0.0)
            aws = float(strength.get(g["away_abbr"], 0.0) or 0.0)
            rows.append(
                {
                    "schedule_season": season_i,
                    "schedule_week": week_i,
                    "home_abbr": g["home_abbr"],
                    "away_abbr": g["away_abbr"],
                    "home_strength": hs,
                    "away_strength": aws,
                    "edge": hs - aws,
                    "result": int(g["result"]),
                    "margin": float(g["score_home"] - g["score_away"]),
                }
            )
    return pd.DataFrame(rows)


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-6, 1 - 1e-6))),
        "away_fav_rate": float((probs < 0.5).mean()),
    }


def fit_form_model(
    off: pd.DataFrame,
    defense: pd.DataFrame,
    cutoff_season: int = EXTRAPOLATION_CUTOFF_SEASON,
    offense_weight: float = 0.55,
) -> FormParams:
    """
    Fit home baseline p0 and edge scale beta on pre-cutoff seasons;
    evaluate on cutoff+ holdout. Saves params to outputs/form_model_params.json.
    """
    # First pass: gather week-1 / early priors to fit z-score scaler on train seasons.
    games = _schedule_games(off)
    train_games = games[games["schedule_season"] < cutoff_season]
    scaler_frames: list[pd.DataFrame] = []
    for (season, week), slate in train_games.groupby(
        ["schedule_season", "schedule_week"], sort=True
    ):
        teams = sorted(set(slate["home_abbr"]) | set(slate["away_abbr"]))
        scaler_frames.append(team_priors(off, defense, int(season), int(week), teams))

    means, stds = fit_feature_scaler(scaler_frames)
    draft = FormParams(
        p0=float(train_games["result"].mean()) if len(train_games) else 0.56,
        beta=1.0,
        offense_weight=offense_weight,
        feature_means=means,
        feature_stds=stds,
    )

    hist = build_historical_edges(off, defense, draft)
    if hist.empty:
        draft.save()
        return draft

    train = hist[hist["schedule_season"] < cutoff_season]
    test = hist[hist["schedule_season"] >= cutoff_season]
    p0 = float(train["result"].mean()) if len(train) else 0.56

    best_beta = 1.0
    best_loss = float("inf")
    for beta in np.linspace(0.1, 3.0, 30):
        trial = FormParams(
            p0=p0,
            beta=float(beta),
            offense_weight=offense_weight,
            feature_means=means,
            feature_stds=stds,
        )
        probs = edge_to_home_prob(train["edge"].to_numpy(), trial)
        loss = log_loss(train["result"], np.clip(probs, 1e-6, 1 - 1e-6))
        if loss < best_loss:
            best_loss = loss
            best_beta = float(beta)

    # Fit expected home margin from form edge (for predicted spreads).
    spread_intercept, spread_scale = 2.5, 4.0
    if "margin" in train.columns and len(train) >= 10:
        edge = train["edge"].to_numpy(dtype=float)
        margin = train["margin"].to_numpy(dtype=float)
        mask = np.isfinite(edge) & np.isfinite(margin)
        if mask.sum() >= 10:
            x = np.column_stack([np.ones(mask.sum()), edge[mask]])
            coef, _, _, _ = np.linalg.lstsq(x, margin[mask], rcond=None)
            spread_intercept = float(coef[0])
            spread_scale = float(coef[1])

    params = FormParams(
        p0=p0,
        beta=best_beta,
        offense_weight=offense_weight,
        spread_intercept=spread_intercept,
        spread_scale=spread_scale,
        feature_means=means,
        feature_stds=stds,
    )
    train_probs = edge_to_home_prob(train["edge"].to_numpy(), params)
    params.train_metrics = _metrics(train["result"].to_numpy(), train_probs)
    if len(test):
        test_probs = edge_to_home_prob(test["edge"].to_numpy(), params)
        params.holdout_metrics = _metrics(test["result"].to_numpy(), test_probs)
    params.save()
    return params
