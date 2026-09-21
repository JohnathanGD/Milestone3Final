#!/usr/bin/env python3
"""Train models and publish week predictions (baseline + form by default)."""

from __future__ import annotations

import argparse

from nfl_predictor.constants import (
    DEFAULT_PREDICT_SEASON,
    DEFAULT_PREDICT_WEEK,
    EXTRAPOLATION_CUTOFF_SEASON,
    OUTPUTS_DIR,
)
from nfl_predictor.data import load_defense, load_offense
from nfl_predictor.form_model import fit_form_model, predict_week_form
from nfl_predictor.model import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NFL predictor; week picks use baseline+form by default"
    )
    parser.add_argument(
        "--cutoff-season",
        type=int,
        default=EXTRAPOLATION_CUTOFF_SEASON,
        help="Test seasons are >= this year (default: 2019)",
    )
    parser.add_argument(
        "--predict-week",
        nargs=2,
        type=int,
        metavar=("SEASON", "WEEK"),
        default=[DEFAULT_PREDICT_SEASON, DEFAULT_PREDICT_WEEK],
        help=(
            "After training, print predictions for this week "
            f"(default: {DEFAULT_PREDICT_SEASON} {DEFAULT_PREDICT_WEEK})"
        ),
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Also train ML models and print holdout metrics; use ML for week preds",
    )
    parser.add_argument(
        "--offense-weight",
        type=float,
        default=0.55,
        help="Form-model weight on offense vs defense (default: 0.55 → 45% defense)",
    )
    args = parser.parse_args()

    print("Fitting baseline + form model (prior season for week 1; in-season after)...")
    off = load_offense()
    defense = load_defense()
    form_params = fit_form_model(
        off,
        defense,
        cutoff_season=args.cutoff_season,
        offense_weight=args.offense_weight,
    )
    print(
        f"Form params: p0={form_params.p0:.3f}  beta={form_params.beta:.3f}  "
        f"offense_weight={form_params.offense_weight:.2f}"
    )
    if form_params.train_metrics:
        tm = form_params.train_metrics
        print(
            f"Form train (<{args.cutoff_season}): "
            f"acc={tm['accuracy']:.3f}  auroc={tm['auroc']:.3f}  "
            f"brier={tm['brier']:.3f}  away_fav={tm['away_fav_rate']:.1%}"
        )
    if form_params.holdout_metrics:
        hm = form_params.holdout_metrics
        print(
            f"Form holdout (>={args.cutoff_season}): "
            f"acc={hm['accuracy']:.3f}  auroc={hm['auroc']:.3f}  "
            f"brier={hm['brier']:.3f}  away_fav={hm['away_fav_rate']:.1%}"
        )
    print(f"Saved: {OUTPUTS_DIR / 'form_model_params.json'}")

    pipe = None
    if args.ml:
        print("\nBuilding lagged ML feature table...")
        pipe = train_pipeline(cutoff_season=args.cutoff_season)
        print(f"\nML features: {len(pipe.feature_columns)} columns")
        print(
            f"Train seasons: < {args.cutoff_season}  |  "
            f"Test seasons: >= {args.cutoff_season}"
        )
        print("\n--- ML holdout metrics ---")
        for name, m in pipe.metrics.items():
            print(
                f"{name:22s}  acc={m['accuracy']:.3f}  "
                f"f1={m['f1']:.3f}  auroc={m['auroc']:.3f}  brier={m['brier']:.3f}"
            )
        em = pipe.extrapolation_metrics
        print(
            f"\nVoting (calibrated)     acc={em['accuracy']:.3f}  "
            f"f1={em['f1']:.3f}  auroc={em['auroc']:.3f}  brier={em['brier']:.3f}"
        )

    if args.predict_week:
        season, week = args.predict_week
        if args.ml and pipe is not None:
            preds = pipe.predict_week(season, week)
            label = "logistic regression"
        else:
            preds = predict_week_form(off, defense, season, week, form_params)
            label = "baseline + form"
        if preds.empty:
            print(f"\nNo games for {season} week {week}.")
        else:
            print(f"\n--- Week {week}, {season} predictions ({label}) ---")
            cols = [
                c
                for c in [
                    "team_home",
                    "team_away",
                    "pred_away_win_prob",
                    "pred_home_win_prob",
                    "predicted_spread",
                    "spread_label",
                    "predicted_winner",
                    "confidence",
                    "edge",
                    "home_strength",
                    "away_strength",
                ]
                if c in preds.columns
            ]
            print(preds[cols].to_string(index=False))
            away_n = int((preds["pred_home_win_prob"] < 0.5).sum())
            print(f"\nAway favorites: {away_n} / {len(preds)}")
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUTS_DIR / f"predictions_{season}_wk{week}.csv"
            preds.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
