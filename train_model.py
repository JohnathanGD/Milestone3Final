#!/usr/bin/env python3
"""Train lagged-feature models and print time-based holdout metrics."""

from __future__ import annotations

import argparse

import pandas as pd

from nfl_predictor.constants import EXTRAPOLATION_CUTOFF_SEASON, OUTPUTS_DIR
from nfl_predictor.model import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL predictor (lagged features)")
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
        help="After training, print predictions for this week",
    )
    args = parser.parse_args()

    print("Building lagged feature table (season-to-date through prior week)...")
    pipe = train_pipeline(cutoff_season=args.cutoff_season)

    print(f"\nFeatures: {len(pipe.feature_columns)} columns")
    print(f"Train seasons: < {args.cutoff_season}  |  Test seasons: >= {args.cutoff_season}")
    print("\n--- Holdout metrics (2019+ style) ---")
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
        preds = pipe.predict_week(season, week)
        if preds.empty:
            print(f"\nNo games for {season} week {week}.")
        else:
            print(f"\n--- Week {week}, {season} predictions (logistic regression) ---")
            cols = [
                "team_home",
                "team_away",
                "pred_home_win_prob",
                "predicted_winner",
                "confidence",
            ]
            print(preds[cols].to_string(index=False))
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUTS_DIR / f"predictions_{season}_wk{week}.csv"
            preds.to_csv(out_path, index=False)
            print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
