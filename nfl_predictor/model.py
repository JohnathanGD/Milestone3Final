"""Train and predict NFL game outcomes with time-aware evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .constants import (
    EXTRAPOLATION_CUTOFF_SEASON,
    SEED,
    TUNED_LR_C,
    TUNED_RF,
    TUNED_XGB,
)
from .data import load_defense, load_merged_game_table, load_offense
from .features import build_week_features


@dataclass
class TrainedPipeline:
    lr: LogisticRegression
    rf: RandomForestClassifier
    xgb: XGBClassifier
    voting: CalibratedClassifierCV
    scaler: ColumnTransformer
    feature_columns: list[str]
    train_means: pd.Series
    metrics: dict = field(default_factory=dict)
    extrapolation_metrics: dict = field(default_factory=dict)

    def predict_proba_home(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.lr.predict_proba(X_scaled)[:, 1]

    def predict_week(self, season: int, week: int) -> pd.DataFrame:
        off = load_offense()
        defense = load_defense()
        X_wk, meta = build_week_features(
            off,
            defense,
            season,
            week,
            self.feature_columns,
            self.train_means,
        )
        if meta.empty:
            return pd.DataFrame()

        probs = self.predict_proba_home(X_wk)
        out = meta.copy()
        out["pred_home_win_prob"] = probs
        out["predicted_winner"] = np.where(
            probs >= 0.5, out["team_home"], out["team_away"]
        )
        out["confidence"] = np.where(probs >= 0.5, probs, 1 - probs)

        if "score_home" in out.columns and "score_away" in out.columns:
            has_scores = out["score_home"].notna() & out["score_away"].notna()
            out.loc[has_scores, "actual_winner"] = np.where(
                out.loc[has_scores, "score_home"] > out.loc[has_scores, "score_away"],
                out.loc[has_scores, "team_home"],
                out.loc[has_scores, "team_away"],
            )
            out.loc[has_scores, "correct"] = (
                out.loc[has_scores, "predicted_winner"]
                == out.loc[has_scores, "actual_winner"]
            )
        return out.sort_values(["team_home"]).reset_index(drop=True)


def time_based_split(
    meta: pd.DataFrame,
    cutoff_season: int = EXTRAPOLATION_CUTOFF_SEASON,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on seasons < cutoff, test on seasons >= cutoff."""
    train_mask = meta["schedule_season"].values < cutoff_season
    test_mask = ~train_mask
    return train_mask, test_mask


def _evaluate_model(name: str, y_true, y_pred, y_proba) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": roc_auc_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def train_pipeline(
    cutoff_season: int = EXTRAPOLATION_CUTOFF_SEASON,
) -> TrainedPipeline:
    """
    Train on pre-cutoff seasons, report metrics on post-cutoff holdout.
    Uses lagged season-to-date features (no same-game defensive leakage).
    """
    X, meta, y = load_merged_game_table()
    train_mask, test_mask = time_based_split(meta, cutoff_season)

    X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    train_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(C=TUNED_LR_C, max_iter=5000, random_state=SEED)
    rf = RandomForestClassifier(
        n_estimators=TUNED_RF["n_estimators"],
        max_depth=TUNED_RF["max_depth"],
        criterion=TUNED_RF["criterion"],
        random_state=SEED,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        learning_rate=TUNED_XGB["learning_rate"],
        max_depth=TUNED_XGB["max_depth"],
        n_estimators=TUNED_XGB["n_estimators"],
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    vote = VotingClassifier(
        estimators=[("xgb", xgb), ("lr", lr), ("rf", rf)],
        voting="soft",
    )
    voting = CalibratedClassifierCV(vote, method="isotonic", cv=3)
    voting.fit(X_train, y_train)

    metrics = {}
    for name, model, scaled in [
        ("Logistic Regression", lr, True),
        ("Random Forest", rf, False),
        ("XGBoost", xgb, False),
    ]:
        Xt = X_test_scaled if scaled else X_test
        y_pred = model.predict(Xt)
        y_proba = model.predict_proba(Xt)[:, 1]
        metrics[name] = _evaluate_model(name, y_test, y_pred, y_proba)

    y_vote = voting.predict(X_test)
    y_vote_proba = voting.predict_proba(X_test)[:, 1]
    extrapolation = _evaluate_model(
        "Voting (calibrated)", y_test, y_vote, y_vote_proba
    )

    return TrainedPipeline(
        lr=lr,
        rf=rf,
        xgb=xgb,
        voting=voting,
        scaler=scaler,
        feature_columns=X.columns.tolist(),
        train_means=train_means,
        metrics=metrics,
        extrapolation_metrics=extrapolation,
    )
