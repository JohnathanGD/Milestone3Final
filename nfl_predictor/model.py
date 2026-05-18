"""Train and predict NFL game outcomes."""

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
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .constants import (
    SEED,
    TUNED_LR_C,
    TUNED_RF,
    TUNED_XGB,
)
from .data import load_merged_game_table, load_offense, load_defense
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

    def predict_proba_home(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.lr.predict_proba(X_scaled)[:, 1]

    def predict_week(
        self, season: int, week: int
    ) -> pd.DataFrame:
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


def train_pipeline(test_size: float = 0.3) -> TrainedPipeline:
    X, _meta, y = load_merged_game_table()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=SEED
    )

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
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
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auroc": roc_auc_score(y_test, y_proba),
        }

    return TrainedPipeline(
        lr=lr,
        rf=rf,
        xgb=xgb,
        voting=voting,
        scaler=scaler,
        feature_columns=X.columns.tolist(),
        train_means=X_train.mean(),
        metrics=metrics,
    )
