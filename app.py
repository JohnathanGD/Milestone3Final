#!/usr/bin/env python3
"""NFL Game Outcome Predictor — reads results from repro_m2.ipynb."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_predictor.constants import OUTPUTS_DIR
from nfl_predictor.data import data_status
from nfl_predictor.team_logos import logo_url, matchup_display_colors
from nfl_predictor.notebook_results import (
    NotebookResults,
    load_notebook_results,
    notebook_mtime_str,
)

st.set_page_config(
    page_title="NFL Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def get_results() -> NotebookResults:
    return load_notebook_results()


def _team_win_bar(
    home_prob: float,
    home_color: str,
    away_color: str,
    caption: str,
) -> None:
    """Split bar: away share (left) in away color, home share (right) in home color."""
    home_prob = max(0.0, min(1.0, home_prob))
    away_prob = 1.0 - home_prob
    away_pct = away_prob * 100
    home_pct = home_prob * 100
    if away_pct > 0 and home_pct > 0:
        grid_cols = f"{away_pct:.4f}fr {home_pct:.4f}fr"
    else:
        grid_cols = "1fr"

    st.markdown(
        f"""
        <div style="margin:0.35rem 0 0.15rem;">
          <div style="
            width:100%;
            height:18px;
            border-radius:9px;
            padding:3px;
            display:grid;
            grid-template-columns:{grid_cols};
            gap:3px;
            background:#ffffff;
            border:2px solid #cbd5e1;
            box-shadow:0 1px 3px rgba(0,0,0,0.12);
          ">
            <div style="
              background:{away_color if away_pct > 0 else 'transparent'};
              border-radius:6px;
              box-shadow:inset 0 -2px 0 rgba(0,0,0,0.22);
            "></div>
            <div style="
              background:{home_color if home_pct > 0 else 'transparent'};
              border-radius:6px;
              box-shadow:inset 0 -2px 0 rgba(0,0,0,0.22);
            "></div>
          </div>
          <div style="
            display:flex;
            justify-content:space-between;
            font-size:0.85rem;
            margin-top:0.4rem;
            font-weight:600;
          ">
            <span style="color:{away_color};text-shadow:0 0 1px #fff;">{away_pct:.0f}% away</span>
            <span style="color:{home_color};text-shadow:0 0 1px #fff;">{home_pct:.0f}% home</span>
          </div>
          <div style="font-size:0.8rem;color:#4b5563;margin-top:0.2rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _team_logo_column(team_name: str, width: int = 56) -> None:
    url = logo_url(team_name)
    if url:
        st.image(url, width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px;height:{width}px;"
            f"border-radius:8px;background:#e5e7eb;'></div>",
            unsafe_allow_html=True,
        )


def show_figure(results: NotebookResults, key: str, caption: str) -> None:
    if key in results.figure_files:
        st.image(str(results.figure_files[key]), caption=caption, use_container_width=True)
    elif key in results.figures:
        st.image(results.figures[key], caption=caption, use_container_width=True)


def render_predictions(results: NotebookResults) -> None:
    preds = results.predictions
    if preds.empty:
        st.warning(
            "No prediction table found in the notebook output. "
            "Open `repro_m2.ipynb`, run all cells, then refresh this page."
        )
        return

    season = int(preds["schedule_season"].iloc[0]) if "schedule_season" in preds.columns else "—"
    week = int(preds["schedule_week"].iloc[0]) if "schedule_week" in preds.columns else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Season", season)
    c2.metric("Week", week)
    c3.metric("Games", len(preds))

    st.subheader(f"Week {week} predictions")
    st.caption(
        "Home win probability from logistic regression in the notebook "
        "(season-to-date offensive + defensive features)."
    )

    for _, row in preds.iterrows():
        home = row.get("team_home", "Home")
        away = row.get("team_away", "Away")
        prob = float(row["pred_home_win_prob"])
        away_prob = 1.0 - prob
        winner = row.get("predicted_winner", home if prob >= 0.5 else away)
        conf = float(row.get("confidence", max(prob, away_prob)))

        home_color, away_color = matchup_display_colors(home, away)
        with st.container(border=True):
            logo_away, text_away, at_col, logo_home, text_home, prob_col = st.columns(
                [0.5, 1.6, 0.25, 0.5, 1.6, 0.9]
            )
            with logo_away:
                _team_logo_column(away)
            with text_away:
                st.markdown(
                    f"<span style='color:{away_color};font-weight:600;'>{away}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{away_prob:.0%} to win")
            with at_col:
                st.markdown("**@**")
            with logo_home:
                _team_logo_column(home)
            with text_home:
                st.markdown(
                    f"<span style='color:{home_color};font-weight:600;'>{home}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{prob:.0%} to win")
            with prob_col:
                st.markdown(f"### {prob:.0%}")
                st.caption("P(home win)")

            _team_win_bar(
                prob,
                home_color,
                away_color,
                f"Model pick: {winner} ({conf:.0%} confidence)",
            )

    st.download_button(
        "Download predictions (CSV)",
        preds.to_csv(index=False),
        file_name="notebook_week13_predictions.csv",
        mime="text/csv",
    )


def render_metrics(results: NotebookResults) -> None:
    st.subheader("Hold-out test performance")
    st.caption("30% random split — copied from the notebook summary output.")

    if results.model_metrics.empty:
        st.warning("Model metrics table not found in notebook output.")
    else:
        display = results.model_metrics.copy()
        for col in display.columns:
            if col == "Model":
                continue
            display[col] = pd.to_numeric(display[col], errors="coerce")
            display[col] = display[col].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)

    if results.cv_scores:
        st.subheader("Cross-validation")
        for line in results.cv_scores.values():
            st.markdown(f"- {line.strip()}")

    if results.extrapolation:
        st.subheader("Out-of-time test (2019+, voting ensemble)")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(
            "Bet win rate",
            results.extrapolation.get("Model Win Percentage", "—"),
        )
        e2.metric(
            "Bets won",
            results.extrapolation.get("Total Number of Bets Won", "—"),
        )
        e3.metric(
            "Bets placed",
            results.extrapolation.get("Total Number of Bets Made", "—"),
        )
        e4.metric(
            "Games (2019+)",
            results.extrapolation.get("Possible Games (2019+)", "—"),
        )

    st.subheader("Training charts")
    chart_tabs = st.tabs(["Random Forest", "XGBoost", "Confusion matrices"])

    with chart_tabs[0]:
        show_figure(results, "rf_tuning", "RF hyperparameter search (notebook)")
    with chart_tabs[1]:
        show_figure(results, "xgb_tuning", "XGBoost hyperparameter search (notebook)")
    with chart_tabs[2]:
        cm_keys = sorted(
            k
            for k in set(results.figure_files) | set(results.figures)
            if "confusion" in k
        )
        if cm_keys:
            for key in cm_keys:
                show_figure(results, key, key.replace("_", " ").title())
        else:
            st.info(
                "Confusion matrix plots appear after the evaluation cells in the notebook. "
                f"Re-run the notebook, or check the `{OUTPUTS_DIR.name}/` folder if figures were saved."
            )


def render_about(results: NotebookResults) -> None:
    status = data_status()

    st.subheader("Data sources")
    st.markdown(
        f"""
        | File | Status |
        |------|--------|
        | `Data/offensive_team_logs_from_nfl_data_py_1999_2025.csv` | {"Ready" if status["offense"] else "Missing"} |
        | `Data/team_defense_game_logs_1999_2025.csv` | {"Ready" if status["defense"] else "Missing"} |
        | `repro_m2.ipynb` | {"Executed" if results.executed else "Not run yet"} |
        """
    )

    st.subheader("How this app works")
    st.markdown(
        """
        This dashboard **does not re-train models**. It reads the saved output cells from
        **`repro_m2.ipynb`** — the same tables and charts you see after running the notebook.

        To refresh predictions and metrics:

        1. Open `repro_m2.ipynb` in Jupyter.
        2. **Run all cells** (Kernel → Restart & Run All).
        3. Reload this page (or click **Refresh from notebook** in the sidebar).
        """
    )

    if st.button("Refresh from notebook", type="primary"):
        get_results.clear()
        st.rerun()


def main() -> None:
    results = get_results()
    status = data_status()

    with st.sidebar:
        st.title("NFL Predictor")
        page = st.radio(
            "Navigate",
            ["Predictions", "Model results", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        if results.executed:
            st.success("Notebook outputs loaded")
            st.caption(f"Last saved: {notebook_mtime_str(results)}")
        else:
            st.error("Notebook has not been run")
            st.caption("Run all cells in repro_m2.ipynb")

        if st.button("Refresh from notebook", use_container_width=True):
            get_results.clear()
            st.rerun()

    st.title("NFL Game Outcome Predictor")
    st.markdown(
        "Probabilistic home/away win forecasts from the Milestone III pipeline "
        "(defensive efficiency + schedule context, 1999–2025)."
    )

    if not results.executed:
        st.error(
            "No executed output found in `repro_m2.ipynb`. "
            "Run the notebook first, then return here."
        )
        render_about(results)
        return

    if page == "Predictions":
        render_predictions(results)
    elif page == "Model results":
        render_metrics(results)
    else:
        render_about(results)


if __name__ == "__main__":
    main()
