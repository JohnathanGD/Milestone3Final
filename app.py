#!/usr/bin/env python3
"""NFL Game Outcome Predictor — reads results from repro_m2.ipynb."""

from __future__ import annotations

import base64

import pandas as pd
import streamlit as st

from pathlib import Path

from nfl_predictor.constants import DEFENSE_PATH, MILESTONE_PDF_PATH, OFFENSE_PATH, OUTPUTS_DIR
from nfl_predictor.data import data_status
from nfl_predictor.pick_results import PickHistory, load_pick_history
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
def get_results(_cache_bust: str = "") -> NotebookResults:
    return load_notebook_results()


@st.cache_data(show_spinner=False)
def get_pick_history(_cache_bust: str = "") -> PickHistory:
    return load_pick_history()


def _predictions_cache_key() -> str:
    """Bust Streamlit cache when prediction CSVs change."""
    parts: list[str] = []
    for path in sorted(OUTPUTS_DIR.glob("predictions_*.csv")):
        parts.append(f"{path.name}:{path.stat().st_mtime_ns}")
    params = OUTPUTS_DIR / "form_model_params.json"
    if params.is_file():
        parts.append(f"params:{params.stat().st_mtime_ns}")
    return "|".join(parts)

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
        "Win probabilities and predicted home spread from the baseline + form model "
        "(in-season form primary; prior season secondary)."
    )
    if "predicted_spread" not in preds.columns:
        st.warning(
            "This prediction file has no spreads yet. Re-run: "
            "`python train_model.py --predict-week 2026 2`"
        )

    for _, row in preds.iterrows():
        home = row.get("team_home", "Home")
        away = row.get("team_away", "Away")
        prob = float(row["pred_home_win_prob"])
        away_prob = float(row["pred_away_win_prob"]) if "pred_away_win_prob" in row.index and pd.notna(row.get("pred_away_win_prob")) else 1.0 - prob
        winner = row.get("predicted_winner", home if prob >= 0.5 else away)
        conf = float(row.get("confidence", max(prob, away_prob)))
        spread = row.get("predicted_spread")
        spread_label = row.get("spread_label")
        if pd.isna(spread_label) or not spread_label:
            if pd.notna(spread):
                from nfl_predictor.form_model import format_home_spread

                spread_label = format_home_spread(float(spread), home, away)
            else:
                spread_label = "—"

        home_color, away_color = matchup_display_colors(home, away)
        with st.container(border=True):
            logo_away, text_away, at_col, logo_home, text_home, meta_col = st.columns(
                [0.5, 1.6, 0.25, 0.5, 1.6, 1.1]
            )
            with logo_away:
                _team_logo_column(away)
            with text_away:
                st.markdown(
                    f"<span style='color:{away_color};font-weight:700;font-size:1.05rem;'>{away}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:1.35rem;font-weight:700;color:{away_color};'>{away_prob:.0%}</div>",
                    unsafe_allow_html=True,
                )
            with at_col:
                st.markdown("**@**")
            with logo_home:
                _team_logo_column(home)
            with text_home:
                st.markdown(
                    f"<span style='color:{home_color};font-weight:700;font-size:1.05rem;'>{home}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:1.35rem;font-weight:700;color:{home_color};'>{prob:.0%}</div>",
                    unsafe_allow_html=True,
                )
            with meta_col:
                st.markdown(
                    f"<div style='font-size:1.15rem;font-weight:800;'>{spread_label}</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Predicted spread")
                if pd.notna(spread):
                    st.caption(f"Home line: {float(spread):+.1f}")

            st.markdown(
                f"<div style='background:#111827;color:#f9fafb;border-radius:8px;"
                f"padding:0.55rem 0.85rem;margin:0.35rem 0 0.5rem 0;"
                f"display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;"
                f"font-weight:700;'>"
                f"<span>{away}: {away_prob:.0%}</span>"
                f"<span>Spread: {spread_label}</span>"
                f"<span>{home}: {prob:.0%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            _team_win_bar(
                prob,
                home_color,
                away_color,
                f"Pick: {winner} ({conf:.0%})",
            )

    st.download_button(
        "Download predictions (CSV)",
        preds.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )


def render_pick_results(history: PickHistory) -> None:
    st.subheader("Weekly pick results")
    st.caption(
        "Graded against final scores in the offense game logs. "
        "Green = correct pick, red = wrong, gray = not played yet."
    )

    if not history.weeks:
        st.warning(
            "No prediction files found under `outputs/predictions_*_wk*.csv`. "
            "Run `python train_model.py --predict-week YYYY W` first."
        )
        return

    # Prefer 2026 form weeks for the default view; still list all files.
    season_options = sorted({w.season for w in history.weeks}, reverse=True)
    season = st.selectbox("Season", season_options, index=0)
    season_weeks = [w for w in history.weeks if w.season == season]
    if not season_weeks:
        st.info("No weeks for that season.")
        return

    graded_season = [w for w in season_weeks if w.n_graded]
    total_correct = sum(w.n_correct for w in graded_season)
    total_wrong = sum(w.n_wrong for w in graded_season)
    total_graded = total_correct + total_wrong
    total_pending = sum(w.n_pending for w in season_weeks)

    m1, m2, m3, m4 = st.columns(4)
    if total_graded:
        m1.metric("Correct", f"{total_correct} ({total_correct / total_graded:.0%})")
        m2.metric("Wrong", f"{total_wrong} ({total_wrong / total_graded:.0%})")
        m3.metric("Accuracy", f"{total_correct / total_graded:.1%}")
    else:
        m1.metric("Correct", "—")
        m2.metric("Wrong", "—")
        m3.metric("Accuracy", "—")
    m4.metric("Pending games", total_pending)

    week_labels = {
        f"Week {w.week}"
        + (
            f" ({w.n_correct}/{w.n_graded})"
            if w.n_graded
            else " (pending)"
        ): w
        for w in season_weeks
    }
    tabs = st.tabs(list(week_labels.keys()))
    for tab, week_res in zip(tabs, week_labels.values()):
        with tab:
            if week_res.n_graded:
                st.markdown(
                    f"**{week_res.n_correct} correct · {week_res.n_wrong} wrong · "
                    f"{week_res.accuracy:.0%} accuracy**"
                    + (
                        f" · {week_res.n_pending} pending"
                        if week_res.n_pending
                        else ""
                    )
                )
            else:
                st.info("No final scores yet for this week — picks are pending.")

            for _, row in week_res.games.iterrows():
                home = row.get("team_home", "Home")
                away = row.get("team_away", "Away")
                pick = row.get("predicted_winner", "—")
                actual = row.get("actual_winner")
                prob = row.get("pred_home_win_prob")
                correct = row.get("correct")

                if pd.isna(correct):
                    bg, border, status = "#f3f4f6", "#d1d5db", "Pending"
                elif bool(correct):
                    bg, border, status = "#dcfce7", "#16a34a", "Correct"
                else:
                    bg, border, status = "#fee2e2", "#dc2626", "Wrong"

                score_txt = ""
                if pd.notna(row.get("score_home")) and pd.notna(row.get("score_away")):
                    score_txt = (
                        f"{int(row['score_away'])}–{int(row['score_home'])} final"
                    )

                prob_txt = f"{float(prob):.0%} home" if pd.notna(prob) else "—"
                actual_txt = actual if pd.notna(actual) else "—"

                st.markdown(
                    f"""
                    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                                padding:0.85rem 1rem;margin-bottom:0.6rem;">
                      <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
                        <div>
                          <div style="font-weight:700;font-size:1.05rem;">{away} @ {home}</div>
                          <div style="opacity:0.85;margin-top:0.2rem;">
                            Pick: <strong>{pick}</strong> · Model: {prob_txt}
                            {" · " + score_txt if score_txt else ""}
                          </div>
                          <div style="opacity:0.85;">Actual winner: <strong>{actual_txt}</strong></div>
                        </div>
                        <div style="font-weight:700;color:{border};align-self:center;">{status}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_metrics(results: NotebookResults) -> None:
    st.subheader("Hold-out test performance")
    st.caption("Seasons before 2019 = train, 2019+ = test — from notebook output when available.")

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


@st.cache_data(show_spinner=False)
def _milestone_pdf_bytes() -> bytes | None:
    if not MILESTONE_PDF_PATH.is_file():
        return None
    return MILESTONE_PDF_PATH.read_bytes()


def _display_pdf(pdf_bytes: bytes, height: int = 900) -> None:
    """Show PDF in-browser (pdf_viewer if available, else embedded iframe)."""
    if hasattr(st, "pdf_viewer"):
        st.pdf_viewer(pdf_bytes, height=height)
        return
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="{height}px" style="border:1px solid #cbd5e1;'
        f'border-radius:8px;"></iframe>',
        unsafe_allow_html=True,
    )


def render_report() -> None:
    st.subheader("Milestone III report")
    st.caption("Full project write-up (methods, results, and discussion).")

    pdf_bytes = _milestone_pdf_bytes()
    if pdf_bytes is None:
        st.warning(
            f"`{MILESTONE_PDF_PATH.name}` was not found at the project root. "
            "Add the PDF next to `app.py` to enable this page."
        )
        return

    st.download_button(
        "Download milestone3.pdf",
        data=pdf_bytes,
        file_name=MILESTONE_PDF_PATH.name,
        mime="application/pdf",
        use_container_width=True,
    )
    _display_pdf(pdf_bytes, height=900)


def render_about(results: NotebookResults) -> None:
    status = data_status()

    st.subheader("Data sources")
    st.markdown(
        f"""
        | File | Status |
        |------|--------|
        | `{OFFENSE_PATH.relative_to(Path(__file__).resolve().parent)}` | {"Ready" if status["offense"] else "Missing"} |
        | `{DEFENSE_PATH.relative_to(Path(__file__).resolve().parent)}` | {"Ready" if status["defense"] else "Missing"} |
        | `repro_m2.ipynb` | {"Executed" if results.executed else "Not run yet"} |
        """
    )

    st.subheader("How this app works")
    st.markdown(
        """
        This dashboard **does not re-train models** in the browser. It reads saved output from
        **`repro_m2.ipynb`** (tables and charts).

        Week predictions use a **home baseline** adjusted by offense/defense form
        (prior season for week 1; season-to-date afterward):

        `python train_model.py --predict-week 2026 2`

        That writes `outputs/predictions_2026_wk2.csv`. Optional `--ml` also reports
        lagged ML holdout metrics. Re-run the notebook afterward if you
        want the Streamlit UI to show notebook stdout metrics as well.

        To refresh notebook-backed views:

        1. Open `repro_m2.ipynb` in Jupyter.
        2. **Run all cells** (Kernel → Restart & Run All).
        3. Reload this page (or click **Refresh from notebook** in the sidebar).

        Use the **Report** page in the sidebar to read or download `milestone3.pdf`.
        """
    )

    if st.button("Refresh from notebook", type="primary"):
        get_results.clear()
        get_pick_history.clear()
        st.rerun()


def main() -> None:
    cache_key = _predictions_cache_key()
    results = get_results(cache_key)
    status = data_status()

    with st.sidebar:
        st.title("NFL Predictor")
        page = st.radio(
            "Navigate",
            ["Predictions", "Results", "Model results", "Report", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        if results.executed:
            st.success("Notebook outputs loaded")
            st.caption(f"Last saved: {notebook_mtime_str(results)}")
        elif results.data_source:
            st.info(f"Using saved data: `{results.data_source}`")
            st.caption("Run repro_m2.ipynb with outputs saved for full metrics.")
        elif results.has_model_views():
            st.info("Charts loaded from outputs/")
        else:
            st.warning("No notebook or saved outputs yet")
            st.caption("Run repro_m2.ipynb or train_model.py")

        if st.button("Refresh from notebook", use_container_width=True):
            get_results.clear()
            get_pick_history.clear()
            st.rerun()

    st.title("NFL Game Outcome Predictor")
    st.markdown(
        "Probabilistic home/away win forecasts from the Milestone III pipeline "
        "(defensive efficiency + schedule context, 1999–2026)."
    )

    if page == "Report":
        render_report()
        return

    if page == "Predictions":
        if not results.has_predictions():
            st.error(
                "No predictions found. Either run all cells in `repro_m2.ipynb` and save "
                "with outputs, or run: `python train_model.py --predict-week 2026 2`"
            )
            render_about(results)
            return
        if not results.executed and results.data_source:
            st.info(f"Showing predictions from `{results.data_source}` (notebook not saved with outputs).")
        render_predictions(results)
    elif page == "Results":
        render_pick_results(get_pick_history(cache_key))
    elif page == "Model results":
        if not results.has_model_views():
            st.warning(
                "Model metrics and charts need notebook stdout or files in `outputs/`. "
                "Run `repro_m2.ipynb` (Kernel → Restart & Run All), then refresh."
            )
        else:
            if not results.executed:
                st.info("Some charts loaded from `outputs/` folder.")
            render_metrics(results)
    else:
        render_about(results)


if __name__ == "__main__":
    main()
