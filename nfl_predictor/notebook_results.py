"""Load metrics and predictions from executed repro_m2.ipynb outputs."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd

from .constants import OUTPUTS_DIR, ROOT

NOTEBOOK_PATH = ROOT / "repro_m2.ipynb"


@dataclass
class NotebookResults:
    notebook_path: Path
    executed: bool
    stdout: str
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_scores: dict[str, str] = field(default_factory=dict)
    extrapolation: dict[str, str] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    figure_files: dict[str, Path] = field(default_factory=dict)
    data_source: str = ""

    def has_predictions(self) -> bool:
        return not self.predictions.empty

    def has_model_views(self) -> bool:
        return (
            not self.model_metrics.empty
            or bool(self.cv_scores)
            or bool(self.extrapolation)
            or bool(self.figures)
            or bool(self.figure_files)
        )

    def can_run_app(self) -> bool:
        return self.has_predictions() or self.has_model_views() or self.executed


def _collect_stdout(cells: list) -> str:
    parts: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                parts.extend(out.get("text", []))
    return "".join(parts)


def _collect_figures(cells: list) -> dict[str, bytes]:
    figures: dict[str, bytes] = {}
    labels = [
        ("rf_tuning", "Random Forest Hyperparameter"),
        ("lr_tuning", "Logistic Regression C Parameter"),
        ("xgb_tuning", "XGBoost Hyperparameter"),
    ]
    cm_idx = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        cell_stdout = ""
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                cell_stdout += "".join(out.get("text", []))

        for out in cell.get("outputs", []):
            if out.get("output_type") != "display_data":
                continue
            data = out.get("data", {})
            png = data.get("image/png")
            if not png:
                continue
            raw = base64.b64decode(png) if isinstance(png, str) else base64.b64decode(
                png[0]
            )

            named = False
            for key, needle in labels:
                if needle in cell_stdout and key not in figures:
                    figures[key] = raw
                    named = True
                    break

            if not named:
                if "confusion" in cell_stdout.lower() or cm_idx < 3:
                    figures[f"confusion_matrix_{cm_idx}"] = raw
                    cm_idx += 1
                else:
                    figures[f"figure_{len(figures)}"] = raw

    return figures


def _read_fwf_table(text: str, header_test) -> pd.DataFrame:
    """header_test: callable(line) -> bool for the header row."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if header_test(line):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    table_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.strip().startswith("="):
            break
        table_lines.append(line.rstrip())

    if len(table_lines) < 2:
        return pd.DataFrame()

    try:
        df = pd.read_fwf(StringIO("\n".join(table_lines)), header=0)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def _load_predictions_from_disk() -> tuple[pd.DataFrame, str]:
    """Fallback: latest outputs/predictions_*.csv from train_model.py."""
    candidates = sorted(
        OUTPUTS_DIR.glob("predictions_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return pd.DataFrame(), ""
    path = candidates[0]
    df = pd.read_csv(path)
    if "pred_home_win_prob" in df.columns:
        df["pred_home_win_prob"] = pd.to_numeric(df["pred_home_win_prob"], errors="coerce")
    if "confidence" not in df.columns and "pred_home_win_prob" in df.columns:
        df["confidence"] = df["pred_home_win_prob"].apply(
            lambda p: max(p, 1 - p) if pd.notna(p) else None
        )
    return df, f"outputs/{path.name}"


def _parse_metrics_eval_blocks(stdout: str) -> pd.DataFrame:
    """Parse evaluate() stdout blocks (Accuracy / Precision / ... / Brier)."""
    rows: list[dict[str, float | str]] = []
    current_name: str | None = None
    metrics: dict[str, float] = {}

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_name and metrics:
                rows.append({"Model": current_name, **metrics})
                current_name, metrics = None, {}
            continue
        if stripped.startswith("Random Forest") or stripped.startswith("Logistic") or stripped.startswith("XGBoost"):
            if current_name and metrics:
                rows.append({"Model": current_name, **metrics})
            current_name = stripped
            metrics = {}
            continue
        if current_name and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip().lower()
            try:
                metrics[key] = float(val.strip())
            except ValueError:
                pass

    if current_name and metrics:
        rows.append({"Model": current_name, **metrics})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {
        "accuracy": "acc",
        "precision": "prec",
        "recall": "rec",
        "f1": "f1",
        "auroc": "auroc",
        "brier": "brier",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _parse_metrics(stdout: str) -> pd.DataFrame:
    df = _parse_metrics_eval_blocks(stdout)
    if not df.empty:
        return df
    return _parse_metrics_legacy(stdout)


def _parse_metrics_legacy(stdout: str) -> pd.DataFrame:
    lines = stdout.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Model" in line and "acc" in line and "prec" in line:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    rows: list[dict[str, str | float]] = []
    aurocs: list[float] = []

    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("="):
            break
        if stripped.replace("\\", "").strip() == "":
            continue
        if stripped.startswith("auroc"):
            continue
        parts = stripped.replace("\\", "").split()
        if len(parts) >= 6 and parts[0].isdigit():
            try:
                acc, prec, rec, f1 = (
                    float(parts[-4]),
                    float(parts[-3]),
                    float(parts[-2]),
                    float(parts[-1]),
                )
            except ValueError:
                continue
            rows.append(
                {
                    "Model": " ".join(parts[1:-4]),
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                }
            )
        elif len(parts) == 2 and parts[0].isdigit():
            aurocs.append(float(parts[1]))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if len(aurocs) == len(df):
        df["auroc"] = aurocs
    else:
        capture = False
        for line in lines[header_idx + 1 :]:
            if line.strip().startswith("auroc"):
                capture = True
                continue
            if capture:
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit():
                    aurocs.append(float(parts[1]))
                elif not line.strip():
                    break
        if len(aurocs) == len(df):
            df["auroc"] = aurocs

    return df


def _parse_cv(stdout: str) -> dict[str, str]:
    cv: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("CV Acc —"):
            cv[line.split("—")[0].replace("CV Acc ", "").strip()] = line
    return cv


def _parse_extrapolation(stdout: str) -> dict[str, str]:
    keys = [
        "Model Win Percentage",
        "Total Number of Bets Won",
        "Total Number of Bets Made",
        "Possible Games (2019+)",
        "Train (pre-",
        "Test (",
    ]
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        for key in keys:
            if stripped.startswith(key) and ":" in stripped:
                label = stripped.split(":", 1)[0].strip()
                result[label] = stripped.split(":", 1)[-1].strip()
    return result


def _figure_files_from_disk() -> dict[str, Path]:
    mapping = {
        "rf_tuning": OUTPUTS_DIR / "rf_hyperopt_tuning.png",
        "xgb_tuning": OUTPUTS_DIR / "xgb_hyperopt_tuning.png",
    }
    found = {k: p for k, p in mapping.items() if p.is_file()}
    for path in sorted(OUTPUTS_DIR.glob("*confusion_matrix*.png")):
        found[path.stem] = path
    return found


def load_notebook_results(
    notebook_path: Path | None = None,
) -> NotebookResults:
    path = notebook_path or NOTEBOOK_PATH
    if not path.is_file():
        return NotebookResults(
            notebook_path=path,
            executed=False,
            stdout="",
        )

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    notebook_executed = any(
        cell.get("outputs") for cell in cells if cell.get("cell_type") == "code"
    )
    stdout = _collect_stdout(cells)

    predictions = _read_fwf_table(
        stdout,
        lambda ln: "pred_home_win_prob" in ln and "team_home" in ln,
    )
    data_source = "repro_m2.ipynb" if notebook_executed else ""

    if predictions.empty:
        predictions, disk_src = _load_predictions_from_disk()
        if disk_src:
            data_source = disk_src

    metrics = _parse_metrics(stdout)

    if not predictions.empty and "pred_home_win_prob" in predictions.columns:
        predictions["pred_home_win_prob"] = pd.to_numeric(
            predictions["pred_home_win_prob"], errors="coerce"
        )
        if "confidence" not in predictions.columns:
            predictions["confidence"] = predictions["pred_home_win_prob"].apply(
                lambda p: max(p, 1 - p) if pd.notna(p) else None
            )

    figure_files = _figure_files_from_disk()
    figures = _collect_figures(cells) if notebook_executed else {}

    return NotebookResults(
        notebook_path=path,
        executed=notebook_executed,
        stdout=stdout,
        predictions=predictions,
        model_metrics=metrics,
        cv_scores=_parse_cv(stdout),
        extrapolation=_parse_extrapolation(stdout),
        figures=figures,
        figure_files=figure_files,
        data_source=data_source,
    )


def notebook_mtime_str(results: NotebookResults) -> str:
    if not results.notebook_path.is_file():
        return "unknown"
    from datetime import datetime

    ts = results.notebook_path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y at %I:%M %p")
