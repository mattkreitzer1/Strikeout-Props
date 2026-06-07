from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pandas as pd

LINES_COLUMNS = [
    "player_id",
    "player_name",
    "book_line",
    "over_odds",
    "under_odds",
    "notes",
]

EXAMPLE_LINE = {
    "player_id": 434378,
    "player_name": "Verlander, Justin",
    "book_line": 5.5,
    "over_odds": -115,
    "under_odds": -105,
    "notes": "example — delete or replace",
}


def lines_path_for_date(
    run_date: Date,
    lines_root: Path = Path("data/lines"),
) -> Path:
    return lines_root / f"{run_date.isoformat()}.csv"


def write_lines_template(
    run_date: Date,
    lines_root: Path = Path("data/lines"),
    overwrite: bool = False,
) -> Path:
    lines_root.mkdir(parents=True, exist_ok=True)
    path = lines_path_for_date(run_date, lines_root=lines_root)
    if path.exists() and not overwrite:
        return path

    df = pd.DataFrame([EXAMPLE_LINE], columns=LINES_COLUMNS)
    df.to_csv(path, index=False)
    return path


def load_lines(
    run_date: Date,
    lines_root: Path = Path("data/lines"),
    lines_path: Path | None = None,
) -> pd.DataFrame:
    path = lines_path or lines_path_for_date(run_date, lines_root=lines_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing lines file: {path}\n"
            f"Run: python -m mlb_kprop init-lines --date {run_date.isoformat()}"
        )

    df = pd.read_csv(path)
    missing = [c for c in LINES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.copy()
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df = df[df["player_name"].ne("") & df["player_name"].ne("nan")]

    if "notes" in df.columns:
        df = df[~df["notes"].astype(str).str.startswith("example")]

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["book_line"] = pd.to_numeric(df["book_line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")

    return df.reset_index(drop=True)
