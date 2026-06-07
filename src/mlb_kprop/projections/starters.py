from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pandas as pd

STARTERS_COLUMNS = [
    "player_id",
    "player_name",
    "pitcher_throws",
    "opp_lhb_pct",
    "batters_faced",
    "opp_team_id",
    "lineup_source",
    "game_status",
    "home_team_abbr",
    "notes",
]

EXAMPLE_ROW = {
    "player_id": 434378,
    "player_name": "Verlander, Justin",
    "pitcher_throws": "R",
    "opp_lhb_pct": 0.42,
    "batters_faced": "",
    "notes": "example — delete or replace",
}


def starters_path_for_date(
    run_date: Date,
    starters_root: Path = Path("data/starters"),
) -> Path:
    return starters_root / f"{run_date.isoformat()}.csv"


def write_starters_template(
    run_date: Date,
    starters_root: Path = Path("data/starters"),
    overwrite: bool = False,
) -> Path:
    """Create a blank starters CSV for the day with column headers + one example row."""
    starters_root.mkdir(parents=True, exist_ok=True)
    path = starters_path_for_date(run_date, starters_root=starters_root)
    if path.exists() and not overwrite:
        return path

    df = pd.DataFrame([EXAMPLE_ROW], columns=STARTERS_COLUMNS)
    df.to_csv(path, index=False)
    return path


def load_starters(
    run_date: Date,
    starters_root: Path = Path("data/starters"),
    starters_path: Path | None = None,
) -> pd.DataFrame:
    path = starters_path or starters_path_for_date(run_date, starters_root=starters_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing starters file: {path}\n"
            f"Run: python -m mlb_kprop init-starters --date {run_date.isoformat()}"
        )

    df = pd.read_csv(path)
    missing = [c for c in STARTERS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    # Drop example / blank rows with no name.
    df = df.copy()
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df = df[df["player_name"].ne("") & df["player_name"].ne("nan")]

    if "notes" in df.columns:
        df = df[~df["notes"].astype(str).str.startswith("example")]

    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    df["opp_lhb_pct"] = pd.to_numeric(df["opp_lhb_pct"], errors="coerce")
    if (df["opp_lhb_pct"] > 1).any():
        df.loc[df["opp_lhb_pct"] > 1, "opp_lhb_pct"] /= 100.0

    if "batters_faced" in df.columns:
        df["batters_faced"] = pd.to_numeric(df["batters_faced"], errors="coerce")

    for col in ("opp_team_id",):
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("lineup_source", "game_status", "home_team_abbr"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()

    if "pitcher_throws" in df.columns:
        df["pitcher_throws"] = (
            df["pitcher_throws"].astype(str).str.strip().str.upper().str[:1]
        )
        df.loc[~df["pitcher_throws"].isin(["R", "L"]), "pitcher_throws"] = ""

    return df.reset_index(drop=True)
