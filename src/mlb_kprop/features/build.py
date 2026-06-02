from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BuildOutputs:
    pitcher_platoon_pitch_type: Path
    pitcher_custom: Path
    batter_custom: Path


def _split_from_filename(path: Path) -> str:
    # Expect filenames like pitcher_R_vs_R.csv
    name = path.stem
    if name.startswith("pitcher_") and "_vs_" in name:
        return name.replace("pitcher_", "", 1)
    return "unknown"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_features(
    run_date: Date,
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
    min_pitches_per_pitch_type_row: int = 100,
) -> BuildOutputs:
    """
    Turn raw Savant exports into smaller, consistent feature tables.

    Why:
    - Statcast Search exports lots of extra columns by default.
    - The meaning of \"min_pitches\" in the UI is confusing for grouped tables.
      We enforce the pitch-type row minimum ourselves using the `pitches` column.
    """
    day_raw = raw_root / run_date.isoformat()
    if not day_raw.exists():
        raise FileNotFoundError(
            f"Missing raw folder: {day_raw}. Run `python -m mlb_kprop fetch-savant` first."
        )

    day_processed = processed_root / run_date.isoformat()
    day_processed.mkdir(parents=True, exist_ok=True)

    # --- Pitcher platoon (Statcast Search) ---
    platoon_files = [
        day_raw / "pitcher_R_vs_R.csv",
        day_raw / "pitcher_R_vs_L.csv",
        day_raw / "pitcher_L_vs_R.csv",
        day_raw / "pitcher_L_vs_L.csv",
    ]

    platoon_frames: list[pd.DataFrame] = []
    for path in platoon_files:
        df = _read_csv(path)
        df["platoon_split"] = _split_from_filename(path)
        platoon_frames.append(df)

    platoon = pd.concat(platoon_frames, ignore_index=True)

    # Keep only the columns we actually intend to use right now.
    keep_cols = [
        "player_id",
        "player_name",
        "platoon_split",
        "pitch_type",
        "pitches",
        "pitch_percent",
        "pa",
        "k_percent",
        "bb_percent",
        "xwoba",
        "swing_miss_percent",
    ]
    platoon = platoon[[c for c in keep_cols if c in platoon.columns]].copy()

    platoon = _to_numeric(
        platoon,
        cols=[
            "player_id",
            "pitches",
            "pitch_percent",
            "pa",
            "k_percent",
            "bb_percent",
            "xwoba",
            "swing_miss_percent",
        ],
    )

    # Enforce pitch-type sample size for THIS row.
    platoon = platoon[platoon["pitches"] >= min_pitches_per_pitch_type_row].copy()

    platoon_out = day_processed / "pitcher_platoon_pitch_type.csv"
    platoon.to_csv(platoon_out, index=False)

    # --- Custom leaderboards ---
    pitcher_custom_in = day_raw / "pitcher_custom_2025_2026.csv"
    batter_custom_in = day_raw / "batter_custom_2025_2026.csv"

    pitcher_custom = _read_csv(pitcher_custom_in).rename(
        columns={"last_name, first_name": "player_name"}
    )
    batter_custom = _read_csv(batter_custom_in).rename(
        columns={"last_name, first_name": "player_name"}
    )

    # Standardize key numeric columns.
    pitcher_custom = _to_numeric(
        pitcher_custom,
        cols=["player_id", "year", "pa", "k_percent", "bb_percent", "woba", "xwoba"],
    )
    batter_custom = _to_numeric(
        batter_custom,
        cols=["player_id", "year", "pa", "k_percent", "bb_percent", "woba", "xwoba"],
    )

    pitcher_custom_out = day_processed / "pitcher_custom.csv"
    batter_custom_out = day_processed / "batter_custom.csv"
    pitcher_custom.to_csv(pitcher_custom_out, index=False)
    batter_custom.to_csv(batter_custom_out, index=False)

    return BuildOutputs(
        pitcher_platoon_pitch_type=platoon_out,
        pitcher_custom=pitcher_custom_out,
        batter_custom=batter_custom_out,
    )

