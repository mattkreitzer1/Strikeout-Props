from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BuildOutputs:
    pitcher_platoon_pitch_type: Path
    batter_platoon_pitch_type: Path
    pitcher_custom: Path
    batter_custom: Path
    batter_hand_summary: Path


def _split_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("pitcher_") and "_vs_" in name:
        return name.replace("pitcher_", "", 1)
    return "unknown"


def _hand_split_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("batter_vs_RHP"):
        return "vs_RHP"
    if name.startswith("batter_vs_LHP"):
        return "vs_LHP"
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


def _normalize_custom(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "last_name, first_name" in out.columns:
        out = out.rename(columns={"last_name, first_name": "player_name"})
    return out


CUSTOM_NUMERIC = [
    "player_id",
    "year",
    "pa",
    "k_percent",
    "bb_percent",
    "woba",
    "xwoba",
    "whiff_percent",
    "swing_percent",
    "zone_percent",
    "o_swing_percent",
    "z_swing_percent",
]

BATTER_HAND_NUMERIC = [
    "player_id",
    "pa",
    "k_percent",
    "bb_percent",
    "swing_miss_percent",
]


def build_features(
    run_date: Date,
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
    min_pitches_per_pitch_type_row: int = 100,
    min_pitches_per_batter_pitch_row: int = 75,
) -> BuildOutputs:
    """
    Turn raw Savant exports into smaller, consistent feature tables.
    """
    day_raw = raw_root / run_date.isoformat()
    if not day_raw.exists():
        raise FileNotFoundError(
            f"Missing raw folder: {day_raw}. Run `python -m mlb_kprop fetch-savant` first."
        )

    day_processed = processed_root / run_date.isoformat()
    day_processed.mkdir(parents=True, exist_ok=True)

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
    platoon = platoon[platoon["pitches"] >= min_pitches_per_pitch_type_row].copy()

    platoon_out = day_processed / "pitcher_platoon_pitch_type.csv"
    platoon.to_csv(platoon_out, index=False)

    pitcher_custom_in = day_raw / "pitcher_custom_2025_2026.csv"
    batter_custom_in = day_raw / "batter_custom_2025_2026.csv"
    pitcher_custom = _normalize_custom(_read_csv(pitcher_custom_in))
    batter_custom = _normalize_custom(_read_csv(batter_custom_in))
    pitcher_custom = _to_numeric(
        pitcher_custom,
        cols=[c for c in CUSTOM_NUMERIC if c in pitcher_custom.columns],
    )
    batter_custom = _to_numeric(
        batter_custom,
        cols=[c for c in CUSTOM_NUMERIC if c in batter_custom.columns],
    )

    batter_hand_files = [
        day_raw / "batter_vs_RHP.csv",
        day_raw / "batter_vs_LHP.csv",
    ]
    batter_hand_frames: list[pd.DataFrame] = []
    for path in batter_hand_files:
        if not path.exists():
            continue
        df = _read_csv(path)
        df["hand_split"] = _hand_split_from_filename(path)
        batter_hand_frames.append(df)

    if batter_hand_frames:
        batter_hand = pd.concat(batter_hand_frames, ignore_index=True)
        batter_keep = [
            "player_id",
            "player_name",
            "hand_split",
            "pa",
            "k_percent",
            "bb_percent",
            "swing_miss_percent",
        ]
        batter_hand = batter_hand[[c for c in batter_keep if c in batter_hand.columns]].copy()
        batter_hand = _to_numeric(batter_hand, cols=BATTER_HAND_NUMERIC)
        batter_hand = batter_hand[batter_hand["pa"] >= 25].copy()
    else:
        batter_hand = pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "hand_split",
                "pa",
                "k_percent",
                "bb_percent",
                "swing_miss_percent",
            ]
        )

    batter_hand_out = day_processed / "batter_hand_summary.csv"
    if not batter_custom.empty and not batter_hand.empty:
        chase = (
            batter_custom.groupby("player_id", as_index=False)
            .agg(o_swing_percent=("o_swing_percent", "mean"))
        )
        batter_hand = batter_hand.merge(chase, on="player_id", how="left")
    batter_hand.to_csv(batter_hand_out, index=False)

    batter_pitch_files = [
        day_raw / "batter_vs_RHP_pitch_type.csv",
        day_raw / "batter_vs_LHP_pitch_type.csv",
    ]
    batter_pitch_frames: list[pd.DataFrame] = []
    for path in batter_pitch_files:
        if not path.exists():
            continue
        df = _read_csv(path)
        df["hand_split"] = _hand_split_from_filename(path)
        batter_pitch_frames.append(df)

    if batter_pitch_frames:
        batter_pitch = pd.concat(batter_pitch_frames, ignore_index=True)
        batter_pitch_keep = [
            "player_id",
            "player_name",
            "hand_split",
            "pitch_type",
            "pitches",
            "pa",
            "k_percent",
            "swing_miss_percent",
        ]
        batter_pitch = batter_pitch[
            [c for c in batter_pitch_keep if c in batter_pitch.columns]
        ].copy()
        batter_pitch = _to_numeric(
            batter_pitch,
            cols=[
                "player_id",
                "pitches",
                "pa",
                "k_percent",
                "swing_miss_percent",
            ],
        )
        batter_pitch = batter_pitch[
            batter_pitch["pitches"] >= min_pitches_per_batter_pitch_row
        ].copy()
    else:
        batter_pitch = pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "hand_split",
                "pitch_type",
                "pitches",
                "pa",
                "k_percent",
                "swing_miss_percent",
            ]
        )

    batter_pitch_out = day_processed / "batter_platoon_pitch_type.csv"
    batter_pitch.to_csv(batter_pitch_out, index=False)

    pitcher_custom_out = day_processed / "pitcher_custom.csv"
    batter_custom_out = day_processed / "batter_custom.csv"
    pitcher_custom.to_csv(pitcher_custom_out, index=False)
    batter_custom.to_csv(batter_custom_out, index=False)

    return BuildOutputs(
        pitcher_platoon_pitch_type=platoon_out,
        batter_platoon_pitch_type=batter_pitch_out,
        pitcher_custom=pitcher_custom_out,
        batter_custom=batter_custom_out,
        batter_hand_summary=batter_hand_out,
    )
