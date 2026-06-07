from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MergeOutputs:
    pitcher_merged_long: Path
    pitcher_split_summary: Path
    pitcher_skill: Path


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    return float((v * w).sum() / w.sum())


def _prefix_custom_columns(custom: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: f"custom_{col}"
        for col in custom.columns
        if col not in ("player_id", "player_name")
    }
    return custom.rename(columns=rename_map)


def _build_pitcher_skill(custom: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher: PA-weighted season whiff/zone/chase from custom leaderboard."""
    if custom.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "pa_total",
                "whiff_percent",
                "zone_percent",
                "o_swing_percent",
                "z_swing_percent",
                "k_percent",
            ]
        )

    df = custom.copy()
    df["pa"] = pd.to_numeric(df["pa"], errors="coerce")
    df = df[df["pa"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["player_id"])

    skill_cols = [
        "whiff_percent",
        "zone_percent",
        "o_swing_percent",
        "z_swing_percent",
        "k_percent",
        "bb_percent",
        "xwoba",
    ]
    rows: list[dict[str, object]] = []
    for player_id, group in df.groupby("player_id", dropna=False):
        name = group["player_name"].iloc[0] if "player_name" in group.columns else ""
        row: dict[str, object] = {
            "player_id": int(player_id),
            "player_name": name,
            "pa_total": int(group["pa"].sum()),
        }
        for col in skill_cols:
            if col in group.columns:
                row[col] = _weighted_mean(group[col], group["pa"])
        rows.append(row)

    return pd.DataFrame(rows)


def merge_pitcher_features(
    run_date: Date,
    processed_root: Path = Path("data/processed"),
) -> MergeOutputs:
    day_dir = processed_root / run_date.isoformat()
    platoon_path = day_dir / "pitcher_platoon_pitch_type.csv"
    custom_path = day_dir / "pitcher_custom.csv"

    if not platoon_path.exists():
        raise FileNotFoundError(
            f"Missing {platoon_path}. Run `python -m mlb_kprop build-features` first."
        )

    platoon = pd.read_csv(platoon_path)
    custom = pd.read_csv(custom_path) if custom_path.exists() else pd.DataFrame()

    if not custom.empty:
        custom_attach = _build_pitcher_skill(custom).rename(columns={"pa_total": "pa"})
        custom_prefixed = _prefix_custom_columns(custom_attach)
        merged_long = platoon.merge(
            custom_prefixed,
            on="player_id",
            how="left",
            suffixes=("", "_dup"),
        )
    else:
        merged_long = platoon.copy()

    merged_long_path = day_dir / "pitcher_merged_long.csv"
    merged_long.to_csv(merged_long_path, index=False)

    summary_rows: list[dict[str, object]] = []
    group_cols = ["player_id", "player_name", "platoon_split"]
    rate_cols = ["k_percent", "bb_percent", "xwoba", "swing_miss_percent"]

    for keys, group in platoon.groupby(group_cols, dropna=False):
        player_id, player_name, platoon_split = keys
        total_pitches = float(group["pitches"].sum())
        row: dict[str, object] = {
            "player_id": player_id,
            "player_name": player_name,
            "platoon_split": platoon_split,
            "pitch_types_used": int(len(group)),
            "pitches_total": total_pitches,
        }
        for col in rate_cols:
            if col in group.columns:
                row[col] = _weighted_mean(group[col], group["pitches"])
        summary_rows.append(row)

    split_summary = pd.DataFrame(summary_rows)
    split_summary_path = day_dir / "pitcher_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)

    pitcher_skill = _build_pitcher_skill(custom)
    skill_path = day_dir / "pitcher_skill.csv"
    pitcher_skill.to_csv(skill_path, index=False)

    return MergeOutputs(
        pitcher_merged_long=merged_long_path,
        pitcher_split_summary=split_summary_path,
        pitcher_skill=skill_path,
    )
