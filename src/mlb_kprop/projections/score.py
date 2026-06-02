from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlb_kprop.projections.names import match_player_name
from mlb_kprop.projections.starters import load_starters

DEFAULT_CONFIG_PATH = Path("config/projection_defaults.yaml")

# Platoon split keys on pitcher_split_summary.csv
SPLIT_RHP_VS_LHB = "R_vs_L"
SPLIT_RHP_VS_RHB = "R_vs_R"
SPLIT_LHP_VS_LHB = "L_vs_L"
SPLIT_LHP_VS_RHB = "L_vs_R"


@dataclass(frozen=True)
class ProjectionOutputs:
    projections_csv: Path


def load_projection_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _round_to_half(value: float) -> float:
    return round(value * 2) / 2


def _split_lookup(
    summary: pd.DataFrame,
    player_id: int,
    split: str,
) -> pd.Series | None:
    rows = summary[
        (summary["player_id"] == player_id) & (summary["platoon_split"] == split)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def infer_pitcher_throws(
    summary: pd.DataFrame,
    player_id: int,
    min_pitches: float,
) -> str | None:
    """
    Guess R/L from which handedness bucket has more pitch volume in the export.
    RHP rows use R_vs_* ; LHP rows use L_vs_*.
    """
    player = summary[summary["player_id"] == player_id]
    if player.empty:
        return None

    r_pitches = player[player["platoon_split"].str.startswith("R_vs_")][
        "pitches_total"
    ].sum()
    l_pitches = player[player["platoon_split"].str.startswith("L_vs_")][
        "pitches_total"
    ].sum()

    if r_pitches < min_pitches and l_pitches < min_pitches:
        return None
    return "R" if r_pitches >= l_pitches else "L"


def blend_platoon_k_percent(
    summary: pd.DataFrame,
    player_id: int,
    pitcher_throws: str,
    opp_lhb_pct: float,
) -> tuple[float, str]:
    """
    Weight K% by expected share of LHB vs RHB faced.

    opp_lhb_pct: fraction of opposing plate appearances from left-handed batters (0–1).
    """
    if pitcher_throws == "R":
        vs_lhb = SPLIT_RHP_VS_LHB
        vs_rhb = SPLIT_RHP_VS_RHB
    elif pitcher_throws == "L":
        vs_lhb = SPLIT_LHP_VS_LHB
        vs_rhb = SPLIT_LHP_VS_RHB
    else:
        raise ValueError(f"pitcher_throws must be R or L, got {pitcher_throws!r}")

    row_l = _split_lookup(summary, player_id, vs_lhb)
    row_r = _split_lookup(summary, player_id, vs_rhb)

    if row_l is None and row_r is None:
        raise ValueError(
            f"No platoon splits for player_id={player_id} ({vs_lhb} / {vs_rhb})"
        )

    if row_l is None:
        return float(row_r["k_percent"]), vs_rhb
    if row_r is None:
        return float(row_l["k_percent"]), vs_lhb

    k_l = float(row_l["k_percent"])
    k_r = float(row_r["k_percent"])
    blended = opp_lhb_pct * k_l + (1.0 - opp_lhb_pct) * k_r
    blend_label = f"{opp_lhb_pct:.0%}*{vs_lhb} + {1-opp_lhb_pct:.0%}*{vs_rhb}"
    return blended, blend_label


def resolve_player_id(
    starters_row: pd.Series,
    summary: pd.DataFrame,
) -> int | None:
    pid = starters_row.get("player_id")
    if pd.notna(pid):
        pid_int = int(pid)
        if (summary["player_id"] == pid_int).any():
            return pid_int

    name = str(starters_row["player_name"]).strip()
    candidates = summary["player_name"].astype(str).unique().tolist()
    matched = match_player_name(name, candidates)
    if matched:
        return int(summary[summary["player_name"] == matched]["player_id"].iloc[0])
    return None


def score_projections(
    run_date: Date,
    processed_root: Path = Path("data/processed"),
    starters_root: Path = Path("data/starters"),
    starters_path: Path | None = None,
    reports_root: Path = Path("reports"),
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> ProjectionOutputs:
    """
    Phase 1 fair K: blended platoon K% × expected batters faced.

    fair_k = (k_percent_blended / 100) * batters_faced
    """
    cfg = load_projection_config(config_path)
    default_bf = int(cfg.get("default_batters_faced", 24))
    round_half = bool(cfg.get("round_fair_k_to_half", True))
    infer_hand = bool(cfg.get("infer_pitcher_throws_from_splits", True))
    min_pitches_infer = float(cfg.get("min_pitches_for_hand_inference", 50))

    day_proc = processed_root / run_date.isoformat()
    summary_path = day_proc / "pitcher_split_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run `python -m mlb_kprop run-daily` first."
        )

    summary = pd.read_csv(summary_path)
    starters = load_starters(
        run_date, starters_root=starters_root, starters_path=starters_path
    )

    if starters.empty:
        raise ValueError("Starters file has no rows (after removing blanks).")

    rows: list[dict[str, object]] = []
    errors: list[str] = []

    for _, starter in starters.iterrows():
        name = starter["player_name"]
        player_id = resolve_player_id(starter, summary)
        if player_id is None:
            errors.append(f"Could not match pitcher: {name}")
            continue

        throws = starter.get("pitcher_throws", "")
        if not throws or throws not in ("R", "L"):
            if infer_hand:
                throws = infer_pitcher_throws(
                    summary, player_id, min_pitches_infer
                )
            if not throws:
                errors.append(
                    f"{name}: set pitcher_throws to R or L in starters CSV"
                )
                continue

        opp_lhb = starter["opp_lhb_pct"]
        if pd.isna(opp_lhb):
            errors.append(f"{name}: set opp_lhb_pct (0–1, e.g. 0.40 for 40% LHB)")
            continue
        opp_lhb = float(max(0.0, min(1.0, opp_lhb)))

        bf = starter.get("batters_faced")
        batters_faced = int(bf) if pd.notna(bf) and bf > 0 else default_bf

        try:
            k_blend, blend_detail = blend_platoon_k_percent(
                summary, player_id, str(throws), opp_lhb
            )
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue

        fair_k = (k_blend / 100.0) * batters_faced
        fair_k_book = _round_to_half(fair_k) if round_half else fair_k

        rows.append(
            {
                "player_id": player_id,
                "player_name": name,
                "pitcher_throws": throws,
                "opp_lhb_pct": opp_lhb,
                "batters_faced": batters_faced,
                "k_percent_blended": round(k_blend, 3),
                "blend_detail": blend_detail,
                "fair_k": round(fair_k, 3),
                "fair_k_line": fair_k_book,
            }
        )

    if errors:
        msg = "Projection errors:\n" + "\n".join(f"  - {e}" for e in errors)
        if not rows:
            raise ValueError(msg)
        print(msg)

    if not rows:
        raise ValueError("No projections produced.")

    out_df = pd.DataFrame(rows).sort_values("player_name")
    reports_root.mkdir(parents=True, exist_ok=True)
    out_path = reports_root / f"projections_{run_date.isoformat()}.csv"
    out_df.to_csv(out_path, index=False)

    return ProjectionOutputs(projections_csv=out_path)
