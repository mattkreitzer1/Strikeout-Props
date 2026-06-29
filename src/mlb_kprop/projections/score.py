from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlb_kprop.mlb.starters import MlbStatsClient, load_mlb_config
from mlb_kprop.mlb.workload import predict_batters_faced
from mlb_kprop.projections.matchup import (
    blend_pitcher_platoon_rates,
    composite_k_percent,
    lineup_opponent_profile,
    parse_lineup_batter_ids,
    pitcher_pitch_mix,
    pitcher_skill_row,
)
from mlb_kprop.projections.names import match_player_name
from mlb_kprop.projections.starters import load_starters

DEFAULT_CONFIG_PATH = Path("config/projection_defaults.yaml")


@dataclass(frozen=True)
class ProjectionOutputs:
    projections_csv: Path


def load_projection_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _round_to_half(value: float) -> float:
    return round(value * 2) / 2


def infer_pitcher_throws(
    summary: pd.DataFrame,
    player_id: int,
    min_pitches: float,
) -> str | None:
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


def load_park_factors(config: dict[str, Any]) -> dict[str, float]:
    path = Path(config.get("park_factors_path", "config/park_factors.yaml"))
    if not path.exists():
        return {"default": 1.0}
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    teams = raw.get("teams") or {}
    factors = {str(k): float(v) for k, v in teams.items()}
    factors.setdefault("default", 1.0)
    return factors


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
    Fair K: composite K% (platoon whiff/K + lineup opponent + zone/chase skill) × BF.
    """
    cfg = load_projection_config(config_path)
    default_bf = int(cfg.get("default_batters_faced", 24))
    round_half = bool(cfg.get("round_fair_k_to_half", True))
    infer_hand = bool(cfg.get("infer_pitcher_throws_from_splits", True))
    min_pitches_infer = float(cfg.get("min_pitches_for_hand_inference", 50))
    league = cfg.get("league_rates") or {}
    league_k = float(cfg.get("league_k_percent", league.get("k_percent", 22.5)))
    league_whiff = float(league.get("whiff_percent", 25.0))
    shrinkage_pitches = float(cfg.get("k_shrinkage_pitches", 400))
    k_model = cfg.get("k_model") or {}
    pitch_matchup = cfg.get("pitch_matchup") or {}
    park_factors = load_park_factors(cfg)
    mlb_cfg = load_mlb_config()
    stats_client = MlbStatsClient(mlb_cfg)

    day_proc = processed_root / run_date.isoformat()
    summary_path = day_proc / "pitcher_split_summary.csv"
    skill_path = day_proc / "pitcher_skill.csv"
    batter_hand_path = day_proc / "batter_hand_summary.csv"
    batter_pitch_path = day_proc / "batter_platoon_pitch_type.csv"
    pitcher_platoon_path = day_proc / "pitcher_platoon_pitch_type.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run morning `run-daily` first "
            "(afternoon refresh reuses that day's processed Savant tables)."
        )

    summary = pd.read_csv(summary_path)
    pitcher_skill = pd.read_csv(skill_path) if skill_path.exists() else pd.DataFrame()
    batter_hand = pd.read_csv(batter_hand_path) if batter_hand_path.exists() else pd.DataFrame()
    batter_pitch = (
        pd.read_csv(batter_pitch_path) if batter_pitch_path.exists() else pd.DataFrame()
    )
    pitcher_platoon = (
        pd.read_csv(pitcher_platoon_path)
        if pitcher_platoon_path.exists()
        else pd.DataFrame()
    )

    league_full = {
        "k_percent": league_k,
        "whiff_percent": league_whiff,
        "o_swing_percent": float(league.get("o_swing_percent", 31.5)),
        "zone_percent": float(league.get("zone_percent", 49.5)),
        "z_swing_percent": float(league.get("z_swing_percent", 66.0)),
    }

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
                throws = infer_pitcher_throws(summary, player_id, min_pitches_infer)
            if not throws:
                errors.append(f"{name}: set pitcher_throws to R or L in starters CSV")
                continue

        opp_lhb = starter["opp_lhb_pct"]
        if pd.isna(opp_lhb):
            errors.append(f"{name}: set opp_lhb_pct (0–1, e.g. 0.40 for 40% LHB)")
            continue
        opp_lhb = float(max(0.0, min(1.0, opp_lhb)))

        bf = starter.get("batters_faced")
        bf_detail = ""
        recent_bf_std = pd.NA
        if pd.notna(bf) and bf > 0:
            batters_faced = int(bf)
            bf_detail = "manual"
        else:
            opp_team_id = starter.get("opp_team_id")
            if pd.notna(opp_team_id):
                bf_pred = predict_batters_faced(
                    stats_client,
                    pitcher_id=int(player_id),
                    opp_team_id=int(opp_team_id),
                    run_date=run_date,
                    config=cfg,
                )
                batters_faced = bf_pred.batters_faced
                bf_detail = bf_pred.detail
                recent_bf_std = bf_pred.recent_bf_std
            else:
                batters_faced = default_bf
                bf_detail = f"default_{default_bf}bf"

        lineup_source = str(starter.get("lineup_source") or "")
        game_status = str(starter.get("game_status") or "")
        game_start = str(starter.get("game_start") or "")
        home_abbr = str(starter.get("home_team_abbr") or "")
        park_factor = float(park_factors.get(home_abbr, park_factors.get("default", 1.0)))

        try:
            k_platoon, whiff_platoon, platoon_label = blend_pitcher_platoon_rates(
                summary,
                player_id,
                str(throws),
                opp_lhb,
                league_k=league_k,
                league_whiff=league_whiff,
                shrinkage_pitches=shrinkage_pitches,
            )
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue

        batter_ids = parse_lineup_batter_ids(starter.get("lineup_batter_ids"))
        pitch_mix = (
            pitcher_pitch_mix(pitcher_platoon, player_id, str(throws), opp_lhb)
            if not pitcher_platoon.empty
            else {}
        )
        opp_profile = lineup_opponent_profile(
            batter_ids,
            str(throws),
            batter_hand,
            league_full,
            batter_pitch=batter_pitch if not batter_pitch.empty else None,
            pitch_mix=pitch_mix or None,
            pitch_matchup_cfg=pitch_matchup,
        )
        skill = pitcher_skill_row(pitcher_skill, player_id, league_full)
        k_blend, model_detail = composite_k_percent(
            k_platoon,
            whiff_platoon,
            float(opp_profile["opp_k_percent"]),
            float(opp_profile["opp_whiff_percent"]),
            float(opp_profile["opp_chase_percent"]),
            skill,
            k_model,
            league_full,
        )

        fair_k = (k_blend / 100.0) * batters_faced * park_factor
        fair_k_book = _round_to_half(fair_k) if round_half else fair_k

        rows.append(
            {
                "player_id": player_id,
                "player_name": name,
                "pitcher_throws": throws,
                "opp_lhb_pct": opp_lhb,
                "lineup_source": lineup_source,
                "game_status": game_status,
                "game_start": game_start,
                "lineup_batters_matched": int(opp_profile["lineup_batters_matched"]),
                "batters_faced": batters_faced,
                "bf_detail": bf_detail,
                "recent_bf_std": recent_bf_std,
                "park_factor": round(park_factor, 3),
                "home_team_abbr": home_abbr,
                "k_percent_platoon": round(k_platoon, 3),
                "whiff_percent_platoon": round(whiff_platoon, 3),
                "opp_k_percent": round(float(opp_profile["opp_k_percent"]), 3),
                "opp_k_percent_hand": round(float(opp_profile["opp_k_percent_hand"]), 3),
                "opp_k_percent_pitch": round(float(opp_profile["opp_k_percent_pitch"]), 3),
                "opp_whiff_percent": round(float(opp_profile["opp_whiff_percent"]), 3),
                "opp_whiff_percent_hand": round(
                    float(opp_profile["opp_whiff_percent_hand"]), 3
                ),
                "opp_whiff_percent_pitch": round(
                    float(opp_profile["opp_whiff_percent_pitch"]), 3
                ),
                "opp_chase_percent": round(float(opp_profile["opp_chase_percent"]), 3),
                "pitcher_whiff_percent": round(skill["whiff_percent"], 3),
                "pitcher_zone_percent": round(skill["zone_percent"], 3),
                "pitcher_chase_percent": round(skill["o_swing_percent"], 3),
                "k_percent_blended": round(k_blend, 3),
                "blend_detail": model_detail,
                "platoon_split_detail": platoon_label,
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
