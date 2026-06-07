from __future__ import annotations

from typing import Any

import pandas as pd

SPLIT_RHP_VS_LHB = "R_vs_L"
SPLIT_RHP_VS_RHB = "R_vs_R"
SPLIT_LHP_VS_LHB = "L_vs_L"
SPLIT_LHP_VS_RHB = "L_vs_R"

BATTER_VS_RHP = "vs_RHP"
BATTER_VS_LHP = "vs_LHP"


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


def shrink_rate(
    rate: float,
    sample: float,
    league: float,
    shrinkage: float,
) -> float:
    if shrinkage <= 0:
        return rate
    weight = sample / (sample + shrinkage)
    return league + weight * (rate - league)


def batter_hand_key(pitcher_throws: str) -> str:
    return BATTER_VS_RHP if pitcher_throws == "R" else BATTER_VS_LHP


def blend_pitcher_platoon_rates(
    summary: pd.DataFrame,
    player_id: int,
    pitcher_throws: str,
    opp_lhb_pct: float,
    league_k: float,
    league_whiff: float,
    shrinkage_pitches: float,
) -> tuple[float, float, str]:
    """Pitch-weighted platoon K% and whiff% for today's expected L/R mix."""
    if pitcher_throws == "R":
        vs_lhb, vs_rhb = SPLIT_RHP_VS_LHB, SPLIT_RHP_VS_RHB
    elif pitcher_throws == "L":
        vs_lhb, vs_rhb = SPLIT_LHP_VS_LHB, SPLIT_LHP_VS_RHB
    else:
        raise ValueError(f"pitcher_throws must be R or L, got {pitcher_throws!r}")

    row_l = _split_lookup(summary, player_id, vs_lhb)
    row_r = _split_lookup(summary, player_id, vs_rhb)
    if row_l is None and row_r is None:
        raise ValueError(f"No platoon splits for player_id={player_id}")

    def _sample(row: pd.Series | None) -> float:
        return float(row["pitches_total"]) if row is not None else 0.0

    def _k(row: pd.Series | None) -> float:
        if row is None:
            return league_k
        return shrink_rate(
            float(row["k_percent"]),
            _sample(row),
            league_k,
            shrinkage_pitches,
        )

    def _whiff(row: pd.Series | None) -> float:
        if row is None or pd.isna(row.get("swing_miss_percent")):
            return league_whiff
        return shrink_rate(
            float(row["swing_miss_percent"]),
            _sample(row),
            league_whiff,
            shrinkage_pitches,
        )

    k_l, k_r = _k(row_l), _k(row_r)
    w_l, w_r = _whiff(row_l), _whiff(row_r)
    k_blend = opp_lhb_pct * k_l + (1.0 - opp_lhb_pct) * k_r
    whiff_blend = opp_lhb_pct * w_l + (1.0 - opp_lhb_pct) * w_r
    label = f"{opp_lhb_pct:.0%}*{vs_lhb}/{vs_rhb}"
    return k_blend, whiff_blend, label


def pitcher_pitch_mix(
    platoon: pd.DataFrame,
    player_id: int,
    pitcher_throws: str,
    opp_lhb_pct: float,
) -> dict[str, float]:
    """Pitch-type shares of today's expected arsenal (L/R platoon mix)."""
    if pitcher_throws == "R":
        vs_lhb, vs_rhb = SPLIT_RHP_VS_LHB, SPLIT_RHP_VS_RHB
    elif pitcher_throws == "L":
        vs_lhb, vs_rhb = SPLIT_LHP_VS_LHB, SPLIT_LHP_VS_RHB
    else:
        return {}

    df_l = platoon[
        (platoon["player_id"] == player_id) & (platoon["platoon_split"] == vs_lhb)
    ]
    df_r = platoon[
        (platoon["player_id"] == player_id) & (platoon["platoon_split"] == vs_rhb)
    ]
    pitch_types = set(df_l.get("pitch_type", pd.Series(dtype=str)).dropna()) | set(
        df_r.get("pitch_type", pd.Series(dtype=str)).dropna()
    )
    if not pitch_types:
        return {}

    weights: dict[str, float] = {}
    total = 0.0
    for pitch_type in pitch_types:
        p_l = float(df_l.loc[df_l["pitch_type"] == pitch_type, "pitches"].sum())
        p_r = float(df_r.loc[df_r["pitch_type"] == pitch_type, "pitches"].sum())
        w = opp_lhb_pct * p_l + (1.0 - opp_lhb_pct) * p_r
        if w > 0:
            weights[str(pitch_type)] = w
            total += w
    if total <= 0:
        return {}
    return {pt: w / total for pt, w in weights.items()}


def batter_k_vs_pitch_mix(
    batter_id: int,
    hand: str,
    pitch_mix: dict[str, float],
    batter_pitch: pd.DataFrame,
    hand_k: float,
    shrinkage_pa: float,
) -> float:
    """Arsenal-weighted batter K% vs hand; shrink pitch rows toward batter hand aggregate."""
    if not pitch_mix:
        return hand_k

    rows = batter_pitch[
        (batter_pitch["player_id"] == batter_id)
        & (batter_pitch["hand_split"] == hand)
    ]
    weighted = 0.0
    for pitch_type, mix_w in pitch_mix.items():
        pt_rows = rows[rows["pitch_type"] == pitch_type]
        if pt_rows.empty:
            k_rate = hand_k
            sample = 0.0
        else:
            row = pt_rows.iloc[0]
            k_rate = float(row["k_percent"])
            sample = float(row.get("pa") or row.get("pitches") or 0.0)
            k_rate = shrink_rate(k_rate, sample, hand_k, shrinkage_pa)
        weighted += mix_w * k_rate
    return weighted


def lineup_opponent_profile(
    batter_ids: list[int],
    pitcher_throws: str,
    batter_hand: pd.DataFrame,
    league: dict[str, float],
    *,
    batter_pitch: pd.DataFrame | None = None,
    pitch_mix: dict[str, float] | None = None,
    pitch_matchup_cfg: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Average opponent K%, whiff%, chase for today's lineup vs pitcher hand."""
    hand = batter_hand_key(pitcher_throws)
    league_k = float(league.get("k_percent", 22.5))
    subset = batter_hand[
        (batter_hand["hand_split"] == hand)
        & (batter_hand["player_id"].isin(batter_ids))
    ] if not batter_hand.empty else pd.DataFrame()
    matched = len(subset)
    if matched == 0:
        return {
            "opp_k_percent": league_k,
            "opp_k_percent_hand": league_k,
            "opp_k_percent_pitch": league_k,
            "opp_whiff_percent": league.get("whiff_percent", 25.0),
            "opp_chase_percent": league.get("o_swing_percent", 31.5),
            "lineup_batters_matched": 0,
        }

    chase = league.get("o_swing_percent", 31.5)
    if "o_swing_percent" in subset.columns and subset["o_swing_percent"].notna().any():
        chase = float(subset["o_swing_percent"].mean())

    opp_k_hand = float(subset["k_percent"].mean())
    opp_k_pitch = opp_k_hand
    cfg = pitch_matchup_cfg or {}
    blend = float(cfg.get("blend_weight", 0.55))
    shrinkage_pa = float(cfg.get("shrinkage_pa", 150))

    if (
        batter_pitch is not None
        and not batter_pitch.empty
        and pitch_mix
        and batter_ids
    ):
        pitch_ks: list[float] = []
        hand_by_id = {
            int(row["player_id"]): float(row["k_percent"])
            for _, row in subset.iterrows()
        }
        for batter_id in batter_ids:
            hand_k = hand_by_id.get(batter_id, opp_k_hand)
            pitch_ks.append(
                batter_k_vs_pitch_mix(
                    batter_id,
                    hand,
                    pitch_mix,
                    batter_pitch,
                    hand_k,
                    shrinkage_pa,
                )
            )
        opp_k_pitch = float(sum(pitch_ks) / len(pitch_ks))
        opp_k = blend * opp_k_pitch + (1.0 - blend) * opp_k_hand
    else:
        opp_k = opp_k_hand

    return {
        "opp_k_percent": opp_k,
        "opp_k_percent_hand": opp_k_hand,
        "opp_k_percent_pitch": opp_k_pitch,
        "opp_whiff_percent": float(subset["swing_miss_percent"].mean()),
        "opp_chase_percent": chase,
        "lineup_batters_matched": matched,
    }


def pitcher_skill_row(
    pitcher_skill: pd.DataFrame,
    player_id: int,
    league: dict[str, float],
) -> dict[str, float]:
    rows = pitcher_skill[pitcher_skill["player_id"] == player_id]
    if rows.empty:
        return {
            "whiff_percent": league.get("whiff_percent", 25.0),
            "zone_percent": league.get("zone_percent", 49.5),
            "o_swing_percent": league.get("o_swing_percent", 31.5),
            "z_swing_percent": league.get("z_swing_percent", 66.0),
        }
    row = rows.iloc[0]
    return {
        "whiff_percent": float(
            row.get("whiff_percent") or league.get("whiff_percent", 25.0)
        ),
        "zone_percent": float(
            row.get("zone_percent") or league.get("zone_percent", 49.5)
        ),
        "o_swing_percent": float(
            row.get("o_swing_percent") or league.get("o_swing_percent", 31.5)
        ),
        "z_swing_percent": float(
            row.get("z_swing_percent") or league.get("z_swing_percent", 66.0)
        ),
    }


def composite_k_percent(
    k_platoon: float,
    whiff_platoon: float,
    opp_k: float,
    opp_whiff: float,
    opp_chase: float,
    pitcher_skill: dict[str, float],
    cfg: dict[str, Any],
    league: dict[str, float],
) -> tuple[float, str]:
    """Blend platoon K%, lineup opponent K%, and whiff/chase skill into one K% estimate."""
    w_plat = float(cfg.get("platoon_weight", 0.35))
    w_match = float(cfg.get("matchup_weight", 0.45))
    w_whiff = float(cfg.get("whiff_weight", 0.20))
    chase_weight = float(cfg.get("chase_interaction_weight", 0.08))

    league_k = float(league.get("k_percent", 22.5))
    league_whiff = float(league.get("whiff_percent", 25.0))
    league_chase = float(league.get("o_swing_percent", 31.5))

    k_whiff_skill = league_k * (whiff_platoon / league_whiff) if league_whiff else league_k
    k_blend = w_plat * k_platoon + w_match * opp_k + w_whiff * k_whiff_skill

    chase_edge = (opp_chase - league_chase) / 100.0
    whiff_edge = (pitcher_skill["whiff_percent"] - league_whiff) / 100.0
    chase_adj = 1.0 + chase_weight * chase_edge * whiff_edge
    k_final = k_blend * chase_adj

    detail = (
        f"plat={k_platoon:.1f}% opp={opp_k:.1f}% whiff={k_whiff_skill:.1f}% "
        f"→ {k_final:.1f}% (w {w_plat:.0%}/{w_match:.0%}/{w_whiff:.0%}, chase×{chase_adj:.3f})"
    )
    return k_final, detail


def parse_lineup_batter_ids(raw: object) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    ids: list[int] = []
    for part in text.split("|"):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids
