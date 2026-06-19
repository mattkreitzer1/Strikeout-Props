from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlb_kprop.projections.lines import load_lines
from mlb_kprop.projections.names import match_player_name

DEFAULT_CONFIG_PATH = Path("config/value_defaults.yaml")


@dataclass(frozen=True)
class ValueOutputs:
    value_csv: Path


def load_value_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied win probability (includes vig)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_to_decimal_odds(odds: float) -> float:
    """Profit multiplier on a $1 stake (decimal odds)."""
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def _norm_cdf(x: float, mean: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("k_sigma must be positive")
    z = (x - mean) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def model_prob_over(book_line: float, fair_k: float, sigma: float) -> float:
    """
    P(strikeouts > book_line) using a normal distribution around fair_k.

    For a 5.5 line, over wins at 6+ K; we use P(X > 5.5) as a continuous approximation.
    """
    return 1.0 - _norm_cdf(book_line, fair_k, sigma)


def model_prob_under(book_line: float, fair_k: float, sigma: float) -> float:
    """P(strikeouts under the book line)."""
    return _norm_cdf(book_line, fair_k, sigma)


def expected_value(model_prob: float, american_odds: float) -> float:
    """EV per $1 wagered: (win prob × decimal payout) − 1."""
    decimal = american_to_decimal_odds(american_odds)
    return model_prob * decimal - 1.0


def no_vig_probs(implied_over: float, implied_under: float) -> tuple[float, float]:
    total = implied_over + implied_under
    if total <= 0:
        return implied_over, implied_under
    return implied_over / total, implied_under / total


def _run_guards(cfg: dict[str, Any], run_mode: str) -> dict[str, Any]:
    section_key = "confirmed_run" if run_mode == "confirmed" else "early_run"
    section = cfg.get(section_key) or {}
    return {
        "min_ev": float(section.get("min_ev", cfg.get("min_ev", 0.0))),
        "max_ev": float(section.get("max_ev", cfg.get("max_ev", 0.35))),
        "max_fair_k_book_gap": float(
            section.get("max_fair_k_book_gap", cfg.get("max_fair_k_book_gap", 1.75))
        ),
        "require_lineup_source": section.get("require_lineup_source"),
        "skip_started_games": bool(section.get("skip_started_games", False)),
    }


def _model_sigma(base_sigma: float, proj: pd.Series, cfg: dict[str, Any]) -> float:
    sigma = base_sigma
    if str(proj.get("lineup_source", "")) == "default":
        sigma += float(cfg.get("sigma_lineup_default_add", 0.35))
    bf_std = proj.get("recent_bf_std")
    if pd.notna(bf_std):
        sigma += float(bf_std) * float(cfg.get("sigma_bf_std_multiplier", 0.08))
    sigma_min = float(cfg.get("k_sigma_min", base_sigma))
    sigma_max = float(cfg.get("k_sigma_max", base_sigma))
    return max(sigma_min, min(sigma_max, sigma))


def _match_projection_row(
    line_row: pd.Series,
    projections: pd.DataFrame,
) -> pd.Series | None:
    pid = line_row.get("player_id")
    if pd.notna(pid):
        matches = projections[projections["player_id"] == int(pid)]
        if len(matches) == 1:
            return matches.iloc[0]

    name = str(line_row["player_name"]).strip()
    candidates = projections["player_name"].astype(str).unique().tolist()
    matched = match_player_name(name, candidates)
    if matched:
        hits = projections[projections["player_name"] == matched]
        if len(hits) == 1:
            return hits.iloc[0]
    return None


def _pick_side(
    edge_over: float,
    edge_under: float,
    min_edge: float,
    ev_over: float,
    ev_under: float,
    fair_k: float,
    book_line: float,
    guards: dict[str, Any],
    lineup_source: str,
    game_status: str,
) -> tuple[str, str]:
    """Return (pick, pass_reason). pick is PASS when filtered out."""
    if guards.get("skip_started_games") and game_status in ("Live", "Final"):
        return "PASS", "game_started"

    required_lineup = guards.get("require_lineup_source")
    if required_lineup and lineup_source != required_lineup:
        return "PASS", f"lineup_{lineup_source or 'unknown'}"

    min_ev = float(guards.get("min_ev", 0.0))
    max_ev = float(guards.get("max_ev", 1.0))
    max_gap = float(guards.get("max_fair_k_book_gap", 99.0))

    candidates: list[tuple[str, float, float, float]] = []
    if edge_over >= min_edge:
        candidates.append(("OVER", edge_over, ev_over, abs(fair_k - book_line)))
    if edge_under >= min_edge:
        candidates.append(("UNDER", edge_under, ev_under, abs(fair_k - book_line)))

    if not candidates:
        return "PASS", "below_min_edge"

    pick, edge, ev, gap = max(candidates, key=lambda row: row[1])
    if ev < min_ev:
        return "PASS", f"below_min_ev({ev:.0%})"
    if ev > max_ev:
        return "PASS", f"ev_cap({ev:.0%})"
    if gap > max_gap:
        return "PASS", f"fair_book_gap({gap:.1f})"
    return pick, ""


def value_props(
    run_date: Date,
    reports_root: Path = Path("reports"),
    lines_root: Path = Path("data/lines"),
    lines_path: Path | None = None,
    projections_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_mode: str = "early",
) -> ValueOutputs:
    """
    Merge projections with book lines; compute model vs implied prob and EV.

    run_mode: "early" (morning) or "confirmed" (afternoon lineup refresh).
    """
    cfg = load_value_config(config_path)
    base_sigma = float(cfg.get("k_sigma", 2.25))
    min_edge = float(cfg.get("min_edge", 0.03))
    use_no_vig = bool(cfg.get("use_no_vig_implied", True))
    guards = _run_guards(cfg, run_mode)

    proj_path = projections_path or (
        reports_root / f"projections_{run_date.isoformat()}.csv"
    )
    if not proj_path.exists():
        raise FileNotFoundError(
            f"Missing {proj_path}. Run `python -m mlb_kprop score-projections` first."
        )

    projections = pd.read_csv(proj_path)
    lines = load_lines(run_date, lines_root=lines_root, lines_path=lines_path)

    if lines.empty:
        raise ValueError("Lines file has no rows (after removing blanks/examples).")

    rows: list[dict[str, object]] = []
    errors: list[str] = []

    for _, line_row in lines.iterrows():
        name = line_row["player_name"]
        proj = _match_projection_row(line_row, projections)
        if proj is None:
            errors.append(f"No projection for: {name} (run score-projections first)")
            continue

        book_line = float(line_row["book_line"])
        over_odds = float(line_row["over_odds"])
        under_odds = float(line_row["under_odds"])
        fair_k = float(proj["fair_k"])
        sigma = _model_sigma(base_sigma, proj, cfg)
        lineup_source = str(proj.get("lineup_source", ""))
        game_status = str(proj.get("game_status", ""))

        p_over = model_prob_over(book_line, fair_k, sigma)
        p_under = model_prob_under(book_line, fair_k, sigma)

        impl_over = american_to_implied_prob(over_odds)
        impl_under = american_to_implied_prob(under_odds)
        if use_no_vig:
            fair_impl_over, fair_impl_under = no_vig_probs(impl_over, impl_under)
        else:
            fair_impl_over, fair_impl_under = impl_over, impl_under

        edge_over = p_over - fair_impl_over
        edge_under = p_under - fair_impl_under
        ev_over = expected_value(p_over, over_odds)
        ev_under = expected_value(p_under, under_odds)
        pick, pass_reason = _pick_side(
            edge_over,
            edge_under,
            min_edge,
            ev_over,
            ev_under,
            fair_k,
            book_line,
            guards,
            lineup_source,
            game_status,
        )

        rows.append(
            {
                "player_id": proj["player_id"],
                "player_name": name,
                "fair_k": round(fair_k, 3),
                "fair_k_line": proj.get("fair_k_line", ""),
                "book_line": book_line,
                "batters_faced": proj.get("batters_faced", ""),
                "lineup_source": lineup_source,
                "game_status": game_status,
                "k_sigma": round(sigma, 3),
                "run_mode": run_mode,
                "model_p_over": round(p_over, 4),
                "model_p_under": round(p_under, 4),
                "implied_p_over": round(impl_over, 4),
                "implied_p_under": round(impl_under, 4),
                "no_vig_p_over": round(fair_impl_over, 4),
                "no_vig_p_under": round(fair_impl_under, 4),
                "edge_over": round(edge_over, 4),
                "edge_under": round(edge_under, 4),
                "ev_over": round(ev_over, 4),
                "ev_under": round(ev_under, 4),
                "over_odds": int(over_odds),
                "under_odds": int(under_odds),
                "pick": pick,
                "pass_reason": pass_reason or "",
            }
        )

    if errors:
        msg = "Value errors:\n" + "\n".join(f"  - {e}" for e in errors)
        if not rows:
            raise ValueError(msg)
        print(msg)

    if not rows:
        raise ValueError("No value rows produced.")

    out_df = pd.DataFrame(rows).sort_values("edge_over", ascending=False)
    reports_root.mkdir(parents=True, exist_ok=True)
    out_path = reports_root / f"value_{run_date.isoformat()}.csv"
    out_df.to_csv(out_path, index=False)

    return ValueOutputs(value_csv=out_path)
