from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlb_kprop.projections.value import american_to_decimal_odds
from mlb_kprop.tracker.stats import MlbGameStatsClient

DEFAULT_CONFIG_PATH = Path("config/tracker_defaults.yaml")

LEDGER_COLUMNS = [
    "slate_date",
    "player_id",
    "player_name",
    "pick",
    "book_line",
    "fair_k",
    "pick_edge",
    "pick_ev",
    "pick_odds",
    "actual_k",
    "result",
    "profit_units",
    "recorded_at",
    "graded_at",
]


@dataclass(frozen=True)
class EvBucket:
    label: str
    min_ev: float
    max_ev: float | None


@dataclass(frozen=True)
class TrackerOutputs:
    ledger_csv: Path
    summary_txt: Path
    daily_rollup_csv: Path
    ev_rollup_csv: Path
    picks_recorded: int
    picks_graded: int
    pending_count: int


def load_tracker_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ledger_path_from_config(config: dict[str, Any]) -> Path:
    return Path(config.get("ledger_path", "data/tracker/ledger.csv"))


def ev_buckets_from_config(config: dict[str, Any]) -> list[EvBucket]:
    raw = config.get("ev_buckets")
    if not raw:
        return [
            EvBucket("Thin (<5% EV)", 0.0, 0.05),
            EvBucket("Moderate (5-15% EV)", 0.05, 0.15),
            EvBucket("Strong (15-25% EV)", 0.15, 0.25),
            EvBucket("Elite (25%+ EV)", 0.25, None),
        ]
    buckets: list[EvBucket] = []
    for entry in raw:
        max_ev = entry.get("max_ev")
        buckets.append(
            EvBucket(
                label=str(entry["label"]),
                min_ev=float(entry["min_ev"]),
                max_ev=None if max_ev is None else float(max_ev),
            )
        )
    return buckets


def assign_ev_bucket(ev: float, buckets: list[EvBucket]) -> str:
    for bucket in buckets:
        if ev >= bucket.min_ev and (bucket.max_ev is None or ev < bucket.max_ev):
            return bucket.label
    return "Other"


def _bucket_stats(df: pd.DataFrame) -> dict[str, float | int]:
    wins = int((df["result"] == "WIN").sum())
    losses = int((df["result"] == "LOSS").sum())
    pushes = int((df["result"] == "PUSH").sum())
    units = float(df["profit_units"].sum())
    bets = wins + losses
    win_pct = (100.0 * wins / bets) if bets else 0.0
    roi = (100.0 * units / bets) if bets else 0.0
    avg_ev = float(df["pick_ev"].mean()) if not df.empty else 0.0
    return {
        "picks": len(df),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "units": units,
        "win_pct": win_pct,
        "roi_pct": roi,
        "avg_ev": avg_ev,
    }


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_csv(path)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ("slate_date", "player_name", "pick", "result", "recorded_at", "graded_at"):
        df[col] = df[col].apply(lambda v: pd.NA if pd.isna(v) else str(v))
    return df[LEDGER_COLUMNS]


def _save_ledger(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["slate_date", "player_name"]).to_csv(path, index=False)


def _pick_metrics(row: pd.Series) -> tuple[float, float, int]:
    pick = str(row["pick"])
    if pick == "OVER":
        return float(row["edge_over"]), float(row["ev_over"]), int(row["over_odds"])
    if pick == "UNDER":
        return float(row["edge_under"]), float(row["ev_under"]), int(row["under_odds"])
    raise ValueError(f"Not a tracked pick: {pick}")


def grade_prop(actual_k: float, book_line: float, pick: str) -> str:
    """Return WIN, LOSS, or PUSH for a strikeout O/U prop."""
    if pick == "OVER":
        if actual_k > book_line:
            return "WIN"
        if actual_k < book_line:
            return "LOSS"
        return "PUSH"
    if pick == "UNDER":
        if actual_k < book_line:
            return "WIN"
        if actual_k > book_line:
            return "LOSS"
        return "PUSH"
    raise ValueError(f"Unknown pick: {pick}")


def profit_units(result: str, american_odds: int) -> float:
    if result == "WIN":
        return american_to_decimal_odds(float(american_odds)) - 1.0
    if result == "LOSS":
        return -1.0
    return 0.0


def record_picks_from_value(
    run_date: Date,
    value_csv: Path,
    ledger_path: Path,
) -> int:
    """Append flagged plays from value_<date>.csv to the ledger."""
    if not value_csv.exists():
        raise FileNotFoundError(f"Missing {value_csv}")

    value_df = pd.read_csv(value_csv)
    picks = value_df[value_df["pick"].isin(["OVER", "UNDER"])].copy()
    ledger = _load_ledger(ledger_path)
    existing_keys = set(
        zip(
            ledger["slate_date"].astype(str),
            ledger["player_id"].astype(str),
            ledger["pick"].astype(str),
            strict=False,
        )
        if not ledger.empty
        else set()
    )

    now = datetime.now().isoformat(timespec="seconds")
    new_rows: list[dict[str, object]] = []
    slate = run_date.isoformat()

    for _, row in picks.iterrows():
        key = (slate, str(int(row["player_id"])), str(row["pick"]))
        if key in existing_keys:
            continue
        edge, ev, odds = _pick_metrics(row)
        new_rows.append(
            {
                "slate_date": slate,
                "player_id": int(row["player_id"]),
                "player_name": row["player_name"],
                "pick": row["pick"],
                "book_line": float(row["book_line"]),
                "fair_k": float(row["fair_k"]),
                "pick_edge": round(edge, 4),
                "pick_ev": round(ev, 4),
                "pick_odds": odds,
                "actual_k": pd.NA,
                "result": "PENDING",
                "profit_units": pd.NA,
                "recorded_at": now,
                "graded_at": pd.NA,
            }
        )

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        _save_ledger(ledger_path, ledger)
    return len(new_rows)


def grade_pending_picks(
    ledger_path: Path,
    stats_client: MlbGameStatsClient | None = None,
    through_date: Date | None = None,
) -> int:
    """Fill in actual K and WIN/LOSS for PENDING rows once games are final."""
    client = stats_client or MlbGameStatsClient()
    ledger = _load_ledger(ledger_path)
    if ledger.empty:
        return 0

    pending = ledger["result"] == "PENDING"
    if through_date is not None:
        pending &= ledger["slate_date"].astype(str) <= through_date.isoformat()

    graded_count = 0
    now = datetime.now().isoformat(timespec="seconds")

    for idx in ledger[pending].index:
        row = ledger.loc[idx]
        slate = Date.fromisoformat(str(row["slate_date"]))
        player_id = int(row["player_id"])

        actual_k = client.pitcher_strikeouts_on_date(player_id, slate)
        if actual_k is None:
            continue

        result = grade_prop(float(actual_k), float(row["book_line"]), str(row["pick"]))
        units = profit_units(result, int(row["pick_odds"]))

        ledger.at[idx, "actual_k"] = actual_k
        ledger.at[idx, "result"] = result
        ledger.at[idx, "profit_units"] = round(units, 4)
        ledger.at[idx, "graded_at"] = now
        graded_count += 1

    if graded_count:
        _save_ledger(ledger_path, ledger)
    return graded_count


def build_ev_bucket_rollup(
    ledger: pd.DataFrame,
    buckets: list[EvBucket],
) -> pd.DataFrame:
    graded = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    columns = [
        "ev_bucket",
        "min_ev",
        "max_ev",
        "picks",
        "wins",
        "losses",
        "pushes",
        "units",
        "win_pct",
        "roi_pct",
        "avg_pick_ev",
    ]
    if graded.empty:
        return pd.DataFrame(columns=columns)

    graded["ev_bucket"] = graded["pick_ev"].astype(float).apply(
        lambda ev: assign_ev_bucket(ev, buckets)
    )

    rows: list[dict[str, object]] = []
    for bucket in buckets:
        subset = graded[graded["ev_bucket"] == bucket.label]
        if subset.empty:
            continue
        stats = _bucket_stats(subset)
        rows.append(
            {
                "ev_bucket": bucket.label,
                "min_ev": bucket.min_ev,
                "max_ev": bucket.max_ev,
                "picks": stats["picks"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "pushes": stats["pushes"],
                "units": round(stats["units"], 4),
                "win_pct": round(stats["win_pct"], 1),
                "roi_pct": round(stats["roi_pct"], 1),
                "avg_pick_ev": round(stats["avg_ev"], 4),
            }
        )

    other = graded[~graded["ev_bucket"].isin([b.label for b in buckets])]
    if not other.empty:
        stats = _bucket_stats(other)
        rows.append(
            {
                "ev_bucket": "Other",
                "min_ev": pd.NA,
                "max_ev": pd.NA,
                "picks": stats["picks"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "pushes": stats["pushes"],
                "units": round(stats["units"], 4),
                "win_pct": round(stats["win_pct"], 1),
                "roi_pct": round(stats["roi_pct"], 1),
                "avg_pick_ev": round(stats["avg_ev"], 4),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_summary_text(
    ledger: pd.DataFrame,
    buckets: list[EvBucket] | None = None,
) -> str:
    graded = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    pending = ledger[ledger["result"] == "PENDING"]
    bucket_defs = buckets or ev_buckets_from_config({})

    lines = ["MLB K prop tracker — performance history", ""]

    if graded.empty:
        lines.append("No graded plays yet.")
    else:
        stats = _bucket_stats(graded)
        lines.extend(
            [
                f"Graded plays: {stats['picks']} "
                f"({stats['wins']}-{stats['losses']}-{stats['pushes']})",
                f"Win rate: {stats['win_pct']:.1f}%  |  "
                f"Units: {stats['units']:+.2f}  |  "
                f"ROI: {stats['roi_pct']:+.1f}% (1u flat)",
                "",
                "By EV bucket:",
            ]
        )

        ev_rollup = build_ev_bucket_rollup(graded, bucket_defs)
        for _, row in ev_rollup.iterrows():
            lines.append(
                f"  {row['ev_bucket']}: {int(row['wins'])}-{int(row['losses'])} "
                f"({int(row['picks'])} picks, {row['units']:+.2f}u, "
                f"{row['roi_pct']:+.1f}% ROI, avg EV {row['avg_pick_ev']:.1%})"
            )

        lines.extend(["", "Last 7 slate days:"])

        rollup = (
            graded.groupby("slate_date", as_index=False)
            .agg(
                picks=("result", "count"),
                wins=("result", lambda s: int((s == "WIN").sum())),
                losses=("result", lambda s: int((s == "LOSS").sum())),
                units=("profit_units", "sum"),
            )
            .sort_values("slate_date", ascending=False)
            .head(7)
        )
        for _, day in rollup.iterrows():
            lines.append(
                f"  {day['slate_date']}: {int(day['wins'])}-{int(day['losses'])} "
                f"({int(day['picks'])} picks, {day['units']:+.2f}u)"
            )

    if not pending.empty:
        lines.extend(["", f"Pending (awaiting final stats): {len(pending)}"])

    lines.append("")
    return "\n".join(lines)


def build_daily_rollup(ledger: pd.DataFrame) -> pd.DataFrame:
    graded = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    if graded.empty:
        return pd.DataFrame(
            columns=["slate_date", "picks", "wins", "losses", "pushes", "units", "roi_pct"]
        )

    rollup = (
        graded.groupby("slate_date", as_index=False)
        .agg(
            picks=("result", "count"),
            wins=("result", lambda s: int((s == "WIN").sum())),
            losses=("result", lambda s: int((s == "LOSS").sum())),
            pushes=("result", lambda s: int((s == "PUSH").sum())),
            units=("profit_units", "sum"),
        )
        .sort_values("slate_date")
    )
    bets = rollup["wins"] + rollup["losses"]
    rollup["roi_pct"] = (100.0 * rollup["units"] / bets.replace(0, pd.NA)).round(1)
    return rollup


def track_performance(
    run_date: Date,
    reports_root: Path = Path("reports"),
    value_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    record_today: bool = True,
    grade_pending: bool = True,
) -> TrackerOutputs:
    """
    Record today's flagged picks, grade older pending rows, write summary files.
    """
    cfg = load_tracker_config(config_path)
    ledger_path = ledger_path_from_config(cfg)
    summary_path = Path(cfg.get("summary_path", "data/tracker/summary.txt"))
    rollup_path = Path(cfg.get("daily_rollup_path", "data/tracker/daily_rollup.csv"))
    ev_rollup_path = Path(cfg.get("ev_rollup_path", "data/tracker/ev_rollup.csv"))
    buckets = ev_buckets_from_config(cfg)

    value_csv = value_path or reports_root / f"value_{run_date.isoformat()}.csv"
    picks_recorded = 0
    if record_today:
        picks_recorded = record_picks_from_value(run_date, value_csv, ledger_path)

    picks_graded = 0
    if grade_pending:
        picks_graded = grade_pending_picks(ledger_path, through_date=run_date)

    ledger = _load_ledger(ledger_path)
    pending_count = int((ledger["result"] == "PENDING").sum())

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary_text(ledger, buckets), encoding="utf-8")

    rollup = build_daily_rollup(ledger)
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    rollup.to_csv(rollup_path, index=False)

    ev_rollup = build_ev_bucket_rollup(ledger, buckets)
    ev_rollup_path.parent.mkdir(parents=True, exist_ok=True)
    ev_rollup.to_csv(ev_rollup_path, index=False)

    return TrackerOutputs(
        ledger_csv=ledger_path,
        summary_txt=summary_path,
        daily_rollup_csv=rollup_path,
        ev_rollup_csv=ev_rollup_path,
        picks_recorded=picks_recorded,
        picks_graded=picks_graded,
        pending_count=pending_count,
    )
