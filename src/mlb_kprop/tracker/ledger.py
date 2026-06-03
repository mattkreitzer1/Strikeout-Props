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
class TrackerOutputs:
    ledger_csv: Path
    summary_txt: Path
    daily_rollup_csv: Path
    picks_recorded: int
    picks_graded: int
    pending_count: int


def load_tracker_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ledger_path_from_config(config: dict[str, Any]) -> Path:
    return Path(config.get("ledger_path", "data/tracker/ledger.csv"))


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


def build_summary_text(ledger: pd.DataFrame) -> str:
    graded = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    pending = ledger[ledger["result"] == "PENDING"]

    lines = ["MLB K prop tracker — performance history", ""]

    if graded.empty:
        lines.append("No graded plays yet.")
    else:
        wins = int((graded["result"] == "WIN").sum())
        losses = int((graded["result"] == "LOSS").sum())
        pushes = int((graded["result"] == "PUSH").sum())
        units = float(graded["profit_units"].sum())
        bets = wins + losses
        win_pct = (100.0 * wins / bets) if bets else 0.0
        roi = (100.0 * units / bets) if bets else 0.0

        lines.extend(
            [
                f"Graded plays: {len(graded)} ({wins}-{losses}-{pushes})",
                f"Win rate: {win_pct:.1f}%  |  Units: {units:+.2f}  |  ROI: {roi:+.1f}% (1u flat)",
                "",
                "Last 7 slate days:",
            ]
        )

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
    summary_path.write_text(build_summary_text(ledger), encoding="utf-8")

    rollup = build_daily_rollup(ledger)
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    rollup.to_csv(rollup_path, index=False)

    return TrackerOutputs(
        ledger_csv=ledger_path,
        summary_txt=summary_path,
        daily_rollup_csv=rollup_path,
        picks_recorded=picks_recorded,
        picks_graded=picks_graded,
        pending_count=pending_count,
    )
