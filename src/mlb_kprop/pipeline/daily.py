from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd
import yaml

from mlb_kprop.features.build import build_features
from mlb_kprop.features.merge import merge_pitcher_features
from mlb_kprop.features.validate import (
    print_validation_report,
    validate_daily_data,
    write_validation_report,
)
from mlb_kprop.mlb.starters import sync_starters_from_mlb
from mlb_kprop.odds.fetch import fetch_oddstrader_lines
from mlb_kprop.odds.oddstrader import DEFAULT_CONFIG_PATH as DEFAULT_ODDSTRADER_CONFIG
from mlb_kprop.projections.score import score_projections
from mlb_kprop.projections.value import value_props
from mlb_kprop.savant.fetch import fetch_all_sources, write_fetch_manifest
from mlb_kprop.tracker.ledger import track_performance

DEFAULT_MLB_CONFIG = Path("config/mlb_defaults.yaml")
DEFAULT_PROJECTION_CONFIG = Path("config/projection_defaults.yaml")
DEFAULT_VALUE_CONFIG = Path("config/value_defaults.yaml")


@dataclass(frozen=True)
class DailyOutputs:
    projections_csv: Path | None
    value_csv: Path | None
    value_plays: int
    tracker_summary: Path | None = None


def _print_value_picks(value_csv: Path, value_config: Path) -> int:
    with value_config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    min_edge = float((cfg or {}).get("min_edge", 0.03))
    df = pd.read_csv(value_csv)
    value_plays = int((df["pick"] != "PASS").sum())
    print(f"  {value_csv} ({value_plays} plays with edge >= {min_edge})")
    if value_plays:
        cols = ["player_name", "pick", "book_line", "lineup_source", "edge_over", "edge_under"]
        picks = df[df["pick"] != "PASS"][[c for c in cols if c in df.columns]]
        print(picks.to_string(index=False))
    return value_plays


def run_lineup_refresh(
    run_date: Date,
    out_dir: Path = Path("reports"),
    lines_root: Path = Path("data/lines"),
    starters_root: Path = Path("data/starters"),
    oddstrader_config: Path = DEFAULT_ODDSTRADER_CONFIG,
    mlb_config: Path = DEFAULT_MLB_CONFIG,
    projection_config: Path = DEFAULT_PROJECTION_CONFIG,
    value_config: Path = DEFAULT_VALUE_CONFIG,
    fetch_odds: bool = True,
    run_mode: str = "confirmed",
    record_tracker_picks: bool = True,
) -> DailyOutputs:
    """
    Afternoon refresh: re-sync lineups, re-score, refresh odds/lines, re-run EV.

    Skips Savant download and feature rebuild (uses morning processed data).
    Day games may already be Live/Final — those are filtered in confirmed mode.
    """
    print(f"Running lineup refresh for {run_date.isoformat()} (mode={run_mode})")

    if fetch_odds:
        print("\nStep 1: Refresh strikeout O/U lines from OddsTrader...")
        odds_outputs = fetch_oddstrader_lines(
            run_date=run_date,
            lines_root=lines_root,
            config_path=oddstrader_config,
            overwrite=True,
        )
        print(
            f"  {odds_outputs.lines_csv} "
            f"({odds_outputs.row_count} rows, {odds_outputs.sportsbook})"
        )

    print("\nStep 2: Sync probable pitchers + lineups from MLB API...")
    starters_outputs = sync_starters_from_mlb(
        run_date=run_date,
        starters_root=starters_root,
        config_path=mlb_config,
        overwrite=True,
    )
    lineup_rows = pd.read_csv(starters_outputs.starters_csv)
    confirmed = int((lineup_rows.get("lineup_source", "") == "lineup").sum())
    print(
        f"  {starters_outputs.starters_csv} "
        f"({starters_outputs.row_count} pitchers, {confirmed} confirmed lineups)"
    )

    print("\nStep 3: Score fair K projections (workload BF + shrinkage + park)...")
    proj_outputs = score_projections(
        run_date=run_date,
        starters_root=starters_root,
        reports_root=out_dir,
        config_path=projection_config,
    )
    projections_csv = proj_outputs.projections_csv
    print(f"  {projections_csv}")

    value_csv: Path | None = None
    value_plays = 0
    tracker_summary: Path | None = None

    if fetch_odds:
        print(f"\nStep 4: Value props (mode={run_mode})...")
        val_outputs = value_props(
            run_date=run_date,
            reports_root=out_dir,
            lines_root=lines_root,
            config_path=value_config,
            run_mode=run_mode,
        )
        value_csv = val_outputs.value_csv
        value_plays = _print_value_picks(value_csv, value_config)

        print("\nStep 5: Track pick performance (record + grade)...")
        tracker_outputs = track_performance(
            run_date=run_date,
            reports_root=out_dir,
            value_path=value_csv,
            record_today=record_tracker_picks,
        )
        tracker_summary = tracker_outputs.summary_txt
        print(
            f"  {tracker_outputs.ledger_csv} "
            f"(+{tracker_outputs.picks_recorded} new, "
            f"graded {tracker_outputs.picks_graded}, "
            f"{tracker_outputs.pending_count} pending)"
        )

    print("\nLineup refresh complete.")
    return DailyOutputs(
        projections_csv=projections_csv,
        value_csv=value_csv,
        value_plays=value_plays,
        tracker_summary=tracker_summary,
    )


def run_daily(
    run_date: Date,
    out_dir: Path,
    min_pitches_per_pitch_type_row: int = 100,
    lines_root: Path = Path("data/lines"),
    starters_root: Path = Path("data/starters"),
    oddstrader_config: Path = DEFAULT_ODDSTRADER_CONFIG,
    mlb_config: Path = DEFAULT_MLB_CONFIG,
    projection_config: Path = DEFAULT_PROJECTION_CONFIG,
    value_config: Path = DEFAULT_VALUE_CONFIG,
    fetch_odds: bool = True,
    run_model: bool = True,
    run_mode: str = "early",
    record_tracker_picks: bool | None = None,
) -> DailyOutputs:
    """
    Full daily pipeline:
    Savant -> features -> validate -> OddsTrader lines -> MLB starters -> projections -> EV.
    """
    if record_tracker_picks is None:
        record_tracker_picks = run_mode == "confirmed"

    print(f"Running daily pipeline for {run_date.isoformat()} (mode={run_mode})")

    print("\nStep 1: Download Baseball Savant CSVs...")
    results = fetch_all_sources(run_date=run_date)
    manifest_path = out_dir / f"savant_fetch_{run_date.isoformat()}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fetch_manifest(results, manifest_path)
    print(f"  Manifest: {manifest_path}")

    print("\nStep 2: Build cleaned feature tables...")
    build_outputs = build_features(
        run_date=run_date,
        min_pitches_per_pitch_type_row=min_pitches_per_pitch_type_row,
    )
    print(f"  {build_outputs.pitcher_platoon_pitch_type}")
    print(f"  {build_outputs.pitcher_custom}")
    print(f"  {build_outputs.batter_custom}")
    print(f"  {build_outputs.batter_hand_summary}")
    print(f"  {build_outputs.batter_platoon_pitch_type}")

    print("\nStep 3: Merge pitcher platoon + custom...")
    merge_outputs = merge_pitcher_features(run_date=run_date)
    print(f"  {merge_outputs.pitcher_merged_long}")
    print(f"  {merge_outputs.pitcher_split_summary}")
    print(f"  {merge_outputs.pitcher_skill}")

    print("\nStep 4: Validate raw + processed files...")
    report = validate_daily_data(
        run_date=run_date,
        reports_root=out_dir,
        min_pitches_per_pitch_type_row=min_pitches_per_pitch_type_row,
    )
    validation_path = out_dir / f"validation_{run_date.isoformat()}.txt"
    write_validation_report(report, validation_path)
    print_validation_report(report)
    print(f"  Report: {validation_path}")

    if not report.passed:
        raise SystemExit(1)

    if fetch_odds:
        print("\nStep 5: Fetch strikeout O/U lines from OddsTrader...")
        odds_outputs = fetch_oddstrader_lines(
            run_date=run_date,
            lines_root=lines_root,
            config_path=oddstrader_config,
            overwrite=True,
        )
        print(
            f"  {odds_outputs.lines_csv} "
            f"({odds_outputs.row_count} rows, {odds_outputs.sportsbook})"
        )

    projections_csv: Path | None = None
    value_csv: Path | None = None
    value_plays = 0
    tracker_summary: Path | None = None

    if run_model:
        print("\nStep 6: Sync probable pitchers + opp LHB% from MLB API...")
        starters_outputs = sync_starters_from_mlb(
            run_date=run_date,
            starters_root=starters_root,
            config_path=mlb_config,
            overwrite=True,
        )
        print(f"  {starters_outputs.starters_csv} ({starters_outputs.row_count} pitchers)")

        print("\nStep 7: Score fair K projections...")
        proj_outputs = score_projections(
            run_date=run_date,
            starters_root=starters_root,
            reports_root=out_dir,
            config_path=projection_config,
        )
        projections_csv = proj_outputs.projections_csv
        print(f"  {projections_csv}")

        if fetch_odds:
            print(f"\nStep 8: Value props (mode={run_mode})...")
            val_outputs = value_props(
                run_date=run_date,
                reports_root=out_dir,
                lines_root=lines_root,
                config_path=value_config,
                run_mode=run_mode,
            )
            value_csv = val_outputs.value_csv
            value_plays = _print_value_picks(value_csv, value_config)

            print("\nStep 9: Track pick performance (grade pending)...")
            tracker_outputs = track_performance(
                run_date=run_date,
                reports_root=out_dir,
                value_path=value_csv,
                record_today=record_tracker_picks,
            )
            tracker_summary = tracker_outputs.summary_txt
            print(
                f"  {tracker_outputs.ledger_csv} "
                f"(+{tracker_outputs.picks_recorded} new, "
                f"graded {tracker_outputs.picks_graded}, "
                f"{tracker_outputs.pending_count} pending)"
            )
            print(f"  {tracker_summary}")
        else:
            print("\nStep 8: Skipped value-props (--skip-odds).")

    print("\nDaily pipeline complete.")
    return DailyOutputs(
        projections_csv=projections_csv,
        value_csv=value_csv,
        value_plays=value_plays,
        tracker_summary=tracker_summary,
    )
