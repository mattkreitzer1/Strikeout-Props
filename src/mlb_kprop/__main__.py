from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd

from mlb_kprop.pipeline.daily import run_daily, run_lineup_refresh
from mlb_kprop.savant.fetch import fetch_all_sources, write_fetch_manifest
from mlb_kprop.features.build import build_features
from mlb_kprop.features.merge import merge_pitcher_features
from mlb_kprop.features.validate import (
    print_validation_report,
    validate_daily_data,
    write_validation_report,
)
from mlb_kprop.projections.score import score_projections
from mlb_kprop.projections.starters import write_starters_template
from mlb_kprop.projections.lines import write_lines_template
from mlb_kprop.projections.value import value_props
from mlb_kprop.odds.fetch import fetch_oddstrader_lines
from mlb_kprop.notify.email import send_daily_digest
from mlb_kprop.tracker.ledger import track_performance


@dataclass(frozen=True)
class CliArgs:
    command: str
    date: Date
    out_dir: Path
    config_path: Path
    data_root: Path
    processed_root: Path
    min_pitches: int
    starters_root: Path = Path("data/starters")
    starters_file: Path | None = None
    projection_config: Path = Path("config/projection_defaults.yaml")
    overwrite_starters: bool = False
    lines_root: Path = Path("data/lines")
    lines_file: Path | None = None
    value_config: Path = Path("config/value_defaults.yaml")
    overwrite_lines: bool = False
    projections_file: Path | None = None
    oddstrader_config: Path = Path("config/oddstrader.yaml")
    skip_odds: bool = False
    skip_model: bool = False
    mlb_config: Path = Path("config/mlb_defaults.yaml")
    email_config: Path = Path("config/email_defaults.yaml")
    tracker_config: Path = Path("config/tracker_defaults.yaml")
    no_record: bool = False
    no_grade: bool = False
    run_mode: str = "early"
    dry_run: bool = False


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="mlb_kprop",
        description="Daily pipeline for pitcher K prop projections.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--date",
            type=Date.fromisoformat,
            default=Date.today(),
            help="Run date (YYYY-MM-DD). Used for the output folder name.",
        )
        subparser.add_argument(
            "--config",
            type=Path,
            default=Path("config/savant_sources.yaml"),
            help="YAML file listing Savant URLs to download.",
        )
        subparser.add_argument(
            "--data-root",
            type=Path,
            default=Path("data/raw"),
            help="Root folder for downloaded Savant CSVs.",
        )
        subparser.add_argument(
            "--processed-root",
            type=Path,
            default=Path("data/processed"),
            help="Root folder for cleaned, processed CSVs.",
        )
        subparser.add_argument(
            "--min-pitches",
            type=int,
            default=100,
            help="Minimum `pitches` per (pitcher, pitch_type, platoon) row.",
        )

    fetch_parser = subparsers.add_parser(
        "fetch-savant",
        help="Download Savant CSVs (platoon splits + custom boards from config).",
    )
    add_shared_options(fetch_parser)

    build_parser = subparsers.add_parser(
        "build-features",
        help="Build cleaned feature tables from raw Savant CSVs.",
    )
    add_shared_options(build_parser)

    merge_parser = subparsers.add_parser(
        "merge-features",
        help="Merge platoon + custom pitcher tables into model-ready files.",
    )
    add_shared_options(merge_parser)

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Run automated sanity checks on raw + processed files for a date.",
    )
    add_shared_options(validate_parser)
    validate_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder for validation report output.",
    )

    run_daily_parser = subparsers.add_parser(
        "run-daily",
        help="Savant fetch + features + validate + OddsTrader K lines.",
    )
    add_shared_options(run_daily_parser)
    run_daily_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder for manifests and validation reports.",
    )
    run_daily_parser.add_argument(
        "--lines-root",
        type=Path,
        default=Path("data/lines"),
        help="Folder for daily sportsbook line CSVs.",
    )
    run_daily_parser.add_argument(
        "--oddstrader-config",
        type=Path,
        default=Path("config/oddstrader.yaml"),
        help="OddsTrader market id, sportsbook paid id, URLs.",
    )
    run_daily_parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip OddsTrader fetch and value-props.",
    )
    run_daily_parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip MLB starters, projections, and value-props.",
    )
    run_daily_parser.add_argument(
        "--starters-root",
        type=Path,
        default=Path("data/starters"),
        help="Folder for daily starter CSVs.",
    )
    run_daily_parser.add_argument(
        "--mlb-config",
        type=Path,
        default=Path("config/mlb_defaults.yaml"),
        help="MLB API defaults (opp_lhb fallback, throttle).",
    )
    run_daily_parser.add_argument(
        "--projection-config",
        type=Path,
        default=Path("config/projection_defaults.yaml"),
        help="Fair K projection defaults.",
    )
    run_daily_parser.add_argument(
        "--value-config",
        type=Path,
        default=Path("config/value_defaults.yaml"),
        help="EV / edge defaults.",
    )
    run_daily_parser.add_argument(
        "--run-mode",
        choices=("early", "confirmed"),
        default="early",
        help="early = morning (stricter EV caps); confirmed = post-lineup refresh.",
    )

    refresh_parser = subparsers.add_parser(
        "run-lineup-refresh",
        help="Afternoon refresh: lineups + projections + EV (skips Savant).",
    )
    add_shared_options(refresh_parser)
    refresh_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder for projections and value CSVs.",
    )
    refresh_parser.add_argument(
        "--lines-root",
        type=Path,
        default=Path("data/lines"),
        help="Folder for daily sportsbook line CSVs.",
    )
    refresh_parser.add_argument(
        "--starters-root",
        type=Path,
        default=Path("data/starters"),
        help="Folder for daily starter CSVs.",
    )
    refresh_parser.add_argument(
        "--oddstrader-config",
        type=Path,
        default=Path("config/oddstrader.yaml"),
        help="OddsTrader market id, sportsbook paid id, URLs.",
    )
    refresh_parser.add_argument(
        "--mlb-config",
        type=Path,
        default=Path("config/mlb_defaults.yaml"),
        help="MLB API defaults.",
    )
    refresh_parser.add_argument(
        "--projection-config",
        type=Path,
        default=Path("config/projection_defaults.yaml"),
        help="Fair K projection defaults.",
    )
    refresh_parser.add_argument(
        "--value-config",
        type=Path,
        default=Path("config/value_defaults.yaml"),
        help="EV / edge defaults.",
    )
    refresh_parser.add_argument(
        "--run-mode",
        choices=("early", "confirmed"),
        default="confirmed",
        help="Use confirmed mode (requires posted lineups for picks).",
    )
    refresh_parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip OddsTrader refresh and value-props.",
    )
    refresh_parser.add_argument(
        "--no-track-record",
        action="store_true",
        help="Grade pending picks only; do not record new picks.",
    )

    init_starters_parser = subparsers.add_parser(
        "init-starters",
        help="Create data/starters/<date>.csv template for today's pitchers.",
    )
    add_shared_options(init_starters_parser)
    init_starters_parser.add_argument(
        "--starters-root",
        type=Path,
        default=Path("data/starters"),
        help="Folder for daily starter input CSVs.",
    )
    init_starters_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace starters file if it already exists.",
    )

    score_parser = subparsers.add_parser(
        "score-projections",
        help="Fair K projections from platoon splits + starters file.",
    )
    add_shared_options(score_parser)
    score_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder for projections CSV output.",
    )
    score_parser.add_argument(
        "--starters-root",
        type=Path,
        default=Path("data/starters"),
        help="Folder containing starters/<date>.csv.",
    )
    score_parser.add_argument(
        "--starters-file",
        type=Path,
        default=None,
        help="Optional path to starters CSV (default: data/starters/<date>.csv).",
    )
    score_parser.add_argument(
        "--projection-config",
        type=Path,
        default=Path("config/projection_defaults.yaml"),
        help="YAML defaults (batters faced, rounding, etc.).",
    )

    init_lines_parser = subparsers.add_parser(
        "init-lines",
        help="Create data/lines/<date>.csv template for FanDuel (or other) K props.",
    )
    add_shared_options(init_lines_parser)
    init_lines_parser.add_argument(
        "--lines-root",
        type=Path,
        default=Path("data/lines"),
        help="Folder for daily sportsbook line CSVs.",
    )
    init_lines_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace lines file if it already exists.",
    )

    value_parser = subparsers.add_parser(
        "value-props",
        help="Compare fair K to book lines; output edge and EV.",
    )
    add_shared_options(value_parser)
    value_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder for value CSV output.",
    )
    value_parser.add_argument(
        "--lines-root",
        type=Path,
        default=Path("data/lines"),
        help="Folder containing lines/<date>.csv.",
    )
    value_parser.add_argument(
        "--lines-file",
        type=Path,
        default=None,
        help="Optional path to lines CSV (default: data/lines/<date>.csv).",
    )
    value_parser.add_argument(
        "--projections-file",
        type=Path,
        default=None,
        help="Optional path to projections CSV.",
    )
    value_parser.add_argument(
        "--value-config",
        type=Path,
        default=Path("config/value_defaults.yaml"),
        help="YAML defaults (k_sigma, min_edge, etc.).",
    )

    fetch_odds_parser = subparsers.add_parser(
        "fetch-odds",
        help="Download pitcher K O/U lines from OddsTrader into data/lines/<date>.csv.",
    )
    add_shared_options(fetch_odds_parser)
    fetch_odds_parser.add_argument(
        "--lines-root",
        type=Path,
        default=Path("data/lines"),
        help="Folder for daily sportsbook line CSVs.",
    )
    fetch_odds_parser.add_argument(
        "--oddstrader-config",
        type=Path,
        default=Path("config/oddstrader.yaml"),
        help="OddsTrader market id, sportsbook paid id, URLs.",
    )
    fetch_odds_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace lines file if it already exists.",
    )

    send_email_parser = subparsers.add_parser(
        "send-email",
        help="Email flagged plays from reports/value_<date>.csv.",
    )
    add_shared_options(send_email_parser)
    send_email_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder containing value_<date>.csv.",
    )
    send_email_parser.add_argument(
        "--email-config",
        type=Path,
        default=Path("config/email_defaults.yaml"),
        help="Email subject and formatting defaults.",
    )
    send_email_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the email instead of sending.",
    )
    send_email_parser.add_argument(
        "--run-mode",
        choices=("early", "confirmed"),
        default="early",
        help="Controls subject line (early vs confirmed lineups).",
    )

    track_parser = subparsers.add_parser(
        "track-performance",
        help="Record flagged picks and grade results (updates data/tracker/).",
    )
    add_shared_options(track_parser)
    track_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Folder containing value_<date>.csv.",
    )
    track_parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path("config/tracker_defaults.yaml"),
        help="Ledger and summary output paths.",
    )
    track_parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip recording today's picks (grade pending only).",
    )
    track_parser.add_argument(
        "--no-grade",
        action="store_true",
        help="Skip grading pending picks (record today only).",
    )

    ns = parser.parse_args()
    out_dir = getattr(ns, "out_dir", Path("reports"))
    return CliArgs(
        command=ns.command,
        date=ns.date,
        out_dir=out_dir,
        config_path=ns.config,
        data_root=ns.data_root,
        processed_root=ns.processed_root,
        min_pitches=ns.min_pitches,
        starters_root=getattr(ns, "starters_root", Path("data/starters")),
        starters_file=getattr(ns, "starters_file", None),
        projection_config=getattr(
            ns, "projection_config", Path("config/projection_defaults.yaml")
        ),
        overwrite_starters=getattr(ns, "overwrite", False),
        lines_root=getattr(ns, "lines_root", Path("data/lines")),
        lines_file=getattr(ns, "lines_file", None),
        value_config=getattr(ns, "value_config", Path("config/value_defaults.yaml")),
        overwrite_lines=getattr(ns, "overwrite", False),
        projections_file=getattr(ns, "projections_file", None),
        oddstrader_config=getattr(
            ns, "oddstrader_config", Path("config/oddstrader.yaml")
        ),
        skip_odds=getattr(ns, "skip_odds", False),
        skip_model=getattr(ns, "skip_model", False),
        mlb_config=getattr(ns, "mlb_config", Path("config/mlb_defaults.yaml")),
        email_config=getattr(ns, "email_config", Path("config/email_defaults.yaml")),
        tracker_config=getattr(
            ns, "tracker_config", Path("config/tracker_defaults.yaml")
        ),
        no_record=getattr(ns, "no_record", False),
        no_grade=getattr(ns, "no_grade", False),
        run_mode=getattr(ns, "run_mode", "early"),
        dry_run=getattr(ns, "dry_run", False),
    )


def _run_fetch_savant(args: CliArgs) -> None:
    results = fetch_all_sources(
        run_date=args.date,
        config_path=args.config_path,
        data_root=args.data_root,
    )
    manifest_path = args.data_root / args.date.isoformat() / "_manifest.csv"
    write_fetch_manifest(results, manifest_path)
    print(f"Manifest: {manifest_path}")

def _run_build_features(args: CliArgs) -> None:
    outputs = build_features(
        run_date=args.date,
        raw_root=args.data_root,
        processed_root=args.processed_root,
        min_pitches_per_pitch_type_row=args.min_pitches,
    )
    print("Wrote processed files:")
    print(f"  {outputs.pitcher_platoon_pitch_type}")
    print(f"  {outputs.pitcher_custom}")
    print(f"  {outputs.batter_custom}")
    print(f"  {outputs.batter_hand_summary}")
    print(f"  {outputs.batter_platoon_pitch_type}")


def _run_validate_data(args: CliArgs) -> None:
    report = validate_daily_data(
        run_date=args.date,
        config_path=args.config_path,
        raw_root=args.data_root,
        processed_root=args.processed_root,
        reports_root=args.out_dir,
        min_pitches_per_pitch_type_row=args.min_pitches,
    )
    print_validation_report(report)
    report_path = (
        args.out_dir / f"validation_{args.date.isoformat()}.txt"
    )
    write_validation_report(report, report_path)
    print(f"\nReport: {report_path}")
    if not report.passed:
        raise SystemExit(1)


def _run_init_starters(args: CliArgs) -> None:
    path = write_starters_template(
        run_date=args.date,
        starters_root=args.starters_root,
        overwrite=args.overwrite_starters,
    )
    print(f"Starters template: {path}")
    print("Edit the file: add today's starters, set opp_lhb_pct, remove the example row.")


def _run_fetch_odds(args: CliArgs) -> None:
    outputs = fetch_oddstrader_lines(
        run_date=args.date,
        lines_root=args.lines_root,
        config_path=args.oddstrader_config,
        overwrite=args.overwrite_lines,
    )
    print(f"Wrote {outputs.row_count} lines: {outputs.lines_csv}")
    print(f"Sportsbook: {outputs.sportsbook}")


def _run_init_lines(args: CliArgs) -> None:
    path = write_lines_template(
        run_date=args.date,
        lines_root=args.lines_root,
        overwrite=args.overwrite_lines,
    )
    print(f"Lines template: {path}")
    print("Edit the file: paste book line and American odds from FanDuel.")


def _run_value_props(args: CliArgs) -> None:
    outputs = value_props(
        run_date=args.date,
        reports_root=args.out_dir,
        lines_root=args.lines_root,
        lines_path=args.lines_file,
        projections_path=args.projections_file,
        config_path=args.value_config,
    )
    print(f"Wrote value sheet: {outputs.value_csv}")
    df = pd.read_csv(outputs.value_csv)
    picks = df[df["pick"] != "PASS"][["player_name", "pick", "book_line", "edge_over", "edge_under", "ev_over", "ev_under"]]
    if not picks.empty:
        print("\nFlagged plays (edge >= min_edge in config):")
        print(picks.to_string(index=False))
    else:
        print("\nNo plays met the minimum edge threshold.")


def _run_score_projections(args: CliArgs) -> None:
    outputs = score_projections(
        run_date=args.date,
        processed_root=args.processed_root,
        starters_root=args.starters_root,
        starters_path=args.starters_file,
        reports_root=args.out_dir,
        config_path=args.projection_config,
    )
    print(f"Wrote projections: {outputs.projections_csv}")


def _run_merge_features(args: CliArgs) -> None:
    outputs = merge_pitcher_features(
        run_date=args.date,
        processed_root=args.processed_root,
    )
    print("Wrote merged pitcher files:")
    print(f"  {outputs.pitcher_merged_long}")
    print(f"  {outputs.pitcher_split_summary}")
    print(f"  {outputs.pitcher_skill}")


def _run_send_email(args: CliArgs) -> None:
    send_daily_digest(
        run_date=args.date,
        reports_root=args.out_dir,
        config_path=args.email_config,
        dry_run=args.dry_run,
        run_mode=args.run_mode,
    )


def _run_lineup_refresh(args: CliArgs) -> None:
    run_lineup_refresh(
        run_date=args.date,
        out_dir=args.out_dir,
        lines_root=args.lines_root,
        starters_root=args.starters_root,
        oddstrader_config=args.oddstrader_config,
        mlb_config=args.mlb_config,
        projection_config=args.projection_config,
        value_config=args.value_config,
        fetch_odds=not args.skip_odds,
        run_mode=args.run_mode,
        record_tracker_picks=not getattr(args, "no_track_record", False),
    )


def _run_track_performance(args: CliArgs) -> None:
    outputs = track_performance(
        run_date=args.date,
        reports_root=args.out_dir,
        config_path=args.tracker_config,
        record_today=not args.no_record,
        grade_pending=not args.no_grade,
    )
    print(
        f"Ledger: {outputs.ledger_csv} "
        f"(+{outputs.picks_recorded} new, graded {outputs.picks_graded}, "
        f"{outputs.pending_count} pending)"
    )
    print(f"Summary: {outputs.summary_txt}")
    print(f"Daily rollup: {outputs.daily_rollup_csv}")
    print(f"EV rollup: {outputs.ev_rollup_csv}")
    print()
    print(outputs.summary_txt.read_text(encoding="utf-8"))


def main() -> None:
    args = _parse_args()

    if args.command == "fetch-savant":
        _run_fetch_savant(args)
        return

    if args.command == "build-features":
        _run_build_features(args)
        return

    if args.command == "merge-features":
        _run_merge_features(args)
        return

    if args.command == "validate-data":
        _run_validate_data(args)
        return

    if args.command == "run-daily":
        run_daily(
            run_date=args.date,
            out_dir=args.out_dir,
            min_pitches_per_pitch_type_row=args.min_pitches,
            lines_root=args.lines_root,
            starters_root=args.starters_root,
            oddstrader_config=args.oddstrader_config,
            mlb_config=args.mlb_config,
            projection_config=args.projection_config,
            value_config=args.value_config,
            fetch_odds=not args.skip_odds,
            run_model=not args.skip_model,
            run_mode=args.run_mode,
        )
        return

    if args.command == "run-lineup-refresh":
        _run_lineup_refresh(args)
        return

    if args.command == "init-starters":
        _run_init_starters(args)
        return

    if args.command == "score-projections":
        _run_score_projections(args)
        return

    if args.command == "init-lines":
        _run_init_lines(args)
        return

    if args.command == "fetch-odds":
        _run_fetch_odds(args)
        return

    if args.command == "value-props":
        _run_value_props(args)
        return

    if args.command == "send-email":
        _run_send_email(args)
        return

    if args.command == "track-performance":
        _run_track_performance(args)
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
