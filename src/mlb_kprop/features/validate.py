from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import pandas as pd

from mlb_kprop.features.merge import _weighted_mean
from mlb_kprop.savant.fetch import load_sources

PLATOON_SOURCE_IDS = (
    "pitcher_R_vs_R",
    "pitcher_R_vs_L",
    "pitcher_L_vs_R",
    "pitcher_L_vs_L",
)
PLATOON_SPLITS = ("R_vs_R", "R_vs_L", "L_vs_R", "L_vs_L")
BATTER_HAND_SPLITS = ("vs_RHP", "vs_LHP")
BATTER_PITCH_SOURCE_IDS = ("batter_vs_RHP_pitch_type", "batter_vs_LHP_pitch_type")

PLATOON_RAW_REQUIRED = (
    "player_id",
    "player_name",
    "pitch_type",
    "pitches",
    "k_percent",
    "bb_percent",
    "xwoba",
    "swing_miss_percent",
)
PLATOON_PROC_REQUIRED = (
    "player_id",
    "player_name",
    "platoon_split",
    "pitch_type",
    "pitches",
    "k_percent",
    "bb_percent",
    "xwoba",
    "swing_miss_percent",
)
CUSTOM_RAW_REQUIRED = ("player_id", "pa", "k_percent")
RATE_COLS = ("k_percent", "bb_percent", "swing_miss_percent")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    run_date: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _record(report: ValidationReport, name: str, ok: bool, detail: str = "") -> None:
    report.checks.append(CheckResult(name=name, passed=ok, detail=detail))


def _looks_like_html(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:500].lower()
    return "<html" in head or "<!doctype" in head


def _split_from_source_id(source_id: str) -> str:
    if source_id.startswith("pitcher_") and "_vs_" in source_id:
        return source_id.replace("pitcher_", "", 1)
    return "unknown"


def validate_daily_data(
    run_date: Date,
    config_path: Path = Path("config/savant_sources.yaml"),
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
    reports_root: Path = Path("reports"),
    min_pitches_per_pitch_type_row: int = 100,
    weighted_k_tolerance: float = 0.01,
) -> ValidationReport:
    """
    Run automated sanity checks on raw downloads and processed tables for one day.
    """
    report = ValidationReport(run_date=run_date.isoformat())
    day_raw = raw_root / run_date.isoformat()
    day_proc = processed_root / run_date.isoformat()

    if not day_raw.is_dir():
        _record(report, "raw_folder_exists", False, f"Missing {day_raw}")
        return report
    _record(report, "raw_folder_exists", True)

    if not day_proc.is_dir():
        _record(report, "processed_folder_exists", False, f"Missing {day_proc}")
        return report
    _record(report, "processed_folder_exists", True)

    # --- Config: every source file present and non-empty ---
    try:
        sources = load_sources(config_path)
    except Exception as exc:  # noqa: BLE001 — surface config errors clearly
        _record(report, "config_load", False, str(exc))
        return report
    _record(report, "config_load", True, f"{len(sources)} sources in config")

    manifest_path = reports_root / f"savant_fetch_{run_date.isoformat()}.csv"
    manifest_rows: dict[str, int] = {}
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        for _, row in manifest_df.iterrows():
            manifest_rows[str(row["source_id"])] = int(row["row_count"])

    for source in sources:
        csv_path = day_raw / f"{source.id}.csv"
        check_name = f"raw/{source.id}"
        if not csv_path.exists():
            _record(report, check_name, False, "file missing")
            continue
        if _looks_like_html(csv_path):
            _record(report, check_name, False, "looks like HTML, not CSV")
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            _record(report, check_name, False, "zero rows")
            continue
        if source.id in manifest_rows and len(df) != manifest_rows[source.id]:
            _record(
                report,
                check_name,
                False,
                f"row count {len(df)} != manifest {manifest_rows[source.id]}",
            )
            continue
        _record(report, check_name, True, f"{len(df)} rows")

    # --- Platoon raw: required columns + filter consistency per split ---
    for source_id in PLATOON_SOURCE_IDS:
        path = day_raw / f"{source_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        missing = [c for c in PLATOON_RAW_REQUIRED if c not in df.columns]
        _record(
            report,
            f"raw/{source_id}/columns",
            not missing,
            f"missing: {missing}" if missing else "ok",
        )
        if "pitches" in df.columns:
            df = df.copy()
            df["pitches"] = pd.to_numeric(df["pitches"], errors="coerce")
            expected = int((df["pitches"] >= min_pitches_per_pitch_type_row).sum())
            split = _split_from_source_id(source_id)
            proc_path = day_proc / "pitcher_platoon_pitch_type.csv"
            if proc_path.exists():
                platoon = pd.read_csv(proc_path)
                actual = int((platoon["platoon_split"] == split).sum())
                _record(
                    report,
                    f"processed/platoon_row_count/{split}",
                    actual == expected,
                    f"processed={actual}, raw>={min_pitches_per_pitch_type_row}={expected}",
                )

    # --- Processed platoon ---
    platoon_path = day_proc / "pitcher_platoon_pitch_type.csv"
    if not platoon_path.exists():
        _record(report, "processed/pitcher_platoon_pitch_type", False, "file missing")
    else:
        platoon = pd.read_csv(platoon_path)
        missing = [c for c in PLATOON_PROC_REQUIRED if c not in platoon.columns]
        _record(
            report,
            "processed/platoon_columns",
            not missing,
            f"missing: {missing}" if missing else "ok",
        )

        below_min = int((platoon["pitches"] < min_pitches_per_pitch_type_row).sum())
        _record(
            report,
            "processed/platoon_min_pitches",
            below_min == 0,
            f"{below_min} rows below {min_pitches_per_pitch_type_row}",
        )

        dup = int(
            platoon.duplicated(["player_id", "platoon_split", "pitch_type"]).sum()
        )
        _record(report, "processed/platoon_no_duplicates", dup == 0, f"{dup} duplicates")

        bad_splits = set(platoon["platoon_split"].unique()) - set(PLATOON_SPLITS)
        _record(
            report,
            "processed/platoon_splits",
            not bad_splits,
            f"unexpected splits: {sorted(bad_splits)}" if bad_splits else "ok",
        )

        for col in RATE_COLS:
            if col not in platoon.columns:
                continue
            nulls = int(platoon[col].isna().sum())
            _record(report, f"processed/platoon_{col}_not_null", nulls == 0, f"{nulls} nulls")
            out_of_range = platoon[col].dropna()
            bad = ((out_of_range < 0) | (out_of_range > 100)).sum()
            _record(
                report,
                f"processed/platoon_{col}_range",
                bad == 0,
                f"{bad} values outside 0–100",
            )

        n_pitchers = platoon["player_id"].nunique()
        _record(
            report,
            "processed/platoon_pitcher_count",
            n_pitchers >= 100,
            f"{n_pitchers} unique pitchers",
        )

    # --- Custom processed ---
    for label, filename in (
        ("pitcher", "pitcher_custom.csv"),
        ("batter", "batter_custom.csv"),
    ):
        path = day_proc / filename
        if not path.exists():
            _record(report, f"processed/{filename}", False, "file missing")
            continue
        custom = pd.read_csv(path)
        if len(custom) == 0:
            _record(report, f"processed/{filename}", False, "zero rows")
            continue
        dup_ids = (
            int(custom.duplicated(["player_id", "year"]).sum())
            if "year" in custom.columns
            else int(custom["player_id"].duplicated().sum())
        )
        _record(
            report,
            f"processed/{label}_unique_players",
            dup_ids == 0,
            f"{dup_ids} duplicate player rows",
        )
        if "year" in custom.columns:
            years = set(pd.to_numeric(custom["year"], errors="coerce").dropna().astype(int))
            _record(
                report,
                f"processed/{label}_seasons",
                years >= {2025, 2026},
                f"years present: {sorted(years)}",
            )
        if "pa" in custom.columns:
            bad_pa = int((pd.to_numeric(custom["pa"], errors="coerce") <= 0).sum())
            _record(
                report,
                f"processed/{label}_pa_positive",
                bad_pa == 0,
                f"{bad_pa} rows with pa <= 0",
            )
        _record(report, f"processed/{filename}", True, f"{len(custom)} rows")

    # --- Merge outputs ---
    merged_path = day_proc / "pitcher_merged_long.csv"
    summary_path = day_proc / "pitcher_split_summary.csv"
    if not platoon_path.exists():
        return report

    platoon = pd.read_csv(platoon_path)

    if not merged_path.exists():
        _record(report, "processed/pitcher_merged_long", False, "file missing")
    else:
        merged = pd.read_csv(merged_path)
        _record(
            report,
            "merge/long_row_count",
            len(merged) == len(platoon),
            f"merged={len(merged)}, platoon={len(platoon)}",
        )
        custom_col = "custom_k_percent"
        if custom_col in merged.columns:
            custom_path = day_proc / "pitcher_custom.csv"
            if custom_path.exists():
                custom = pd.read_csv(custom_path)
                expected_rows = platoon["player_id"].isin(custom["player_id"]).sum()
                actual_rows = int(merged[custom_col].notna().sum())
                _record(
                    report,
                    "merge/custom_attach_count",
                    actual_rows == expected_rows,
                    f"attached={actual_rows}, expected={expected_rows}",
                )

    if not summary_path.exists():
        _record(report, "processed/pitcher_split_summary", False, "file missing")
    else:
        summary = pd.read_csv(summary_path)
        expected_groups = platoon.groupby(
            ["player_id", "platoon_split"], dropna=False
        ).ngroups
        _record(
            report,
            "merge/summary_row_count",
            len(summary) == expected_groups,
            f"summary={len(summary)}, expected={expected_groups}",
        )

        mismatches = 0
        for keys, group in platoon.groupby(
            ["player_id", "platoon_split"], dropna=False
        ):
            player_id, platoon_split = keys
            expected_k = _weighted_mean(group["k_percent"], group["pitches"])
            rows = summary[
                (summary["player_id"] == player_id)
                & (summary["platoon_split"] == platoon_split)
            ]
            if rows.empty:
                mismatches += 1
                continue
            actual_k = float(rows["k_percent"].iloc[0])
            if pd.isna(expected_k) or pd.isna(actual_k):
                if pd.isna(expected_k) != pd.isna(actual_k):
                    mismatches += 1
            elif abs(expected_k - actual_k) > weighted_k_tolerance:
                mismatches += 1

        _record(
            report,
            "merge/weighted_k_percent",
            mismatches == 0,
            f"{mismatches} split(s) off by > {weighted_k_tolerance}",
        )

    batter_hand_path = day_proc / "batter_hand_summary.csv"
    if not batter_hand_path.exists():
        _record(report, "processed/batter_hand_summary", False, "file missing")
    else:
        bh = pd.read_csv(batter_hand_path)
        _record(
            report,
            "processed/batter_hand_summary",
            len(bh) > 0,
            f"{len(bh)} rows",
        )
        bad = set(bh.get("hand_split", pd.Series(dtype=str)).unique()) - set(BATTER_HAND_SPLITS)
        _record(
            report,
            "processed/batter_hand_splits",
            not bad or bh.empty,
            f"unexpected: {sorted(bad)}" if bad else "ok",
        )

    batter_pitch_path = day_proc / "batter_platoon_pitch_type.csv"
    if not batter_pitch_path.exists():
        _record(report, "processed/batter_platoon_pitch_type", False, "file missing")
    else:
        bp = pd.read_csv(batter_pitch_path)
        _record(
            report,
            "processed/batter_platoon_pitch_type",
            len(bp) > 0,
            f"{len(bp)} rows",
        )
        bad = set(bp.get("hand_split", pd.Series(dtype=str)).unique()) - set(
            BATTER_HAND_SPLITS
        )
        _record(
            report,
            "processed/batter_pitch_splits",
            not bad or bp.empty,
            f"unexpected: {sorted(bad)}" if bad else "ok",
        )
        dup = int(bp.duplicated(["player_id", "hand_split", "pitch_type"]).sum())
        _record(
            report,
            "processed/batter_pitch_no_duplicates",
            dup == 0,
            f"{dup} duplicates",
        )

    skill_path = day_proc / "pitcher_skill.csv"
    if not skill_path.exists():
        _record(report, "processed/pitcher_skill", False, "file missing")
    else:
        skill = pd.read_csv(skill_path)
        _record(
            report,
            "processed/pitcher_skill",
            len(skill) > 0,
            f"{len(skill)} pitchers",
        )

    return report


def write_validation_report(
    report: ValidationReport,
    output_path: Path,
) -> Path:
    """Save pass/fail lines to a text report for the day."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# validation {report.run_date}",
        f"overall: {'PASS' if report.passed else 'FAIL'}",
        "",
    ]
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        suffix = f" — {check.detail}" if check.detail else ""
        lines.append(f"{status}\t{check.name}{suffix}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def print_validation_report(report: ValidationReport) -> None:
    """Print a human-readable summary to the terminal."""
    print(f"\nValidation for {report.run_date}")
    print("-" * 50)
    for check in report.checks:
        mark = "ok" if check.passed else "FAIL"
        suffix = f" ({check.detail})" if check.detail else ""
        print(f"  [{mark}] {check.name}{suffix}")
    print("-" * 50)
    if report.passed:
        print(f"All {len(report.checks)} checks passed.")
    else:
        print(f"{len(report.failures)} of {len(report.checks)} checks failed.")
