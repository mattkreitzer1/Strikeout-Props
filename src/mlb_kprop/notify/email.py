from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import date as Date
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("config/email_defaults.yaml")


@dataclass(frozen=True)
class EmailDigestOutputs:
    run_date: Date
    plays_sent: int
    recipient: str


@dataclass(frozen=True)
class FailureAlertOutputs:
    run_date: Date
    recipient: str
    validation_failures: int


def load_email_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _smtp_settings() -> dict[str, str | int]:
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    to_addr = os.environ.get("EMAIL_TO", "")
    from_addr = os.environ.get("EMAIL_FROM", user)
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "to": to_addr,
        "from": from_addr,
    }


def _deliver_message(msg: EmailMessage, smtp: dict[str, str | int], dry_run: bool) -> None:
    if dry_run:
        print("--- DRY RUN ---")
        print(f"To: {smtp['to']}")
        print(f"Subject: {msg['Subject']}")
        print(msg.get_content())
        return

    with smtplib.SMTP(str(smtp["host"]), int(smtp["port"])) as server:
        server.starttls()
        server.login(str(smtp["user"]), str(smtp["password"]))
        server.send_message(msg)


def _resolve_smtp(dry_run: bool) -> dict[str, str | int] | None:
    smtp = _smtp_settings()
    if not smtp["password"] or not smtp["to"]:
        if dry_run:
            return {
                **smtp,
                "to": smtp["to"] or "you@example.com",
                "from": smtp["from"] or "mlb-kprop@example.com",
                "password": "dry-run",
            }
        print(
            "Email skipped: set SMTP_USER, SMTP_PASSWORD, and EMAIL_TO "
            "(see README → Email to your phone)."
        )
        return None
    return smtp


def parse_validation_failures(validation_path: Path) -> list[str]:
    """Return human-readable failure lines from reports/validation_<date>.txt."""
    if not validation_path.exists():
        return []
    failures: list[str] = []
    for line in validation_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FAIL\t"):
            failures.append(line.removeprefix("FAIL\t"))
    return failures


def build_failure_body(
    run_date: Date,
    run_mode: str,
    workflow_label: str,
    validation_failures: list[str],
    run_url: str | None = None,
) -> str:
    lines = [
        f"MLB K prop pipeline FAILED — {run_date.isoformat()}",
        f"Workflow: {workflow_label} ({run_mode})",
        "",
        "The daily digest was NOT sent. Do not use today's sheet until this is fixed.",
    ]
    if run_url:
        lines.extend(["", f"GitHub Actions run: {run_url}"])

    if validation_failures:
        lines.extend(["", "Validation failures:"])
        for detail in validation_failures[:25]:
            lines.append(f"  • {detail}")
        if len(validation_failures) > 25:
            lines.append(f"  … and {len(validation_failures) - 25} more")
    else:
        lines.extend(
            [
                "",
                "No validation report found — failure may be Savant fetch,",
                "OddsTrader, MLB API, scoring, or an earlier pipeline step.",
            ]
        )

    lines.extend(["", "— mlb_kprop"])
    return "\n".join(lines)


def _failure_subject(
    cfg: dict[str, Any],
    run_date: Date,
    run_mode: str,
) -> str:
    if run_mode == "confirmed":
        template = str(
            cfg.get("subject_failure_confirmed", "MLB K props FAILED (confirmed) — {date}")
        )
    else:
        template = str(cfg.get("subject_failure_early", "MLB K props FAILED (early) — {date}"))
    return template.format(date=run_date.isoformat())


def send_failure_alert(
    run_date: Date,
    reports_root: Path = Path("reports"),
    config_path: Path = DEFAULT_CONFIG_PATH,
    dry_run: bool = False,
    run_mode: str = "early",
    workflow_label: str = "daily pipeline",
    run_url: str | None = None,
) -> FailureAlertOutputs | None:
    """
    Email when GitHub Actions (or a local run) fails before the success digest.

    Reads validation failures from reports/validation_<date>.txt when present.
    """
    cfg = load_email_config(config_path)
    smtp = _resolve_smtp(dry_run)
    if smtp is None:
        return None

    validation_path = reports_root / f"validation_{run_date.isoformat()}.txt"
    validation_failures = parse_validation_failures(validation_path)
    body = build_failure_body(
        run_date,
        run_mode,
        workflow_label,
        validation_failures,
        run_url=run_url,
    )
    subject = _failure_subject(cfg, run_date, run_mode)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = str(smtp["from"])
    msg["To"] = str(smtp["to"])
    msg.set_content(body)

    if validation_path.exists():
        msg.add_attachment(
            validation_path.read_bytes(),
            maintype="text",
            subtype="plain",
            filename=validation_path.name,
        )

    _deliver_message(msg, smtp, dry_run)
    if not dry_run:
        print(f"Sent failure alert to {smtp['to']}.")

    return FailureAlertOutputs(
        run_date=run_date,
        recipient=str(smtp["to"]),
        validation_failures=len(validation_failures),
    )


def _format_play_row(row: pd.Series) -> str:
    pick = str(row["pick"])
    edge = float(row["edge_over"] if pick == "OVER" else row["edge_under"])
    ev = float(row["ev_over"] if pick == "OVER" else row["ev_under"])
    odds = int(row["over_odds"] if pick == "OVER" else row["under_odds"])
    return (
        f"{row['player_name']}: {pick} {row['book_line']} K "
        f"(fair {row['fair_k']}, edge {edge:+.1%}, EV {ev:+.2f}, odds {odds:+d})"
    )


def build_digest_body(
    run_date: Date,
    value_df: pd.DataFrame,
    max_plays: int,
) -> tuple[str, int]:
    picks = value_df[value_df["pick"] != "PASS"].copy()
    if picks.empty:
        body = f"No plays met the edge threshold for {run_date.isoformat()}.\n"
        return body, 0

    picks["best_edge"] = picks.apply(
        lambda r: max(float(r["edge_over"]), float(r["edge_under"])),
        axis=1,
    )
    picks = picks.sort_values("best_edge", ascending=False).head(max_plays)

    lines = [
        f"MLB pitcher strikeout props — {run_date.isoformat()}",
        f"Flagged plays: {len(value_df[value_df['pick'] != 'PASS'])}",
        "",
    ]
    for _, row in picks.iterrows():
        lines.append(f"• {_format_play_row(row)}")
    lines.extend(
        [
            "",
            "Full sheet attached as CSV.",
        ]
    )

    summary_path = Path("data/tracker/summary.txt")
    if summary_path.exists():
        tracker_text = summary_path.read_text(encoding="utf-8").strip()
        if tracker_text:
            lines.extend(["", "---", tracker_text])

    lines.extend(["", "— mlb_kprop"])
    return "\n".join(lines), len(picks)


def send_daily_digest(
    run_date: Date,
    reports_root: Path = Path("reports"),
    value_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dry_run: bool = False,
    run_mode: str = "early",
) -> EmailDigestOutputs | None:
    """
    Email today's flagged K prop plays.

    Required environment variables:
      SMTP_USER, SMTP_PASSWORD, EMAIL_TO
    Optional:
      SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587), EMAIL_FROM
    """
    cfg = load_email_config(config_path)
    smtp = _resolve_smtp(dry_run)
    if smtp is None:
        return None

    path = value_path or reports_root / f"value_{run_date.isoformat()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m mlb_kprop run-daily` first."
        )

    value_df = pd.read_csv(path)
    max_plays = int(cfg.get("max_plays_in_body", 15))
    body, play_count = build_digest_body(run_date, value_df, max_plays)

    subject_template = str(cfg.get("subject", "MLB K props — {date}"))
    if run_mode == "confirmed":
        subject_template = str(
            cfg.get("subject_confirmed", "MLB K props (confirmed lineups) — {date}")
        )
    elif run_mode == "early":
        subject_template = str(
            cfg.get("subject_early", "MLB K props (early) — {date}")
        )
    subject = subject_template.format(date=run_date.isoformat())

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = str(smtp["from"])
    msg["To"] = str(smtp["to"])
    msg.set_content(body)

    if cfg.get("attach_value_csv", True):
        msg.add_attachment(
            path.read_bytes(),
            maintype="text",
            subtype="csv",
            filename=path.name,
        )

    if dry_run:
        _deliver_message(msg, smtp, dry_run=True)
        return EmailDigestOutputs(
            run_date=run_date,
            plays_sent=play_count,
            recipient=str(smtp["to"]),
        )

    _deliver_message(msg, smtp, dry_run=False)
    print(f"Sent digest to {smtp['to']} ({play_count} plays in body).")
    return EmailDigestOutputs(
        run_date=run_date,
        plays_sent=play_count,
        recipient=str(smtp["to"]),
    )
