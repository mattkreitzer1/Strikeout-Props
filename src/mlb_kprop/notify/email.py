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
) -> EmailDigestOutputs | None:
    """
    Email today's flagged K prop plays.

    Required environment variables:
      SMTP_USER, SMTP_PASSWORD, EMAIL_TO
    Optional:
      SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587), EMAIL_FROM
    """
    cfg = load_email_config(config_path)
    smtp = _smtp_settings()

    if not smtp["password"] or not smtp["to"]:
        if dry_run:
            smtp = {
                **smtp,
                "to": smtp["to"] or "you@example.com",
                "from": smtp["from"] or "mlb-kprop@example.com",
                "password": "dry-run",
            }
        else:
            print(
                "Email skipped: set SMTP_USER, SMTP_PASSWORD, and EMAIL_TO "
                "(see README → Email to your phone)."
            )
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
        print("--- DRY RUN ---")
        print(f"To: {smtp['to']}")
        print(f"Subject: {subject}")
        print(body)
        return EmailDigestOutputs(
            run_date=run_date,
            plays_sent=play_count,
            recipient=str(smtp["to"]),
        )

    with smtplib.SMTP(str(smtp["host"]), int(smtp["port"])) as server:
        server.starttls()
        server.login(str(smtp["user"]), str(smtp["password"]))
        server.send_message(msg)

    print(f"Sent digest to {smtp['to']} ({play_count} plays in body).")
    return EmailDigestOutputs(
        run_date=run_date,
        plays_sent=play_count,
        recipient=str(smtp["to"]),
    )
