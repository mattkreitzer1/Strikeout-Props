from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd
import requests

from mlb_kprop.odds.oddstrader import (
    DEFAULT_CONFIG_PATH,
    OddsTraderClient,
    load_oddstrader_auth,
    load_oddstrader_config,
)
from mlb_kprop.projections.lines import LINES_COLUMNS, lines_path_for_date


@dataclass(frozen=True)
class FetchOddsOutputs:
    lines_csv: Path
    row_count: int
    sportsbook: str


def fetch_oddstrader_lines(
    run_date: Date,
    lines_root: Path = Path("data/lines"),
    config_path: Path = DEFAULT_CONFIG_PATH,
    overwrite: bool = False,
) -> FetchOddsOutputs:
    """
    Pull pitcher strikeout O/U lines from OddsTrader GraphQL into data/lines/<date>.csv.
    """
    config = load_oddstrader_config(config_path)
    lines_root.mkdir(parents=True, exist_ok=True)
    out_path = lines_path_for_date(run_date, lines_root=lines_root)

    if out_path.exists() and not overwrite:
        existing = pd.read_csv(out_path)
        return FetchOddsOutputs(
            lines_csv=out_path,
            row_count=len(existing),
            sportsbook=config.sportsbook_name,
        )

    session = requests.Session()
    auth = load_oddstrader_auth(session=session, props_page_url=config.props_page_url)
    client = OddsTraderClient(auth=auth, config=config, session=session)
    strikeout_lines = client.fetch_strikeout_lines_for_date(run_date)

    if not strikeout_lines:
        raise ValueError(
            f"No strikeout props returned for {run_date.isoformat()} "
            f"(mtid={config.strikeout_mtid}, paid={config.sportsbook_paid})."
        )

    rows = [
        {
            "player_id": line.player_id,
            "player_name": line.player_name,
            "book_line": line.book_line,
            "over_odds": line.over_odds,
            "under_odds": line.under_odds,
            "notes": f"{line.sportsbook} | {line.matchup} | eid={line.event_id}",
        }
        for line in strikeout_lines
    ]
    df = pd.DataFrame(rows, columns=LINES_COLUMNS)
    df = df.sort_values("player_name").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    return FetchOddsOutputs(
        lines_csv=out_path,
        row_count=len(df),
        sportsbook=config.sportsbook_name,
    )
