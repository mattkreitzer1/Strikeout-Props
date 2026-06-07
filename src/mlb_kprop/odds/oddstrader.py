from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import requests
import yaml

DEFAULT_CONFIG_PATH = Path("config/oddstrader.yaml")
DEFAULT_PROPS_PAGE_URL = "https://www.oddstrader.com/mlb/player-props/?m=766"
DEFAULT_MATCHUPS_PAGE_URL = "https://www.oddstrader.com/mlb/matchups/"
_CONFIG_PATTERN = re.compile(r"window\.__config\s*=\s*(\{.+?\});", re.DOTALL)
_MATCHUP_EID_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})-e(\d{7})")
_USER_AGENT = (
    "mlb-kprop/0.1 (personal research; oddstrader graphql; not for redistribution)"
)

EVENTS_QUERY = """
query EventsWithStarters($eid: [Int]!) {
  events(eid: $eid) {
    eid
    des
    participants {
      startingPitcher {
        pid
        fn
        lnam
      }
    }
  }
}
"""

BEST_LINES_QUERY = """
query StrikeoutBestLines($eid: [Int]!, $mtid: [Int]!, $paid: [Int]!) {
  bestLines(eid: $eid, mtid: $mtid, paid: $paid)
}
"""


@dataclass(frozen=True)
class OddsTraderConfig:
    strikeout_mtid: int
    sportsbook_paid: int
    sportsbook_name: str
    props_page_url: str
    matchups_page_url: str
    seconds_between_requests: float


@dataclass(frozen=True)
class OddsTraderAuth:
    graphql_url: str
    authorization: str
    access_policy: str


@dataclass(frozen=True)
class StrikeoutLine:
    player_id: int
    player_name: str
    book_line: float
    over_odds: int
    under_odds: int
    event_id: int
    matchup: str
    sportsbook: str


def load_oddstrader_config(config_path: Path = DEFAULT_CONFIG_PATH) -> OddsTraderConfig:
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return OddsTraderConfig(
        strikeout_mtid=int(raw["strikeout_mtid"]),
        sportsbook_paid=int(raw["sportsbook_paid"]),
        sportsbook_name=str(raw.get("sportsbook_name", "")),
        props_page_url=str(raw["props_page_url"]),
        matchups_page_url=str(raw["matchups_page_url"]),
        seconds_between_requests=float(raw.get("seconds_between_requests", 0.35)),
    )


def _parse_page_config(html: str) -> dict[str, Any]:
    match = _CONFIG_PATTERN.search(html)
    if not match:
        raise ValueError("OddsTrader page did not include window.__config")
    return json.loads(match.group(1))


def load_oddstrader_auth(
    session: requests.Session | None = None,
    props_page_url: str | None = None,
) -> OddsTraderAuth:
    """Guest JWT + headers from the public OddsTrader props page."""
    client = session or requests.Session()
    url = props_page_url or DEFAULT_PROPS_PAGE_URL

    response = client.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=45,
    )
    response.raise_for_status()

    cfg = _parse_page_config(response.text)
    gateway = cfg["apigateway"]["headers"]
    odds_base = cfg["dataProvider"]["oddsV2"].rstrip("/")

    return OddsTraderAuth(
        graphql_url=f"{odds_base}/graphql",
        authorization=str(gateway["Authorization"]),
        access_policy=str(gateway["X-Access-Policy"]),
    )


def discover_event_ids_for_date(
    run_date: Date,
    session: requests.Session | None = None,
    matchups_page_url: str | None = None,
) -> list[int]:
    """
    Parse OddsTrader MLB matchups HTML for event ids (eid) on the given date.

    URLs look like: .../matchups/...-2026-05-30-e4782656/...
    """
    client = session or requests.Session()
    url = matchups_page_url or DEFAULT_MATCHUPS_PAGE_URL
    response = client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=45)
    response.raise_for_status()

    target = run_date.isoformat()
    eids: list[int] = []
    seen: set[int] = set()
    for date_str, eid_str in _MATCHUP_EID_PATTERN.findall(response.text):
        if date_str != target:
            continue
        eid = int(eid_str)
        if eid not in seen:
            seen.add(eid)
            eids.append(eid)
    return eids


def _savant_style_name(first: str, last: str) -> str:
    return f"{last.strip()}, {first.strip()}"


class OddsTraderClient:
    def __init__(
        self,
        auth: OddsTraderAuth,
        config: OddsTraderConfig,
        session: requests.Session | None = None,
    ) -> None:
        self.auth = auth
        self.config = config
        self.session = session or requests.Session()
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        delay = self.config.seconds_between_requests
        if delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        response = self.session.post(
            self.auth.graphql_url,
            headers={
                "Authorization": self.auth.authorization,
                "X-Access-Policy": self.auth.access_policy,
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
            json={"query": query, "variables": variables or {}},
            timeout=45,
        )
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            messages = "; ".join(
                err.get("message", str(err)) for err in payload["errors"]
            )
            raise RuntimeError(f"OddsTrader GraphQL error: {messages}")
        return payload

    def fetch_strikeout_lines_for_event(self, event_id: int) -> list[StrikeoutLine]:
        events_payload = self.graphql(EVENTS_QUERY, {"eid": [event_id]})
        events = events_payload.get("data", {}).get("events") or []
        if not events:
            return []

        event = events[0]
        matchup = str(event.get("des", ""))
        pitchers: dict[int, str] = {}
        for participant in event.get("participants") or []:
            starter = participant.get("startingPitcher")
            if not starter:
                continue
            pid = int(starter["pid"])
            pitchers[pid] = _savant_style_name(
                str(starter.get("fn", "")),
                str(starter.get("lnam", "")),
            )

        lines_payload = self.graphql(
            BEST_LINES_QUERY,
            {
                "eid": [event_id],
                "mtid": [self.config.strikeout_mtid],
                "paid": [self.config.sportsbook_paid],
            },
        )
        raw_lines = lines_payload.get("data", {}).get("bestLines") or []
        if not raw_lines:
            return []

        by_player: dict[int, dict[str, dict[str, Any]]] = {}
        for row in raw_lines:
            player_id = int(row["entrid"])
            side = "over" if float(row["sort"]) < 0 else "under"
            by_player.setdefault(player_id, {})[side] = row

        results: list[StrikeoutLine] = []
        book_name = self.config.sportsbook_name or f"paid:{self.config.sportsbook_paid}"
        for player_id, sides in by_player.items():
            over = sides.get("over")
            under = sides.get("under")
            if not over or not under:
                continue
            name = pitchers.get(player_id)
            if not name:
                continue
            results.append(
                StrikeoutLine(
                    player_id=player_id,
                    player_name=name,
                    book_line=float(over["adj"]),
                    over_odds=int(over["ap"]),
                    under_odds=int(under["ap"]),
                    event_id=event_id,
                    matchup=matchup,
                    sportsbook=book_name,
                )
            )
        return results

    def fetch_strikeout_lines_for_date(self, run_date: Date) -> list[StrikeoutLine]:
        event_ids = discover_event_ids_for_date(
            run_date,
            session=self.session,
            matchups_page_url=self.config.matchups_page_url,
        )
        if not event_ids:
            raise ValueError(
                f"No OddsTrader matchup eids found for {run_date.isoformat()} "
                f"on {self.config.matchups_page_url}"
            )

        all_lines: list[StrikeoutLine] = []
        for event_id in event_ids:
            all_lines.extend(self.fetch_strikeout_lines_for_event(event_id))
        return all_lines
