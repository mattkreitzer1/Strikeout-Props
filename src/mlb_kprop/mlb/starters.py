from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from mlb_kprop.projections.names import mlb_full_name_to_savant
from mlb_kprop.projections.starters import STARTERS_COLUMNS, starters_path_for_date

DEFAULT_CONFIG_PATH = Path("config/mlb_defaults.yaml")
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_USER_AGENT = "mlb-kprop/0.1 (personal research; statsapi.mlb.com)"


@dataclass(frozen=True)
class StarterRow:
    player_id: int
    player_name: str
    pitcher_throws: str
    opp_lhb_pct: float
    batters_faced: float | None
    notes: str


@dataclass(frozen=True)
class SyncStartersOutputs:
    starters_csv: Path
    row_count: int


def load_mlb_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class MlbStatsClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers["User-Agent"] = _USER_AGENT
        self._people_cache: dict[int, dict[str, Any]] = {}
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        delay = float(self.config.get("seconds_between_requests", 0.15))
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        response = self.session.get(url, params=params, timeout=45)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response.json()

    def fetch_person(self, player_id: int) -> dict[str, Any]:
        if player_id not in self._people_cache:
            payload = self._get(f"{MLB_API_BASE}/people/{player_id}")
            people = payload.get("people") or []
            if not people:
                raise ValueError(f"No MLB person record for id={player_id}")
            self._people_cache[player_id] = people[0]
        return self._people_cache[player_id]

    def pitcher_throws_code(self, player_id: int) -> str:
        person = self.fetch_person(player_id)
        code = (person.get("pitchHand") or {}).get("code", "")
        return code if code in ("R", "L") else ""

    def effective_bat_side(self, batter_id: int, pitcher_throws: str) -> str:
        """
        Return L or R for platoon purposes (switch hitters bat opposite the pitcher).
        """
        person = self.fetch_person(batter_id)
        code = (person.get("batSide") or {}).get("code", "")
        if code == "L":
            return "L"
        if code == "R":
            return "R"
        if code == "S":
            return "L" if pitcher_throws == "R" else "R"
        return "R"

    def schedule_games(self, run_date: Date) -> list[dict[str, Any]]:
        hydrate = str(self.config.get("schedule_hydrate", "probablePitcher"))
        payload = self._get(
            f"{MLB_API_BASE}/schedule",
            {
                "sportId": 1,
                "date": run_date.isoformat(),
                "hydrate": hydrate,
            },
        )
        dates = payload.get("dates") or []
        if not dates:
            return []
        return dates[0].get("games") or []

    def batting_order(self, game_pk: int) -> dict[str, list[int]]:
        payload = self._get(f"{MLB_API_BASE}.1/game/{game_pk}/feed/live")
        teams = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
        result: dict[str, list[int]] = {}
        for side in ("home", "away"):
            order = teams.get(side, {}).get("battingOrder") or []
            result[side] = [int(b) for b in order if b]
        return result

    def opp_lhb_pct_for_lineup(
        self,
        batter_ids: list[int],
        pitcher_throws: str,
    ) -> float | None:
        min_batters = int(self.config.get("min_lineup_batters", 5))
        if len(batter_ids) < min_batters:
            return None
        if pitcher_throws not in ("R", "L"):
            return None

        lhb = 0
        for batter_id in batter_ids:
            if self.effective_bat_side(batter_id, pitcher_throws) == "L":
                lhb += 1
        return lhb / len(batter_ids)


def _probable_pitcher(team: dict[str, Any]) -> dict[str, Any] | None:
    probable = team.get("probablePitcher") or {}
    if probable.get("id"):
        return probable
    return None


def build_starters_for_date(
    run_date: Date,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[StarterRow]:
    config = load_mlb_config(config_path)
    client = MlbStatsClient(config)
    default_opp_lhb = float(config.get("default_opp_lhb_pct", 0.40))

    rows: list[StarterRow] = []
    for game in client.schedule_games(run_date):
        game_pk = int(game["gamePk"])
        away = game["teams"]["away"]
        home = game["teams"]["home"]
        away_name = away["team"]["name"]
        home_name = home["team"]["name"]
        matchup = f"{away_name} @ {home_name}"

        try:
            orders = client.batting_order(game_pk)
        except requests.HTTPError:
            orders = {"home": [], "away": []}

        for pitching_team, batting_team, batting_side in (
            (away, home, "home"),
            (home, away, "away"),
        ):
            probable = _probable_pitcher(pitching_team)
            if not probable:
                continue

            pid = int(probable["id"])
            savant_name = mlb_full_name_to_savant(probable.get("fullName", ""))
            throws = client.pitcher_throws_code(pid)

            opp_lhb = client.opp_lhb_pct_for_lineup(
                orders.get(batting_side, []),
                throws,
            )
            opp_source = "lineup"
            if opp_lhb is None:
                opp_lhb = default_opp_lhb
                opp_source = "default"

            rows.append(
                StarterRow(
                    player_id=pid,
                    player_name=savant_name,
                    pitcher_throws=throws,
                    opp_lhb_pct=round(float(opp_lhb), 4),
                    batters_faced=None,
                    notes=f"{matchup} | opp_lhb_{opp_source}",
                )
            )

    return rows


def sync_starters_from_mlb(
    run_date: Date,
    starters_root: Path = Path("data/starters"),
    config_path: Path = DEFAULT_CONFIG_PATH,
    overwrite: bool = True,
) -> SyncStartersOutputs:
    """Write data/starters/<date>.csv from MLB probables + lineup LHB share."""
    starters_root.mkdir(parents=True, exist_ok=True)
    path = starters_path_for_date(run_date, starters_root=starters_root)

    if path.exists() and not overwrite:
        existing = pd.read_csv(path)
        return SyncStartersOutputs(starters_csv=path, row_count=len(existing))

    starter_rows = build_starters_for_date(run_date, config_path=config_path)
    if not starter_rows:
        raise ValueError(
            f"No probable pitchers found for {run_date.isoformat()} on MLB schedule."
        )

    df = pd.DataFrame(
        [
            {
                "player_id": row.player_id,
                "player_name": row.player_name,
                "pitcher_throws": row.pitcher_throws,
                "opp_lhb_pct": row.opp_lhb_pct,
                "batters_faced": row.batters_faced if row.batters_faced is not None else "",
                "notes": row.notes,
            }
            for row in starter_rows
        ],
        columns=STARTERS_COLUMNS,
    )
    df = df.sort_values("player_name").reset_index(drop=True)
    df.to_csv(path, index=False)

    return SyncStartersOutputs(starters_csv=path, row_count=len(df))
