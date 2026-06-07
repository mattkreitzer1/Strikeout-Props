from __future__ import annotations

import time
from datetime import date as Date

import requests

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_USER_AGENT = "mlb-kprop/0.1 (personal research; statsapi.mlb.com)"


class MlbGameStatsClient:
    def __init__(self, seconds_between_requests: float = 0.15) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = _USER_AGENT
        self.delay = seconds_between_requests
        self._last_request_at = 0.0
        self._schedule_cache: dict[str, list[dict]] = {}

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _get(self, url: str, params: dict | None = None) -> dict:
        self._throttle()
        response = self.session.get(url, params=params, timeout=45)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response.json()

    def games_on_date(self, slate_date: Date) -> list[dict]:
        key = slate_date.isoformat()
        if key not in self._schedule_cache:
            payload = self._get(
                f"{MLB_API_BASE}/schedule",
                {"sportId": 1, "date": key, "hydrate": "team"},
            )
            dates = payload.get("dates") or []
            self._schedule_cache[key] = dates[0].get("games") if dates else []
        return self._schedule_cache[key]

    def all_games_final(self, slate_date: Date) -> bool:
        games = self.games_on_date(slate_date)
        if not games:
            return False
        return all(
            g.get("status", {}).get("abstractGameState") == "Final" for g in games
        )

    def pitcher_strikeouts_on_date(self, player_id: int, slate_date: Date) -> int | None:
        """
        Sum strikeouts for a pitcher on a calendar date.

        Returns None if games are not all final yet, or pitcher did not appear.
        """
        games = self.games_on_date(slate_date)
        if not games:
            return None

        if not all(g.get("status", {}).get("abstractGameState") == "Final" for g in games):
            return None

        total_k = 0
        found = False
        for game in games:
            pk = int(game["gamePk"])
            feed = self._get(f"{MLB_API_BASE}.1/game/{pk}/feed/live")
            for side in ("home", "away"):
                players = (
                    feed.get("liveData", {})
                    .get("boxscore", {})
                    .get("teams", {})
                    .get(side, {})
                    .get("players", {})
                )
                player_key = f"ID{player_id}"
                if player_key not in players:
                    continue
                pitching = players[player_key].get("stats", {}).get("pitching", {})
                if not pitching or pitching.get("gamesPitched", 0) == 0:
                    continue
                total_k += int(pitching.get("strikeOuts", 0))
                found = True
        return total_k if found else None
